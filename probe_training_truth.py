# %%
import torch as t
import accelerate
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sae_lens import SAE, HookedSAETransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
from warnings import warn
import os
import time  # For timing the training loops

###############################################################################
# Probe Definition
###############################################################################
class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits

###############################################################################
# Data Helpers
###############################################################################
def train_test_split_df(df, test_size=0.2, seed=123):
    np.random.seed(seed)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int((1 - test_size) * len(shuffled))
    return shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]

def tokenize_data(df, tokenizer, text_column="statement"):
    texts = df[text_column].tolist()
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return tokenized

###############################################################################
# Last Token Extraction Helpers
###############################################################################
def get_last_token_indices(attention_mask, offset=1):
    """
    Given an attention mask of shape (batch, seq_len) where valid tokens are 1
    and padded tokens are 0, compute the index of the token `offset` positions from the end.
    
    Args:
        attention_mask: Tensor of shape (batch, seq_len) with 1s for valid tokens and 0s for padding
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    
    Returns:
        Tensor of indices for the specified token position
    """
    token_counts = attention_mask.sum(dim=1)
    indices = token_counts - offset
    # Make sure we don't go below 0 (if a sequence is too short)
    indices = t.clamp(indices, min=0)
    return indices

def extract_last_token_acts(act_tensor, attention_mask, offset=1):
    """
    Given a tensor of activations [batch, seq_len, dim] and the corresponding
    attention mask, select for each sample the activation at the specified token position.
    
    Args:
        act_tensor: Activation tensor of shape (batch, seq_len, dim)
        attention_mask: Tensor of shape (batch, seq_len) with 1s for valid tokens and 0s for padding
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    
    Returns:
        Tensor of activations at the specified position
    """
    indices = get_last_token_indices(attention_mask, offset)
    batch_indices = t.arange(act_tensor.size(0), device=act_tensor.device)
    activations = act_tensor[batch_indices, indices, :]
    return activations

###############################################################################
# Feature Generation
###############################################################################
def generate_probing_features(tokenized, model, sae, batch_size=8, device='cuda', offset=1):
    """
    Runs the model (with run_with_cache_with_saes) in batches on the tokenized input.
    For each batch it extracts the three features:
      - hook_sae_input, hook_sae_recons, and (sae_input - sae_recons)
    with the extraction done only at the specified token position.
    
    Args:
        tokenized: Tokenized input
        model: The model to run
        sae: The sparse autoencoder
        batch_size: Batch size for processing
        device: Device to use for computation
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    """
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    all_feats_input = []
    all_feats_recons = []
    all_feats_diff = []
    n = input_ids.size(0)
    for i in tqdm(range(0, n, batch_size), desc="Generating features"):
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        batch_out = model.run_with_cache_with_saes(
            batch_ids,
            saes=sae,
            names_filter=lambda name: name in [
                'blocks.19.hook_resid_post.hook_sae_input',
                'blocks.19.hook_resid_post.hook_sae_recons'
            ]
        )[1]
        act_input = extract_last_token_acts(batch_out['blocks.19.hook_resid_post.hook_sae_input'], batch_mask, offset)
        act_recons = extract_last_token_acts(batch_out['blocks.19.hook_resid_post.hook_sae_recons'], batch_mask, offset)
        act_diff = act_input - act_recons

        all_feats_input.append(act_input.detach().cpu())
        all_feats_recons.append(act_recons.detach().cpu())
        all_feats_diff.append(act_diff.detach().cpu())

    feats_input = t.cat(all_feats_input, dim=0)
    feats_recons = t.cat(all_feats_recons, dim=0)
    feats_diff = t.cat(all_feats_diff, dim=0)
    return feats_input, feats_recons, feats_diff

###############################################################################
# Probe Training and Evaluation
###############################################################################
def train_probe_model(features, labels, dim, lr=1e-2, epochs=5, batch_size=8, device='cuda'):
    """
    Trains a linear probe (a one-layer model) on the provided features to predict
    the binary labels. Returns the trained probe and a list of loss values.
    """
    probe = Probe(dim).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        probe.train()
        for batch_feats, batch_labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device).float()
            logits = probe(batch_feats)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return probe, losses

def evaluate_probe_full(probe, features, labels, device='cuda'):
    """
    Evaluates the probe on the given features and labels.
    Returns the loss and accuracy.
    """
    probe.eval()
    criterion = nn.BCEWithLogitsLoss()
    with t.no_grad():
        feats = features.to(device)
        lbls = labels.to(device).float()
        logits = probe(feats)
        loss = criterion(logits, lbls)
        preds = (logits > 0).float()
        accuracy = (preds == lbls).float().mean().item()
    return loss.item(), accuracy

###############################################################################
# Simple Test Case for Last Token Extraction
###############################################################################
def test_last_token_extraction():
    """
    This test creates a dummy tokenized batch with padded input and a dummy
    activation tensor. It then checks that only the activations corresponding to
    the last valid token are returned.
    """
    # Create dummy tokenized batch with varying sequence lengths
    dummy_input = {
        "input_ids": t.tensor([
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 0, 0, 0],
            [8, 9, 10, 11, 12, 0]
        ]),
        "attention_mask": t.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0]
        ])
    }
    # Create a dummy activations tensor with shape (batch, seq_len, dim)
    dim = 4
    dummy_acts = t.arange(3 * 6 * dim, dtype=t.float).reshape(3, 6, dim)
    # The last valid indices for each sample are: sample0 -> index 3, sample1 -> index 2, sample2 -> index 4.
    last_acts = extract_last_token_acts(dummy_acts, dummy_input["attention_mask"])
    expected_0 = dummy_acts[0, 3, :]
    expected_1 = dummy_acts[1, 2, :]
    expected_2 = dummy_acts[2, 4, :]
    assert t.allclose(last_acts[0], expected_0), "Test failed for sample 0"
    assert t.allclose(last_acts[1], expected_1), "Test failed for sample 1"
    assert t.allclose(last_acts[2], expected_2), "Test failed for sample 2"
    print("Test last_token_extraction passed.")


# Get active latents and their activations

def record_active_latents(tokenized, model, sae, batch_size=8, device='cuda', offset=1):
    """
    Runs the model in batches and records which latents are active (nonzero) at the specified token position,
    along with their activation values.
    
    Args:
        tokenized: Tokenized input
        model: The model to run
        sae: The sparse autoencoder
        batch_size: Batch size for processing
        device: Device to use for computation
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)
    
    Returns:
    - List of tuples (nonzero_indices, activation_values) for each input sample
    """
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    results = []
    n = input_ids.size(0)
    
    for i in tqdm(range(0, n, batch_size), desc="Recording active latents"):
        batch_ids = input_ids[i:i + batch_size]
        batch_mask = attention_mask[i:i + batch_size]
        
        # Get activations from the model
        batch_out = model.run_with_cache_with_saes(
            batch_ids,
            saes=sae,
            names_filter=lambda name: name in ['blocks.19.hook_resid_post.hook_sae_acts_post']
        )[1]
        
        # Extract specified token activations
        acts = extract_last_token_acts(
            batch_out['blocks.19.hook_resid_post.hook_sae_acts_post'], 
            batch_mask,
            offset
        )
        
        # Process each sample in the batch
        for sample_acts in acts:
            # Find nonzero indices and their values
            nonzero_mask = sample_acts != 0
            nonzero_indices = nonzero_mask.nonzero().squeeze(-1)
            nonzero_values = sample_acts[nonzero_mask]
            
            # Convert to CPU and regular Python types for storage
            results.append((
                nonzero_indices.cpu().tolist(),
                nonzero_values.cpu().tolist()
            ))
    
    return results


# %%

def run_probing_pipeline(df, tokenized_all, model, sae, device, 
                        label_columns, features_map,
                        n_seeds=50, save_probes_count=20, 
                        probe_save_dir="trained_probes", 
                        results_csv="probe_results.csv",
                        similarities_csv="probe_similarities.csv"):
    """
    Runs the complete probing pipeline: training probes, evaluating them,
    and computing similarities between probe weight vectors.
    
    Args:
        df: DataFrame with the dataset
        tokenized_all: Tokenized input data
        model: The model to use
        sae: The sparse autoencoder
        device: Device to use for computation
        label_columns: List of columns to use as labels
        features_map: Dictionary mapping feature types to feature tensors
        n_seeds: Number of seeds to train for
        save_probes_count: Number of seeds for which to save probes
        probe_save_dir: Directory to save trained probes
        results_csv: Filename for the results CSV
        similarities_csv: Filename for the similarities CSV
    """
    os.makedirs(probe_save_dir, exist_ok=True)
    
    results = []
    n = df.shape[0]
    train_size = int(0.8 * n)  # 80% for training
    
    for seed in range(n_seeds):
        print(f"\nStarting training loop with seed {seed}...")
        start_time = time.time()
        
        # Compute a train/test split for the entire dataset using this seed
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Loop over each label
        for label_col in label_columns:
            # Prepare the labels (same for all feature types)
            labels_all = t.tensor(df[label_col].values)
            train_labels = labels_all[train_indices]
            test_labels = labels_all[test_indices]
            
            # Dictionary to hold the probes for the three feature types for cosine similarity computation
            probes_for_label = {}
            # Temporary storage for the per-probe results for this seed & label
            temp_results = {}
            
            # Train a probe for each feature type
            for feature_type, feats_all in features_map.items():
                train_feats = feats_all[train_indices]
                test_feats = feats_all[test_indices]
                
                # Set the torch seed to ensure probe initialization consistency
                t.manual_seed(seed)
                probe, _ = train_probe_model(
                    train_feats, train_labels, dim=train_feats.size(1),
                    epochs=2, batch_size=8, device=device, lr=0.005
                )
                train_loss, train_acc = evaluate_probe_full(probe, train_feats, train_labels, device=device)
                test_loss, test_acc = evaluate_probe_full(probe, test_feats, test_labels, device=device)
                weight_norm = probe.net.weight.norm().item()
                
                probes_for_label[feature_type] = probe
                temp_results[feature_type] = {
                    "Seed": seed,
                    "Feature Type": feature_type,
                    "Label": label_col,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "Weight Norm": weight_norm
                }
                
                # Save the probe if this seed is among the first N
                if seed < save_probes_count:
                    safe_label = label_col.replace(' ', '_')
                    model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                    t.save(probe.state_dict(), os.path.join(probe_save_dir, model_filename))
            
            # Compute cosine similarities between the weight vectors for the three probes
            # Extract the weight vectors (flattening them)
            w_input = probes_for_label["sae_input"].net.weight.view(-1)
            w_recons = probes_for_label["sae_recons"].net.weight.view(-1)
            w_diff = probes_for_label["sae_diff"].net.weight.view(-1)
            
            cos_sim_input_recons = F.cosine_similarity(w_input, w_recons, dim=0).item()
            cos_sim_input_diff = F.cosine_similarity(w_input, w_diff, dim=0).item()
            cos_sim_recons_diff = F.cosine_similarity(w_recons, w_diff, dim=0).item()
            
            # Add the cosine similarity metrics to each probe's result
            for feature_type in features_map.keys():
                temp_results[feature_type]["Cosine Sim Input-Recons"] = cos_sim_input_recons
                temp_results[feature_type]["Cosine Sim Input-Diff"] = cos_sim_input_diff
                temp_results[feature_type]["Cosine Sim Recons-Diff"] = cos_sim_recons_diff
                
                # Append the result to the main results list
                results.append(temp_results[feature_type])
            
            t.cuda.empty_cache()
        
        loop_duration = time.time() - start_time
        print(f"Training loop with seed {seed} completed in {loop_duration:.2f} seconds.")
    
    # Create a results table and print it
    results_df = pd.DataFrame(results)
    print("\nFinal Evaluation Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(results_csv, index=False)
    
    # Compute average cosine similarities across saved probes
    similarity_results = []
    # For each combination of feature type and label, load the saved probes from the first N seeds
    for label_col in label_columns:
        safe_label = label_col.replace(' ', '_')
        for feature_type in features_map.keys():
            weight_vectors = []
            for seed in range(save_probes_count):
                model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                filepath = os.path.join(probe_save_dir, model_filename)
                if os.path.exists(filepath):
                    # Initialize a probe and load its state dict
                    dummy_probe = Probe(activation_dim=features_map[feature_type].size(1)).to('cpu')
                    state_dict = t.load(filepath, map_location='cpu')
                    dummy_probe.load_state_dict(state_dict)
                    weight_vectors.append(dummy_probe.net.weight.view(-1))
                else:
                    warn(f"Probe file {filepath} does not exist.")
            
            # Compute pairwise cosine similarities among these weight vectors
            sims = []
            num = len(weight_vectors)
            for i in range(num):
                for j in range(i+1, num):
                    sim = F.cosine_similarity(weight_vectors[i], weight_vectors[j], dim=0).item()
                    sims.append(sim)
            if sims:
                avg_sim = np.mean(sims)
                std_sim = np.std(sims)
            else:
                avg_sim = None
                std_sim = None
            similarity_results.append({
                "Feature Type": feature_type,
                "Label": label_col,
                "Average Cosine Similarity": avg_sim,
                "Cosine Similarity Std": std_sim
            })
    
    similarities_df = pd.DataFrame(similarity_results)
    print("\nProbe Similarities across saved probes:")
    print(similarities_df.to_string(index=False))
    similarities_df.to_csv(similarities_csv, index=False)
    
    return results_df, similarities_df


#Setup
# %%
if __name__ == "__main__":
    # Run simple test for the last-token extraction helper.
    device = t.device('cuda:0')
    test_last_token_extraction()
    
    # Read datasets and combine them.
    df = pd.read_csv("all_cities.csv")
    
    # Load SAE and the model.
    print('Load SAE')
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_19/width_16k/canonical",
        device="cpu"
    )
    sae = sae.to(device)
    print('Load Model')
    model2b = HookedSAETransformer.from_pretrained("gemma-2-2b", device='cpu')
    model2b = model2b.to(device)
    
    # Tokenize the entire dataset at once.
    print("Tokenizing entire dataset...")
    tokenized_all = tokenize_data(df, model2b.tokenizer)
    label_columns = ['label']
    
    features_possibilities = ['sae_input', 'sae_recons', 'sae_diff']

        # Directory to save trained probes.
    probe_save_dir = "trained_probes_truth"
    os.makedirs(probe_save_dir, exist_ok=True)
    

# %%
if __name__ == "__main__":
    # Run the original probing pipeline on last token (offset=1)
    print("Running probing pipeline on LAST tokens...")
    
    # Generate features for the entire dataset
    print("Generating features for entire dataset (last token)...")
    feats_all_input, feats_all_recons, feats_all_diff = generate_probing_features(
        tokenized_all, model2b, sae, batch_size=8, device=device, offset=1
    )
    
    # Map each feature type to its corresponding feature tensor
    features_map_last = {
        "sae_input": feats_all_input,
        "sae_recons": feats_all_recons,
        "sae_diff": feats_all_diff
    }
    
    # Run the probing pipeline
    results_df_last, similarities_df_last = run_probing_pipeline(
        df=df,
        tokenized_all=tokenized_all,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=label_columns,
        features_map=features_map_last,
        n_seeds=50,
        save_probes_count=20,
        probe_save_dir="trained_probes_truth_last",
        results_csv="probe_results_truth_last.csv",
        similarities_csv="probe_similarities_truth_last.csv"
    )

# %%
if __name__ == "__main__":
    # Run the probing pipeline on second-to-last tokens (offset=2)
    print("Running probing pipeline on SECOND-TO-LAST tokens...")
    
    # Generate features for the entire dataset using second-to-last tokens
    print("Generating features for entire dataset (second-to-last token)...")
    feats_all_input_2nd, feats_all_recons_2nd, feats_all_diff_2nd = generate_probing_features(
        tokenized_all, model2b, sae, batch_size=8, device=device, offset=2
    )
    
    # Map each feature type to its corresponding feature tensor
    features_map_2nd = {
        "sae_input": feats_all_input_2nd,
        "sae_recons": feats_all_recons_2nd,
        "sae_diff": feats_all_diff_2nd
    }
    
    # Run the probing pipeline
    results_df_2nd, similarities_df_2nd = run_probing_pipeline(
        df=df,
        tokenized_all=tokenized_all,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=label_columns,
        features_map=features_map_2nd,
        n_seeds=50,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_second_last",
        results_csv="probe_results_truth_second_last.csv",
        similarities_csv="probe_similarities_truth_second_last.csv"
    )
# %%
#Run probe on twitter happiness

if __name__ == "__main__":
    tw_happiness = pd.read_csv('149_twt_emotion_happiness.csv')
    tw_happiness_tokenized = tokenize_data(tw_happiness, model2b.tokenizer, 'prompt')

    feats_tw_input, feats_tw_recons, feats_tw_diff = generate_probing_features(
        tw_happiness_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_tw = {
        "sae_input": feats_tw_input,
        "sae_recons": feats_tw_recons,
        "sae_diff": feats_tw_diff
    }
    
    # Run the probing pipeline
    results_df_tw, similarities_df_tw = run_probing_pipeline(
        df=tw_happiness,
        tokenized_all=tw_happiness_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_tw,
        n_seeds=50,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_tw",
        results_csv="probe_results_tw_happiness.csv",
        similarities_csv="probe_similarities_tw_happiness.csv"
    )
    results_df_tw.groupby(['Feature Type', 'Label'])['Test Accuracy'].mean().reset_index()
    
# %%
#Run probe on headline front page

if __name__ == "__main__":
    hl_frontp = pd.read_csv('headline_frontpage_sample.csv')
    hl_frontp_tokenized = tokenize_data(hl_frontp, model2b.tokenizer, 'prompt')

    feats_hl_input, feats_hl_recons, feats_hl_diff = generate_probing_features(
        hl_frontp_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_hl = {
        "sae_input": feats_hl_input,
        "sae_recons": feats_hl_recons,
        "sae_diff": feats_hl_diff
    }
    
    # Run the probing pipeline
    results_df_hl, similarities_df_hl = run_probing_pipeline(
        df=hl_frontp,
        tokenized_all=hl_frontp_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_hl,
        n_seeds=50,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_hl",
        results_csv="probe_results_hl_frontp.csv",
        similarities_csv="probe_similarities_hl_frontp.csv"
    )

# %%
#Run probe on if the borough is manhattan

if __name__ == "__main__":
    manhattan = pd.read_csv('114_nyc_borough_Manhattan.csv')
    manhattan_tokenized = tokenize_data(manhattan, model2b.tokenizer, 'prompt')

    feats_man_input, feats_man_recons, feats_man_diff = generate_probing_features(
        manhattan_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_man = {
        "sae_input": feats_man_input,
        "sae_recons": feats_man_recons,
        "sae_diff": feats_man_diff
    }
    
    # Run the probing pipeline
    results_df_man, similarities_df_man = run_probing_pipeline(
        df=manhattan,
        tokenized_all=manhattan_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_man,
        n_seeds=50,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_man",
        results_csv="probe_results_man_borough.csv",
        similarities_csv="probe_similarities_man_borough.csv"
    )
    results_df_man.groupby(['Feature Type', 'Label'])['Test Accuracy', 'Test Loss'].mean().reset_index()

# %%
# run probe on athlete sport basketball 155_athlete_sport_basketball.csv

if __name__ == "__main__":
    athlete_sport = pd.read_csv('155_athlete_sport_basketball.csv')
    athlete_sport_tokenized = tokenize_data(athlete_sport, model2b.tokenizer, 'prompt')
    
    feats_ath_input, feats_ath_recons, feats_ath_diff = generate_probing_features(
        athlete_sport_tokenized, model2b, sae, batch_size=8, device=device, offset=1
    )

    features_map_ath = {
        "sae_input": feats_ath_input,
        "sae_recons": feats_ath_recons,
        "sae_diff": feats_ath_diff
    }
    
    # Run the probing pipeline
    results_df_ath, similarities_df_ath = run_probing_pipeline(
        df=athlete_sport,
        tokenized_all=athlete_sport_tokenized,
        model=model2b,
        sae=sae,
        device=device,
        label_columns=['target'],
        features_map=features_map_ath,
        n_seeds=50,
        save_probes_count=0,
        probe_save_dir="trained_probes_truth_ath",
        results_csv="probe_results_ath_sport.csv",
        similarities_csv="probe_similarities_ath_sport.csv"
    )
    results_df_ath.groupby(['Feature Type', 'Label'])['Test Loss'].mean().reset_index()
    

# %%
if __name__ == "__main__":
    # Compare results between last token and second-to-last token
    print("\nComparison of average test accuracies:")
    
    # Compute average test accuracies for each feature type and token position
    avg_acc_last = results_df_last.groupby(['Feature Type', 'Label'])['Test Accuracy'].mean().reset_index()
    avg_acc_2nd = results_df_2nd.groupby(['Feature Type', 'Label'])['Test Accuracy'].mean().reset_index()
    
    # Rename for clarity
    avg_acc_last['Token Position'] = 'Last'
    avg_acc_2nd['Token Position'] = 'Second-to-last'
    
    # Combine the results
    comparison = pd.concat([avg_acc_last, avg_acc_2nd])
    
    # Display the comparison
    print(comparison.pivot_table(
        index=['Feature Type', 'Label'], 
        columns='Token Position', 
        values='Test Accuracy'
    ).reset_index())
    
    # Save the comparison
    comparison.to_csv("token_position_comparison.csv", index=False)




##### Old code

# %%





#%%
if __name__ == "__main__":
    #Code to run the probing pipeline
    # Generate features for the entire dataset.
    print("Generating features for entire dataset...")
    feats_all_input, feats_all_recons, feats_all_diff = generate_probing_features(
        tokenized_all, model2b, sae, batch_size=8, device=device
    )
    
    # Map each feature type to its corresponding feature tensor.
    features_map = {
        "sae_input": feats_all_input,
        "sae_recons": feats_all_recons,
        "sae_diff": feats_all_diff
    }
    
    
    results = []
    n = df.shape[0]
    train_size = int(0.8 * n)  # 80% for training
    
    N_SEEDS = 50
    for seed in range(N_SEEDS):
        print(f"\nStarting training loop with seed {seed}...")
        start_time = time.time()
        
        # Compute a train/test split for the entire dataset using this seed.
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Loop over each label.
        for label_col in label_columns:
            # Prepare the labels (same for all feature types).
            labels_all = t.tensor(df[label_col].values)
            train_labels = labels_all[train_indices]
            test_labels = labels_all[test_indices]
            
            # Dictionary to hold the probes for the three feature types for cosine similarity computation.
            probes_for_label = {}
            # Temporary storage for the per-probe results for this seed & label.
            temp_results = {}
            
            # Train a probe for each feature type.
            for feature_type, feats_all in features_map.items():
                train_feats = feats_all[train_indices]
                test_feats = feats_all[test_indices]
                
                # Set the torch seed to ensure probe initialization consistency.
                t.manual_seed(seed)
                probe, _ = train_probe_model(
                    train_feats, train_labels, dim=train_feats.size(1),
                    epochs=2, batch_size=8, device=device, lr=0.005
                )
                train_loss, train_acc = evaluate_probe_full(probe, train_feats, train_labels, device=device)
                test_loss, test_acc = evaluate_probe_full(probe, test_feats, test_labels, device=device)
                weight_norm = probe.net.weight.norm().item()
                
                probes_for_label[feature_type] = probe
                temp_results[feature_type] = {
                    "Seed": seed,
                    "Feature Type": feature_type,
                    "Label": label_col,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "Weight Norm": weight_norm
                }
                
                # Save the probe if this seed is among the first 20.
                if seed < 20:
                    safe_label = label_col.replace(' ', '_')
                    model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                    t.save(probe.state_dict(), os.path.join(probe_save_dir, model_filename))
            
            # Compute cosine similarities between the weight vectors for the three probes.
            # Extract the weight vectors (flattening them)
            w_input = probes_for_label["sae_input"].net.weight.view(-1)
            w_recons = probes_for_label["sae_recons"].net.weight.view(-1)
            w_diff = probes_for_label["sae_diff"].net.weight.view(-1)
            
            cos_sim_input_recons = F.cosine_similarity(w_input, w_recons, dim=0).item()
            cos_sim_input_diff = F.cosine_similarity(w_input, w_diff, dim=0).item()
            cos_sim_recons_diff = F.cosine_similarity(w_recons, w_diff, dim=0).item();
            
            # Add the cosine similarity metrics to each probe's result.
            for feature_type in features_map.keys():
                temp_results[feature_type]["Cosine Sim Input-Recons"] = cos_sim_input_recons
                temp_results[feature_type]["Cosine Sim Input-Diff"] = cos_sim_input_diff
                temp_results[feature_type]["Cosine Sim Recons-Diff"] = cos_sim_recons_diff
                
                # Append the result to the main results list.
                results.append(temp_results[feature_type])
            
            t.cuda.empty_cache()
        
        loop_duration = time.time() - start_time
        print(f"Training loop with seed {seed} completed in {loop_duration:.2f} seconds.")
    
    # Create a results table and print it.
    results_df = pd.DataFrame(results)
    print("\nFinal Evaluation Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv("probe_results_truth.csv", index=False)

# %%
if __name__ == "__main__":
    ###########################################################################
    # Compute average cosine similarities across saved probes
    ###########################################################################
    similarity_results = []
    # For each combination of feature type and label, load the saved probes from the first 20 seeds.
    for label_col in label_columns:
        safe_label = label_col.replace(' ', '_')
        for feature_type in features_map.keys():
            weight_vectors = []
            for seed in range(20):
                model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                filepath = os.path.join(probe_save_dir, model_filename)
                if os.path.exists(filepath):
                    # Initialize a probe and load its state dict.
                    dummy_probe = Probe(activation_dim=features_map[feature_type].size(1)).to('cpu')
                    state_dict = t.load(filepath, map_location='cpu')
                    dummy_probe.load_state_dict(state_dict)
                    weight_vectors.append(dummy_probe.net.weight.view(-1))
                else:
                    warn(f"Probe file {filepath} does not exist.")
            
            # Compute pairwise cosine similarities among these weight vectors.
            sims = []
            num = len(weight_vectors)
            for i in range(num):
                for j in range(i+1, num):
                    sim = F.cosine_similarity(weight_vectors[i], weight_vectors[j], dim=0).item()
                    sims.append(sim)
            if sims:
                avg_sim = np.mean(sims)
                std_sim = np.std(sims)
            else:
                avg_sim = None
                std_sim = None
            similarity_results.append({
                "Feature Type": feature_type,
                "Label": label_col,
                "Average Cosine Similarity": avg_sim,
                "Cosine Similarity Std": std_sim
            })
    
    similarities_df = pd.DataFrame(similarity_results)
    print("\nProbe Similarities across saved probes:")
    print(similarities_df.to_string(index=False))
    similarities_df.to_csv("probe_similarities_truth.csv", index=False)

# %%

#Probe steering results:


#Ok I want to write a new function, generate steering results. This function would look similar to generate probing features. For now, here are the differences:



# %%
if __name__ == "__main__":
    #Code for SAE reconstruction analysis
    #This is early because I'm scared that tokenized_all will get scrambled somehow below.

    active_latent_list = record_active_latents(tokenized_all, model = model2b, sae = sae)
    tot_lats = set()
    for latent_and_acts in active_latent_list:
        for latent in latent_and_acts[0]:
            tot_lats.add(latent)
    df_w_latent_data = df.copy()
    df_w_latent_data['active_latents'] = [l[0] for l in active_latent_list]
    print(len(tot_lats))
    lat_list = list(tot_lats)
    lat_list.sort()
    out_dirs = sae.W_dec[lat_list,:]
    # Initialize list to store results
    cos_sim_results = []
    
    # Load each probe from the first 20 seeds
    for seed in range(20):
        for feature_type in features_possibilities:
            for label_col in label_columns:
                safe_label = label_col.replace(' ', '_')
                model_filename = f"probe_{feature_type}_{safe_label}_seed_{seed}.pt"
                filepath = os.path.join(probe_save_dir, model_filename)
                
                if os.path.exists(filepath):
                    # Load probe
                    probe = Probe(activation_dim=2304).to('cuda')  # 2304 is the model dimension
                    probe.load_state_dict(t.load(filepath, map_location='cuda'))
                    
                    # Get probe weights and normalize them
                    probe_weights = probe.net.weight.view(-1)
                    probe_weights_norm = F.normalize(probe_weights, dim=0)
                    
                    # Calculate cosine similarities with all output directions
                    # Normalize output directions
                    out_dirs_norm = F.normalize(t.tensor(out_dirs), dim=1)
                    
                    # Compute all cosine similarities at once
                    similarities = F.linear(out_dirs_norm, probe_weights_norm.unsqueeze(0)).squeeze()
                    
                    # Create result dictionary
                    result = {
                        'seed': seed,
                        'feature_type': feature_type,
                        'label': label_col,
                    }
                    # Add similarities for each latent
                    for i, sim in enumerate(similarities):
                        result[f'latent_{lat_list[i]}'] = sim.item()
                    
                    cos_sim_results.append(result)
                else:
                    warn(f"Probe file {filepath} does not exist.")
    
    # Create DataFrame
    cos_sim_df = pd.DataFrame(cos_sim_results)
    
    # Save results
    cos_sim_df.to_csv("probe_latent_similarities.csv", index=False)
    print("Cosine similarities computed and saved!")
