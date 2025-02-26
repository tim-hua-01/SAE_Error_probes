# %%
import torch as t
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

# For reproducibility

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
def get_last_token_indices(attention_mask):
    """
    Given an attention mask of shape (batch, seq_len) where valid tokens are 1
    and padded tokens are 0, compute the index of the last token for each sample.
    """
    token_counts = attention_mask.sum(dim=1)
    last_indices = token_counts - 1
    return last_indices

def extract_last_token_acts(act_tensor, attention_mask):
    """
    Given a tensor of activations [batch, seq_len, dim] and the corresponding
    attention mask, select for each sample the activation at the last token.
    """
    last_indices = get_last_token_indices(attention_mask)
    batch_indices = t.arange(act_tensor.size(0), device=act_tensor.device)
    last_activations = act_tensor[batch_indices, last_indices, :]
    return last_activations

###############################################################################
# Feature Generation
###############################################################################
def generate_probing_features(tokenized, model, sae, batch_size=8, device='cuda'):
    """
    Runs the model (with run_with_cache_with_saes) in batches on the tokenized input.
    For each batch it extracts the three features:
      - hook_sae_input, hook_sae_recons, and (sae_input - sae_recons)
    with the extraction done only at the last valid token.
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
        act_input = extract_last_token_acts(batch_out['blocks.19.hook_resid_post.hook_sae_input'], batch_mask)
        act_recons = extract_last_token_acts(batch_out['blocks.19.hook_resid_post.hook_sae_recons'], batch_mask)
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

def record_active_latents(tokenized, model, sae, batch_size=8, device='cuda'):
    """
    Runs the model in batches and records which latents are active (nonzero) at the last token,
    along with their activation values.
    
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
        
        # Extract last token activations
        acts = extract_last_token_acts(
            batch_out['blocks.19.hook_resid_post.hook_sae_acts_post'], 
            batch_mask
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



#Setup
# %%
if __name__ == "__main__":
    # Run simple test for the last-token extraction helper.
    test_last_token_extraction()
    
    # Read datasets and combine them.
    data = pd.read_csv("cities_alice.csv")
    neg_data = pd.read_csv("neg_cities_alice.csv")
    df = pd.concat([data, neg_data])
    
    # Load SAE and the model.
    print('Load SAE')
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_19/width_16k/canonical",
        device="cuda"
    )
    print('Load Model')
    model2b = HookedSAETransformer.from_pretrained("gemma-2-2b", device='cuda')
    
    # Tokenize the entire dataset at once.
    print("Tokenizing entire dataset...")
    tokenized_all = tokenize_data(df, model2b.tokenizer)
    label_columns = ['label', 'has_alice', 'has_not',
                     'has_alice xor has_not', 'has_alice xor label', 'has_not xor label']
    
    features_possibilities = ['sae_input', 'sae_recons', 'sae_diff']

        # Directory to save trained probes.
    probe_save_dir = "trained_probes"
    os.makedirs(probe_save_dir, exist_ok=True)
    

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


    

#%%
if __name__ == "__main__":
    #Code to run the probing pipeline
    # Generate features for the entire dataset.
    print("Generating features for entire dataset...")
    feats_all_input, feats_all_recons, feats_all_diff = generate_probing_features(
        tokenized_all, model2b, sae, batch_size=8, device='cuda'
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
                    epochs=2, batch_size=8, device='cuda', lr=0.005
                )
                train_loss, train_acc = evaluate_probe_full(probe, train_feats, train_labels, device='cuda')
                test_loss, test_acc = evaluate_probe_full(probe, test_feats, test_labels, device='cuda')
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
    results_df.to_csv("probe_results.csv", index=False)
    
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
    similarities_df.to_csv("probe_similarities.csv", index=False)


