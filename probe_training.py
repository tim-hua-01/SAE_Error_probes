# %%
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sae_lens import SAE, HookedSAETransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time  # For timing the training loops

# For reproducibility
t.manual_seed(123)

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

###############################################################################
# Main Pipeline
###############################################################################

# %%
if __name__ == "__main__":
    # Run simple test for the last-token extraction helper.
    test_last_token_extraction()
    
    # Read datasets and combine them.
    data = pd.read_csv("cities_alice.csv")
    neg_data = pd.read_csv("neg_cities_alice.csv")
    df = pd.concat([data, neg_data])
    
    # Load SAE and the model.
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_19/width_16k/canonical",
        device="cuda"
    )
    model2b = HookedSAETransformer.from_pretrained("gemma-2-2b", device='cuda')
    
    # Tokenize the entire dataset at once.
    print("Tokenizing entire dataset...")
    tokenized_all = tokenize_data(df, model2b.tokenizer)
    
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
    
    label_columns = ['label', 'has_alice', 'has_not',
                     'has_alice xor has_not', 'has_alice xor label', 'has_not xor label']
    
    # Directory to save trained probes.
    probe_save_dir = "trained_probes"
    os.makedirs(probe_save_dir, exist_ok=True)
    
    results = []
    n = df.shape[0]
    train_size = int(0.8 * n)  # 80% for training
    
    # Loop over different seeds.
    N_SEEDS = 30
    for seed in range(N_SEEDS):
        print(f"\nStarting training loop with seed {seed}...")
        start_time = time.time()
        
        # Randomly scramble indices for splitting features.
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        for feature_type, feats_all in features_map.items():
            # Create train/test splits for the features.
            train_feats = feats_all[train_indices]
            test_feats = feats_all[test_indices]
            
            for label_col in label_columns:
                # Create corresponding label splits.
                labels_all = t.tensor(df[label_col].values)
                train_labels = labels_all[train_indices]
                test_labels = labels_all[test_indices]
                
                #print(f"Training probe on {feature_type} features for label '{label_col}' with seed {seed}...")
                # Set the random seed right before initialization/training so the probe
                # gets the same initialization and same batch ordering.
                t.manual_seed(seed)
                
                # Train the probe.
                probe, _ = train_probe_model(
                    train_feats, train_labels, dim=train_feats.size(1),
                    epochs=2, batch_size=8, device='cuda', lr=0.005
                )
                
                # Evaluate on both train and test splits.
                train_loss, train_acc = evaluate_probe_full(probe, train_feats, train_labels, device='cuda')
                test_loss, test_acc = evaluate_probe_full(probe, test_feats, test_labels, device='cuda')
                
                # Only save probes for the last seed
                if seed == N_SEEDS - 1:
                    model_filename = f"probe_{feature_type}_{label_col.replace(' ', '_')}_seed_{seed}.pt"
                    t.save(probe.state_dict(), os.path.join(probe_save_dir, model_filename))
                
                # Append results.
                results.append({
                    "Seed": seed,
                    "Feature Type": feature_type,
                    "Label": label_col,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                })
                t.cuda.empty_cache()
        
        loop_duration = time.time() - start_time
        # Update loop time for all results of this seed.
        print(f"Training loop with seed {seed} completed in {loop_duration:.2f} seconds.")
    
    # Create a results table and print it.
    results_df = pd.DataFrame(results)
    print("\nFinal Evaluation Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv("probe_results.csv", index=False)
# %%
