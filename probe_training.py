
# %%
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sae_lens import SAE, HookedSAETransformer
import pandas as pd
import numpy as np
from tqdm import tqdm

# For reproducibility
SEED = 123

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
def train_test_split_df(df, test_size=0.2, seed=SEED):
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
    # attention_mask: (batch, seq_len)
    # sum gives the count of valid tokens per sample; subtract one to get the index.
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
def generate_probing_features(tokenized, model, sae, batch_size=2, device='cuda'):
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
        # Run the model. It is assumed that model2b (our hooked transformer) has the
        # method run_with_cache_with_saes and that the sae (pre-loaded) is provided.
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
    the binary labels. The features are expected to have shape (N, dim) and labels
    shape (N,). Returns the trained probe and a list of loss values.
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

def evaluate_probe(probe, features, labels, device='cuda'):
    """
    Evaluates the given probe on the test features and returns the loss.
    """
    probe.eval()
    criterion = nn.BCEWithLogitsLoss()
    with t.no_grad():
        logits = probe(features.to(device))
        loss = criterion(logits, labels.to(device).float())
    return loss.item()

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
    # Run the main probe training pipeline.
    data = pd.read_csv("cities_alice.csv")
    neg_data = pd.read_csv("neg_cities_alice.csv")
    df = pd.concat([data, neg_data]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res-canonical",
    sae_id = "layer_19/width_16k/canonical",
    device = "cuda"
    )
    model2b = HookedSAETransformer.from_pretrained("gemma-2-2b", device = 'cuda')
    
    # ----------------------
    # Train/test split.
    # ----------------------
    train_df, test_df = train_test_split_df(df, test_size=0.2, seed=SEED)
    
    # ----------------------
    # Tokenize the statements.
    # ----------------------
    tokenized_train = tokenize_data(train_df, model2b.tokenizer)
    tokenized_test = tokenize_data(test_df, model2b.tokenizer)
    
    # ----------------------
    # Convert labels to tensors.
    # ----------------------
    # We assume that the DataFrame has a column called "label" (a binary 0/1 value).
    train_labels = t.tensor(train_df["has_alice xor has_not"].values)
    test_labels = t.tensor(test_df["has_alice xor has_not"].values)
    
    # ----------------------
    # Generate features using our hooked SAE in mini-batches.
    # ----------------------
    print("Generating training features...")
    feats_train_input, feats_train_recons, feats_train_diff = generate_probing_features(
        tokenized_train, model2b, sae, batch_size=4, device='cuda'
    )
    print("Generating test features...")
    feats_test_input, feats_test_recons, feats_test_diff = generate_probing_features(
        tokenized_test, model2b, sae, batch_size=4, device='cuda'
    )
    
    # ----------------------
    # Determine the feature (model) dimension.
    # ----------------------
    act_dim = feats_train_input.size(1)
    
    # ----------------------
    # Train probes on the three feature types.
    # ----------------------
    print("Training probe on hook_sae_input activations...")
    probe_input, losses_input = train_probe_model(
        feats_train_input, train_labels, dim=act_dim,
        epochs=1, batch_size=8, device='cuda'
    )
    test_loss_input = evaluate_probe(probe_input, feats_test_input, test_labels, device='cuda')
    
    print("Training probe on hook_sae_recons activations...")
    probe_recons, losses_recons = train_probe_model(
        feats_train_recons, train_labels, dim=act_dim,
        epochs=1, batch_size=8, device='cuda'
    )
    test_loss_recons = evaluate_probe(probe_recons, feats_test_recons, test_labels, device='cuda')
    
    print("Training probe on (input - recons) activations...")
    probe_diff, losses_diff = train_probe_model(
        feats_train_diff, train_labels, dim=act_dim,
        epochs=1, batch_size=8, device='cuda'
    )
    test_loss_diff = evaluate_probe(probe_diff, feats_test_diff, test_labels, device='cuda')
    
    # ----------------------
    # Save trained probes.
    # ----------------------
    t.save(probe_input.state_dict(), "probe_sae_input.pt")
    t.save(probe_recons.state_dict(), "probe_sae_recons.pt")
    t.save(probe_diff.state_dict(), "probe_sae_diff.pt")
    
    # ----------------------
    # Report results.
    # ----------------------
    print("Out-of-sample (test) losses:")
    print(f"Probe (sae_input):   test loss = {test_loss_input}")
    print(f"Probe (sae_recons):  test loss = {test_loss_recons}")
    print(f"Probe (sae_diff):    test loss = {test_loss_diff}")