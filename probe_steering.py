from probe_training_main import *

if __name__ == "__main__":
    print("Setting up the probing pipeline")
    device = t.device('cuda:0')
    test_last_token_extraction()
    login()
    # Read datasets and combine them.
    df = pd.read_csv("all_cities.csv")
    
    # Load SAE and the model.
    print('Load SAE')
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_21/width_16k/average_l0_139",
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




# %%
#Applying steering
if __name__ == "__main__":
    # Example usage of the steering function
    scaling_range = [-20,-10,-8,-5.0, 5.0, 8, 10, 20]
    print("Generating steering results")
    # Get token IDs for "True" and "False"
      # Note the space before "False"
    np.random.seed(32)
    df_sample = df.sample(480)
    df_sample.to_csv("df_sample.csv", index=False)
    sample_tokenized = tokenize_data(df_sample, model2b.tokenizer, 'twoshot_prompt')
    # Generate steering results
    steering_results = generate_steering_results(
        saved_probe_dir="trained_probes_truth_twoshot",
        tokenized_text=sample_tokenized,
        model=model2b,
        scaling_range=scaling_range,
        ref_token_1=true_token_id,
        ref_token_2=false_token_id,
        random_seed=32,
        batch_size=16, 
        device=device,
        offset=1,  # Last token
        output_csv="steering_results_truth.csv",
        label_name="label",
        n_probes=25
    )

