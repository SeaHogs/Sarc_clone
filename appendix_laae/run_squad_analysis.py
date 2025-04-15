"""
BERT Model Interpretation for Sarcasm Detection

This script uses Captum to interpret a pre-loaded BERT sarcasm detection model.
It assumes the model, tokenizer, and device variables are already defined in the environment.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import random

# Import Captum for interpretability
from captum.attr import LayerIntegratedGradients, visualization as viz
from captum.attr import LayerConductance, IntegratedGradients

# Define forward function for interpretability
def sarcasm_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    """Forward function for attribution analysis, returns sarcasm score (class 1)"""
    # Check if inputs are embeddings or token ids
    if inputs.dim() == 3:  # [batch_size, seq_len, hidden_dim]
        # Inputs are embeddings
        output = model(inputs_embeds=inputs, token_type_ids=token_type_ids,
                    position_ids=position_ids, attention_mask=attention_mask)
    else:
        # Inputs are token ids
        output = model(inputs, token_type_ids=token_type_ids,
                    position_ids=position_ids, attention_mask=attention_mask)

    # Get the logits
    if hasattr(output, 'logits'):
        logits = output.logits
    else:
        logits = output

    # Return score for sarcasm class (index 1)
    return logits[:, 1]

# Helper function to summarize attributions
def summarize_attributions(attributions):
    """Summarize attributions by summing across embedding dimension"""
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

# Function to construct input reference pair
def construct_input_ref_pair(text, tokenizer, device):
    """Construct input and reference token ids for interpretation"""
    text_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move to device
    input_ids = text_ids['input_ids'].to(device)
    token_type_ids = text_ids['token_type_ids'].to(device) if 'token_type_ids' in text_ids else None
    attention_mask = text_ids['attention_mask'].to(device)

    # Create a reference filled with pad token id
    ref_input_ids = torch.ones_like(input_ids) * tokenizer.pad_token_id
    ref_token_type_ids = torch.zeros_like(token_type_ids) if token_type_ids is not None else None

    return input_ids, ref_input_ids, token_type_ids, ref_token_type_ids, attention_mask, text_ids

# Function to visualize attributions
def visualize_attributions(attributions, tokens, pred_score, pred_class, true_class=None, delta=None):
    """Visualize token attributions"""
    # Filter out special tokens
    special_tokens = ['[CLS]', '[SEP]', '[PAD]']
    filtered_tokens = []
    filtered_attrs = []
    for token, attr in zip(tokens, attributions):
        if token not in special_tokens:
            filtered_tokens.append(token)
            filtered_attrs.append(attr.item()) # Convert tensor to scalar

    # Convert filtered attributions to a tensor if needed, or handle as a list
    # For viz.VisualizationDataRecord, it expects a tensor or list of numbers.
    # Let's use the list of scalars we created.
    if not filtered_tokens: # Avoid errors if all tokens are special tokens
        print("Warning: No content tokens left after filtering special tokens.")
        return

    # Convert filtered_attrs back to a tensor for the record if required by viz
    # Note: viz.VisualizationDataRecord can often handle lists directly
    # Depending on the exact Captum version/requirements, this might need adjustment
    filtered_attrs_tensor = torch.tensor(filtered_attrs, device=attributions.device)

    # Convert attributions to visualization format using filtered data
    vis_data = viz.VisualizationDataRecord(
        filtered_attrs_tensor, # Use filtered attributions
        pred_score,
        pred_class,
        pred_class, # Using pred_class for attribution label as before
        str(true_class) if true_class is not None else "",
        filtered_attrs_tensor.sum(), # Sum of filtered attributions
        filtered_tokens, # Use filtered tokens
        delta # Convergence delta remains the same
    )

    # Print visualization
    print('\n Visualized attributions:')
    viz.visualize_text([vis_data])

# Function to analyze a single sample
def analyze_sample(text, label=None):
    """Analyze a single text sample and visualize attributions"""
    # Prepare inputs
    input_ids, ref_input_ids, token_type_ids, ref_token_type_ids, attention_mask, text_ids = construct_input_ref_pair(text, tokenizer, device)

    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        pred_score = probs[0, pred_class].item()

    print(f"\nText: {text}")
    print(f"Predicted class: {'Sarcastic' if pred_class == 1 else 'Not sarcastic'} (probability: {pred_score:.4f})")
    if label is not None:
        print(f"True class: {'Sarcastic' if label == 1 else 'Not sarcastic'}")

    # Initialize LayerIntegratedGradients
    lig = LayerIntegratedGradients(sarcasm_forward_func, model.bert.embeddings)

    # Compute attributions
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=ref_input_ids,
        additional_forward_args=(token_type_ids, None, attention_mask),
        return_convergence_delta=True
    )

    # Summarize attributions
    attributions_sum = summarize_attributions(attributions)

    # Visualize attributions
    visualize_attributions(attributions_sum, tokens, pred_score, pred_class, true_class=label, delta=delta)

    return attributions, tokens, pred_class, pred_score

# Function to analyze a layer
def analyze_layer(text, layer_idx):
    """Analyze attributions for a specific layer"""
    # Prepare inputs
    input_ids, ref_input_ids, token_type_ids, ref_token_type_ids, attention_mask, text_ids = construct_input_ref_pair(text, tokenizer, device)

    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Initialize LayerConductance for the specific layer
    layer = model.bert.encoder.layer[layer_idx]
    lc = LayerConductance(sarcasm_forward_func, layer)

    # Get embedding outputs to use as inputs for layer conductance
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids)

    # Compute attributions - ensure attention mask is correctly passed
    # LayerConductance might return just attributions or (attributions, delta)
    layer_conductance_output = lc.attribute(
        inputs=input_embeddings,
        baselines=ref_input_embeddings,
        additional_forward_args=(token_type_ids, None, attention_mask)
    )

    # Extract attributions (handle tuple output if necessary)
    if isinstance(layer_conductance_output, tuple):
        attributions = layer_conductance_output[0]
    else:
        attributions = layer_conductance_output

    # Summarize attributions
    attributions_sum = summarize_attributions(attributions)
    attributions_sum_np = attributions_sum.cpu().detach().numpy()

    # Filter out special tokens before plotting
    special_tokens = ['[CLS]', '[SEP]', '[PAD]']
    filtered_indices = [i for i, token in enumerate(tokens) if token not in special_tokens]
    filtered_tokens = [tokens[i] for i in filtered_indices]
    
    if not filtered_tokens:
        print(f"Warning: No content tokens left after filtering for Layer {layer_idx}.")
        # Return unfiltered raw attributions and original tokens if needed, or handle error
        return attributions, tokens

    filtered_attributions_sum_np = attributions_sum_np[filtered_indices]

    # Plot attributions for this layer using filtered data
    plt.figure(figsize=(max(10, len(filtered_tokens)*0.6), 5)) # Dynamic width
    plt.bar(range(len(filtered_tokens)), filtered_attributions_sum_np)
    plt.xticks(range(len(filtered_tokens)), filtered_tokens, rotation=90)
    plt.title(f"Layer {layer_idx} Attributions (Content Tokens Only)")
    plt.ylabel("Attribution Score")
    plt.tight_layout()
    plt.show()

    # Return the raw, unfiltered attributions and original tokens, 
    # as analyze_all_layers expects the full set for its own filtering logic before heatmap
    return attributions, tokens # Return original full attributions for heatmap processing

# Function to analyze all layers
def analyze_all_layers(text):
    """Analyze attributions across all layers"""
    # Prepare inputs
    input_ids, ref_input_ids, token_type_ids, ref_token_type_ids, attention_mask, text_ids = construct_input_ref_pair(text, tokenizer, device)

    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Store attributions for all layers
    layer_attrs_list = []

    # Iterate through all layers
    layer_indices = list(range(model.config.num_hidden_layers))
    print(f"Analyzing {len(layer_indices)} layers...")

    for i in layer_indices:
        # Get attributions for this layer
        # Ensure analyze_layer returns the necessary structure, assuming attr[0] is the tensor
        attr, _ = analyze_layer(text, i)
        # Summarize attributions - check if attr[0] is correct based on analyze_layer output
        if isinstance(attr, tuple): # If analyze_layer returns tuple (attributions, delta)
            attr_sum = summarize_attributions(attr[0])
        else: # If it returns just attributions tensor
             attr_sum = summarize_attributions(attr)

        layer_attrs_list.append(attr_sum.cpu().detach().numpy())

    # Convert list of layer attributions to a 2D numpy array
    layer_attrs_np = np.array(layer_attrs_list)

    # Filter out special tokens and corresponding attributions
    special_tokens = ['[CLS]', '[SEP]', '[PAD]']
    filtered_indices = [i for i, token in enumerate(tokens) if token not in special_tokens]
    filtered_tokens = [tokens[i] for i in filtered_indices]
    
    if not filtered_tokens:
        print("Warning: No content tokens left after filtering special tokens for layer analysis.")
        return [], [] # Return empty lists if no content
        
    # Filter columns of the attribution array
    filtered_layer_attrs_np = layer_attrs_np[:, filtered_indices]

    # Create heatmap for all layers using filtered data
    plt.figure(figsize=(max(15, len(filtered_tokens)), max(5, len(layer_indices) / 2))) # Adjust size
    sns.heatmap(
        filtered_layer_attrs_np, # Use filtered attributions
        xticklabels=filtered_tokens, # Use filtered tokens
        yticklabels=layer_indices,
        linewidth=0.2,
        cmap="viridis" # Use a common colormap
    )
    plt.xlabel('Tokens')
    plt.ylabel('Layers')
    plt.title('Token Attribution Across All Layers (Content Tokens Only)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Return the filtered versions if needed elsewhere
    return filtered_layer_attrs_np, filtered_tokens

# Function to download sample data if not already present
def ensure_data_loaded():
    """Make sure we have access to the sarcasm dataset"""
    global df  # Use global variable to match existing setup

    try:
        # Check if df is already defined
        if 'df' in globals():
            print(f"Using existing dataset with {len(df)} samples")
            return df
    except:
        pass

    # If not, try to load it
    try:
        import kagglehub
        print("Downloading dataset...")
        dataset_path = kagglehub.dataset_download("rmisra/news-headlines-dataset-for-sarcasm-detection")
        path = f"{dataset_path}/Sarcasm_Headlines_Dataset_v2.json"
        df = pd.read_json(path, lines=True)
        print(f"Dataset loaded with {len(df)} samples")
        return df
    except:
        # Fallback to a simple dataset
        print("Creating a simple dataset for demo purposes")
        data = {
            'headline': [
                "mom starting to fear son's web series closest thing she will have to grandchild",
                "area man smoking like he's been to fucking war or something",
                "report: only 47,000 social justice milestones to go before u.s. achieves full equality",
                "former versace store clerk sues over secret 'black code' for minority shoppers",
                "j.k. rowling wishes snape happy birthday in the most magical way",
                "the 'roseanne' revival catches up to our thorny political mood, for better and worse"
            ],
            'is_sarcastic': [1, 1, 1, 0, 0, 0]
        }
        df = pd.DataFrame(data)
        print(f"Created simple dataset with {len(df)} samples")
        return df


"""Run interpretability analysis on the pre-loaded model"""
# Ensure we have a dataset
df = ensure_data_loaded()

# Ensure the global variables are accessible
global model, tokenizer, device

print(f"Using model: {type(model).__name__} on {device}")

# Analyze a few examples from each class
print("\n=== Analyzing individual examples ===")
sarcastic_samples = df[df['is_sarcastic'] == 1]['headline'].sample(3).tolist()
non_sarcastic_samples = df[df['is_sarcastic'] == 0]['headline'].sample(3).tolist()

for text in sarcastic_samples:
    analyze_sample(text, label=1)

for text in non_sarcastic_samples:
    analyze_sample(text, label=0)

# Pick a specific example for detailed layer analysis
detailed_example = sarcastic_samples[0]

# Analyze key layers for this example
print("\n=== Layer-by-layer analysis ===")
layer_attrs, tokens = analyze_all_layers(detailed_example)

print("\nInterpretability analysis complete.")

