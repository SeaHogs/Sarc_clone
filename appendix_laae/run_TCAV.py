import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer # Assuming HF Transformers
from torch.utils.data import Dataset, DataLoader
from captum.concept import Concept, TCAV
from captum.attr import LayerGradientXActivation
from sklearn.linear_model import SGDClassifier
import glob
import warnings
import logging
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import logging

def normalize_tcav_score(concept_score, random_sarc_score, random_non_sarc_score):
    """
    Normalizes the TCAV score for a concept based on the scores of random counterparts.
    Uses the logic from eval.py.

    Args:
        concept_score (float): The raw magnitude score for the concept.
        random_sarc_score (float): The raw magnitude score for the random_sarcastic concept.
        random_non_sarc_score (float): The raw magnitude score for the random_non_sarcastic concept.

    Returns:
        float: The normalized score. Returns 0 if the random scores are too close.
    """
    width = abs(random_sarc_score - random_non_sarc_score)
    if width < 1e-6: # Avoid division by zero or near-zero
        logging.warning(f"Normalization width is near zero ({width}). Random concepts might be too similar. Returning 0 for normalized score.")
        return 0.0
    else:
        # Original logic: (weight - random_sarcastic_weight) / width
        # Let's consider if this is the best normalization.
        # Alternative: Center around the midpoint of randoms? (weight - (r1+r2)/2) / (width/2)
        # Let's stick to the eval.py logic for now:
        normalized_weight = (concept_score - random_sarc_score) / width
        return normalized_weight

def plot_normalized_tcav_results(results_dict, text_sample, target_class, layer_name=""):
    """
    Plots the normalized TCAV results for a single sample.

    Args:
        results_dict (dict): A dictionary where keys are concept names and values are
                             dictionaries {'score': float, 'rand_sarc': float, 'rand_non_sarc': float}.
        text_sample (str): The text sample being analyzed.
        target_class (int or str): The target class for the interpretation.
        layer_name (str, optional): Name of the layer analyzed. Defaults to "".
    """
    if not results_dict:
        logging.warning("No TCAV results provided to plot.")
        return

    normalized_scores = {}
    for concept_name, scores in results_dict.items():
        try:
            normalized_scores[concept_name] = normalize_tcav_score(
                scores['score'],
                scores['rand_sarc'],
                scores['rand_non_sarc']
            )
        except KeyError:
            logging.warning(f"Missing score data for concept '{concept_name}'. Skipping normalization.")
            continue # Skip this concept if data is incomplete

    if not normalized_scores:
        logging.warning("No concepts could be normalized. Plotting skipped.")
        return

    concept_names = list(normalized_scores.keys())
    norm_values = list(normalized_scores.values())

    fig, ax = plt.subplots(figsize=(10, max(6, len(concept_names) * 0.5))) # Adjust height
    colors = ['forestgreen' if s > 0 else 'firebrick' for s in norm_values]
    bars = ax.barh(concept_names, norm_values, color=colors)

    ax.set_xlabel('Normalized TCAV Score (Relative to Random)')
    ax.set_ylabel('Concept')
    title = f'Normalized Concept Importance for Target: {target_class}'
    if layer_name:
        title += f' (Layer: {layer_name})'
    if text_sample:
         # Truncate long text
        display_text = text_sample[:60] + '...' if len(text_sample) > 60 else text_sample
        title += f'\nSample: "{display_text}"'
    ax.set_title(title)
    ax.axvline(0, color='grey', linewidth=0.8) # Add line at zero

    # Add labels to bars
    ax.bar_label(bars, fmt='%.3f', padding=3)

    plt.tight_layout() # Adjust layout
    plt.show()



model_directory = "/content/cs4248-project/weight"
tokenizer = BertTokenizerFast.from_pretrained(model_directory)
config = AutoConfig.from_pretrained(model_directory, num_labels=2)
model = BertForSequenceClassification.from_pretrained(model_directory, config=config)
model = BertForSequenceClassification.from_pretrained(model_directory, config=config)
model.eval()
dataset_path = kagglehub.dataset_download("rmisra/news-headlines-dataset-for-sarcasm-detection")
path = f"{dataset_path}/Sarcasm_Headlines_Dataset_v2.json"
df = pd.read_json(path, lines=True)
df.iloc[0]["article_link"]
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress specific warnings if needed (e.g., from sklearn)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


# --- Add this section after loading the DataFrame 'df' ---
NUM_RANDOM_SAMPLES = 10 # Number of samples for each random concept
# Use the CONCEPTS_DIR defined earlier
RANDOM_CONCEPTS_DIR = CONCEPTS_DIR

# Ensure the concepts directory exists
os.makedirs(RANDOM_CONCEPTS_DIR, exist_ok=True)

# Get all headlines if df is loaded
if 'df' in locals() and not df.empty:
    all_headlines = df['headline'].tolist()

    if len(all_headlines) > 0:
        # Create random_sarcastic concept file
        # Sample min(NUM_RANDOM_SAMPLES, len(all_headlines)) to avoid errors if dataset is smaller
        num_samples_sarcastic = min(NUM_RANDOM_SAMPLES, len(all_headlines))
        random_sarcastic_samples = random.sample(all_headlines, num_samples_sarcastic)
        random_sarcastic_path = os.path.join(RANDOM_CONCEPTS_DIR, "random_sarcastic.txt")
        with open(random_sarcastic_path, 'w', encoding='utf-8') as f:
            for line in random_sarcastic_samples:
                f.write(line + '\n')
        logging.info(f"Created random concept file: {random_sarcastic_path} with {len(random_sarcastic_samples)} samples.")

        # Create random_non_sarcastic concept file (using different random samples)
        num_samples_non_sarcastic = min(NUM_RANDOM_SAMPLES, len(all_headlines))
        # Re-sample to get a different set for the second file
        random_non_sarcastic_samples = random.sample(all_headlines, num_samples_non_sarcastic)
        random_non_sarcastic_path = os.path.join(RANDOM_CONCEPTS_DIR, "random_non_sarcastic.txt")
        with open(random_non_sarcastic_path, 'w', encoding='utf-8') as f:
            for line in random_non_sarcastic_samples:
                f.write(line + '\n')
        logging.info(f"Created random concept file: {random_non_sarcastic_path} with {len(random_non_sarcastic_samples)} samples.")
    else:
        logging.warning("DataFrame 'df' is empty or 'headline' column not found. Cannot create random concept files.")
else:
    logging.warning("DataFrame 'df' not loaded. Cannot create random concept files.")
# --- End of added section ---


# --- Configuration ---
# <<< --- START --- >>>
# <<< Specify your Hugging Face model name here >>>
MODEL_NAME = "bert-base-uncased"
# MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" # Example alternative
# <<< Specify the directory containing your concept files (.txt) >>>
CONCEPTS_DIR = "./concepts"
# <<< Specify where to save TCAV results (activations, CAVs) >>>
# SAVE_PATH = "./tcav_results" # Removed as saving is not required
# <<< Specify the names of the layers you want to analyze >>>
# Example: Layer before classification head in BERT. Adjust based on your model structure.
# Use print(model) to inspect layer names if unsure.
TARGET_LAYER_NAMES = ["bert.pooler.dense"]
# Example for DistilBERT: ["pre_classifier"]
# Example for deeper layer: ["bert.encoder.layer.11.output.dense"]
# <<< --- END --- >>>

BATCH_SIZE = 16 # For concept dataloaders and interpretation inputs
MAX_LENGTH = 128 # Tokenizer max length

# --- Helper Functions ---

class TextDataset(Dataset):
    """Simple Dataset for text lines from a file."""
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]
            if not self.texts:
                logging.warning(f"Concept file is empty: {file_path}")
        except FileNotFoundError:
            logging.error(f"Concept file not found: {file_path}")
            self.texts = []
        except Exception as e:
            logging.error(f"Error reading concept file {file_path}: {e}")
            self.texts = []

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize and ensure PyTorch tensors
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        # Remove the batch dimension added by the tokenizer
        return {key: val.squeeze(0) for key, val in inputs.items()}

def load_concepts(concepts_dir, tokenizer, max_length, batch_size, min_examples=10):
    """Loads concepts from text files in a directory."""
    concepts = []
    concept_files = glob.glob(os.path.join(concepts_dir, "*.txt"))
    if not concept_files:
        raise ValueError(f"No concept files (.txt) found in {concepts_dir}")

    logging.info(f"Found concept files: {concept_files}")

    for i, file_path in enumerate(sorted(concept_files)): # Sort for consistent IDs
        concept_name = os.path.splitext(os.path.basename(file_path))[0]
        dataset = TextDataset(file_path, tokenizer, max_length)

        if len(dataset) < min_examples:
            logging.warning(
                f"Skipping concept '{concept_name}' from {file_path} "
                f"due to insufficient examples ({len(dataset)} < {min_examples})."
            )
            continue

        # This collate function prepares batches for the model's forward pass
        # It needs to match what Captum expects when extracting activations.
        # Returning a tuple matching the model's forward args is required here.
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            # Include token_type_ids if the model uses them (e.g., BERT)
            token_type_ids = None
            if 'token_type_ids' in batch[0]:
                 token_type_ids = torch.stack([item['token_type_ids'] for item in batch])

            # Return a tuple in the order expected by model.forward
            # (input_ids, attention_mask, token_type_ids, ...)
            # Adjust if your specific model expects a different order or args.
            if token_type_ids is not None:
                return (input_ids, attention_mask, token_type_ids)
            else:
                # If model doesn't use/need token_type_ids, just return the required ones
                return (input_ids, attention_mask)

        # Important: drop_last=True can prevent errors if the last batch is smaller
        # than what the classifier expects, especially with parallel processing.
        data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

        concepts.append(Concept(id=i, name=concept_name, data_iter=data_iter))
        logging.info(f"Loaded concept: '{concept_name}' (ID: {i}) with {len(dataset)} examples from {file_path}")

    if not concepts:
         raise ValueError(f"Could not load any valid concepts from {concepts_dir}")

    return concepts

def get_layer_from_name(model, layer_name):
    """Helper to get a layer module object from its string name."""
    try:
        # Split the name and traverse the model attributes
        layers = layer_name.split('.')
        module = model
        for layer in layers:
            module = getattr(module, layer)
        return module
    except AttributeError:
        logging.error(f"Layer '{layer_name}' not found in model.")
        # You could print available layer names here for debugging:
        # for name, _ in model.named_modules(): print(name)
        return None
    except Exception as e:
        logging.error(f"Error accessing layer {layer_name}: {e}")
        return None


# --- Main Script ---


"""Runs the TCAV analysis."""
# Define a wrapper for the model's forward pass to return only logits
def forward_wrapper(*args, **kwargs):
    # Remove potential 'output_attentions' and 'output_hidden_states' if Captum adds them
    kwargs.pop('output_attentions', None)
    kwargs.pop('output_hidden_states', None)
    # Ensure inputs are on the correct device
    args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

    output = model(*args, **kwargs)
    # Extract logits from the output object
    if hasattr(output, 'logits'):
        return output.logits
    else:
        # Fallback if the output is already logits (less likely for SeqClassification)
        return output

# --- Add this function definition somewhere after helper functions ---

def interpret_single_sample(text_sample, mytcav_instance, tokenizer, device, all_concepts, target_class, max_len):
    """Tokenizes a single text sample and runs TCAV interpretation against multiple concept sets."""
    logging.info(f"\n--- Interpreting Single Sample ---")
    logging.info(f"Text: {text_sample}")

    # 1. Tokenize the single sample
    inputs_tokenized = tokenizer(
        [text_sample],
        return_tensors='pt',
        max_length=max_len,
        padding='max_length',
        truncation=True
    )
    inputs_tokenized = {key: val.to(device) for key, val in inputs_tokenized.items()}

    # 2. Prepare inputs and additional args
    interpret_inputs = inputs_tokenized['input_ids']
    interpret_add_args = []
    if 'attention_mask' in inputs_tokenized:
        interpret_add_args.append(inputs_tokenized['attention_mask'])
    if 'token_type_ids' in inputs_tokenized:
        interpret_add_args.append(inputs_tokenized['token_type_ids'])
    interpret_add_args = tuple(interpret_add_args) if interpret_add_args else None

    # 3. Find random and original concepts
    random_sarcastic_concept = next((c for c in all_concepts if c.name == "random_sarcastic"), None)
    random_non_sarcastic_concept = next((c for c in all_concepts if c.name == "random_non_sarcastic"), None)
    if not random_sarcastic_concept or not random_non_sarcastic_concept:
        logging.error("Could not find random concepts. Ensure random_sarcastic.txt and random_non_sarcastic.txt exist.")
        return

    original_concepts = [c for c in all_concepts if c.name not in ["random_sarcastic", "random_non_sarcastic"]]
    skip_concepts = {"juxtaposition", "low_expectations", "playing_on_stereotypes", "self-deprecation"} # Keep skipping problematic ones

    if not original_concepts:
        logging.warning("No non-random concepts found.")
        return

    # Store results for ONE layer only for this specific plotting function
    # TCAV typically returns scores per layer. We need to pick one or aggregate.
    # Let's assume we use the *first* layer specified in mytcav_instance.layers
    target_layer_for_plot = mytcav_instance.layers[0] if mytcav_instance.layers else None
    if not target_layer_for_plot:
        logging.error("No target layer found in TCAV instance for plotting.")
        return

    logging.info(f"Collecting scores for plotting from layer: {target_layer_for_plot}")
    results_for_plot = {} # <<< Changed structure

    # 4. Run interpretation loop (Modified result collection)
    print(f"\n--- TCAV Results for Sample: '{text_sample[:50]}...' ---")
    print(f"Target Class: {target_class}")

    for original_concept in original_concepts:
        if original_concept.name in skip_concepts:
            logging.info(f"Skipping known problematic concept: {original_concept.name}")
            continue

        print(f"\nProcessing concept: {original_concept.name} (ID: {original_concept.id})")
        # Ensure the random concepts have different IDs if they came from the same load_concepts call
        current_experimental_set = [original_concept, random_sarcastic_concept, random_non_sarcastic_concept]
        concept_names_in_set = "-".join([c.name for c in current_experimental_set])


        try:
            single_tcav_scores = mytcav_instance.interpret(
                inputs=interpret_inputs,
                experimental_sets=[current_experimental_set], # List with one set
                target=target_class,
                additional_forward_args=interpret_add_args
            )

            # Extract scores for the target layer
            set_key = list(single_tcav_scores.keys())[0] # Should be the concept name string
            layer_scores = single_tcav_scores[set_key].get(target_layer_for_plot) # Get scores for the specific layer

            if layer_scores:
                print(f"  Layer: {target_layer_for_plot}")
                magnitude = layer_scores.get('magnitude')
                sign_count = layer_scores.get('sign_count') # For sensitivity later if needed

                # <<< --- Store results for plotting --- >>>
                if magnitude is not None and magnitude.numel() >= 3:
                    # Assuming order in magnitude tensor matches current_experimental_set
                    # Shape might be [1, 3] or [3]
                    score_val = magnitude[0, 0].item() if magnitude.dim() == 2 else magnitude[0].item()
                    rand_sarc_score = magnitude[0, 1].item() if magnitude.dim() == 2 else magnitude[1].item()
                    rand_non_sarc_score = magnitude[0, 2].item() if magnitude.dim() == 2 else magnitude[2].item()

                    results_for_plot[original_concept.name] = {
                        'score': score_val,
                        'rand_sarc': rand_sarc_score,
                        'rand_non_sarc': rand_non_sarc_score
                    }
                    print(f"    Raw Magnitude Scores: Concept={score_val:.4f}, RandSarc={rand_sarc_score:.4f}, RandNonSarc={rand_non_sarc_score:.4f}")
                else:
                    logging.warning(f"Could not extract sufficient magnitude scores for concept '{original_concept.name}' in layer '{target_layer_for_plot}'.")
                 # <<< --- End storing results --- >>>

                # Print sensitivity if needed (optional)
                # ... (sensitivity calculation code can remain if desired) ...

            else:
                 logging.warning(f"No scores found for layer '{target_layer_for_plot}' for concept '{original_concept.name}'.")


        except Exception as e:
            logging.error(f"Error during interpretation for concept '{original_concept.name}': {e}", exc_info=True) # Keep exc_info for debugging interpret errors
            continue

    # --- Call the new plotting function from plot.py ---
    if results_for_plot:
        plot_normalized_tcav_results(
            results_dict=results_for_plot,
            text_sample=text_sample,
            target_class=target_class,
            layer_name=target_layer_for_plot # Pass the layer name used
        )
    else:
        logging.warning(f"No results collected for plotting for sample: {text_sample[:60]}...")


# --- Example call in the main script execution part ---

# 1. Load Model and Tokenizer
logging.info(f"Loading model and tokenizer: {MODEL_NAME}")
# Verify target layers exist and get module objects
logging.info(f"Verifying target layers: {TARGET_LAYER_NAMES}")
# Keep this verification loop to ensure names are valid
valid_layer_names = []
for layer_name in TARGET_LAYER_NAMES:
    layer_module = get_layer_from_name(model, layer_name)
    if layer_module is not None:
        valid_layer_names.append(layer_name) # Store the valid name
    else:
        logging.warning(f"Could not find layer: {layer_name}, skipping.")

if not valid_layer_names:
    raise ValueError("No valid target layers found. Please check TARGET_LAYER_NAMES.")

# 2. Load Concepts
logging.info(f"\nLoading concepts from: {CONCEPTS_DIR}")
concepts = load_concepts(CONCEPTS_DIR, tokenizer, MAX_LENGTH, BATCH_SIZE, min_examples=10)
# 3. Initialize TCAV - Pass the list of layer NAMES
#    and configure the attribution method to use the wrapper
logging.info(f"\nInitializing TCAV for layers: {valid_layer_names}")

# Explicitly create the default attribution method but with our wrapper
# Layer will be set internally by TCAV for each layer in valid_layer_names
layer_attrib = LayerGradientXActivation(forward_func=forward_wrapper, layer=None)

mytcav = TCAV(model=model, # TCAV still needs the model for activation generation
                layers=valid_layer_names, # Pass the list of layer names (strings)
                layer_attr_method=layer_attrib) # Pass our configured attribution method


# Example: Pick one sarcastic and one non-sarcastic sample
sample_to_interpret_sarcastic = df[df['is_sarcastic'] == 1]['headline'].iloc[0]
sample_to_interpret_non_sarcastic = df[df['is_sarcastic'] == 0]['headline'].iloc[0]

# Pass the full 'concepts' list to the function now
interpret_single_sample(sample_to_interpret_sarcastic, mytcav, tokenizer, device, concepts, target_class=1, max_len=MAX_LENGTH) # Assuming target=1 for sarcasm
interpret_single_sample(sample_to_interpret_non_sarcastic, mytcav, tokenizer, device, concepts, target_class=0, max_len=MAX_LENGTH) # Assuming target=0 for non-sarcasm


# ... (Continue with the rest of your script, e.g., the batch interpretation) ...

                    
