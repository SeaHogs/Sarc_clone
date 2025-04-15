import json
import os
# from openai import client  # Assuming client is initialized in openai.py
from openai import OpenAI, Client as openai_client # Use aliased import
from sklearn.metrics import confusion_matrix, f1_score
import time
import threading # Add threading
from concurrent.futures import ThreadPoolExecutor, as_completed # Add concurrent futures
openai_api_key = ""
'''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the dataset
'''
'''
sarcastic_headlines = []  
non_sarcastic_headlines = []
with open('Sarcasm_Headlines_Dataset.json', 'r') as file:
    for line in file:
        obj = json.loads(line)
        if obj['is_sarcastic'] == 1:
            sarcastic_headlines.append(obj['headline'])
        else:
            non_sarcastic_headlines.append(obj['headline'])

import random
sarcastic_headlines = random.sample(sarcastic_headlines, 50)
non_sarcastic_headlines = random.sample(non_sarcastic_headlines, 50)

# save ramdon 50 sarcastic and non sarcastic headlines to a file
with open('validation/val_random_sarcastic.txt', 'w') as file:
    for headline in sarcastic_headlines:
        file.write(headline + '\n')

with open('validation/val_random_non_sarcastic.txt', 'w') as file:
    for headline in non_sarcastic_headlines:
        file.write(headline + '\n')
'''

# Modified to optionally load only sarcastic headlines
def load_validation_data(sarcastic_file="validation/val_random_sarcastic.txt", non_sarcastic_file="validation/val_random_non_sarcastic.txt", load_only_sarcastic=False):
    """Loads headlines and their labels from validation files.
       Can optionally load only sarcastic headlines."""
    data = []
    with open(sarcastic_file, 'r') as f:
        for line in f:
            headline = line.strip()
            if headline:
                data.append((headline, 1))  # 1 for sarcastic

    if not load_only_sarcastic:
        with open(non_sarcastic_file, 'r') as f:
            for line in f:
                headline = line.strip()
                if headline:
                    data.append((headline, 0))  # 0 for non-sarcastic
    # If only loading sarcastic, return list of headlines directly
    return [item[0] for item in data] if load_only_sarcastic else data


def classify_headline(headline: str, model_name: str, client: OpenAI) -> int:
    """Classifies a single headline using the specified OpenAI model."""
    prompt = f"""Is the following headline sarcastic or not sarcastic? Answer with only 'sarcastic' or 'not sarcastic'.

Headline: {headline}
Answer:"""
    try:
        response = client.chat.completions.create( # Use aliased client
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1.0 if model_name == "o1-mini" else 0.0
        )
        answer = response.choices[0].message.content.strip().lower()
        print(f"Headline: {headline} | Model: {model_name} | Raw Answer: {answer}") # Debug print
        if "not sarcastic" in answer:
            return 0
        elif "sarcastic" in answer:
            return 1
        else:
            print(f"Warning: Could not parse answer for '{headline}': {answer}. Defaulting to 0 (not sarcastic).")
            return 0 # Default or handle uncertainty
    except Exception as e:
        print(f"Error classifying headline '{headline}' with model {model_name}: {e}")
        # Optional: Add retry logic or specific error handling
        time.sleep(1) # Basic rate limiting/error backoff
        return -1 # Indicate error


def evaluate_model(model_name: str, validation_data: list):
    """Evaluates a model on the validation data using threads and returns metrics."""
    predictions = []
    true_labels = []
    # Use a list of dictionaries for results_data to avoid ordering issues with threads
    results_data_list = []
    results_lock = threading.Lock() # Lock for thread safety

    print(f"--- Evaluating model: {model_name} ---")

    # Worker function to be executed by each thread
    def worker(headline, label, client):
        prediction = classify_headline(headline, model_name, client)
        with results_lock: # Ensure only one thread modifies lists at a time
            if prediction != -1: # Only include successful classifications
                # Append results associated with this specific headline
                 results_data_list.append({
                     "headline": headline,
                     "true_label": label,
                     "prediction": prediction,
                     # Store the original index if needed for ordering later, though not strictly necessary for metrics
                 })
            else:
                 results_data_list.append({
                     "headline": headline,
                     "true_label": label,
                     "prediction": "Error"
                 })
        # No sleep needed here as ThreadPoolExecutor manages threads

    # Use ThreadPoolExecutor for parallel execution
    futures = []
    client = OpenAI(api_key=openai_api_key)
    # Limit to 10 threads as requested
    with ThreadPoolExecutor(max_workers=10) as executor:
        for headline, label in validation_data:
            futures.append(executor.submit(worker, headline, label, client))

        # Wait for all futures to complete (optional: add progress tracking here)
        for future in as_completed(futures):
            try:
                future.result() # Check for exceptions raised in worker threads
            except Exception as exc:
                print(f'A headline classification generated an exception: {exc}')
                # Decide how to handle exceptions, e.g., log them

    # Process results collected from all threads
    # Sort results_data_list by original order if necessary, though calculation doesn't require it
    # For metric calculation, extract predictions and labels where prediction was successful
    for result in results_data_list:
        if result["prediction"] != "Error":
            predictions.append(result["prediction"])
            true_labels.append(result["true_label"])

    if not true_labels:
        print(f"No successful predictions for model {model_name}. Cannot calculate metrics.")
        return None, None, results_data_list # Return the collected data list

    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    f1 = f1_score(true_labels, predictions, average='binary') # Use binary for sarcastic class (1)

    print(f"Model: {model_name}")
    # Fix: Print multi-line f-string content separately or format differently
    print("Confusion Matrix:")
    print(cm)
    print(f"F1 Score (Sarcastic): {f1:.4f}")
    print("-" * 30)

    return cm, f1, results_data_list # Return the collected data list


def save_results(model_name: str, cm, f1, results_data: list, output_dir="results"):
    """Saves the evaluation results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_results.json")

    # Convert numpy arrays/types to native Python types for JSON serialization
    if cm is not None:
        cm_list = cm.tolist()
    else:
        cm_list = None

    if f1 is not None:
        f1_float = float(f1)
    else:
        f1_float = None

    results = {
        "model": model_name,
        "confusion_matrix (TN, FP, FN, TP)": cm_list, # [[TN, FP], [FN, TP]] for labels [0, 1]
        "f1_score_sarcastic": f1_float,
        "individual_results": results_data
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")


# --- New Function for Sarcasm Explanation ---
def explain_sarcasm(headline: str, client: OpenAI) -> dict:
    """Asks gpt-4o to explain sarcasm and extract keywords for a headline."""
    model_name = "gpt-4o" # Use gpt-4o for explanation
    prompt = f"""Analyze the following headline and explain why it is sarcastic. Also, identify the top 3 keywords or short phrases that most strongly signal the sarcasm.

Headline: {headline}

Respond ONLY with a valid JSON object containing two keys:
1.  "explanation": A string explaining the sarcasm.
2.  "keywords": A list of 3 strings representing the top sarcastic keywords/phrases.

Example JSON format:
{{
  "explanation": "The sarcasm arises from...",
  "keywords": ["keyword1", "phrase 2", "key word 3"]
}}

JSON Response:"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing text for sarcasm and providing structured JSON output."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5, # Allow some creativity but keep it focused
                response_format={ "type": "json_object" } # Request JSON output directly
            )
            content = response.choices[0].message.content.strip()
            print(f"Headline: {headline} | Explanation Raw Response: {content}") # Debug print

            # Attempt to parse the JSON response
            explanation_data = json.loads(content)

            # Validate structure
            if isinstance(explanation_data, dict) and \
               "explanation" in explanation_data and \
               "keywords" in explanation_data and \
               isinstance(explanation_data["explanation"], str) and \
               isinstance(explanation_data["keywords"], list) and \
               len(explanation_data["keywords"]) <= 3 and \
               all(isinstance(kw, str) for kw in explanation_data["keywords"]):
                return {
                    "headline": headline,
                    "explanation": explanation_data["explanation"],
                    "keywords": explanation_data["keywords"]
                }
            else:
                print(f"Warning: Invalid JSON structure received for '{headline}': {content}")
                # Fall through to retry or return error

        except json.JSONDecodeError as json_err:
            print(f"Error decoding JSON response for '{headline}': {json_err}. Response: {content}")
            # Fall through to retry or return error
        except Exception as e:
            print(f"Error getting explanation for headline '{headline}' with model {model_name}: {e}")
            time.sleep(1 + attempt) # Exponential backoff for retries

    # If all retries fail
    return {
        "headline": headline,
        "explanation": "Error: Could not get explanation.",
        "keywords": []
    }

def save_explanations(explanations: list, output_dir="results", filename="sarcasm_explanations.json"):
    """Saves the sarcasm explanations to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        json.dump(explanations, f, indent=4)
    print(f"Sarcasm explanations saved to {output_path}")


# --- New Function for Concept Classification ---
def classify_concepts(headline: str, concepts_list: list, client: OpenAI) -> dict:
    """Asks gpt-4o to classify a headline into up to 3 predefined concepts."""
    model_name = "gpt-4o"
    concepts_str = "\n".join([f"- {c}" for c in concepts_list])

    prompt = f"""Consider the following sarcastic headline:

Headline: {headline}

From the list of concepts below, select up to 3 concepts that best describe the type or theme of sarcasm used in the headline. 

IMPORTANT: You MUST respond with a valid JSON array containing only the concept names. For example: ["Exaggeration", "Everyday Banality"]

Available Concepts:
{concepts_str}

JSON Response (must be a JSON array):"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing sarcastic text. When asked to classify concepts, ALWAYS respond with a JSON array of concept names, never with key-value pairs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content.strip()
            print(f"Headline: {headline} | Concept Classification Raw Response: {content}")

            # Parse the JSON response
            raw_output = json.loads(content)
            
            # Handle different response structures
            selected_concepts = []
            
            # If we got a list directly (preferred format)
            if isinstance(raw_output, list):
                selected_concepts = raw_output
            # If we got an object with a list inside
            elif isinstance(raw_output, dict):
                # Try to find a list in the object
                for key, value in raw_output.items():
                    if isinstance(value, list):
                        selected_concepts = value
                        break
                
                # If we couldn't find a list, check if keys are concepts themselves
                if not selected_concepts:
                    # Use the keys from the dict as concepts (e.g., {"Exaggeration": "Absurdity"} -> ["Exaggeration"])
                    concept_keys = [k for k in raw_output.keys() if k in concepts_list]
                    if concept_keys:
                        selected_concepts = concept_keys[:3]  # Take up to 3
                    
                    # Also check values if they're strings and in the concepts list
                    concept_values = [v for v in raw_output.values() 
                                     if isinstance(v, str) and v in concepts_list]
                    selected_concepts.extend(concept_values)
                    selected_concepts = list(set(selected_concepts))[:3]  # Deduplicate and limit to 3
            
            # Validate and ensure only valid concepts
            valid_concepts = [c for c in selected_concepts if isinstance(c, str) and c in concepts_list]
            final_concepts = valid_concepts[:3]  # Ensure max 3
            
            return {
                "headline": headline,
                "concepts": final_concepts
            }

        except Exception as e:
            print(f"Error classifying concepts for '{headline}': {e}")
            time.sleep(1 + attempt)
    
    return {
        "headline": headline,
        "concepts": ["Error: Could not classify concepts."]
    }

def save_concept_classifications(classifications: list, output_dir="results", filename="classify_concepts_using_gpt4o.json"):
    """Saves the concept classifications to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        json.dump(classifications, f, indent=4)
    print(f"Concept classifications saved to {output_path}")

# --- Main Execution Logic ---
if __name__ == "__main__":

    run_evaluation = False # Set to True to run the classification evaluation
    run_explanation = False# Set to True to run the sarcasm explanation
    run_concept_classification = True # Set to True to run the concept classification
    client = OpenAI(api_key=openai_api_key) # Create client instance

    # --- Classification Evaluation ---
    if run_evaluation:
        print("--- Starting Classification Evaluation ---")
        validation_data = load_validation_data() # Load both sarcastic and non-sarcastic
        models_to_evaluate = ["gpt-4o-mini", "gpt-4o", "o1-mini"] # Add "o1-mini" back if needed, but expect errors

        all_results = {}
        for model in models_to_evaluate:
            cm, f1, results_data = evaluate_model(model, validation_data)
            all_results[model] = {"confusion_matrix": cm, "f1_score": f1}
            save_results(model, cm, f1, results_data)

        print("\n--- Evaluation Summary ---")
        for model, metrics in all_results.items():
            cm_str = metrics["confusion_matrix"].tolist() if metrics["confusion_matrix"] is not None else "N/A"
            f1_str = f"{metrics['f1_score']:.4f}" if metrics['f1_score'] is not None else "N/A"
            print(f"Model: {model} | F1 (Sarcastic): {f1_str} | CM: {cm_str}")
        print("-" * 30 + "\n")


    # --- Sarcasm Explanation ---
    if run_explanation:
        print("--- Starting Sarcasm Explanation ---")
        # Load only sarcastic headlines
        sarcastic_headlines = load_validation_data(load_only_sarcastic=True)
        print(f"Loaded {len(sarcastic_headlines)} sarcastic headlines for explanation.")

        explanation_results = []
        explanation_lock = threading.Lock()

        def explanation_worker(headline):
            result = explain_sarcasm(headline, client)
            with explanation_lock:
                explanation_results.append(result)

        # Use ThreadPoolExecutor for parallel explanation requests
        # Fewer workers might be better due to potentially longer response times for explanations
        explanation_futures = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for headline in sarcastic_headlines:
                explanation_futures.append(executor.submit(explanation_worker, headline))

            # Wait for all futures to complete
            for future in as_completed(explanation_futures):
                try:
                    future.result() # Check for exceptions
                except Exception as exc:
                    print(f'An explanation worker generated an exception: {exc}')

        # Save the collected explanations
        save_explanations(explanation_results)
    
    if run_concept_classification:
        print("--- Starting Concept Classification ---")
        sarcastic_headlines = load_validation_data(load_only_sarcastic=True)
        if not sarcastic_headlines:
            print("Concept classification skipped: Could not load sarcastic headlines.")
        else:
            print(f"Loaded {len(sarcastic_headlines)} sarcastic headlines for concept classification.")

            concept_results = []
            concept_lock = threading.Lock()
            SARCASTIC_CONCEPTS = ["Exaggeration",
                                "Understatement",
                                "Absurdity",
                                "Dark Humor",
                                "Mocking Authority",
                                "Mocking Celebrities",
                                "Mocking Corporations",
                                "Political Satire",
                                "Everyday Banality",
                                "Critique of Media/Consumerism",
                                "Failure/Disappointment",
                                "Personification/Anthropomorphism",
                                "Weird News",
                                "Taboo Subjects"]

            def concept_worker(headline):
                
                # Pass the initialized client and the global concepts list
                result = classify_concepts(headline, SARCASTIC_CONCEPTS, client)
                with concept_lock:
                    concept_results.append(result)

            concept_futures = []
            # Use ThreadPoolExecutor similar to explanations
            with ThreadPoolExecutor(max_workers=5) as executor:
                for headline in sarcastic_headlines:
                    concept_futures.append(executor.submit(concept_worker, headline))

                # Wait for all futures to complete
                for future in as_completed(concept_futures):
                    try:
                        future.result() # Check for exceptions
                    except Exception as exc:
                        print(f'A concept classification worker generated an exception: {exc}')

            # Save the collected concept classifications
            save_concept_classifications(concept_results)
            print("-" * 30 + "\n")

    print("--- Script Finished ---")












