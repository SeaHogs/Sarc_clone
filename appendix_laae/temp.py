import json
import logging # Added for consistency with plot.py logging
from plot import plot_normalized_tcav_results # Assuming plot.py is accessible

# Configure logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

target_text = "sun thinking of just collapsing now and getting this all over with"
json_file = 'explain_using_tcav.json'
# Assuming the TCAV results were generated for this layer, as specified in run_TCAV.py
layer_name_to_use = "bert.pooler.dense"

# 1. Load the data from the JSON file
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        all_explanations = json.load(f)
    logging.info(f"Successfully loaded data from {json_file}")
except FileNotFoundError:
    logging.error(f"Error: File not found at {json_file}")
    exit()
except json.JSONDecodeError:
    logging.error(f"Error: Could not decode JSON from {json_file}")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred while loading the file: {e}")
    exit()

# 2. Find the specific explanation object based on the text
target_explanation_obj = None
for explanation in all_explanations:
    if explanation.get('text') == target_text:
        target_explanation_obj = explanation
        break

# 3. Process and plot if the explanation was found
if target_explanation_obj:
    logging.info(f"Found explanation for: '{target_text}'")

    # Extract necessary data
    raw_explanation_data = target_explanation_obj.get('explanation')
    prediction_label = target_explanation_obj.get('prediction') # e.g., "Sarcastic"
    text_sample = target_explanation_obj.get('text')

    if not all([raw_explanation_data, prediction_label, text_sample]):
        logging.error("Error: Found entry is missing required fields ('explanation', 'prediction', 'text'). Cannot plot.")
    else:
        # Convert prediction label to target class index (assuming 1 for Sarcastic, 0 for Not Sarcastic)
        target_class_index = 1 if prediction_label == "Sarcastic" else 0
        logging.info(f"Prediction: {prediction_label} (Target Class Index: {target_class_index})")

        # Parse the raw explanation data into the format needed by the plot function
        # Expected format: {'concept_name': {'score': float, 'rand_sarc': float, 'rand_non_sarc': float}}
        results_for_plot = {}
        try:
            for concept_set in raw_explanation_data:
                # Expecting [[concept_name, score], [rand_sarc_name, score], [rand_non_sarc_name, score]]
                if isinstance(concept_set, list) and len(concept_set) == 3 and \
                   all(isinstance(item, list) and len(item) == 2 for item in concept_set):

                    concept_name = concept_set[0][0]
                    concept_score = float(concept_set[0][1])
                    # Assuming the order stored was: original, random_sarcastic, random_non_sarcastic
                    random_sarc_score = float(concept_set[1][1])
                    random_non_sarc_score = float(concept_set[2][1])

                    results_for_plot[concept_name] = {
                        'score': concept_score,
                        'rand_sarc': random_sarc_score,
                        'rand_non_sarc': random_non_sarc_score
                    }
                else:
                    logging.warning(f"Skipping malformed concept set in explanation: {concept_set}")
        except (TypeError, IndexError, ValueError) as parse_error:
             logging.error(f"Error parsing explanation data: {parse_error}. Data: {raw_explanation_data}")
             results_for_plot = {} # Clear potentially corrupt data

        # 4. Call the plotting function if data was successfully parsed
        if results_for_plot:
            logging.info(f"Plotting results for {len(results_for_plot)} concepts.")
            plot_normalized_tcav_results(
                results_dict=results_for_plot,
                text_sample=text_sample,
                target_class=target_class_index, # Use the numeric index
                layer_name=layer_name_to_use
            )
        else:
            logging.error("Could not parse any valid concept data from the explanation. Plotting skipped.")

else:
    logging.warning(f"Explanation for the text '{target_text}' not found in {json_file}.")