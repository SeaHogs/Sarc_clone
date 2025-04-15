import json
import numpy as np
gpt_explanations = None
ig_explanations = None
lime_explanations = None

with open('explain_using_gpt4o.json', 'r') as f:
    gpt_explanations = json.load(f)

with open('explain_using_ig.json', 'r') as f:
    ig_explanations = json.load(f)

with open('explain_using_lime.json', 'r') as f:
    lime_explanations = json.load(f)

with open('explain_using_tcav.json', 'r') as f:
    tcav_explanations = json.load(f)

def get_ratio_of_correct_constructive_words(classified_constructive_words, keywords, verbose=False):
    if verbose:
        print(f"classified_constructive_words: {classified_constructive_words}")
        print(f"keywords: {keywords}")
    num_constructive_words_in_keywords = len([word for word in classified_constructive_words if word in keywords])
    num_constructive_words = len(classified_constructive_words)
    return num_constructive_words_in_keywords / num_constructive_words if num_constructive_words > 0 else 1

def get_ratio_of_not_wrong_contradictory_words(classified_contradictory_words, keywords, verbose=False):
    if verbose:
        print(f"classified_contradictory_words: {classified_contradictory_words}")
        print(f"keywords: {keywords}")
    num_contradictory_words_not_in_keywords = len([word for word in classified_contradictory_words if word not in keywords])
    num_contradictory_words = len(classified_contradictory_words)
    return num_contradictory_words_not_in_keywords / num_contradictory_words if num_contradictory_words > 0 else 1

def get_ratio_of_comprehensiveness(words, keyword_groups):
   touched = [0 for _ in range(len(keyword_groups))]
   for i in range(len(keyword_groups)):
       keyword_group = keyword_groups[i]
       keywords = keyword_group.split(' ')
       for word in words:
           if word in keywords:
               touched[i] = 1
   return sum(touched) / len(touched)

def get_keywords(explanation):
    res = []
    for keyword in explanation['keywords']:
        res.extend(keyword.split(' '))
    return res

def get_keyword_groups(explanation):
    return explanation['keywords']

def get_concept_groups(explanation):
    # lowercase, replace spaces with underscores    
    def transform_concept(concept):
        return concept.lower().replace(' ', '_')
    return [transform_concept(concept) for concept in explanation['concepts']]

def get_90_confidence_interval(weights):
    average_weight = sum(weights) / len(weights)
    std_weight = np.std(weights)
    z_score = 1.645
    abs_to_avg = z_score * std_weight
    return average_weight, abs_to_avg

def get_lime_word_groups(lime_obj):
    lime_explanation = lime_obj['explanation']
    objs =  {word: weight for word, weight in lime_explanation}
    sorted_objs = sorted(objs.items(), key=lambda x: x[1], reverse=True)
    constructive_words = [word for word, weight in sorted_objs if weight > 0]
    contradictory_words = [word for word, weight in sorted_objs if weight < 0]
    average_weight, abs_to_avg = get_90_confidence_interval(list(objs.values()))
    def is_significantly_positive(weight):
        return weight > abs_to_avg
    def is_significantly_negative(weight):
        return weight < -abs_to_avg
    significantly_constructive_words = [word for word, weight in sorted_objs if is_significantly_positive(weight)]
    significantly_contradictory_words = [word for word, weight in sorted_objs if is_significantly_negative(weight)]
    return constructive_words, contradictory_words, significantly_constructive_words, significantly_contradictory_words

def get_ig_word_groups(ig_obj):
    ig_explanation = ig_obj['explanation']
    objs =  {obj["token"]: obj["weight"] for obj in ig_explanation}
    objs_filtered = {word: weight for word, weight in objs.items() if word not in ['[CLS]', '[SEP]']}
    constructive_words = [word for word, weight in objs_filtered.items() if weight > 0]
    contradictory_words = [word for word, weight in objs_filtered.items() if weight < 0]

    average_weight, abs_to_avg = get_90_confidence_interval(list(objs_filtered.values()))
    def is_significantly_positive(weight):
        return weight > abs_to_avg
    def is_significantly_negative(weight):
        return weight < -abs_to_avg
    
    significantly_constructive_words = [word for word, weight in objs_filtered.items() if is_significantly_positive(weight)]
    significantly_contradictory_words = [word for word, weight in objs_filtered.items() if is_significantly_negative(weight)]
    return constructive_words, contradictory_words, significantly_constructive_words, significantly_contradictory_words

def get_tcav_word_groups(tcav_obj):
    tcav_explanation = tcav_obj['explanation']
    def normalize_single_exp_obj(exp_arr):
        
        concept_array, random_sarcastic_array, random_non_sarcastic_array = exp_arr[0], exp_arr[1], exp_arr[2]
        
        concept_name, weight = concept_array[0], concept_array[1]
        random_sarcastic_weight = random_sarcastic_array[1]
        random_non_sarcastic_weight = random_non_sarcastic_array[1]
       
        width = abs(random_sarcastic_weight - random_non_sarcastic_weight)
        normalized_weight = (weight - random_sarcastic_weight) / width
        return {"word": concept_name, "weight": normalized_weight}
    objs =  [normalize_single_exp_obj(exp_arr) for exp_arr in tcav_explanation]
    constructive_words = [obj["word"] for obj in objs if obj["weight"] > 0]
    contradictory_words = [obj["word"] for obj in objs if obj["weight"] < 0]
    average_weight, abs_to_avg = get_90_confidence_interval([obj["weight"] for obj in objs])
    def is_significantly_positive(weight):
        return weight > abs_to_avg
    def is_significantly_negative(weight):
        return weight < -abs_to_avg
    significantly_constructive_words = [obj["word"] for obj in objs if is_significantly_positive(obj["weight"])]
    significantly_contradictory_words = [obj["word"] for obj in objs if is_significantly_negative(obj["weight"])]
    return constructive_words, contradictory_words, significantly_constructive_words, significantly_contradictory_words

with open('results/classify_concepts_using_gpt4o.json', 'r') as f:
    gpt_concepts = json.load(f)
key_word_to_obj_map = {}
for obj in gpt_explanations:
    headline = obj['headline']
    key_word_to_obj_map[headline] = obj

for obj in gpt_concepts:
    headline = obj['headline']
    key_word_to_obj_map[headline]['concepts'] = obj['concepts']

def eval_comprehensizeness(test_explanations, get_word_groups_func, get_targets_func):
    count = 0
    total_constructive_comprehensiveness = 0
    total_significant_constructive_comprehensiveness = 0
    for test_obj in test_explanations:
        if test_obj['prediction'] == "Not Sarcastic":
            continue
        count += 1
        text = test_obj['text']
        keyword_groups = get_targets_func(key_word_to_obj_map[text])
        constructive_words, _, significantly_constructive_words, _ = get_word_groups_func(test_obj)
        ratio_of_correct_constructive_words = get_ratio_of_comprehensiveness(constructive_words, keyword_groups)
        ratio_of_correct_significant_constructive_words = get_ratio_of_comprehensiveness(significantly_constructive_words, keyword_groups)
        total_constructive_comprehensiveness += ratio_of_correct_constructive_words
        total_significant_constructive_comprehensiveness += ratio_of_correct_significant_constructive_words
    result = {
        "ratio_of_constructive_comprehensiveness": total_constructive_comprehensiveness / count if count > 0 else 1,
        "ratio_of_significant_constructive_comprehensiveness": total_significant_constructive_comprehensiveness / count if count > 0 else 1
    }
    return result


def eval_robustness(test_explanations, get_word_groups_func, get_targets_func):
    count = 0
    total_correct_constructive_words = 0
    total_not_wrong_contradictory_words = 0
    total_significant_constructive_words = 0
    total_significant_contradictory_words = 0
    for test_obj in test_explanations:
        if test_obj['prediction'] == "Not Sarcastic":
            continue
        count += 1
        text = test_obj['text']
        print(f"text: {text}")
        constructive_words, contradictory_words, significantly_constructive_words, significantly_contradictory_words = get_word_groups_func(test_obj)
        print(f"constructive_words: {constructive_words}\ncontradictory_words: {contradictory_words}\nsignificantly_constructive_words: {significantly_constructive_words}\nsignificantly_contradictory_words: {significantly_contradictory_words}")
        targets = get_targets_func(key_word_to_obj_map[text])
        print(f"targets: {targets}")
        ratio_of_correct_constructive_words = get_ratio_of_correct_constructive_words(constructive_words, targets)
        ratio_of_not_wrong_contradictory_words = get_ratio_of_not_wrong_contradictory_words(contradictory_words, targets)
        ratio_of_significant_constructive_words = get_ratio_of_correct_constructive_words(significantly_constructive_words, targets)
        ratio_of_significant_contradictory_words = get_ratio_of_not_wrong_contradictory_words(significantly_contradictory_words, targets)
        total_correct_constructive_words += ratio_of_correct_constructive_words
        total_not_wrong_contradictory_words += ratio_of_not_wrong_contradictory_words
        total_significant_constructive_words += ratio_of_significant_constructive_words
        total_significant_contradictory_words += ratio_of_significant_contradictory_words
    result = {
        "ratio_of_correct_constructive_words": total_correct_constructive_words / count if count > 0 else 1,
        "ratio_of_not_wrong_contradictory_words": total_not_wrong_contradictory_words / count if count > 0 else 1,
        "ratio_of_correct_significantly_constructive_words": total_significant_constructive_words / count if count > 0 else 1,
        "ratio_of_not_wrong_significantly_contradictory_words": total_significant_contradictory_words / count if count > 0 else 1
    }
    return result

def eval_and_write_results(eval_func, test_explanations, get_word_groups_func, get_targets_func, file_name):
    result = eval_func(test_explanations, get_word_groups_func, get_targets_func)
    print(f"{file_name}: {result}")
    with open(file_name, 'w') as f:
        json.dump(result, f)
import os
def eval_and_write_robustness(test_explanations, get_word_groups_func, get_targets_func, file_name):
    os.makedirs(f"results/robustness", exist_ok=True)
    eval_and_write_results(eval_robustness, test_explanations, get_word_groups_func, get_targets_func, f"results/robustness/{file_name}")

def eval_and_write_comprehensizeness(test_explanations, get_word_groups_func, get_targets_func, file_name):
    os.makedirs(f"results/comprehensizeness", exist_ok=True)
    eval_and_write_results(eval_comprehensizeness, test_explanations, get_word_groups_func, get_targets_func, f"results/comprehensizeness/{file_name}")

def calculate_noise_ratio(constructive_words, keywords):
    """Calculate the ratio of constructive words that are actually in keywords (noise ratio).
    
    A lower noise ratio is better - it means fewer constructive words were irrelevant.
    """
    if not constructive_words:
        return 0.0
        
    # Count how many constructive words are actually in the keywords (hits)
    constructive_hits = len([word for word in constructive_words if word in keywords])
    return constructive_hits / len(constructive_words)

def calculate_info_score(constructive_words, keyword_groups):
    """Calculate how many keyword groups were hit by constructive words.
    
    A higher info score is better - more keyword groups were covered.
    """
    if not keyword_groups:
        return 1.0  # If no keyword groups, perfect coverage
        
    # Check if any word in the constructive list hits each keyword group
    hits = 0
    for keyword_group in keyword_groups:
        keywords = keyword_group.split(' ')
        for word in constructive_words:
            if word in keywords:
                hits += 1
                break  # Count only one hit per keyword group
    
    return hits / len(keyword_groups)

def calculate_misleading_score(contradictory_words, keywords):
    """Calculate how many contradictory words are actually in keywords (misleading).
    
    A lower misleading score is better - fewer contradictory words were wrong.
    """
    if not contradictory_words:
        return 0.0  # If no contradictory words, no misleading
        
    # Count contradictory words that appear in keywords (should be non-contradictory)
    misleading_count = len([word for word in contradictory_words if word in keywords])
    return misleading_count / len(contradictory_words)

def calculate_usefulness(constructive_words, contradictory_words, keyword_groups, keywords):
    """Calculate the overall usefulness metric.
    
    usefulness = (1 - noise_ratio) * (info_score - misleading_score)
    """
    noise_ratio = calculate_noise_ratio(constructive_words, keywords)
    info_score = calculate_info_score(constructive_words, keyword_groups)
    misleading_score = calculate_misleading_score(contradictory_words, keywords)
    
    usefulness = (1 - noise_ratio) * (info_score - misleading_score)
    return usefulness, noise_ratio, info_score, misleading_score

def eval_usefulness(test_explanations, get_word_groups_func, get_targets_func, get_target_groups_func):
    """Evaluate usefulness across test explanations for both general and significant word groups."""
    count = 0
    # General metrics
    total_usefulness = 0
    total_noise_ratio = 0
    total_info_score = 0
    total_misleading_score = 0
    
    # Significant metrics
    total_sig_usefulness = 0
    total_sig_noise_ratio = 0
    total_sig_info_score = 0
    total_sig_misleading_score = 0
    
    for test_obj in test_explanations:
        if test_obj['prediction'] == "Not Sarcastic":
            continue
        count += 1
        text = test_obj['text']
        
        # Get keyword data
        keywords = get_targets_func(key_word_to_obj_map[text])
        keyword_groups = get_target_groups_func(key_word_to_obj_map[text])
        
        # Get all word groups from explanation
        constructive_words, contradictory_words, sig_constructive_words, sig_contradictory_words = get_word_groups_func(test_obj)
        
        # Calculate metrics for general words
        usefulness, noise_ratio, info_score, misleading_score = calculate_usefulness(
            constructive_words, contradictory_words, keyword_groups, keywords
        )
        
        # Calculate metrics for significant words
        sig_usefulness, sig_noise_ratio, sig_info_score, sig_misleading_score = calculate_usefulness(
            sig_constructive_words, sig_contradictory_words, keyword_groups, keywords
        )
        
        # Accumulate general metrics
        total_usefulness += usefulness
        total_noise_ratio += noise_ratio
        total_info_score += info_score
        total_misleading_score += misleading_score
        
        # Accumulate significant metrics
        total_sig_usefulness += sig_usefulness
        total_sig_noise_ratio += sig_noise_ratio
        total_sig_info_score += sig_info_score
        total_sig_misleading_score += sig_misleading_score
        
        print(f"Text: {text}")
        print(f"General - Usefulness: {usefulness:.4f}, Noise Ratio: {noise_ratio:.4f}, "
              f"Info Score: {info_score:.4f}, Misleading Score: {misleading_score:.4f}")
        print(f"Significant - Usefulness: {sig_usefulness:.4f}, Noise Ratio: {sig_noise_ratio:.4f}, "
              f"Info Score: {sig_info_score:.4f}, Misleading Score: {sig_misleading_score:.4f}")
        
    # Create result dictionary with both sets of metrics
    result = {
        "general": {
            "usefulness": total_usefulness / count if count > 0 else 0,
            "noise_ratio": total_noise_ratio / count if count > 0 else 0,
            "info_score": total_info_score / count if count > 0 else 0,
            "misleading_score": total_misleading_score / count if count > 0 else 0
        },
        "significant": {
            "usefulness": total_sig_usefulness / count if count > 0 else 0,
            "noise_ratio": total_sig_noise_ratio / count if count > 0 else 0,
            "info_score": total_sig_info_score / count if count > 0 else 0,
            "misleading_score": total_sig_misleading_score / count if count > 0 else 0
        }
    }
    return result

def eval_and_write_usefulness(test_explanations, get_word_groups_func, get_targets_func, get_target_groups_func, file_name):
    """Evaluate usefulness and write results to file."""
    os.makedirs(f"results/usefulness", exist_ok=True)
    result = eval_usefulness(test_explanations, get_word_groups_func, get_targets_func, get_target_groups_func)
    
    # Print results in a structured way
    print(f"\nUsefulness metrics for {file_name}:")
    print("=== General Words ===")
    for metric, value in result["general"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("=== Significant Words ===")
    for metric, value in result["significant"].items():
        print(f"  {metric}: {value:.4f}")
    
    # Write the complete results to file
    with open(f"results/usefulness/{file_name}", 'w') as f:
        json.dump(result, f, indent=2)
'''
#eval_and_write_comprehensizeness(lime_explanations, get_lime_word_groups, get_keyword_groups, 'results_lime.json')
#eval_and_write_comprehensizeness(ig_explanations, get_ig_word_groups, get_keyword_groups, 'results_ig.json')
eval_and_write_comprehensizeness(tcav_explanations, get_tcav_word_groups, get_concept_groups, 'results_tcav.json')
#eval_and_write_robustness(lime_explanations, get_lime_word_groups, get_keywords, 'results_lime.json')
#eval_and_write_robustness(ig_explanations, get_ig_word_groups, get_keywords, 'results_ig.json')
eval_and_write_robustness(tcav_explanations, get_tcav_word_groups, get_concept_groups, 'results_tcav.json')
'''
eval_and_write_usefulness(lime_explanations, get_lime_word_groups, get_keywords, get_keyword_groups, 'results_lime.json')
eval_and_write_usefulness(ig_explanations, get_ig_word_groups, get_keywords, get_keyword_groups, 'results_ig.json')
eval_and_write_usefulness(tcav_explanations, get_tcav_word_groups, get_concept_groups, get_concept_groups, 'results_tcav.json')
