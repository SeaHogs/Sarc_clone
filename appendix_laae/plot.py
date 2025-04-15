import matplotlib.pyplot as plt
import numpy as np
import json
import os
import logging
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.patches import Patch

def ensure_plot_directory(directory="plot"):
    """Ensure the plot directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

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

def plot_normalized_tcav_results(results_dict, text_sample, target_class, layer_name="", save_dir="plot"):
    """
    Plots the normalized TCAV results for a single sample. Shows the full text sample in the title.
    Saves the plot to the specified directory instead of displaying.

    Args:
        results_dict (dict): A dictionary where keys are concept names and values are
                             dictionaries {'score': float, 'rand_sarc': float, 'rand_non_sarc': float}.
        text_sample (str): The text sample being analyzed.
        target_class (int or str): The target class for the interpretation.
        layer_name (str, optional): Name of the layer analyzed. Defaults to "".
        save_dir (str, optional): Directory to save the plot. Defaults to "plot".
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
        display_text = text_sample
        title += f'\nSample: "{display_text}"'
    ax.set_title(title)
    ax.axvline(0, color='grey', linewidth=0.8) # Add line at zero

    # Add labels to bars
    ax.bar_label(bars, fmt='%.3f', padding=3)

    plt.tight_layout() # Adjust layout
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Create a safe filename from the text sample
    safe_text = text_sample[:30].replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    filename = f"{save_dir}/tcav_normalized_{safe_text}_{target_class}.png"
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved TCAV plot to {filename}")

# New functions for visualizing evaluation results

def load_results(directory="results"):
    """Load all result files from the specified directory structure."""
    results = {
        "usefulness": {},
        "robustness": {},
        "comprehensizeness": {}
    }
    
    # Load usefulness results
    usefulness_dir = os.path.join(directory, "usefulness")
    if os.path.exists(usefulness_dir):
        for file in os.listdir(usefulness_dir):
            if file.endswith(".json"):
                method = file.replace("results_", "").replace(".json", "")
                with open(os.path.join(usefulness_dir, file), 'r') as f:
                    results["usefulness"][method] = json.load(f)
    
    # Load robustness results
    robustness_dir = os.path.join(directory, "robustness")
    if os.path.exists(robustness_dir):
        for file in os.listdir(robustness_dir):
            if file.endswith(".json"):
                method = file.replace("results_", "").replace(".json", "")
                with open(os.path.join(robustness_dir, file), 'r') as f:
                    results["robustness"][method] = json.load(f)
    
    # Load comprehensiveness results
    comprehensiveness_dir = os.path.join(directory, "comprehensizeness")
    if os.path.exists(comprehensiveness_dir):
        for file in os.listdir(comprehensiveness_dir):
            if file.endswith(".json"):
                method = file.replace("results_", "").replace(".json", "")
                with open(os.path.join(comprehensiveness_dir, file), 'r') as f:
                    results["comprehensizeness"][method] = json.load(f)
    
    return results

def plot_usefulness_score_comparison(results, fig_size=(10, 6), save_dir="plot"):
    """Plot comparison of only the usefulness scores across methods."""
    if not results.get("usefulness"):
        print("No usefulness results found to plot.")
        return
    
    methods = list(results["usefulness"].keys())
    
    # Create a figure with 2 subplots (general and significant)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Extract just the usefulness scores
    general_usefulness = []
    significant_usefulness = []
    
    for method in methods:
        if "general" in results["usefulness"][method]:
            general_usefulness.append(results["usefulness"][method]["general"].get("usefulness", 0))
        else:
            general_usefulness.append(0)
            
        if "significant" in results["usefulness"][method]:
            significant_usefulness.append(results["usefulness"][method]["significant"].get("usefulness", 0))
        else:
            significant_usefulness.append(0)
    
    # Set width of bars
    bar_width = 0.35
    
    # Create positions for grouped bars
    positions = np.arange(len(methods))
    
    # Plot general and significant usefulness scores side by side
    general_bars = ax.bar(positions - bar_width/2, general_usefulness, 
                        width=bar_width, label='General', color='#3498db')
    sig_bars = ax.bar(positions + bar_width/2, significant_usefulness, 
                   width=bar_width, label='Significant', color='#e74c3c')
    
    # Add value labels on top of bars
    for bars in [general_bars, sig_bars]:
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Usefulness Score')
    ax.set_title('Overall Usefulness Score Comparison')
    ax.set_xticks(positions)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Save the figure
    filename = f"{save_dir}/usefulness_score_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved usefulness score comparison to {filename}")

def plot_usefulness_breakdown_comparison(results, fig_size=(15, 10), save_dir="plot"):
    """Plot breakdown of usefulness component metrics (excluding overall usefulness)."""
    if not results.get("usefulness"):
        print("No usefulness results found to plot.")
        return
    
    methods = list(results["usefulness"].keys())
    
    # Create a figure with 2 rows for general and significant metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)
    
    # Component metrics to extract (exclude usefulness itself)
    component_metrics = ["noise_ratio", "info_score", "misleading_score"]
    
    # Prepare data for general metrics
    general_data = {metric: [] for metric in component_metrics}
    for method in methods:
        if "general" in results["usefulness"][method]:
            for metric in component_metrics:
                general_data[metric].append(results["usefulness"][method]["general"].get(metric, 0))
        else:
            for metric in component_metrics:
                general_data[metric].append(0)
    
    # Prepare data for significant metrics
    significant_data = {metric: [] for metric in component_metrics}
    for method in methods:
        if "significant" in results["usefulness"][method]:
            for metric in component_metrics:
                significant_data[metric].append(results["usefulness"][method]["significant"].get(metric, 0))
        else:
            for metric in component_metrics:
                significant_data[metric].append(0)
    
    # Set width of bars
    bar_width = 0.25
    
    # Generate positions for bars
    positions = np.arange(len(methods))
    
    # Colors for metrics
    colors = {'noise_ratio': '#f39c12', 'info_score': '#2ecc71', 'misleading_score': '#e74c3c'}
    
    # Plot general metrics
    for i, metric in enumerate(component_metrics):
        bars = ax1.bar(positions + (i-1)*bar_width, general_data[metric], 
                width=bar_width, label=metric.replace('_', ' ').title(), color=colors[metric])
        # Add small value labels
        ax1.bar_label(bars, fmt='%.2f', padding=3, fontsize=8, rotation=90)
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Score')
    ax1.set_title('General Metrics Breakdown')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([m.upper() for m in methods])
    ax1.legend()
    
    # Plot significant metrics
    for i, metric in enumerate(component_metrics):
        bars = ax2.bar(positions + (i-1)*bar_width, significant_data[metric], 
                width=bar_width, label=metric.replace('_', ' ').title(), color=colors[metric])
        # Add small value labels
        ax2.bar_label(bars, fmt='%.2f', padding=3, fontsize=8, rotation=90)
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Score')
    ax2.set_title('Significant Metrics Breakdown')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([m.upper() for m in methods])
    ax2.legend()
    
    # Add a note explaining the metrics
    note = ("Note: Usefulness = (1 - Noise Ratio) * (Info Score - Misleading Score)\n"
            "• Lower Noise Ratio is better (fewer irrelevant words)\n"
            "• Higher Info Score is better (more keyword groups covered)\n"
            "• Lower Misleading Score is better (fewer wrong contradictory words)")
    plt.figtext(0.5, 0.01, note, ha='center', fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the note
    plt.suptitle('Usefulness Components Breakdown', fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Save the figure
    filename = f"{save_dir}/usefulness_breakdown_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved usefulness breakdown comparison to {filename}")

def plot_robustness_comparison(results, fig_size=(12, 6), save_dir="plot"):
    """Plot comparison of robustness metrics across methods."""
    if not results.get("robustness"):
        print("No robustness results found to plot.")
        return
    
    methods = list(results["robustness"].keys())
    metrics = []
    
    # Get all metrics from the first method
    if methods:
        metrics = list(results["robustness"][methods[0]].keys())
    
    # Extract data for plotting
    data = {}
    for metric in metrics:
        data[metric] = []
        for method in methods:
            data[metric].append(results["robustness"][method].get(metric, 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Set width of bars
    bar_width = 0.2
    
    # Generate positions for bars
    positions = np.arange(len(methods))
    
    # Colors for metrics
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        ax.bar(positions + i*bar_width, data[metric], 
               width=bar_width, label=metric.replace("ratio_of_", "").replace("_", " ").title(), 
               color=colors[i % len(colors)])
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Robustness Metrics Comparison')
    ax.set_xticks(positions + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    
    plt.tight_layout()
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Save the figure
    filename = f"{save_dir}/robustness_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved robustness comparison to {filename}")

def plot_comprehensiveness_comparison(results, fig_size=(10, 6), save_dir="plot"):
    """Plot comparison of comprehensiveness metrics across methods."""
    if not results.get("comprehensizeness"):
        print("No comprehensiveness results found to plot.")
        return
    
    methods = list(results["comprehensizeness"].keys())
    metrics = []
    
    # Get all metrics from the first method
    if methods:
        metrics = list(results["comprehensizeness"][methods[0]].keys())
    
    # Extract data for plotting
    data = {}
    for metric in metrics:
        data[metric] = []
        for method in methods:
            data[metric].append(results["comprehensizeness"][method].get(metric, 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Set width of bars
    bar_width = 0.35
    
    # Generate positions for bars
    positions = np.arange(len(methods))
    
    # Colors for metrics
    colors = ['#3498db', '#e74c3c']
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        display_name = metric.replace("ratio_of_", "").replace("_comprehensiveness", "").replace("_", " ").title()
        ax.bar(positions + i*bar_width, data[metric], 
               width=bar_width, label=display_name, 
               color=colors[i % len(colors)])
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Comprehensiveness Metrics Comparison')
    ax.set_xticks(positions + bar_width / 2)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    
    plt.tight_layout()
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Save the figure
    filename = f"{save_dir}/comprehensiveness_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved comprehensiveness comparison to {filename}")

def plot_radar_chart(results, fig_size=(12, 10), save_dir="plot"):
    """Plot radar chart comparing key metrics across methods."""
    # Extract key metrics from each result type
    metrics = {
        "Usefulness": {},
        "Robustness": {},
        "Comprehensiveness": {}
    }
    
    methods = set()
    
    # Extract usefulness metrics (main usefulness score only)
    if results.get("usefulness"):
        for method, data in results["usefulness"].items():
            methods.add(method)
            metrics["Usefulness"][method] = data.get("general", {}).get("usefulness", 0)
    
    # Extract robustness metrics (average of all scores)
    if results.get("robustness"):
        for method, data in results["robustness"].items():
            methods.add(method)
            metrics["Robustness"][method] = sum(data.values()) / len(data) if data else 0
    
    # Extract comprehensiveness metrics (general comprehensiveness)
    if results.get("comprehensizeness"):
        for method, data in results["comprehensizeness"].items():
            methods.add(method)
            comp_key = "ratio_of_constructive_comprehensiveness"
            metrics["Comprehensiveness"][method] = data.get(comp_key, 0)
    
    # Convert methods set to sorted list
    methods = sorted(list(methods))
    
    # Prepare data for radar chart
    categories = list(metrics.keys())
    N = len(categories)
    
    # Set up radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
    
    # Add lines and fill area for each method
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, method in enumerate(methods):
        values = [metrics[cat].get(method, 0) for cat in categories]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method.upper(), color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set radial limits
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Overall Performance Comparison', size=15, pad=20)
    plt.tight_layout()
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Save the figure
    filename = f"{save_dir}/radar_chart_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved radar chart comparison to {filename}")

def plot_all_metrics_comparison(results, fig_size=(15, 12), save_dir="plot"):
    """Create a comprehensive visualization comparing all metrics across methods."""
    # Check if we have results to plot
    if not any(results.values()):
        print("No results found to plot.")
        return
    
    # Create a list of all available methods
    all_methods = set()
    for result_type in results.values():
        all_methods.update(result_type.keys())
    methods = sorted(list(all_methods))
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(3, 2, figure=fig)
    
    # Create subplots for different metric categories
    ax_usefulness_general = fig.add_subplot(gs[0, 0])
    ax_usefulness_significant = fig.add_subplot(gs[0, 1])
    ax_robustness = fig.add_subplot(gs[1, :])
    ax_comprehensiveness = fig.add_subplot(gs[2, :])
    
    # Plot usefulness metrics (general)
    if results.get("usefulness"):
        usefulness_general_data = {}
        for method in methods:
            if method in results["usefulness"] and "general" in results["usefulness"][method]:
                for metric, value in results["usefulness"][method]["general"].items():
                    if metric not in usefulness_general_data:
                        usefulness_general_data[metric] = []
                    # Ensure all methods have a value (fill with 0 if missing)
                    while len(usefulness_general_data[metric]) < methods.index(method):
                        usefulness_general_data[metric].append(0)
                    usefulness_general_data[metric].append(value)
        
        # Ensure all lists have the same length
        for metric in usefulness_general_data:
            while len(usefulness_general_data[metric]) < len(methods):
                usefulness_general_data[metric].append(0)
        
        # Plot general usefulness metrics
        bar_width = 0.2
        positions = np.arange(len(methods))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(usefulness_general_data.keys()):
            ax_usefulness_general.bar(positions + i*bar_width, usefulness_general_data[metric], 
                    width=bar_width, label=metric.capitalize(), color=colors[i % len(colors)])
        
        ax_usefulness_general.set_title('General Usefulness Metrics')
        ax_usefulness_general.set_xticks(positions + bar_width * (len(usefulness_general_data) - 1) / 2)
        ax_usefulness_general.set_xticklabels([m.upper() for m in methods])
        ax_usefulness_general.legend()
    
    # Plot usefulness metrics (significant)
    if results.get("usefulness"):
        usefulness_significant_data = {}
        for method in methods:
            if method in results["usefulness"] and "significant" in results["usefulness"][method]:
                for metric, value in results["usefulness"][method]["significant"].items():
                    if metric not in usefulness_significant_data:
                        usefulness_significant_data[metric] = []
                    # Ensure all methods have a value (fill with 0 if missing)
                    while len(usefulness_significant_data[metric]) < methods.index(method):
                        usefulness_significant_data[metric].append(0)
                    usefulness_significant_data[metric].append(value)
        
        # Ensure all lists have the same length
        for metric in usefulness_significant_data:
            while len(usefulness_significant_data[metric]) < len(methods):
                usefulness_significant_data[metric].append(0)
        
        # Plot significant usefulness metrics
        bar_width = 0.2
        positions = np.arange(len(methods))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(usefulness_significant_data.keys()):
            ax_usefulness_significant.bar(positions + i*bar_width, usefulness_significant_data[metric], 
                    width=bar_width, label=metric.capitalize(), color=colors[i % len(colors)])
        
        ax_usefulness_significant.set_title('Significant Usefulness Metrics')
        ax_usefulness_significant.set_xticks(positions + bar_width * (len(usefulness_significant_data) - 1) / 2)
        ax_usefulness_significant.set_xticklabels([m.upper() for m in methods])
        ax_usefulness_significant.legend()
    
    # Plot robustness metrics
    if results.get("robustness"):
        robustness_data = {}
        for method in methods:
            if method in results["robustness"]:
                for metric, value in results["robustness"][method].items():
                    if metric not in robustness_data:
                        robustness_data[metric] = []
                    # Ensure all methods have a value (fill with 0 if missing)
                    while len(robustness_data[metric]) < methods.index(method):
                        robustness_data[metric].append(0)
                    robustness_data[metric].append(value)
        
        # Ensure all lists have the same length
        for metric in robustness_data:
            while len(robustness_data[metric]) < len(methods):
                robustness_data[metric].append(0)
        
        # Plot robustness metrics
        bar_width = 0.2
        positions = np.arange(len(methods))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(robustness_data.keys()):
            display_name = metric.replace("ratio_of_", "").replace("_", " ").title()
            ax_robustness.bar(positions + i*bar_width, robustness_data[metric], 
                   width=bar_width, label=display_name, color=colors[i % len(colors)])
        
        ax_robustness.set_title('Robustness Metrics')
        ax_robustness.set_xticks(positions + bar_width * (len(robustness_data) - 1) / 2)
        ax_robustness.set_xticklabels([m.upper() for m in methods])
        ax_robustness.legend()
    
    # Plot comprehensiveness metrics
    if results.get("comprehensizeness"):
        comprehensiveness_data = {}
        for method in methods:
            if method in results["comprehensizeness"]:
                for metric, value in results["comprehensizeness"][method].items():
                    if metric not in comprehensiveness_data:
                        comprehensiveness_data[metric] = []
                    # Ensure all methods have a value (fill with 0 if missing)
                    while len(comprehensiveness_data[metric]) < methods.index(method):
                        comprehensiveness_data[metric].append(0)
                    comprehensiveness_data[metric].append(value)
        
        # Ensure all lists have the same length
        for metric in comprehensiveness_data:
            while len(comprehensiveness_data[metric]) < len(methods):
                comprehensiveness_data[metric].append(0)
        
        # Plot comprehensiveness metrics
        bar_width = 0.35
        positions = np.arange(len(methods))
        colors = ['#3498db', '#e74c3c']
        
        for i, metric in enumerate(comprehensiveness_data.keys()):
            display_name = metric.replace("ratio_of_", "").replace("_comprehensiveness", "").replace("_", " ").title()
            ax_comprehensiveness.bar(positions + i*bar_width, comprehensiveness_data[metric], 
                   width=bar_width, label=display_name, color=colors[i % len(colors)])
        
        ax_comprehensiveness.set_title('Comprehensiveness Metrics')
        ax_comprehensiveness.set_xticks(positions + bar_width / 2)
        ax_comprehensiveness.set_xticklabels([m.upper() for m in methods])
        ax_comprehensiveness.legend()
    
    plt.suptitle('Comprehensive Metrics Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Save the figure
    filename = f"{save_dir}/all_metrics_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved comprehensive metrics comparison to {filename}")

def create_summary_table(results, save_dir="plot"):
    """Create a summary table with key metrics for each method."""
    # Get all methods
    all_methods = set()
    for result_type in results.values():
        all_methods.update(result_type.keys())
    methods = sorted(list(all_methods))
    
    # Initialize table data
    table_data = {
        'Method': methods,
        'Usefulness': [],
        'Noise Ratio': [],
        'Info Score': [],
        'Misleading Score': [],
        'Robustness': [],
        'Comprehensiveness': []
    }
    
    # Fill usefulness metrics
    for method in methods:
        if method in results.get("usefulness", {}):
            general = results["usefulness"][method].get("general", {})
            table_data['Usefulness'].append(general.get("usefulness", 0))
            table_data['Noise Ratio'].append(general.get("noise_ratio", 0))
            table_data['Info Score'].append(general.get("info_score", 0))
            table_data['Misleading Score'].append(general.get("misleading_score", 0))
        else:
            table_data['Usefulness'].append(0)
            table_data['Noise Ratio'].append(0)
            table_data['Info Score'].append(0)
            table_data['Misleading Score'].append(0)
    
    # Fill robustness (average of all metrics)
    for method in methods:
        if method in results.get("robustness", {}):
            avg_robustness = sum(results["robustness"][method].values()) / len(results["robustness"][method])
            table_data['Robustness'].append(avg_robustness)
        else:
            table_data['Robustness'].append(0)
    
    # Fill comprehensiveness (general metric)
    for method in methods:
        if method in results.get("comprehensizeness", {}):
            comp_key = "ratio_of_constructive_comprehensiveness"
            table_data['Comprehensiveness'].append(results["comprehensizeness"][method].get(comp_key, 0))
        else:
            table_data['Comprehensiveness'].append(0)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Plot table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Format numbers to 3 decimal places
    cell_text = []
    for i in range(len(df)):
        row = [df.iloc[i, 0]]  # Method name
        row.extend([f"{float(val):.3f}" for val in df.iloc[i, 1:]])
        cell_text.append(row)
    
    table = ax.table(cellText=cell_text, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Summary of Key Metrics', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Ensure directory exists
    ensure_plot_directory(save_dir)
    
    # Save the figure
    filename = f"{save_dir}/summary_table.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved summary table to {filename}")
    
    # Also save as CSV for easier access
    csv_filename = f"{save_dir}/summary_table.csv"
    df.to_csv(csv_filename, index=False, float_format='%.3f')
    logging.info(f"Saved summary table data to {csv_filename}")
    
    # Print table for easy copy-paste
    print("Summary Table (for copy-paste):")
    print(df.to_string(float_format="{:.3f}".format))

def visualize_all_results(save_dir="plot"):
    """Load results and generate all visualizations, saving to the specified directory."""
    print("Loading evaluation results...")
    results = load_results()
    
    # Ensure the plot directory exists
    ensure_plot_directory(save_dir)
    print(f"\nGenerating visualizations in directory: {save_dir}")
    
    # Summary table first
    print("\n1. Creating Summary Table...")
    create_summary_table(results, save_dir=save_dir)
    
    # Individual metric comparisons
    print("\n2. Creating Usefulness Score Comparison...")
    plot_usefulness_score_comparison(results, save_dir=save_dir)
    
    print("\n3. Creating Usefulness Breakdown Comparison...")
    plot_usefulness_breakdown_comparison(results, save_dir=save_dir)
    
    print("\n4. Creating Robustness Metrics Comparison...")
    plot_robustness_comparison(results, save_dir=save_dir)
    
    print("\n5. Creating Comprehensiveness Metrics Comparison...")
    plot_comprehensiveness_comparison(results, save_dir=save_dir)
    
    # Overall comparisons
    print("\n6. Creating Radar Chart Comparison...")
    plot_radar_chart(results, save_dir=save_dir)
    
    print("\n7. Creating Comprehensive Metrics Comparison...")
    plot_all_metrics_comparison(results, save_dir=save_dir)
    
    print(f"\nAll visualizations completed and saved to {save_dir}/")

# Run all visualizations when the script is executed directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    visualize_all_results()
