import os
import json
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from deepseek import query_llm_json

# Create concepts directory if it doesn't exist
os.makedirs("concepts", exist_ok=True)

# Load headlines from file
def load_headlines(filename="sarcastic_headlines_200.txt"):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]

# Predefined list of concepts
concepts = ["Exaggeration",
"Understatement",
"Absurdity",
"Dark Humor",
"Mocking Authority",
"Mocking Celebrities",
"Mocking Corporations",
"Political Satire",
"Everyday Banality",
"Juxtaposition",
"Playing on Stereotypes",
"Self-deprecation",
"Critique of Media/Consumerism",
"Failure/Disappointment",
"Low Expectations",
"Personification/Anthropomorphism",
"Weird News",
"Taboo Subjects"]

# Process a single headline and extract concepts
def process_headline(headline):
    prompt = f"""
    Analyze this headline and identify which of the following concepts it contains.
    Return a JSON object with the headline and an array of concepts that apply.
    Only include concepts from the list below that are clearly present in the headline.
    
    Headline: "{headline}"
    
    Available concepts:
    {', '.join(concepts)}
    
    Example format:
    {{
        "headline": "example headline",
        "concepts": ["concept1", "concept2"]
    }}
    """
    
    try:
        result = query_llm_json(prompt, temperature=0)
        return result
    except Exception as e:
        print(f"Error processing headline '{headline}': {e}")
        return {"headline": headline, "concepts": []}

# Worker function for thread pool
def worker(headlines, start_idx, end_idx, results):
    for i in range(start_idx, end_idx):
        if i < len(headlines):
            headline = headlines[i]
            print(f"Processing headline {i+1}/200: '{headline}'")
            result = process_headline(headline)
            results[i] = result
            time.sleep(0.5)  # Small delay to avoid rate limiting

# Main execution
def main():
    # Load headlines
    headlines = load_headlines()
    print(f"Loaded {len(headlines)} headlines")
    
    # Initialize results array
    results = [None] * len(headlines)
    
    # Calculate chunk size for each thread
    num_threads = 40
    chunk_size = len(headlines) // num_threads
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_threads - 1 else len(headlines)
        
        thread = threading.Thread(
            target=worker,
            args=(headlines, start_idx, end_idx, results)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Initialize concept files dictionary with all predefined concepts
    concept_files = {concept: [] for concept in concepts}
    
    # Process results and organize concepts
    for result in results:
        if result and "concepts" in result:
            for concept in result["concepts"]:
                if concept in concept_files:
                    concept_files[concept].append(result["headline"])
    
    # Write results to concept files
    for concept, headlines in concept_files.items():
        # Create a safe filename from the concept
        safe_filename = concept.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = f"concepts/{safe_filename}.txt"
        
        with open(filename, "w") as f:
            for headline in headlines:
                f.write(f"{headline}\n")
        
        print(f"Created file: {filename} with {len(headlines)} headlines")
    
    # Save all results to a JSON file for reference
    with open("concepts/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete. Created {len(concept_files)} concept files.")
    print("All results saved to concepts/all_results.json")

# generate ramdon sarcastic and non sarcastic concepts
    
    # Load non-sarcastic headlines
    
if __name__ == "__main__":
    main()
