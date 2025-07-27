import argparse
import os
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from utils import load_model, create_example, check_example, grade_coherence

os.environ["TOKENIZERS_PARALLELISM"] = "false"

script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2-small", help="The model to use for the evaluation.")
parser.add_argument("--num_trials", type=int, default=1, help="The number of times to test coherence for each concept.")
args = parser.parse_args()

# Load concepts from csv
concepts_df = pd.read_csv(os.path.join(script_dir, "concepts.csv"))
concepts = concepts_df["concept"].tolist()

# Create save directory within the Activations folder
save_dir = os.path.join(script_dir, f"results/{args.model}")
os.makedirs(save_dir, exist_ok=True)

# Load model
model = load_model(args.model)

all_coherence_scores = []

for concept in tqdm(concepts, desc="Processing concepts"):
    # Create save directory for each concept
    concept_save_dir = os.path.join(save_dir, concept.replace(' ', '_'))
    os.makedirs(concept_save_dir, exist_ok=True)

    concept_trials_results = {}
    for trial_idx in range(args.num_trials):
        # Generate an example of the concept
        prompt = f"Please provide an example of {concept}."
        example, activations = create_example(prompt, model)

        # Get and save activations for the generation
        activations_path = os.path.join(concept_save_dir, f"activations_trial_{trial_idx}.pt")
        torch.save(activations, activations_path)

        # Check for coherence
        coherence_response = check_example(concept, example, model)
        
        # Grade coherence
        coherent = grade_coherence(coherence_response)
        all_coherence_scores.append(coherent)

        # Store the results for the current trial
        trial_results = {
            "example": example,
            "coherence_response": coherence_response,
            "coherent": coherent,
        }
        concept_trials_results[f"trial_{trial_idx}"] = trial_results

    # Save all trials for the current concept to a single JSON file
    concept_results_path = os.path.join(concept_save_dir, "results.json")
    with open(concept_results_path, "w") as f:
        json.dump(concept_trials_results, f, indent=4)

# Calculate and save the overall coherence rate
overall_coherence_rate = np.mean(all_coherence_scores)
coherence_rate_path = os.path.join(save_dir, "overall_coherence_rate.txt")
with open(coherence_rate_path, "w") as f:
    f.write(str(overall_coherence_rate))

print(f"Overall coherence rate: {overall_coherence_rate}")
print(f"Results saved to {save_dir}")
