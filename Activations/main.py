import argparse
import os
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from utils import load_model, create_example, check_example, grade_coherence
import warnings

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="You are not using LayerNorm, so the writing weights can't be centered! Skipping")

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

def setup_directories(model_name, concept):
    save_dir = os.path.join(script_dir, f"results/{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    concept_save_dir = os.path.join(save_dir, concept.replace(' ', '_'))
    os.makedirs(concept_save_dir, exist_ok=True)
    return save_dir, concept_save_dir

def save_concept_results(concept_save_dir, concept_trials_results):
    concept_results_path = os.path.join(concept_save_dir, "results.json")
    with open(concept_results_path, "w") as f:
        json.dump(concept_trials_results, f, indent=4)

def save_overall_coherence(save_dir, all_coherence_scores):
    overall_coherence_rate = np.mean(all_coherence_scores)
    coherence_rate_path = os.path.join(save_dir, "overall_coherence_rate.txt")
    with open(coherence_rate_path, "w") as f:
        f.write(str(overall_coherence_rate))
    print(f"Overall coherence rate: {overall_coherence_rate}")
    print(f"Results saved to {save_dir}")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2-small", help="The model to use for the evaluation.")
parser.add_argument("--num_trials", type=int, default=1, help="The number of times to test coherence for each concept.")
parser.add_argument("--no_activations", action="store_true", help="If set, activations will not be captured.")
args = parser.parse_args()

# Load concepts from csv
concepts_df = pd.read_csv(os.path.join(script_dir, "concepts.csv"))
concepts = concepts_df["concept"].tolist()

# Load model
model = load_model(args.model)

# Loop through concepts: test coherence and store activations
all_coherence_scores = []
for concept in tqdm(concepts, desc="Processing concepts"):
    save_dir, concept_save_dir = setup_directories(args.model, concept)

    concept_trials_results = {}
    for trial_idx in range(args.num_trials):
        # Generate an example of the concept
        all_activations_for_concept = {}
    for trial_idx in range(args.num_trials):
        # Generate an example of the concept
        example, activations = create_example(concept, model, no_activations=args.no_activations)

        # Store activations for the current trial
        if not args.no_activations:
            all_activations_for_concept[f"trial_{trial_idx}"] = activations

        # Check for coherence
        coherence_response = check_example(concept, example, model, no_activations=args.no_activations)
        
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

    # Save all activations for the current concept to a single file
    if not args.no_activations:
        activations_path = os.path.join(concept_save_dir, "all_activations.pt")
        torch.save(all_activations_for_concept, activations_path)

    save_concept_results(concept_save_dir, concept_trials_results)

save_overall_coherence(save_dir, all_coherence_scores)