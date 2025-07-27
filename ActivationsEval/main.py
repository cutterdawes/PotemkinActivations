import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd

from ActivationsEval.utils import (
    answer_and_grade_benchmark_question,
    answer_open_ended_question,
    edit_to_introduce_error,
    generate_subquestions,
    grade_open_ended_question,
    relies_on_concept,
    sample_question
)

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default="gpt2-small", # Default to a HookedTransformer model
                    help="Name of the HookedTransformer model to test (e.g., gpt2-small, gpt2-medium)")
parser.add_argument("--benchmark",
                    type=str,
                    default="mmlu",
                    choices=["mmlu", "bbh"])
parser.add_argument("--num_subquestions",
                    type=int,
                    default=5,
                    help="number of subquestions (m) to use from the benchmark (for now assume k =1)")
parser.add_argument("--num_trials",
                    type=int,
                    default=10,
                    help="number of concepts to test")
parser.add_argument("--judge_model",
                    type=str,
                    default="gpt-4o", # External LLM for helper tasks
                    help="External LLM model to use for judge/helper tasks (e.g., gpt-4o, gemini-1.5-pro)")
args = parser.parse_args()

# Set up save directory
results_dir = f"results_transformer_lens/{args.model}/{args.benchmark}"
os.makedirs(results_dir, exist_ok=True)

index_to_category = {0: "initial", 1: "edited_with_error"}
category_to_coherence = defaultdict(list)
overall_coherence = []
score_per_concept = []
bar = tqdm(range(args.num_trials))
score_per_concept_std_err = float("inf")
for trial_index in bar:
    # Sample questions until we find one that depends on a concept and is answered correctly.
    while True:
        question, answer, subject = sample_question(
            args.benchmark, 
        )
        # See if question relies on a concept.
        concept_classification, concept = relies_on_concept(question, args.judge_model)
        if concept_classification:
            # See if question is answered correctly.
            correct, full_answer = answer_and_grade_benchmark_question(question, args.model, answer, args.benchmark == "mmlu")
            if correct:
                break
    # Generate subquestions.
    max_attempts = 10
    while max_attempts > 0:
        subquestions = generate_subquestions(question, concept, args.judge_model, args.num_subquestions)
        if len(subquestions) == args.num_subquestions:
            break
        print(f"Failed to generate {args.num_subquestions} subquestions (generated {len(subquestions)} instead). {max_attempts} attempts remaining. Retrying...")
        max_attempts -= 1
    subquestion_bar = tqdm(enumerate(subquestions), total=len(subquestions), desc="Subquestions")
    for index, subquestion in subquestion_bar:
        extracted_answer = answer_open_ended_question(subquestion, args.model)
        # Modify answer to either introduce or remove errors. 
        _, answer_with_error = edit_to_introduce_error(subquestion, extracted_answer, args.judge_model)
        # Self-grading
        expected_answers = ["correct", "incorrect"]
        all_answers = [extracted_answer, answer_with_error]
        for i, answer in enumerate(all_answers):
            judge_answer, full_judge_answer = grade_open_ended_question(subquestion, answer, args.judge_model)
            # Only add answer if it's not empty. Sometimes R1 doesn't finish.
            if judge_answer.strip().lower()[:7] == "correct" or judge_answer.strip().lower()[:9] == "incorrect":
                expected_answer = expected_answers[i]
                coherent = 1 if judge_answer.strip().lower()[:len(expected_answer)] == expected_answer.strip().lower() else 0
                category_to_coherence[index_to_category[i]].append(coherent)
                overall_coherence.append(coherent)
                log_dict = {
                    "original_question": question,
                    "concept": concept,
                    "subquestion": subquestion,
                    "answer": answer,
                    "judge_answer": full_judge_answer,
                    "category": index_to_category[i],
                    "expected_answer": expected_answer,
                    "coherent": coherent,
                }
                save_dir = f"results_transformer_lens/{args.model}/{args.benchmark}/{trial_index}/{index_to_category[i]}"
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, f"{concept.replace('/','_').split(' ')[0]}_{index}_{i}.json"), "w") as f:
                    json.dump(log_dict, f)
    # Potemkin rate is normalized incoherence so 1 is random and 0 is perfect.
    potemkin_rate = 2 * (1 - np.mean(overall_coherence))
    score_per_concept.append(potemkin_rate)
    score_per_concept_std_err = np.std(score_per_concept) / np.sqrt(len(score_per_concept))
    # Log results to bar
    bar.set_description(f"Model: {args.model} "
                        f"Potemkin rate (lower bound): {np.mean(score_per_concept):.2f} ({score_per_concept_std_err:.2f}). ")

print(f"Potemkin rate (lower bound): {np.mean(score_per_concept):.2f} ({score_per_concept_std_err:.2f}).")
