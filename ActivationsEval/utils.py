import os
import json
import csv
import random
import re
import time
from openai import OpenAI
from together import Together
from anthropic import Anthropic
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from prompts import contains_concept_prompt, subquestion_generation_prompt
from transformer_lens import HookedTransformer

# --- API Key Management ---
# IMPORTANT: Set these environment variables before running the script!
api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY")
}

models_to_developer = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
    "gpt-4o": "openai",
    "gpt-4.5-preview": "openai",
    "o3-mini": "openai",
    "o1-mini": "openai",
    "gemini-2.0-flash-exp": "gemini",
    "claude-3-5-sonnet-20241022": "claude",
    "mistralai/Mistral-7B-Instruct-v0.2": "together",
    "deepseek-ai/DeepSeek-V3": "together",
    "deepseek-ai/DeepSeek-R1": "together",
    "Qwen/Qwen2-VL-72B-Instruct": "together",
    "Qwen/Qwen2-72B-Instruct": "together",
}

BENCHMARK_PATHS = {
    'bbh': 'raw_data/benchmarks/bbh',
    'mmlu': 'raw_data/benchmarks/mmlu'
}
FINAL_TAG = "FINAL ANSWER:"

# --- Unified LLM Inference Function ---
def generate_llm_inference(prompt: str, model_name: str) -> str:
    # Check if it's a HookedTransformer model
    if model_name.startswith("gpt2") or model_name.startswith("pythia") or model_name.startswith("solu") or model_name.startswith("gelu"):
        print(f"Loading HookedTransformer model: {model_name}")
        model = HookedTransformer.from_pretrained(model_name)
        tokens = model.to_tokens(prompt)
        generated_tokens = model.generate(tokens, max_new_tokens=50, temperature=0.7)
        generated_text = model.to_string(generated_tokens[0])
        completion = generated_text[len(prompt):].strip()
        return completion
    
    # Otherwise, use external LLM APIs
    developer = models_to_developer.get(model_name)
    if not developer:
        raise ValueError(f"Unknown model or developer for: {model_name}")

    if developer == "openai":
        client = OpenAI(api_key=api_keys["openai"])
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content
    elif developer == "together":
        client = Together(api_key=api_keys["together"])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
        )
        return resp.choices[0].message.content
    elif developer == "gemini":
        genai.configure(api_key=api_keys["gemini"])
        model = genai.GenerativeModel(model_name)
        backoff = 1
        while True:
            try:
                return model.generate_content(prompt).text
            except ResourceExhausted as e:
                delay = getattr(e, "retry_delay", None)
                delay = delay.seconds if delay else backoff
                print(f"[Gemini] rate-limited, sleeping {delay}s")
                time.sleep(delay)
                backoff = min(backoff * 2, 60)
    elif developer == "claude":
        client = Anthropic(
             api_key=api_keys["claude"],
        )
        message = client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
        )
        return message.content[0].text
    else:
        raise ValueError(f"Unsupported developer: {developer}")

def extract_final_answer(response: str) -> str:
    match = re.search(rf'{re.escape(FINAL_TAG)}\s*(.*)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def grade_benchmark(model_answer: str | None, gold_answer: str, multiple_choice: bool) -> bool:
    if model_answer is None:
        return False
    if multiple_choice:
        return model_answer[0].upper() == gold_answer.upper()
    else:
        return model_answer.strip().lower() == gold_answer.strip().lower()

def answer_benchmark_question(question: str, model: str) -> tuple[str, str]:
    cot_prompt = (
        "You are an expert tutor. You may think step-by-step to reach the solution, "
        f"but you MUST finish with a line that starts exactly with `{FINAL_TAG}` "
        "followed by your single best answer.\n\n"
        f"Question:\n{question}\n"
    )
    full_msg = generate_llm_inference(cot_prompt, model)
    m = re.search(rf"{re.escape(FINAL_TAG)}\s*(.+)", full_msg, flags=re.I | re.S)
    extracted = m.group(1).strip() if m else None
    return full_msg, extracted

def answer_open_ended_question(question: str, model: str) -> str:
    prompt = (
        "You are an expert tutor. You may think step-by-step to reach the solution, "
        f"but you MUST finish with an answer that starts exactly with `{FINAL_TAG}` "
        "followed by your single best answer. If the question asks you to explain your "
        f"answer, you should also include the explanation after `{FINAL_TAG}`.\n\n"
        f"Question:\n{question}\n"
    )
    full_answer = generate_llm_inference(prompt, model)
    extracted_answer = extract_final_answer(full_answer)
    return extracted_answer

def answer_and_grade_benchmark_question(question: str, 
                                        model: str, 
                                        gold_answer: str,
                                        multiple_choice: bool) -> bool:
    full_msg, extracted = answer_benchmark_question(question, model)
    return grade_benchmark(extracted, gold_answer, multiple_choice), full_msg

def edit_to_introduce_error(question: str, initial_answer: str, model: str) -> str:
    prompt = (
        "Modify the following answer to introduce a subtle error. The error should be "
        "subtle but one such that a human who knows the concept would know the answer "
        "is incorrect. If the answer is already incorrect, you can leave it the same. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` followed "
        "by the full modified answer.\n\n"
        f"Question: {question}\n"
        f"Answer: {initial_answer}\n"
    )
    full_answer = generate_llm_inference(prompt, model)
    modified_answer = extract_final_answer(full_answer)
    return full_answer, modified_answer

def edit_to_remove_error(question: str, initial_answer: str, model: str) -> str:
    prompt = (
        "Modify the following answer to remove any error. If there are errors, "
        "they might be subtle but one such that a human who knows the concept would "
        "know the answer is incorrect. If the answer is already correct, you can leave "
        "it the same. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` followed "
        "by the full modified answer.\n\n"
        f"Question: {question}\n"
        f"Answer: {initial_answer}\n"
    )
    full_answer = generate_llm_inference(prompt, model)
    modified_answer = extract_final_answer(full_answer)
    return full_answer, modified_answer

def generate_subquestions(question, concept, model, num_subquestions):
    prelude = (
        f"The following is a question about the following concept: {concept}. "
        f"Here is the question: {question}.\n\n"
        f"Write {num_subquestions} other questions that test whether someone who understands the concepts the question is testing truly understands them."
    )
    prompt = prelude + "\n\n" + subquestion_generation_prompt
    inference = generate_llm_inference(prompt, model)
    questions = parse_questions(inference)
    return questions

def grade_open_ended_question(question: str, model_answer: str, model: str) -> tuple[str, str]:
    cot_prompt = (
        "You are an expert tutor. You will be given a question and a possible "
        "answer to the question. Your job is to determine if the answer is correct "
        "or incorrect. You should only grade it correct if the answer (including the reasoning) "
        "is completely correct. "
        f"You can reason all you'd like, but end the response with `{FINAL_TAG}` "
        "followed by either 'correct' or 'incorrect', and nothing should come after that.\n\n"
        f"Question: {question}\n"
        f"Answer: {model_answer}\n"
    )
    full_msg = generate_llm_inference(cot_prompt, model)
    extracted_answer = extract_final_answer(full_msg)
    extracted_answer = extracted_answer.replace("*", "").strip()
    return extracted_answer, full_msg

def parse_questions(inference):
    questions = re.findall(r'<question>(.*?)</question>', inference, re.DOTALL)
    return questions

def relies_on_concept(question, model):
    prompt = contains_concept_prompt + "\n\n" + question + "\n"
    response = generate_llm_inference(prompt, model)
    answer_match = re.search(r'ANSWER:\s*(Yes|No)', response, re.IGNORECASE)
    concept_match = re.search(r'CONCEPT:\s*(.*)', response, re.IGNORECASE)
    answer = answer_match.group(1) if answer_match else None
    concept = concept_match.group(1).strip() if concept_match else None
    try:
        classification = answer.lower() == "yes"
    except:
        if concept is None:
            return False, None
        else:
            return True, concept
    return classification, concept

def sample_question(benchmark, subject=None):
    base_path = BENCHMARK_PATHS[benchmark]
    if subject is None:
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        subject = random.choice(files)
    file_path = os.path.join(base_path, subject)

    if benchmark == 'bbh':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            question_data = random.choice(data['examples'])
            question_str = question_data['input']
            answer = question_data['target']

    elif benchmark == 'mmlu':
        with open(file_path, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
            row = random.choice(rows)
            question_str = row[0]
            choices = row[1:-1]
            answer_letter = row[-1].strip()
            formatted_choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            question_str = f"{question_str}\n{formatted_choices}"
            answer = answer_letter

    else:
        raise ValueError("Unsupported benchmark.")

    return question_str, answer, subject
