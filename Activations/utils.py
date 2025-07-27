import torch
from transformer_lens import HookedTransformer
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str):
    """Loads a model, trying TransformerLens preloaded models first, then Hugging Face models."""
    try:
        # Try loading as a TransformerLens preloaded model
        model = HookedTransformer.from_pretrained(model_name)
        print(f"Loaded preloaded TransformerLens model: {model_name}")
        return model
    except Exception as e:
        print(f"Could not load {model_name} as a preloaded TransformerLens model: {e}")
        print(f"Attempting to load {model_name} from Hugging Face and convert to HookedTransformer...")
        try:
            # Load model and tokenizer from Hugging Face
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Convert to HookedTransformer
            model = HookedTransformer.from_pretrained(hf_model=hf_model, tokenizer=hf_tokenizer)
            print(f"Loaded {model_name} from Hugging Face and converted to HookedTransformer.")
            return model
        except Exception as e_hf:
            raise Exception(f"Failed to load model {model_name} from both TransformerLens preloaded and Hugging Face: {e_hf}")

def extract_generated_text(model, full_response_tokens, prompt_tokens_length):
    """Extracts the newly generated text from the full response, excluding the prompt."""
    # model.to_string returns a list of strings if given a batch, so we take the first element
    generated_text = model.to_string(full_response_tokens[:, prompt_tokens_length:])[0]
    
    # Remove leading <|endoftext|> token if present
    if generated_text.startswith(model.tokenizer.eos_token):
        generated_text = generated_text[len(model.tokenizer.eos_token):]

    return generated_text

def create_example(prompt, model):
    """Generates an example of a concept using the provided model and returns the activations."""
    activation_cache = {}
    def store_activation_hook(activation, hook):
        activation_cache[hook.name] = activation.detach().cpu()

    prompt_tokens = model.to_tokens(prompt)
    with model.hooks(fwd_hooks=[(hook_name, store_activation_hook) for hook_name in get_hook_names(model)]):
        response_tokens = model.generate(prompt_tokens, max_new_tokens=100, verbose=False)
    
    response = extract_generated_text(model, response_tokens, prompt_tokens.shape[1])
    return response, activation_cache

def check_example(concept, example, model):
    """Checks if the model agrees that the example is a valid example of the concept."""
    prompt = f"Is this an example of {concept} (answer 'yes' or 'no')?: {example}"
    prompt_tokens = model.to_tokens(prompt)
    response_tokens = model.generate(prompt_tokens, max_new_tokens=10, verbose=False)

    response = extract_generated_text(model, response_tokens, prompt_tokens.shape[1])
    return response

def grade_coherence(response):
    """Grades the coherence of the model's response."""
    return bool(re.search(r"yes", response, re.IGNORECASE))

def get_hook_names(model):
    """Gets the names of all the hook points in the model."""
    return [name for name, _ in model.named_modules() if "hook" in name]
