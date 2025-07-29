import torch
from transformer_lens import HookedTransformer
import re
from transformers import AutoModelForCausalLM, AutoTokenizer



def load_model(model_name: str):
    """Loads a model, trying TransformerLens preloaded models first, then Hugging Face models."""
    try:
        model = HookedTransformer.from_pretrained(model_name)
        print(f"Loaded preloaded TransformerLens model: {model_name}")
        return model
    except Exception as e:
        print(f"Could not load {model_name} as a preloaded TransformerLens model: {e}")
        print(f"Attempting to load {model_name} from Hugging Face and convert to HookedTransformer...")
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = HookedTransformer.from_pretrained(hf_model=hf_model, tokenizer=hf_tokenizer)
            print(f"Loaded {model_name} from Hugging Face and converted to HookedTransformer.")
            return model
        except Exception as e_hf:
            raise Exception(f"Failed to load model {model_name} from both TransformerLens preloaded and Hugging Face: {e_hf}")

def extract_generated_text(model, full_response_tokens, prompt_tokens_length):
    """Extracts the newly generated text from the full response, excluding the prompt."""
    generated_tokens = full_response_tokens[0, prompt_tokens_length:]
    return model.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()



def generate_and_capture_activations(model, prompt, max_new_tokens, temperature, no_activations=False):
    """A general-purpose function to generate text and capture activations."""
    current_activation_cache = {}

    prompt_tokens = model.to_tokens(prompt)
    if no_activations:
        response_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            verbose=False,
            temperature=temperature,
            eos_token_id=model.tokenizer.eos_token_id
        )
    else:
        def store_activation_hook_local(activation, hook):
            if torch.is_tensor(activation):
                current_activation_cache[hook.name] = activation.detach().cpu()

        with model.hooks(fwd_hooks=[(hook_name, store_activation_hook_local) for hook_name in model.hook_points()]):
            response_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                verbose=False,
                temperature=temperature,
                eos_token_id=model.tokenizer.eos_token_id
            )
    
    generated_text = extract_generated_text(model, response_tokens, prompt_tokens.shape[1])
    return generated_text, current_activation_cache

def create_example(concept, model, no_activations=False):
    """Generates an example of a concept using the provided model and returns the activations."""
    prompt = f"Please provide a single, clear, and concise example of the following concept: {concept}. Please answer with only the example. Example:"
    return generate_and_capture_activations(model, prompt, max_new_tokens=100, temperature=0.5, no_activations=no_activations)

def check_example(concept, example, model, no_activations=False):
    """Checks if the model agrees that the example is a valid example of the concept."""
    prompt = f"Is the following text an example of the concept '{concept}'? Please answer with only the word 'yes' or 'no'.\nText: '{example}'\nAnswer:"
    response, _ = generate_and_capture_activations(model, prompt, max_new_tokens=3, temperature=0.0, no_activations=no_activations)
    return response

def grade_coherence(response):
    """Grades the coherence of the model's response."""
    return bool(re.search(r"yes", response, re.IGNORECASE))