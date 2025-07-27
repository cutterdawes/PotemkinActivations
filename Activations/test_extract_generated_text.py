import torch
from transformer_lens import HookedTransformer
from utils import load_model, extract_generated_text

def run_test():
    print("Running test for extract_generated_text...")

    # 1. Load a small model
    model = load_model("gpt2-small")

    # 2. Define a sample prompt and a simulated generated response
    test_prompt = "The quick brown fox jumps over the lazy dog. "
    simulated_generated_text = "This is a test of the emergency broadcast system."

    # 3. Convert both to tokens
    prompt_tokens = model.to_tokens(test_prompt)
    generated_tokens = model.to_tokens(simulated_generated_text)

    # 4. Combine them to mimic full_response_tokens
    # Ensure they are on the same device as the model if necessary, though to_tokens usually handles this.
    full_response_tokens = torch.cat((prompt_tokens, generated_tokens), dim=-1)

    # 5. Call extract_generated_text
    extracted_text = extract_generated_text(model, full_response_tokens, prompt_tokens.shape[1])

    # 6. Assert that the extracted text matches the simulated generated text
    # Note: model.to_string might add a leading space or handle whitespace differently.
    # We'll strip whitespace for a more robust comparison.
    assert extracted_text.strip() == simulated_generated_text.strip(), \
        f"Test failed! Expected '{simulated_generated_text.strip()}', got '{extracted_text.strip()}'"

    print("Test passed: extract_generated_text is working correctly.")

if __name__ == "__main__":
    run_test()
