import torch
from transformer_lens import HookedTransformer
from utils import load_model, extract_generated_text

def run_test():
    print("Running comprehensive test for extract_generated_text...")

    # 1. Load a small model for general tests
    gpt2_model = load_model("gpt2-small")

    # 2. Load Llama model for Llama-specific special token tests
    llama_model = None
    try:
        llama_model = load_model("meta-llama/Llama-3.2-1B-Instruct")
    except Exception as e:
        print(f"Warning: Could not load Llama model for specific tests: {e}")
        print("Llama-specific special token tests will be skipped.")

    # Test Case 1: Basic extraction without special tokens
    test_prompt_1 = "Hello, how are you? "
    simulated_generated_text_1 = "I am fine, thank you."
    prompt_tokens_1 = gpt2_model.to_tokens(test_prompt_1)
    generated_tokens_1 = gpt2_model.to_tokens(simulated_generated_text_1)
    full_response_tokens_1 = torch.cat((prompt_tokens_1, generated_tokens_1), dim=-1)
    extracted_text_1 = extract_generated_text(gpt2_model, full_response_tokens_1, prompt_tokens_1.shape[1])
    assert extracted_text_1 == simulated_generated_text_1.strip(), \
        f"Test Case 1 Failed! Expected '{simulated_generated_text_1.strip()}', got '{extracted_text_1}'"
    print("Test Case 1 Passed: Basic extraction.")

    # Test Case 2: Extraction with EOS token
    test_prompt_2 = "What is the capital of France? "
    simulated_generated_text_2 = "Paris" + gpt2_model.tokenizer.eos_token
    prompt_tokens_2 = gpt2_model.to_tokens(test_prompt_2)
    generated_tokens_2 = gpt2_model.to_tokens(simulated_generated_text_2)
    full_response_tokens_2 = torch.cat((prompt_tokens_2, generated_tokens_2), dim=-1)
    extracted_text_2 = extract_generated_text(gpt2_model, full_response_tokens_2, prompt_tokens_2.shape[1])
    assert extracted_text_2 == "Paris", \
        f"Test Case 2 Failed! Expected 'Paris', got '{extracted_text_2}'"
    print("Test Case 2 Passed: EOS token removed.")

    # Test Case 3: Extraction with Llama-specific special tokens (simulated with Llama model)
    if llama_model:
        test_prompt_3 = "Tell me about Llama. "
        # Manually create tokens for the simulated generated text including Llama special tokens
        # These tokens are part of Llama's vocabulary and should be handled by skip_special_tokens=True
        simulated_generated_text_3_raw = "Llama is a large language model.<|eot_id|><|start_header_id|>user<|end_header_id|>"
        
        prompt_tokens_3 = llama_model.to_tokens(test_prompt_3)
        generated_tokens_3 = llama_model.to_tokens(simulated_generated_text_3_raw)
        full_response_tokens_3 = torch.cat((prompt_tokens_3, generated_tokens_3), dim=-1)
        
        extracted_text_3 = extract_generated_text(llama_model, full_response_tokens_3, prompt_tokens_3.shape[1])
        assert extracted_text_3 == "Llama is a large language model.", \
            f"Test Case 3 Failed! Expected 'Llama is a large language model.', got '{extracted_text_3}'"
        print("Test Case 3 Passed: Llama special tokens removed (simulated).")
    else:
        print("Test Case 3 Skipped: Llama model not loaded.")

    # Test Case 4: Extraction with leading/trailing whitespace
    test_prompt_4 = "  Question: "
    simulated_generated_text_4 = "  Answer.  "
    prompt_tokens_4 = gpt2_model.to_tokens(test_prompt_4)
    generated_tokens_4 = gpt2_model.to_tokens(simulated_generated_text_4)
    full_response_tokens_4 = torch.cat((prompt_tokens_4, generated_tokens_4), dim=-1)
    extracted_text_4 = extract_generated_text(gpt2_model, full_response_tokens_4, prompt_tokens_4.shape[1])
    assert extracted_text_4 == "Answer.", \
        f"Test Case 4 Failed! Expected 'Answer.', got '{extracted_text_4}'"
    print("Test Case 4 Passed: Leading/trailing whitespace handled.")

    print("All tests for extract_generated_text passed successfully.")

if __name__ == "__main__":
    run_test()
