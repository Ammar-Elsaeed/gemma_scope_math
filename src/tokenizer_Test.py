import torch
from transformers import AutoTokenizer

def test_number_tokenization(model_name: str = "google/gemma-2-2b-it"):
    """Test how the Gemma 2 tokenizer handles various numerical inputs."""
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example numerical inputs to test
    test_inputs = [
        "123",                      # Simple integer
        "123456789",               # Large integer
        # "3.14159",                 # Decimal number (e.g., pi)
        # "2025-05-28",              # Date format
        "1000000",                 # Large number with repeating digits
        "19+28=",                  # Simple addition equation
        "123-45=",                 # Simple subtraction equation
        "19 + 28 = 47",           # Complete arithmetic equation with spaces
        "19+28=47",               # Complete arithmetic equation without spaces
        "123456789+987654321=",               # Simple addition equation
        # "0xFF",                    # Hexadecimal (code-like)
        # "2^10",                    # Mathematical expression
        # "1,234,567",               # Number with commas
        # "42.0%",                   # Percentage
    ]
    
    print(f"Testing Gemma 2 Tokenizer ({model_name}) for Numbers\n")
    
    for text in test_inputs:
        # Tokenize the input
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        # Print the input, tokens, and token IDs
        print(f"Input: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Number of tokens: {len(tokens)}")
        print("-" * 50)

if __name__ == "__main__":
    # Disable gradients for memory efficiency
    torch.set_grad_enabled(False)
    
    # Run the tokenization test
    test_number_tokenization()