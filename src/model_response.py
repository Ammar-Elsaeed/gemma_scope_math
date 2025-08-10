import os
import time
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["USE_TRITON"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # "nf4" is recommended, "fp4" is also possible
    bnb_4bit_compute_dtype=torch.bfloat16  # or torch.float16 if bfloat16 is not supported
)

torch.set_grad_enabled(False) # avoid blowing up mem

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map='auto',
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# --- Load prompts ---
prompts_path = "data/addition.txt"  # Change this if your file is elsewhere or a different format
with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# --- Run for first 10 prompts ---
for i, prompt in enumerate(prompts[:10]):
    print(f"\nPrompt {i+1}: {prompt}")
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    
    start_time = time.time()
    outputs = model.generate(input_ids=inputs, max_new_tokens=50)
    gen_time = time.time() - start_time
    
    decoded = tokenizer.decode(outputs[0])
    print(f"Generation time: {gen_time:.3f} seconds")
    print(f"Output: {decoded}")

    # Optionally, get hidden states for the generated sequence
    all_tokens = outputs[0].unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=all_tokens, output_hidden_states=True, return_dict=True)
        print(f"Shape of last layer hidden state: {out.hidden_states[-1].shape}")

# To return hidden states for each generated token, you need to use model.forward directly, not model.generate.
# Example (for the input prompt only):
with torch.no_grad():
    out = model(input_ids=inputs, output_hidden_states=True, return_dict=True)
    # out.hidden_states is a tuple: (layer_0, layer_1, ..., layer_n)
    # Each is shape (batch, seq_len, hidden_dim)
    print(f"Number of layers: {len(out.hidden_states)}")
    print(f"Shape of last layer hidden state: {out.hidden_states[-1].shape}")

# Note: To get hidden states for generated tokens, you need to feed the generated sequence back into the model with output_hidden_states=True.