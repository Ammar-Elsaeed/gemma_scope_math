import torch
import numpy as np
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import gc
from typing import List, Optional, Union
import os

class FeedForwardModule:
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        batch_size: int = 800,
        device: str = None,
        output_dir: str = "./activations"
    ):
        """Initialize the feedforward module with model and processing parameters."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Disable gradients for memory efficiency
        torch.set_grad_enabled(False)
        
        # Initialize model and tokenizer
        self._initialize_model()
        self._initialize_tokenizer()

    def _initialize_model(self):
        """Load the quantized model with 4-bit precision."""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _initialize_tokenizer(self):
        """Load the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_dataset(self, input_file: str) -> List[str]:
        """Load and clean the dataset from a text file."""
        with open(input_file, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]

    def process_batch(
        self,
        examples: List[str],
        layers: Union[int, List[int]] = 20,
        save_activations: bool = True,
        batch_idx: Optional[int] = None
    ) -> dict:
        """Process a single batch and extract activations for specified layers."""
        if isinstance(layers, int):
            layers = [layers]
        
        # Tokenize the batch
        batch = self.tokenizer(
            examples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True)

        # Initialize results
        activations = {}

        # Process each specified layer
        for layer_idx in layers:
            # Adjust for 0-based indexing (user inputs 1-based layer numbers)
            layer_idx_internal = layer_idx + 1  # Gemma-2-2b includes embeddings as layer 0
            layer_hidden = outputs.hidden_states[layer_idx_internal]  # (batch_size, seq_len, hidden_size)

            # Find last meaningful token positions
            attention_mask = batch["attention_mask"]
            reversed_mask = (attention_mask == 1).int().flip(dims=[1])
            reversed_idx = reversed_mask.argmax(dim=1)
            last_idx = attention_mask.size(1) - 1 - reversed_idx

            batch_size = layer_hidden.shape[0]
            batch_indices = torch.arange(batch_size, device=layer_hidden.device)
            last_meaningful_hidden_states = layer_hidden[batch_indices, last_idx, :]  # (batch_size, hidden_size)

            # Convert to numpy for saving
            activations_array = last_meaningful_hidden_states.cpu().numpy()

            if save_activations and batch_idx is not None:
                # Save activations
                layer_dir = os.path.join(self.output_dir, f"layer_{layer_idx}")
                os.makedirs(layer_dir, exist_ok=True)
                file_path = os.path.join(layer_dir, f"batch_{batch_idx:04d}.npy")
                np.save(file_path, activations_array)

            activations[f"layer_{layer_idx}"] = activations_array

        # Clean up
        del batch, outputs, layer_hidden, last_meaningful_hidden_states
        gc.collect()
        torch.cuda.empty_cache()

        return activations

    def process_dataset(
        self,
        input_file: str,
        layers: Union[int, List[int]] = 20,
        start_at_batch: int = 0,
        end_at_batch: Optional[int] = None,
        save_activations: bool = True
    ) -> dict:
        """Process the dataset in batches with checkpointing support.
        
        Args:
            input_file: Path to the input text file
            layers: Layer number(s) to extract activations from
            start_at_batch: Batch number to start processing from (0-based)
            end_at_batch: Batch number to end processing at (inclusive). If None, process until end.
            save_activations: Whether to save activations to disk
        
        Returns:
            Dictionary of activations for each layer
        """
        # Load dataset
        dataset = self.load_dataset(input_file)
        
        # Calculate total number of batches
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        
        # Validate and adjust batch ranges
        start_at_batch = max(0, min(start_at_batch, total_batches - 1))
        if end_at_batch is None:
            end_at_batch = total_batches - 1
        else:
            end_at_batch = max(start_at_batch, min(end_at_batch, total_batches - 1))
        
        print(f"Processing batches {start_at_batch} to {end_at_batch} (total batches: {total_batches})")
        print(f"Each batch contains {self.batch_size} examples")

        # Process specified batch range
        all_activations = {}
        for batch_idx in range(start_at_batch, end_at_batch + 1):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(dataset))
            batch_examples = dataset[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx}/{end_at_batch} ({len(batch_examples)} examples)")
            
            batch_activations = self.process_batch(
                batch_examples,
                layers=layers,
                save_activations=save_activations,
                batch_idx=batch_idx
            )
            
            # Aggregate activations
            for layer, acts in batch_activations.items():
                if layer not in all_activations:
                    all_activations[layer] = []
                all_activations[layer].append(acts)
            
            # Clean up batch data
            del batch_examples, batch_activations
            gc.collect()
            torch.cuda.empty_cache()

        # Concatenate activations across processed batches
        for layer in all_activations:
            all_activations[layer] = np.concatenate(all_activations[layer], axis=0)

        return all_activations

    def cleanup(self):
        """Clean up model and release resources."""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import timeit
    datasets = ["addition_full_range", "subtraction_full_range"]  # Adjust as needed for your use case
    start_time = timeit.default_timer()
    # dataset = "random_addition" # "addition" or "subtraction" or random_addition or random_subtraction or correct_arithmetic or incorrect_arithmetic

    for dataset in datasets:
        # Example usage with batch control
        module = FeedForwardModule(batch_size=500, output_dir=f"./activations/{dataset}")
    
        # Process specific batch range (e.g., batches 5-10)
        activations = module.process_dataset(
            input_file=f"./data/{dataset}.txt",
            layers=range(0, 26),  # Process layers 1 to 26
            # start_at_batch=5,   # Start from batch 5 (skips first 4000 examples)
            # end_at_batch=10     # Process until batch 10 (up to example 8800)
        )
        
        module.cleanup()
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")