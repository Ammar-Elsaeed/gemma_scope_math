import numpy as np
import torch
import os
import glob
import gc
from typing import Dict, List, Union
from scipy import sparse

class AnalysisModule:
    def __init__(
        self,
        dataset: str = "addition",
        layer: int = 20
    ):
        """Initialize the analysis module for loading and analyzing latents."""
        self.dataset = dataset
        self.layer = layer
        self.active_counts = None  # Will be initialized as a numpy array
        self.total_examples = 0

    def load_latents(self):
        """Load latent file for the specified layer."""
        latents_path = os.path.join("latents", self.dataset, f"layer_{self.layer}.npz")
        if not os.path.exists(latents_path):
            raise FileNotFoundError(f"No latent file found at {latents_path}")
        
        print(f"Loading latents from {latents_path}")
        return sparse.load_npz(latents_path)

    def analyze_latents(self, latents: sparse.csr_matrix) -> np.ndarray:
        """Analyze latents to count active features.
        
        Args:
            latents: Sparse matrix of shape (n_examples, n_features)
            
        Returns:
            np.ndarray: Array of activation counts per feature
        """
        # Initialize counts array if not already done
        if self.active_counts is None:
            self.active_counts = np.zeros(latents.shape[1], dtype=np.uint16)
            print(f"Initialized counts array with shape: {self.active_counts.shape}")
            
        # Get indices where values are > 0.01
        active_mask = latents.data > 0.01
        active_indices = latents.indices[active_mask]
        
        # Count occurrences of each feature index
        unique_indices, counts = np.unique(active_indices, return_counts=True)
        
        # Safely add counts with overflow checking
        for idx, count in zip(unique_indices, counts):
            current_count = self.active_counts[idx]
            new_count = current_count + count
            if new_count > np.iinfo(np.uint16).max:
                print(f"Warning: Overflow detected for feature {idx}. Current: {current_count}, Adding: {count}")
                new_count = np.iinfo(np.uint16).max
            self.active_counts[idx] = new_count
        
        # Update total examples count
        self.total_examples += latents.shape[0]
        
        return self.active_counts

    def process_latents(self) -> np.ndarray:
        """Process latents and return activation frequencies.
        
        Returns:
            np.ndarray: Array of activation frequencies per feature (between 0 and 1)
        """
        # Load and process latents
        latents = self.load_latents()
        active_counts = self.analyze_latents(latents)
        
        # Calculate activation frequencies
        activation_frequencies = active_counts / self.total_examples
        
        # Save activation frequencies
        output_dir = os.path.join("activ_freq", self.dataset)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"layer_{self.layer}.npy")
        np.save(output_path, activation_frequencies)
        
        # np.save( os.path.join(output_dir, f"test_SAve.npy"), np.array([0,1,2,3]))
        # Clean up
        del latents
        gc.collect()
        
        return activation_frequencies

if __name__ == "__main__":
    # List of all datasets to process
    datasets = [
        # "addition", 
        # "subtraction", 
        # "random_addition", 
        # "random_subtraction", 
        # "correct_arithmetic", 
        # "incorrect_arithmetic"
        "addition_full_range",
        "subtraction_full_range",
    ]
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Get available layers for this dataset
        layers = range(0, 26)  # Assuming layers 0 to 25 are available
        if not layers:
            print(f"No latent files found for dataset {dataset}")
            continue
            
        # Process each layer
        for layer in layers:
            print(f"Processing layer {layer}")
            analysis_module = AnalysisModule(dataset=dataset, layer=layer)
            
            try:
                # Process latents to get activation frequencies
                activation_frequencies = analysis_module.process_latents()
                print(f"Processed layer {layer}: {len(activation_frequencies)} features analyzed")
                
            except Exception as e:
                print(f"Error processing {dataset} layer {layer}: {str(e)}")
                continue
    
    print(f"\nAnalysis complete. Results saved in activ_freq directory.")