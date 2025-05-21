import numpy as np
import torch
import os
import gc
from typing import Dict, List, Union
from scipy import sparse

class AnalysisModule:
    def __init__(
        self,
        dataset: str = "addition",
        layer: int = 20,
        sensitivity: float = 0.95
    ):
        """Initialize the analysis module for loading and analyzing latents."""
        self.dataset = dataset
        self.layer = layer
        self.sensitivity = sensitivity
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
        print(f"Max count in this batch: {counts.max()}")
        print(f"Number of active features in this batch: {len(unique_indices)}")
        
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
        print(f"Total examples processed so far: {self.total_examples}")
        
        return self.active_counts

    def process_latents(self) -> Dict[str, List[int]]:
        """Process latents, track active counts, and identify consistently active features."""
        # Load and process latents
        latents = self.load_latents()
        active_counts = self.analyze_latents(latents)
        
        # Calculate which features meet sensitivity threshold
        threshold = self.total_examples * self.sensitivity
        consistent_features = np.where(active_counts >= threshold)[0].tolist()
        
        # Save active counts
        output_dir = os.path.join("activ_freq", self.dataset, f"layer_{self.layer}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "active_counts.npy")
        np.save(output_path, active_counts)
        
        # Print statistics
        print(f"\nAnalysis Results for {self.dataset} layer_{self.layer}:")
        print(f"Total examples processed: {self.total_examples}")
        print(f"Features with sensitivity > {self.sensitivity*100}%: {len(consistent_features)}")
        print(f"Active counts saved to: {output_path}")
        
        # Clean up
        del latents
        gc.collect()
        
        return {f"layer_{self.layer}": consistent_features}

if __name__ == "__main__":
    import timeit
    start_time = timeit.default_timer()
    
    # Initialize module with dataset and layer
    dataset = "random_subtraction" # "addition" or "subtraction" or random_addition or random_subtraction  or correct_arithmetic or incorrect_arithmetic
    layer = 20
    analysis_module = AnalysisModule(dataset=dataset, layer=layer)
    
    # Process latents to get consistent features and save active counts
    consistent_features = analysis_module.process_latents()

    # Print detailed results
    for layer_name, features in consistent_features.items():
        print(f"\n{layer_name} consistently active features ({len(features)} total):")
        if len(features) > 0:
            print(f"Feature indices: {features[:10]}{'...' if len(features) > 10 else ''}")
            
        print("\nTop 10 most frequently active features:")
        top_indices = np.argsort(analysis_module.active_counts)[-10:][::-1]
        for idx in top_indices:
            count = analysis_module.active_counts[idx]
            percentage = (count / analysis_module.total_examples) * 100
            print(f"Feature {idx}: active {count} times ({percentage:.2f}% of examples)")

    end_time = timeit.default_timer()
    print(f"\nTime taken for processing latents: {end_time - start_time:.2f} seconds")