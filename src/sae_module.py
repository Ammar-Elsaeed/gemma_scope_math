import torch
import numpy as np
from sae_lens import SAE
import os
import gc
from typing import Dict, List, Union
import glob
from scipy import sparse

class SAEModule:
    def __init__(
        self,
        sae_release: str = "gemma-scope-2b-pt-res-canonical",
        sae_id: str = "layer_20/width_16k/canonical",
        device: str = None,
        output_dir: str = "./latents",
        dataset: str = "default"
    ):
        """Initialize the SAE module."""
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.dataset = dataset
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Disable gradients for memory efficiency
        torch.set_grad_enabled(False)
        
        # Load SAE
        self._initialize_sae()

    def _initialize_sae(self):
        """Load the SAE and move it to the specified device."""
        self.sae, _, _ = SAE.from_pretrained(
            release=self.sae_release,
            sae_id=self.sae_id
        )
        self.sae = self.sae.to(self.device)
        self.sae.eval()

    def process_activations(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        batch_idx,
        layer_num: int = None,
        save_latents: bool = True
    ) -> np.ndarray:
        """Process a batch of activations through the SAE and save latents."""
        # Convert numpy to torch if necessary
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations).to(self.device, dtype=torch.float16)
        
        # Compute latents
        with torch.no_grad():
            latents = self.sae.encode(activations)  # (batch_size, num_features)
        
        # Convert to numpy for saving
        latents_array = latents.cpu().numpy()
        
        if save_latents and layer_num is not None:
            # Create dataset directory if it doesn't exist
            dataset_dir = os.path.join(self.output_dir, self.dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Convert to sparse matrix and save directly to layer file
            sparse_latents = sparse.csr_matrix(latents_array)
            file_path = os.path.join(dataset_dir, f"layer_{layer_num}.npz")
            sparse.save_npz(file_path, sparse_latents)
        
        # Clean up
        del latents, activations
        gc.collect()
        torch.cuda.empty_cache()
        
        return latents_array

    def process_dataset_activations(
        self,
        activations_dict: Dict[str, np.ndarray],
        layers: Union[int, List[int]] = 20,
        save_latents: bool = True
    ) -> Dict[str, np.ndarray]:
        """Process activations from multiple layers and save latents."""
        if isinstance(layers, int):
            layers = [layers]
        
        all_latents = {}
        for layer in layers:
            layer_key = f"layer_{layer}"
            if layer_key not in activations_dict:
                continue
            
            activations = activations_dict[layer_key]
            layer_latents = []
            
            # Process in batches to match feedforward module's batch_size
            batch_size = 600  # Should match FeedForwardModule's batch_size
            for i in range(0, len(activations), batch_size):
                batch_activations = activations[i:i + batch_size]
                batch_latents = self.process_activations(
                    batch_activations,
                    batch_idx=i // batch_size,
                    layer_num=layer,
                    save_latents=save_latents
                )
                layer_latents.append(batch_latents)
            
            # Concatenate latents across batches
            all_latents[layer_key] = np.concatenate(layer_latents, axis=0)
        
        return all_latents

    def cleanup(self):
        """Clean up SAE and release resources."""
        del self.sae
        gc.collect()
        torch.cuda.empty_cache()

def load_activations(path, batch_size=600):
    """Load activations from files batch by batch to conserve memory."""
    activations_dict = {}
    
    # Check if path contains a wildcard
    if "*" in path:
        # Get all matching directories
        layer_dirs = glob.glob(path)
    else:
        # Use the specific directory
        layer_dirs = [path]
        
    for layer_dir in layer_dirs:
        # Extract layer name from the path
        layer_name = os.path.basename(layer_dir)
        
        # Get all batch files for this layer
        batch_files = glob.glob(os.path.join(layer_dir, "batch*.npy"))
        
        if not batch_files:
            print(f"No activation files found in {layer_dir}")
            continue
            
        # Initialize list to store batch file paths
        activations_dict[layer_name] = []
        
        # Load batches incrementally
        for batch_file in sorted(batch_files):
            # Store file path instead of loading immediately
            activations_dict[layer_name].append(batch_file)
        
        print(f"Found {len(activations_dict[layer_name])} batch files for {layer_name}")
    
    # Yield batches for each layer
    for layer_name, batch_files in activations_dict.items():
        for i in range(0, len(batch_files), batch_size):
            batch_paths = batch_files[i:i + batch_size]
            batch_activations = []
            for batch_path in batch_paths:
                batch_data = np.load(batch_path)
                batch_activations.append(batch_data)
            
            # Concatenate only the current batch group
            if batch_activations:
                yield layer_name, np.concatenate(batch_activations, axis=0)
                # Clean up
                del batch_activations
                gc.collect()

def load_latents(path, return_sparse=False):
    """Load latents from sparse matrix files.
    
    Args:
        path: Path to the latents directory (e.g., 'latents/addition')
        return_sparse: If True, return sparse matrices, otherwise convert to dense
    
    Returns:
        Dictionary mapping layer names to their latent representations
    """
    latents_dict = {}
    
    # Get all layer files
    layer_files = glob.glob(os.path.join(path, "layer_*.npz"))
    
    if not layer_files:
        print(f"No latent files found in {path}")
        return latents_dict
    
    for layer_file in sorted(layer_files):
        # Extract layer name from filename
        layer_name = os.path.basename(layer_file).split('.')[0]  # Remove .npz extension
        
        # Load sparse matrix
        sparse_latents = sparse.load_npz(layer_file)
        
        if return_sparse:
            latents_dict[layer_name] = sparse_latents
        else:
            latents_dict[layer_name] = sparse_latents.toarray()
        
        print(f"Loaded {layer_name} with shape {latents_dict[layer_name].shape}")
    
    return latents_dict

if __name__ == "__main__":
    import timeit
    start_time = timeit.default_timer()
    
    # Define the path to load activations from and dataset name
    dataset = "incorrect_arithmetic" # "addition" or "subtraction" or random_addition or random_subtraction or correct_arithmetic or incorrect_arithmetic
    activations_path = f"activations/{dataset}/layer_20"
    
    # Initialize SAE module with dataset name
    sae_module = SAEModule(output_dir="./latents", dataset=dataset)
    
    # Process activations batch by batch
    latents = {}
    for layer_name, batch_activations in load_activations(activations_path, batch_size=600):
        # Extract layer number from layer name
        layer_num = int(layer_name.split('_')[1])
        
        if layer_name not in latents:
            latents[layer_name] = []
        
        # Process the batch
        batch_latents = sae_module.process_activations(
            batch_activations,
            batch_idx=len(latents[layer_name]),
            layer_num=layer_num,
            save_latents=True
        )
        latents[layer_name].append(batch_latents)
        
        # Clean up
        del batch_activations, batch_latents
        gc.collect()
        torch.cuda.empty_cache()
    
    # Concatenate latents for each layer if needed
    for layer_name in latents:
        latents[layer_name] = np.concatenate(latents[layer_name], axis=0)
        print(f"{layer_name} latents shape: {latents[layer_name].shape}")
    
    # Clean up
    sae_module.cleanup()
    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time} seconds")