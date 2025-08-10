import torch
import torch.nn.functional as F
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import random
import json
import numpy as np
import glob

# Disable gradients for memory efficiency
torch.set_grad_enabled(False)

# Set environment variables
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["USE_TRITON"] = "0"

# Base directory for finding feature arrays
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_features_from_npy(layer_idx, operation, top_n, base_dir=None):
    """
    Load feature indices from .npy array for a specific layer and operation.
    
    Args:
        layer_idx: Layer index (e.g., 0, 1, 2, ...)
        operation: Operation type ('add' or 'sub')
        top_n: Number of top features (e.g., 10)
        base_dir: Base directory containing top_features folder
    
    Returns:
        numpy array of feature indices
    """
    if base_dir is None:
        base_dir = BASE_DIR
    
    feature_file = os.path.join(base_dir, "top_features", f"{operation}_layer-{layer_idx}_top-{top_n}.npy")
    
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    return np.load(feature_file)

def load_all_layers_features(operation, top_n, base_dir=None):
    """
    Load feature indices for all available layers for a specific operation.
    
    Args:
        operation: Operation type ('add' or 'sub') 
        top_n: Number of top features (e.g., 10)
        base_dir: Base directory containing top_features folder
    
    Returns:
        dict: {layer_idx: feature_indices_array, ...}
    """
    if base_dir is None:
        base_dir = BASE_DIR
    
    # Find all matching files
    pattern = os.path.join(base_dir, "top_features", f"{operation}_layer-*_top-{top_n}.npy")
    feature_files = sorted(glob.glob(pattern))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found matching pattern: {pattern}")
    
    layer_features = {}
    for file_path in feature_files:
        # Extract layer index from filename
        filename = os.path.basename(file_path)
        # e.g., "add_layer-5_top-10.npy" -> extract "5"
        layer_idx = int(filename.split('_layer-')[1].split('_')[0])
        layer_features[layer_idx] = np.load(file_path)
    
    return layer_features

def load_features(feature_file):
    """Load feature indices from a text file."""
    with open(feature_file, 'r') as f:
        features = [int(line.strip()) for line in f if line.strip()]
    return features

def load_dataset(dataset_file):
    """Load prompts from a text file."""
    with open(dataset_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def save_results(save_dir, results, params, ablation_stats=None):
    """Save experiment results and parameters to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save experiment parameters
    with open(os.path.join(save_dir, 'experiment_params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save ablation statistics if provided
    if ablation_stats:
        with open(os.path.join(save_dir, 'ablation_stats.json'), 'w') as f:
            json.dump(ablation_stats, f, indent=4)
    
    # Save results for each scenario and layer
    scenarios = ['first_n_ablation', 'random_n_ablation']
    if 'no_ablation' in results:
        scenarios.insert(0, 'no_ablation')
        
    for scenario in scenarios:
        with open(os.path.join(save_dir, f'{scenario}.txt'), 'w') as f:
            if isinstance(results[scenario], dict):
                # Multi-layer results
                for layer_idx, layer_results in results[scenario].items():
                    f.write(f"=== LAYER {layer_idx} ===\n")
                    for prompt, output in layer_results.items():
                        f.write(f"Prompt: {prompt}\nOutput: {output}\n\n")
                    f.write("\n")
            else:
                # Single layer results (backward compatibility)
                for prompt, output in results[scenario].items():
                    f.write(f"Prompt: {prompt}\nOutput: {output}\n\n")

class TargetedAblationHook:
    def __init__(self, sae, candidate_features, n_features_to_ablate, lambda_scale, activation_threshold=0.001):
        """
        Initialize the targeted ablation hook that selects first N active features from candidates.
        
        Args:
            sae: Sparse Autoencoder instance
            candidate_features: List of candidate feature indices (e.g., top 10 features)
            n_features_to_ablate: Number of features to ablate (e.g., 3)
            lambda_scale: Scaling factor for ablation
            activation_threshold: Minimum activation value to consider a feature as active
        """
        self.sae = sae
        self.candidate_features = candidate_features
        self.n_features_to_ablate = n_features_to_ablate
        self.lambda_scale = lambda_scale
        self.activation_threshold = activation_threshold
        self.selected_active_features = []  # Will be set dynamically per forward pass

    def find_first_n_active_features(self, activations):
        """Find first N active features from the candidate list."""
        active_features = []
        for idx in self.candidate_features:
            if torch.any(activations[..., idx] > self.activation_threshold):
                active_features.append(idx)
                if len(active_features) >= self.n_features_to_ablate:
                    break
        
        self.selected_active_features = active_features
        return active_features

    def targeted_latent_hook(self, module, input, output):
        """Forward hook to keep only the first N active features from candidates."""
        # Find first N active features for this forward pass
        selected_features = self.find_first_n_active_features(output)
        
        modified_acts = torch.zeros_like(output)
        for idx in selected_features:
            modified_acts[..., idx] = output[..., idx]
        
        return modified_acts

    def activation_modification_hook(self, module, input, output):
        hidden_states = output[0]
        
        # Compute reconstruction for first N active features from candidates
        with torch.no_grad():
            hook_handle = self.sae.hook_sae_acts_post.register_forward_hook(
                self.targeted_latent_hook
            )
            try:
                targeted_reconstruction = self.sae(hidden_states)
            finally:
                hook_handle.remove()
        
        # Check if we found enough active features for ablation
        self.ablation_occurred = len(self.selected_active_features) >= self.n_features_to_ablate
        
        modified_hidden_states = hidden_states - self.lambda_scale * targeted_reconstruction
        return (modified_hidden_states,) + output[1:] if isinstance(output, tuple) else modified_hidden_states

class DynamicRandomAblationHook:
    def __init__(self, sae, excluded_features, n_random_features, lambda_scale, activation_threshold=0.001):
        """
        Initialize the dynamic random ablation hook.
        
        Args:
            sae: Sparse Autoencoder instance
            excluded_features: List of all feature indices to exclude from random selection (entire candidate array)
            n_random_features: Number of random features to ablate
            lambda_scale: Scaling factor for ablation
            activation_threshold: Minimum activation value to consider a feature as active
        """
        self.sae = sae
        self.excluded_features = set(excluded_features)
        self.n_random_features = n_random_features
        self.lambda_scale = lambda_scale
        self.activation_threshold = activation_threshold
        self.selected_random_features = []  # Will be set dynamically per forward pass

    def find_random_active_features(self, activations):
        """Find random active features excluding the entire candidate array."""
        # Get all active feature indices above threshold
        active_mask = torch.any(activations > self.activation_threshold, dim=tuple(range(activations.dim()-1)))
        active_indices = torch.where(active_mask)[0].cpu().numpy().tolist()
        
        # Filter out ALL features from the candidate array (not just the first N)
        available_features = [idx for idx in active_indices if idx not in self.excluded_features]
        
        # Randomly sample from available features
        if len(available_features) >= self.n_random_features:
            self.selected_random_features = random.sample(available_features, self.n_random_features)
        else:
            self.selected_random_features = available_features
        
        return self.selected_random_features

    def random_latent_hook(self, module, input, output):
        """Forward hook to keep only dynamically selected random latent activations."""
        # Find random active features for this forward pass
        selected_features = self.find_random_active_features(output)
        
        modified_acts = torch.zeros_like(output)
        for idx in selected_features:
            modified_acts[..., idx] = output[..., idx]
        
        return modified_acts

    def activation_modification_hook(self, module, input, output):
        hidden_states = output[0]
        
        # Compute reconstruction for dynamically selected random latents
        with torch.no_grad():
            hook_handle = self.sae.hook_sae_acts_post.register_forward_hook(
                self.random_latent_hook
            )
            try:
                random_latent_reconstruction = self.sae(hidden_states)
            finally:
                hook_handle.remove()
        
        # Check if we found enough random features for ablation
        self.ablation_occurred = len(self.selected_random_features) >= self.n_random_features
        
        modified_hidden_states = hidden_states - self.lambda_scale * random_latent_reconstruction
        return (modified_hidden_states,) + output[1:] if isinstance(output, tuple) else modified_hidden_states
    def __init__(self, sae, latent_indices, lambda_scale, activation_threshold=0.001):
        """
        Initialize the ablation hook for multiple latent indices.
        
        Args:
            sae: Sparse Autoencoder instance
            latent_indices: List of indices of latents to ablate
            lambda_scale: Scaling factor for ablation
            activation_threshold: Minimum activation value to consider a feature as active
        """
        self.sae = sae
        self.latent_indices = latent_indices
        self.lambda_scale = lambda_scale
        self.activation_threshold = activation_threshold

    def single_latent_hook(self, module, input, output):
        """Forward hook to keep only specified latent activations if they are active above threshold."""
        modified_acts = torch.zeros_like(output)
        
        # Check which indices have activations above threshold
        active_indices = []
        for idx in self.latent_indices:
            if torch.any(output[..., idx] > self.activation_threshold):
                modified_acts[..., idx] = output[..., idx]
                active_indices.append(idx)
        
        # Store active indices for logging/debugging (optional)
        self.last_active_indices = active_indices
        
        # Only return modified_acts if at least one feature is active
        if len(active_indices) > 0:
            return modified_acts
        else:
            # If no features are active above threshold, return zeros (no ablation)
            return torch.zeros_like(output)

    def activation_modification_hook(self, module, input, output):
        hidden_states = output[0]
        
        # Compute reconstruction for specified latents
        with torch.no_grad():
            hook_handle = self.sae.hook_sae_acts_post.register_forward_hook(
                self.single_latent_hook
            )
            try:
                single_latent_reconstruction = self.sae(hidden_states)
            finally:
                hook_handle.remove()
        
        # Check if any features were actually active for this forward pass
        self.ablation_occurred = hasattr(self, 'last_active_indices') and len(self.last_active_indices) > 0
        
        modified_hidden_states = hidden_states - self.lambda_scale * single_latent_reconstruction
        return (modified_hidden_states,) + output[1:] if isinstance(output, tuple) else modified_hidden_states

def run_targeted_ablation_experiment(model, tokenizer, sae, prompts, layer_idx, candidate_features, n_features_to_ablate, lambda_scale, max_new_tokens, device, activation_threshold=0.001):
    """
    Run targeted ablation experiment that finds first N active features from candidates.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sae: Sparse autoencoder
        prompts: List of input prompts
        layer_idx: Layer index to ablate
        candidate_features: List of candidate feature indices (e.g., full top 10 array)
        n_features_to_ablate: Number of features to ablate (e.g., 3)
        lambda_scale: Ablation strength
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        activation_threshold: Minimum activation value to consider a feature as active
    
    Returns:
        Tuple: (results_dict, ablation_stats)
        - results_dict: Dictionary mapping prompts to generated outputs
        - ablation_stats: Dictionary with ablation statistics
    """
    results = {}
    ablation_stats = {
        'total_prompts': len(prompts),
        'insufficient_active_features': 0,
        'successful_ablation': 0,
        'insufficient_features_prompts': [],  # List of prompts where insufficient features were active
        'selected_features_per_prompt': {},  # Track which features were selected for each prompt
        'n_features_to_ablate': n_features_to_ablate,
        'total_candidate_features': len(candidate_features)
    }
    hook_name = f"model.layers.{layer_idx}"
    
    for prompt in prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        
        # Create targeted ablation hook
        ablation_hook = TargetedAblationHook(sae, candidate_features, n_features_to_ablate, lambda_scale, activation_threshold)
        hook_handle = model.get_submodule(hook_name).register_forward_hook(
            ablation_hook.activation_modification_hook
        )
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if we found enough active features for ablation
            if hasattr(ablation_hook, 'ablation_occurred') and ablation_hook.ablation_occurred:
                results[prompt] = generated_text
                ablation_stats['successful_ablation'] += 1
                ablation_stats['selected_features_per_prompt'][prompt] = ablation_hook.selected_active_features
            else:
                # Mark that insufficient features were active for ablation
                results[prompt] = f"[INSUFFICIENT_ACTIVE_FEATURES] {generated_text}"
                ablation_stats['insufficient_active_features'] += 1
                ablation_stats['insufficient_features_prompts'].append(prompt)
                ablation_stats['selected_features_per_prompt'][prompt] = ablation_hook.selected_active_features if hasattr(ablation_hook, 'selected_active_features') else []
                
        finally:
            hook_handle.remove()
    
    # Calculate percentage
    if ablation_stats['total_prompts'] > 0:
        ablation_stats['insufficient_features_percentage'] = (ablation_stats['insufficient_active_features'] / ablation_stats['total_prompts']) * 100
    else:
        ablation_stats['insufficient_features_percentage'] = 0.0
    
    return results, ablation_stats

def run_dynamic_random_ablation_experiment(model, tokenizer, sae, prompts, layer_idx, excluded_features, n_random_features, lambda_scale, max_new_tokens, device, activation_threshold=0.001):
    """
    Run dynamic random ablation experiment where random features are selected per prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sae: Sparse autoencoder
        prompts: List of input prompts
        layer_idx: Layer index to ablate
        excluded_features: List of all feature indices to exclude from random selection (entire candidate array)
        n_random_features: Number of random features to ablate per prompt
        lambda_scale: Ablation strength
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        activation_threshold: Minimum activation value to consider a feature as active
    
    Returns:
        Tuple: (results_dict, ablation_stats)
        - results_dict: Dictionary mapping prompts to generated outputs
        - ablation_stats: Dictionary with ablation statistics
    """
    results = {}
    ablation_stats = {
        'total_prompts': len(prompts),
        'insufficient_random_features': 0,
        'successful_random_ablation': 0,
        'insufficient_random_prompts': [],  # List of prompts where insufficient random features were available
        'selected_features_per_prompt': {},  # Track which features were selected for each prompt
        'n_random_features': n_random_features,
        'total_excluded_features': len(excluded_features)
    }
    hook_name = f"model.layers.{layer_idx}"
    
    for prompt in prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        
        # Create dynamic random ablation hook
        ablation_hook = DynamicRandomAblationHook(sae, excluded_features, n_random_features, lambda_scale, activation_threshold)
        hook_handle = model.get_submodule(hook_name).register_forward_hook(
            ablation_hook.activation_modification_hook
        )
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if we found enough random features for ablation
            if hasattr(ablation_hook, 'ablation_occurred') and ablation_hook.ablation_occurred:
                results[prompt] = generated_text
                ablation_stats['successful_random_ablation'] += 1
                ablation_stats['selected_features_per_prompt'][prompt] = ablation_hook.selected_random_features
            else:
                # Mark that insufficient random features were available for ablation
                results[prompt] = f"[INSUFFICIENT_RANDOM_FEATURES] {generated_text}"
                ablation_stats['insufficient_random_features'] += 1
                ablation_stats['insufficient_random_prompts'].append(prompt)
                ablation_stats['selected_features_per_prompt'][prompt] = ablation_hook.selected_random_features if hasattr(ablation_hook, 'selected_random_features') else []
                
        finally:
            hook_handle.remove()
    
    # Calculate percentage
    if ablation_stats['total_prompts'] > 0:
        ablation_stats['insufficient_random_percentage'] = (ablation_stats['insufficient_random_features'] / ablation_stats['total_prompts']) * 100
    else:
        ablation_stats['insufficient_random_percentage'] = 0.0
    
    return results, ablation_stats

def run_no_ablation_experiment(model, tokenizer, prompts, max_new_tokens, device):
    """
    Run baseline experiment with no ablation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
    
    Returns:
        Dictionary mapping prompts to generated outputs
    """
    results = {}
    
    for prompt in prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[prompt] = generated_text
                
        except Exception as e:
            results[prompt] = f"[ERROR] {str(e)}"
    
    return results

def main():
    # ==== HARDCODE YOUR VARIABLES HERE ====
    operation = "add"  # "add" or "sub" - which operation to ablate
    top_n = 10  # Number of top features to use
    dataset_file = None  # Path to dataset file (e.g., "data/addition.txt")
    lambda_scale = 1.0  # Scaling factor for ablation
    max_new_tokens = 50  # Maximum tokens to generate
    save_dir = None  # Directory to save results
    n_features = 10  # Number of features to ablate in each experiment
    run_all_layers = True  # If True, run for all available layers; if False, specify layer_idx
    layer_idx = None  # Specific layer to run (only used if run_all_layers=False)
    run_no_ablation = True  # If False, skip no ablation baseline (saves time for repeated experiments)
    activation_threshold = 0.001  # Minimum activation value to consider a feature as active
    # =======================================

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model.eval()

    # Load dataset
    prompts = load_dataset(dataset_file)

    if run_all_layers:
        # Load features for all layers
        all_layer_features = load_all_layers_features(operation, top_n)
        available_layers = sorted(all_layer_features.keys())
        
        # Initialize results structure for multi-layer
        results = {
            'first_n_ablation': {},
            'random_n_ablation': {}
        }
        ablation_stats = {
            'first_n_ablation': {},
            'random_n_ablation': {}
        }
        if run_no_ablation:
            results['no_ablation'] = {}
            ablation_stats['no_ablation'] = {}
        
        for current_layer in available_layers:
            print(f"Running experiments for layer {current_layer}...")
            
            # Load SAE for current layer
            sae_id = f"layer_{current_layer}/width_16k/canonical"
            sae, _, _ = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=sae_id
            )
            model_dtype = next(model.parameters()).dtype
            sae = sae.to(device=device, dtype=model_dtype)
            sae.eval()
            
            # Get features for current layer  
            candidate_features = all_layer_features[current_layer].tolist()
            
            # Run experiments for current layer
            if run_no_ablation:
                layer_results = run_no_ablation_experiment(model, tokenizer, prompts, max_new_tokens, device)
                results['no_ablation'][current_layer] = layer_results
                ablation_stats['no_ablation'][current_layer] = {'total_prompts': len(prompts)}
                
            layer_results, layer_stats = run_targeted_ablation_experiment(
                model, tokenizer, sae, prompts, current_layer, candidate_features, n_features, lambda_scale, max_new_tokens, device, activation_threshold
            )
            results['first_n_ablation'][current_layer] = layer_results
            ablation_stats['first_n_ablation'][current_layer] = layer_stats
            
            # Use dynamic random ablation excluding the entire candidate array
            layer_results, layer_stats = run_dynamic_random_ablation_experiment(
                model, tokenizer, sae, prompts, current_layer, candidate_features, n_features, lambda_scale, max_new_tokens, device, activation_threshold
            )
            results['random_n_ablation'][current_layer] = layer_results
            ablation_stats['random_n_ablation'][current_layer] = layer_stats
        
        # Save experiment parameters
        params = {
            'operation': operation,
            'top_n': top_n,
            'dataset_file': dataset_file,
            'lambda_scale': lambda_scale,
            'max_new_tokens': max_new_tokens,
            'n_features': n_features,
            'run_all_layers': run_all_layers,
            'run_no_ablation': run_no_ablation,
            'activation_threshold': activation_threshold,
            'available_layers': available_layers,
            'layer_features': {str(layer): features.tolist() for layer, features in all_layer_features.items()},
            'ablation_method': 'targeted_first_n_active',
            'random_ablation_method': 'dynamic_per_prompt_excluding_candidates'
        }
        
    else:
        # Single layer mode (backward compatibility)
        if layer_idx is None:
            raise ValueError("layer_idx must be specified when run_all_layers=False")
            
        # Load SAE for specified layer
        sae_id = f"layer_{layer_idx}/width_16k/canonical"
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=sae_id
        )
        model_dtype = next(model.parameters()).dtype
        sae = sae.to(device=device, dtype=model_dtype)
        sae.eval()

        # Load features for specified layer
        candidate_features = load_features_from_npy(layer_idx, operation, top_n).tolist()

        # Run experiments
        first_results, first_stats = run_targeted_ablation_experiment(model, tokenizer, sae, prompts, layer_idx, candidate_features, n_features, lambda_scale, max_new_tokens, device, activation_threshold)
        # Use dynamic random ablation excluding the entire candidate array
        random_results, random_stats = run_dynamic_random_ablation_experiment(model, tokenizer, sae, prompts, layer_idx, candidate_features, n_features, lambda_scale, max_new_tokens, device, activation_threshold)
        
        results = {
            'first_n_ablation': first_results,
            'random_n_ablation': random_results
        }
        ablation_stats = {
            'first_n_ablation': first_stats,
            'random_n_ablation': random_stats
        }
        
        if run_no_ablation:
            no_ablation_results = run_no_ablation_experiment(model, tokenizer, prompts, max_new_tokens, device)
            results['no_ablation'] = no_ablation_results
            ablation_stats['no_ablation'] = {'total_prompts': len(prompts)}

        # Save experiment parameters
        params = {
            'operation': operation,
            'top_n': top_n,
            'layer_idx': layer_idx,
            'dataset_file': dataset_file,
            'lambda_scale': lambda_scale,
            'max_new_tokens': max_new_tokens,
            'n_features': n_features,
            'run_all_layers': run_all_layers,
            'run_no_ablation': run_no_ablation,
            'activation_threshold': activation_threshold,
            'candidate_features': candidate_features,
            'ablation_method': 'targeted_first_n_active',
            'random_ablation_method': 'dynamic_per_prompt_excluding_candidates'
        }

    # Save results
    save_results(save_dir, results, params, ablation_stats)

if __name__ == "__main__":
    main()
