import numpy as np
import glob
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_activation_frequencies():
    """
    Load activation frequencies from files.
    
    Returns:
        tuple: (addition_freq, subtraction_freq, random_addition_freq, random_subtraction_freq)
    """
    # Load regular activation frequencies
    addition_files = sorted(glob.glob(os.path.join(BASE_DIR, "activ_freq/addition/layer_*.npy")))
    subtraction_files = sorted(glob.glob(os.path.join(BASE_DIR, "activ_freq/subtraction/layer_*.npy")))
    
    addition_freq = np.array([np.load(f) for f in addition_files])
    subtraction_freq = np.array([np.load(f) for f in subtraction_files])
    
    # Load random activation frequencies
    random_addition_files = sorted(glob.glob(os.path.join(BASE_DIR, "activ_freq/random_addition/layer_*.npy")))
    random_subtraction_files = sorted(glob.glob(os.path.join(BASE_DIR, "activ_freq/random_subtraction/layer_*.npy")))
    
    random_addition_freq = np.array([np.load(f) for f in random_addition_files])
    random_subtraction_freq = np.array([np.load(f) for f in random_subtraction_files])
    
    return addition_freq, subtraction_freq, random_addition_freq, random_subtraction_freq


def calculate_feature_indices(addition_freq, subtraction_freq, random_addition_freq, random_subtraction_freq, 
                            upper_bound=0.7, lower_bound=0.2):
    """
    Calculate feature indices based on activation frequencies and thresholds.
    
    Args:
        addition_freq: Regular addition activation frequencies
        subtraction_freq: Regular subtraction activation frequencies
        random_addition_freq: Random addition activation frequencies
        random_subtraction_freq: Random subtraction activation frequencies
        upper_bound: Upper threshold for activation frequency
        lower_bound: Lower threshold for activation frequency
        
    Returns:
        tuple: (addition_indices, subtraction_indices) as numpy arrays
    """
    addition_indices = []
    subtraction_indices = []

    for layer in range(addition_freq.shape[0]):
        # Regular frequencies
        add_layer = addition_freq[layer]
        sub_layer = subtraction_freq[layer]
        
        # Random frequencies
        random_add_layer = random_addition_freq[layer]
        random_sub_layer = random_subtraction_freq[layer]

        # Get indices for regular data using bounds
        regular_add_indices = set(np.where((add_layer > upper_bound) & (sub_layer < lower_bound))[0])
        regular_sub_indices = set(np.where((add_layer < lower_bound) & (sub_layer > upper_bound))[0])
        
        # Get indices for random data using bounds
        random_add_indices = set(np.where((random_add_layer > upper_bound) & (random_sub_layer < lower_bound))[0])
        random_sub_indices = set(np.where((random_add_layer < lower_bound) & (random_sub_layer > upper_bound))[0])
        
        # Take intersection of regular and random indices
        final_add_indices = np.array(list(regular_add_indices.intersection(random_add_indices)))
        final_sub_indices = np.array(list(regular_sub_indices.intersection(random_sub_indices)))

        addition_indices.append(final_add_indices)
        subtraction_indices.append(final_sub_indices)

    return np.array(addition_indices, dtype=object), np.array(subtraction_indices, dtype=object)


def get_top_bottom_features(addition_freq, subtraction_freq, random_addition_freq, random_subtraction_freq, top_n=3):
    """
    Get top and bottom features based on difference scores.
    
    Returns:
        tuple: (top_features, bottom_features) as numpy arrays
    """
    top_features = []
    bottom_features = []

    for layer in range(addition_freq.shape[0]):
        diff = (addition_freq[layer] - subtraction_freq[layer]) + (random_addition_freq[layer] - random_subtraction_freq[layer])
        sorted_indices = np.argsort(diff)

        top_indices = sorted_indices[-top_n:].astype(int)
        bottom_indices = sorted_indices[:top_n].astype(int)

        top_features.append(top_indices)
        bottom_features.append(bottom_indices)

    return np.array(top_features, dtype=object), np.array(bottom_features, dtype=object)


def save_feature_arrays(features_array, save_dir, prefix, suffix=""):
    """
    Split feature arrays along first dimension and save each layer separately.
    
    Args:
        features_array: Numpy array of features per layer
        save_dir: Directory to save the arrays
        prefix: Prefix for filenames (e.g., "add", "sub")
        suffix: Suffix for filenames containing hyperparameters
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_idx, layer_features in enumerate(features_array):
        filename = f"{prefix}_layer-{layer_idx}{suffix}.npy"
        filepath = os.path.join(save_dir, filename)
        np.save(filepath, layer_features)


def save_threshold_based_features(addition_indices, subtraction_indices, upper_bound, lower_bound, base_save_dir=None):
    """
    Save threshold-based features with proper directory structure.
    
    Args:
        addition_indices: Addition feature indices from calculate_feature_indices
        subtraction_indices: Subtraction feature indices from calculate_feature_indices
        upper_bound: Upper threshold used
        lower_bound: Lower threshold used
        base_save_dir: Base directory for saving
    """
    if base_save_dir is None:
        base_save_dir = os.path.join(BASE_DIR, "top_features")
        
    # Create directory path
    save_dir = os.path.join(base_save_dir, "threshold_based")
    
    # Create suffix with hyperparameters
    suffix = f"_sens-{upper_bound}_spec-{lower_bound}"
    
    # Save arrays
    save_feature_arrays(addition_indices, save_dir, "add", suffix)
    save_feature_arrays(subtraction_indices, save_dir, "sub", suffix)


def save_top_n_features(top_features, bottom_features, top_n, base_save_dir=None):
    """
    Save top-N features with proper directory structure.
    
    Args:
        top_features: Top features from get_top_bottom_features
        bottom_features: Bottom features from get_top_bottom_features
        top_n: Number of top features used
        base_save_dir: Base directory for saving
    """
    if base_save_dir is None:
        base_save_dir = os.path.join(BASE_DIR, "top_features")
        
    # Create directory path
    save_dir = os.path.join(base_save_dir, "top_N")
    
    # Create suffix with hyperparameters
    suffix = f"_top-{top_n}"
    
    # Save arrays
    save_feature_arrays(top_features, save_dir, "add", suffix)
    save_feature_arrays(bottom_features, save_dir, "sub", suffix)


# Example usage:
if __name__ == "__main__":
    # Load frequencies
    frequencies = load_activation_frequencies()
    addition_freq, subtraction_freq, random_addition_freq, random_subtraction_freq = frequencies

    # Calculate threshold-based features
    upper_bound = 0.6
    lower_bound = 0.3
    addition_indices, subtraction_indices = calculate_feature_indices(
        addition_freq, subtraction_freq, random_addition_freq, random_subtraction_freq,
        upper_bound=upper_bound, lower_bound=lower_bound
    )

    # Save threshold-based features
    save_threshold_based_features(addition_indices, subtraction_indices, upper_bound, lower_bound)

    # Calculate top-N features
    top_n = 10
    add_features, sub_features = get_top_bottom_features(
        addition_freq, subtraction_freq, random_addition_freq, random_subtraction_freq, top_n=top_n
    )

    # Save top-N features
    save_top_n_features(add_features, sub_features, top_n)