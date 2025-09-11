import numpy as np
import os
from scipy import sparse
import glob
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_latents_for_layer(layer_idx, datasets=None):
    """
    Load latents for a specific layer across all datasets.
    
    Args:
        layer_idx: Layer index to load
        datasets: List of dataset names (default: all four datasets)
        
    Returns:
        dict: Dictionary with dataset names as keys and sparse matrices as values
    """
    if datasets is None:
        datasets = ['addition', 'subtraction', 'random_addition', 'random_subtraction']
    
    latents = {}
    for dataset in datasets:
        filepath = os.path.join(BASE_DIR, "latents", dataset, f"layer_{layer_idx}.npz")
        if os.path.exists(filepath):
            latents[dataset] = sparse.load_npz(filepath)
        else:
            print(f"Warning: File not found: {filepath}")
            latents[dataset] = None
    
    return latents


def calculate_median_activations(sparse_matrix):
    """
    Calculate median activation for each feature across all examples.
    
    Args:
        sparse_matrix: Scipy sparse matrix of shape (examples, features)
        
    Returns:
        numpy array: Median activations for each feature
    """
    if sparse_matrix is None:
        return None
    
    # Convert to dense for median calculation (memory intensive but necessary)
    # For very large matrices, we might need to process in chunks
    dense_matrix = sparse_matrix.toarray()
    medians = np.median(dense_matrix, axis=0)
    return medians


def calculate_addition_metric(median_add, median_sub, median_rand_add, median_rand_sub):
    """
    Calculate the heuristic metric for addition-focused features.
    
    Args:
        median_add: Median activation for addition
        median_sub: Median activation for subtraction  
        median_rand_add: Median activation for random addition
        median_rand_sub: Median activation for random subtraction
        
    Returns:
        float: Calculated addition metric
    """
    metric = (median_add + median_rand_add 
              - abs(median_add - median_rand_add) 
              - (median_sub + median_rand_sub))
    return metric


def calculate_subtraction_metric(median_add, median_sub, median_rand_add, median_rand_sub):
    """
    Calculate the heuristic metric for subtraction-focused features.
    
    Args:
        median_add: Median activation for addition
        median_sub: Median activation for subtraction  
        median_rand_add: Median activation for random addition
        median_rand_sub: Median activation for random subtraction
        
    Returns:
        float: Calculated subtraction metric
    """
    metric = (median_sub + median_rand_sub 
              - abs(median_sub - median_rand_sub) 
              - (median_add + median_rand_add))
    return metric


def process_layer(layer_idx):
    """
    Process a single layer and calculate both addition and subtraction metrics for all features.
    
    Args:
        layer_idx: Layer index to process
        
    Returns:
        tuple: (addition_df, subtraction_df) DataFrames with feature indices, metrics, and median values
    """
    print(f"Processing layer {layer_idx}...")
    
    # Load latents for this layer
    latents = load_latents_for_layer(layer_idx)
    
    # Check if all datasets are available
    if any(latents[dataset] is None for dataset in latents.keys()):
        print(f"Skipping layer {layer_idx} due to missing data")
        return None, None
    
    # Calculate median activations for each dataset
    print(f"  Calculating median activations...")
    medians = {}
    for dataset, matrix in latents.items():
        medians[dataset] = calculate_median_activations(matrix)
    
    # Get number of features (should be same across all datasets)
    num_features = medians['addition'].shape[0]
    
    # Calculate metrics for all features
    print(f"  Calculating metrics for {num_features} features...")
    addition_results = []
    subtraction_results = []
    
    for feature_idx in range(num_features):
        # Get median values for this feature
        med_add = medians['addition'][feature_idx]
        med_sub = medians['subtraction'][feature_idx]
        med_rand_add = medians['random_addition'][feature_idx]
        med_rand_sub = medians['random_subtraction'][feature_idx]
        
        # Calculate both metrics
        addition_metric = calculate_addition_metric(med_add, med_sub, med_rand_add, med_rand_sub)
        subtraction_metric = calculate_subtraction_metric(med_add, med_sub, med_rand_add, med_rand_sub)
        
        # Store addition results
        addition_results.append({
            'feature_idx': feature_idx,
            'metric': addition_metric,
            'median_addition': med_add,
            'median_subtraction': med_sub,
            'median_random_addition': med_rand_add,
            'median_random_subtraction': med_rand_sub
        })
        
        # Store subtraction results
        subtraction_results.append({
            'feature_idx': feature_idx,
            'metric': subtraction_metric,
            'median_addition': med_add,
            'median_subtraction': med_sub,
            'median_random_addition': med_rand_add,
            'median_random_subtraction': med_rand_sub
        })
    
    # Convert to DataFrames and sort by metric (descending)
    addition_df = pd.DataFrame(addition_results)
    addition_df = addition_df.sort_values('metric', ascending=False).reset_index(drop=True)
    
    subtraction_df = pd.DataFrame(subtraction_results)
    subtraction_df = subtraction_df.sort_values('metric', ascending=False).reset_index(drop=True)
    
    return addition_df, subtraction_df


def save_addition_results(df, layer_idx, output_dir):
    """
    Save addition-focused results for a single layer to a CSV file.
    
    Args:
        df: DataFrame with addition results
        layer_idx: Layer index
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"layer_{layer_idx}_feature_metrics.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False, float_format='%.6f')
    print(f"Saved addition results for layer {layer_idx} to {filepath}")


def save_subtraction_results(df, layer_idx, output_dir):
    """
    Save subtraction-focused results for a single layer to a CSV file.
    
    Args:
        df: DataFrame with subtraction results
        layer_idx: Layer index
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"layer_{layer_idx}_subtraction_features.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False, float_format='%.6f')
    print(f"Saved subtraction results for layer {layer_idx} to {filepath}")


def get_available_layers():
    """
    Get list of available layer indices by checking the addition directory.
    
    Returns:
        list: Sorted list of available layer indices
    """
    pattern = os.path.join(BASE_DIR, "latents", "addition", "layer_*.npz")
    files = glob.glob(pattern)
    layer_indices = []
    
    for file in files:
        filename = os.path.basename(file)
        # Extract layer number from filename
        layer_num = int(filename.split('_')[1].split('.')[0])
        layer_indices.append(layer_num)
    
    return sorted(layer_indices)


def main():
    """
    Main function to process all layers and calculate both addition and subtraction feature metrics.
    """
    print("Feature Metric Calculator")
    print("=" * 40)
    
    # Get available layers
    available_layers = get_available_layers()
    print(f"Found {len(available_layers)} layers: {available_layers}")
    
    # Output directory
    output_dir = os.path.join(BASE_DIR, "feature_metrics")
    print(f"Output directory: {output_dir}")
    
    # Process each layer
    for layer_idx in tqdm(available_layers, desc="Processing layers"):
        try:
            # Process layer (returns both addition and subtraction DataFrames)
            addition_df, subtraction_df = process_layer(layer_idx)
            
            if addition_df is not None and subtraction_df is not None:
                # Save both sets of results
                save_addition_results(addition_df, layer_idx, output_dir)
                save_subtraction_results(subtraction_df, layer_idx, output_dir)
                
                # Print some stats
                print(f"  Layer {layer_idx}: {len(addition_df)} features processed")
                print(f"    Addition - Top metric: {addition_df.iloc[0]['metric']:.6f}, Bottom metric: {addition_df.iloc[-1]['metric']:.6f}")
                print(f"    Subtraction - Top metric: {subtraction_df.iloc[0]['metric']:.6f}, Bottom metric: {subtraction_df.iloc[-1]['metric']:.6f}")
            
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {str(e)}")
            continue
    
    print("\nProcessing complete!")
    print(f"Results saved in: {output_dir}")
    print("Files generated:")
    print("  - layer_X_feature_metrics.csv (addition-focused)")
    print("  - layer_X_subtraction_features.csv (subtraction-focused)")


if __name__ == "__main__":
    main()
