import numpy as np
import os
from typing import List, Dict, Tuple
import argparse

def load_counts_for_layer(layer: int, dataset: str) -> np.ndarray:
    """Load activation counts for a specific layer and dataset."""
    path = os.path.join("activ_freq", dataset, f"layer_{layer}", "active_counts.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No activation counts found at {path}")
    return np.load(path)

def calculate_selectivity(counts_a: np.ndarray, counts_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate how selective each feature is for dataset A vs B.
    
    Args:
        counts_a: Activation counts for first dataset
        counts_b: Activation counts for second dataset
        
    Returns:
        Tuple of (sensitivity_to_a, specificity_to_a) where:
        - sensitivity_to_a = activation rate in dataset A
        - specificity_to_a = non-activation rate in dataset B
    """
    total_examples = 10000  # As we know each dataset has 10k examples
    
    sensitivity = counts_a / total_examples
    specificity = (total_examples - counts_b) / total_examples
    
    return sensitivity, specificity

def analyze_layer(layer: int) -> Dict:
    """Analyze feature selectivity for both operations in a layer."""
    # Load counts for both datasets
    addition_counts = load_counts_for_layer(layer, "addition")
    subtraction_counts = load_counts_for_layer(layer, "subtraction")
    
    # Calculate metrics for addition-selective features
    add_sensitivity, add_specificity = calculate_selectivity(addition_counts, subtraction_counts)
    
    # Calculate metrics for subtraction-selective features
    sub_sensitivity, sub_specificity = calculate_selectivity(subtraction_counts, addition_counts)
    
    # Find highly selective features (both sensitivity and specificity > 0.95)
    addition_selective = np.where((add_sensitivity > 0.95) & (add_specificity > 0.95))[0]
    subtraction_selective = np.where((sub_sensitivity > 0.95) & (sub_specificity > 0.95))[0]
    
    # Find features highly active in both operations (high sensitivity in both)
    common_features = np.where((add_sensitivity > 0.95) & (sub_sensitivity > 0.95))[0]
    
    return {
        "addition_selective": addition_selective,
        "subtraction_selective": subtraction_selective,
        "common_features": common_features,
        "add_sensitivity": add_sensitivity,
        "add_specificity": add_specificity,
        "sub_sensitivity": sub_sensitivity,
        "sub_specificity": sub_specificity,
        "addition_counts": addition_counts,
        "subtraction_counts": subtraction_counts
    }

def print_layer_results(layer: int, results: Dict):
    """Print analysis results for a layer."""
    print(f"\n=== Layer {layer} Analysis ===")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Mean Addition Sensitivity: {results['add_sensitivity'].mean():.3f}")
    print(f"Mean Addition Specificity: {results['add_specificity'].mean():.3f}")
    print(f"Mean Subtraction Sensitivity: {results['sub_sensitivity'].mean():.3f}")
    print(f"Mean Subtraction Specificity: {results['sub_specificity'].mean():.3f}")
    
    # Print addition-selective features
    print(f"\nAddition-Selective Features (sens > 0.95, spec > 0.95): {len(results['addition_selective'])}")
    if len(results['addition_selective']) > 0:
        top_add = results['addition_selective'][:10]
        print("Top 10 addition-selective feature indices:", top_add)
        for idx in top_add:
            print(f"Feature {idx}: "
                  f"Add(sens={results['add_sensitivity'][idx]:.3f}, count={results['addition_counts'][idx]}), "
                  f"Sub(sens={results['sub_sensitivity'][idx]:.3f}, count={results['subtraction_counts'][idx]})")
    
    # Print subtraction-selective features
    print(f"\nSubtraction-Selective Features (sens > 0.95, spec > 0.95): {len(results['subtraction_selective'])}")
    if len(results['subtraction_selective']) > 0:
        top_sub = results['subtraction_selective'][:10]
        print("Top 10 subtraction-selective feature indices:", top_sub)
        for idx in top_sub:
            print(f"Feature {idx}: "
                  f"Sub(sens={results['sub_sensitivity'][idx]:.3f}, count={results['subtraction_counts'][idx]}), "
                  f"Add(sens={results['add_sensitivity'][idx]:.3f}, count={results['addition_counts'][idx]})")
    
    # Print common features
    print(f"\nCommon Features (sens > 0.95 in both): {len(results['common_features'])}")
    if len(results['common_features']) > 0:
        top_common = results['common_features'][:10]
        print("Top 10 common feature indices:", top_common)
        for idx in top_common:
            print(f"Feature {idx}: "
                  f"Add(sens={results['add_sensitivity'][idx]:.3f}, count={results['addition_counts'][idx]}), "
                  f"Sub(sens={results['sub_sensitivity'][idx]:.3f}, count={results['subtraction_counts'][idx]})")

def main():
    parser = argparse.ArgumentParser(description='Analyze feature selectivity across layers')
    parser.add_argument('layers', type=int, nargs='+', help='Layer numbers to analyze')
    args = parser.parse_args()
    
    for layer in args.layers:
        try:
            results = analyze_layer(layer)
            print_layer_results(layer, results)
            
            # Save results
            output_dir = os.path.join("feature_metrics", f"layer_{layer}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics
            np.save(os.path.join(output_dir, "addition_selective.npy"), results["addition_selective"])
            np.save(os.path.join(output_dir, "subtraction_selective.npy"), results["subtraction_selective"])
            np.save(os.path.join(output_dir, "common_features.npy"), results["common_features"])
            
            # Save detailed metrics
            np.save(os.path.join(output_dir, "add_sensitivity.npy"), results["add_sensitivity"])
            np.save(os.path.join(output_dir, "add_specificity.npy"), results["add_specificity"])
            np.save(os.path.join(output_dir, "sub_sensitivity.npy"), results["sub_sensitivity"])
            np.save(os.path.join(output_dir, "sub_specificity.npy"), results["sub_specificity"])
            
        except FileNotFoundError as e:
            print(f"\nError analyzing layer {layer}: {e}")

if __name__ == "__main__":
    main() 