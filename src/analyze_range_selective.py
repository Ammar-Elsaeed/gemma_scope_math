import numpy as np
import os
from typing import Dict, List, Set, Tuple
import argparse

def load_counts_for_layer(layer: int, dataset: str) -> np.ndarray:
    """Load activation counts for a specific layer and dataset."""
    path = os.path.join("activ_freq", dataset, f"layer_{layer}", "active_counts.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No activation counts found at {path}")
    return np.load(path)

def find_active_features(counts: np.ndarray, threshold: float = 0.95) -> Set[int]:
    """Find features that are active above the threshold."""
    total_examples = 10000  # Known dataset size
    return set(np.where(counts / total_examples >= threshold)[0])

def find_range_selective_features(normal_counts: np.ndarray, random_counts: np.ndarray, 
                                threshold: float = 0.95) -> Tuple[Set[int], Set[int]]:
    """Find features that are selective for either normal or random range.
    
    Returns:
        Tuple of (normal_selective, random_selective) where:
        - normal_selective: Features active in normal range but not in random range
        - random_selective: Features active in random range but not in normal range
    """
    normal_features = find_active_features(normal_counts, threshold)
    random_features = find_active_features(random_counts, threshold)
    
    normal_selective = normal_features - random_features  # Active only in normal range
    random_selective = random_features - normal_features  # Active only in random range
    
    return normal_selective, random_selective

def analyze_layer(layer: int, threshold: float = 0.95) -> Dict:
    """Analyze range selectivity for both operations."""
    # Load counts for all datasets
    add_counts = load_counts_for_layer(layer, "addition")
    random_add_counts = load_counts_for_layer(layer, "random_addition")
    sub_counts = load_counts_for_layer(layer, "subtraction")
    random_sub_counts = load_counts_for_layer(layer, "random_subtraction")
    
    # Find range-selective features for each operation
    add_normal_selective, add_random_selective = find_range_selective_features(
        add_counts, random_add_counts, threshold
    )
    sub_normal_selective, sub_random_selective = find_range_selective_features(
        sub_counts, random_sub_counts, threshold
    )
    
    # Find features that are range-selective for both operations
    both_normal_selective = add_normal_selective.intersection(sub_normal_selective)
    both_random_selective = add_random_selective.intersection(sub_random_selective)
    
    # Calculate activation rates for each feature
    total_examples = 10000
    results = {
        "addition": {
            "normal_selective": [(idx, 
                add_counts[idx]/total_examples,
                random_add_counts[idx]/total_examples) for idx in add_normal_selective],
            "random_selective": [(idx,
                add_counts[idx]/total_examples,
                random_add_counts[idx]/total_examples) for idx in add_random_selective]
        },
        "subtraction": {
            "normal_selective": [(idx,
                sub_counts[idx]/total_examples,
                random_sub_counts[idx]/total_examples) for idx in sub_normal_selective],
            "random_selective": [(idx,
                sub_counts[idx]/total_examples,
                random_sub_counts[idx]/total_examples) for idx in sub_random_selective]
        },
        "both_operations": {
            "normal_selective": [(idx,
                (add_counts[idx]/total_examples, random_add_counts[idx]/total_examples),
                (sub_counts[idx]/total_examples, random_sub_counts[idx]/total_examples)) 
                for idx in both_normal_selective],
            "random_selective": [(idx,
                (add_counts[idx]/total_examples, random_add_counts[idx]/total_examples),
                (sub_counts[idx]/total_examples, random_sub_counts[idx]/total_examples))
                for idx in both_random_selective]
        }
    }
    
    return results

def print_layer_results(layer: int, results: Dict):
    """Print analysis results for a layer."""
    print(f"\n=== Layer {layer} Range Selectivity Analysis ===")
    
    # Addition range-selective features
    print("\nAddition Range-Selective Features:")
    print(f"Normal range selective: {len(results['addition']['normal_selective'])} features")
    for idx, normal_rate, random_rate in results['addition']['normal_selective'][:10]:
        print(f"Feature {idx}: Normal={normal_rate:.3f}, Random={random_rate:.3f}")
    
    print(f"\nRandom range selective: {len(results['addition']['random_selective'])} features")
    for idx, normal_rate, random_rate in results['addition']['random_selective'][:10]:
        print(f"Feature {idx}: Normal={normal_rate:.3f}, Random={random_rate:.3f}")
    
    # Subtraction range-selective features
    print("\nSubtraction Range-Selective Features:")
    print(f"Normal range selective: {len(results['subtraction']['normal_selective'])} features")
    for idx, normal_rate, random_rate in results['subtraction']['normal_selective'][:10]:
        print(f"Feature {idx}: Normal={normal_rate:.3f}, Random={random_rate:.3f}")
    
    print(f"\nRandom range selective: {len(results['subtraction']['random_selective'])} features")
    for idx, normal_rate, random_rate in results['subtraction']['random_selective'][:10]:
        print(f"Feature {idx}: Normal={normal_rate:.3f}, Random={random_rate:.3f}")
    
    # Features range-selective for both operations
    print("\nFeatures Range-Selective for Both Operations:")
    print(f"Normal range selective: {len(results['both_operations']['normal_selective'])} features")
    for idx, add_rates, sub_rates in results['both_operations']['normal_selective'][:10]:
        print(f"Feature {idx}:")
        print(f"  Addition: Normal={add_rates[0]:.3f}, Random={add_rates[1]:.3f}")
        print(f"  Subtraction: Normal={sub_rates[0]:.3f}, Random={sub_rates[1]:.3f}")
    
    print(f"\nRandom range selective: {len(results['both_operations']['random_selective'])} features")
    for idx, add_rates, sub_rates in results['both_operations']['random_selective'][:10]:
        print(f"Feature {idx}:")
        print(f"  Addition: Normal={add_rates[0]:.3f}, Random={add_rates[1]:.3f}")
        print(f"  Subtraction: Normal={sub_rates[0]:.3f}, Random={sub_rates[1]:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze range-selective features across layers')
    parser.add_argument('layers', type=int, nargs='+', help='Layer numbers to analyze')
    parser.add_argument('--threshold', type=float, default=0.95, help='Activation threshold (default: 0.95)')
    args = parser.parse_args()
    
    for layer in args.layers:
        try:
            results = analyze_layer(layer, args.threshold)
            print_layer_results(layer, results)
            
            # Save results
            output_dir = os.path.join("range_selective_metrics", f"layer_{layer}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save single-operation results
            for operation in ['addition', 'subtraction']:
                for selectivity_type in ['normal_selective', 'random_selective']:
                    if results[operation][selectivity_type]:
                        # Create structured array for single operation data
                        dtype = [('idx', 'i4'), ('normal_rate', 'f4'), ('random_rate', 'f4')]
                        data = np.array(results[operation][selectivity_type], dtype=dtype)
                        np.save(os.path.join(output_dir, f"{operation}_{selectivity_type}.npy"), data)
            
            # Save both-operations results
            for selectivity_type in ['normal_selective', 'random_selective']:
                if results['both_operations'][selectivity_type]:
                    # Create structured array for both operations data
                    dtype = [
                        ('idx', 'i4'),
                        ('add_normal_rate', 'f4'),
                        ('add_random_rate', 'f4'),
                        ('sub_normal_rate', 'f4'),
                        ('sub_random_rate', 'f4')
                    ]
                    
                    # Convert the data to the right format
                    formatted_data = [
                        (idx, 
                         add_rates[0], add_rates[1],  # Addition rates
                         sub_rates[0], sub_rates[1])  # Subtraction rates
                        for idx, add_rates, sub_rates in results['both_operations'][selectivity_type]
                    ]
                    
                    data = np.array(formatted_data, dtype=dtype)
                    np.save(os.path.join(output_dir, f"both_operations_{selectivity_type}.npy"), data)
            
        except FileNotFoundError as e:
            print(f"\nError analyzing layer {layer}: {e}")

if __name__ == "__main__":
    main() 