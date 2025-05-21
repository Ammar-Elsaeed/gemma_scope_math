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

def find_consistent_features(counts: np.ndarray, threshold: float = 0.95) -> Set[int]:
    """Find features that are active above the threshold."""
    total_examples = 10000  # Known dataset size
    return set(np.where(counts / total_examples >= threshold)[0])

def find_range_agnostic_features(normal_counts: np.ndarray, random_counts: np.ndarray, 
                               threshold: float = 0.95) -> Set[int]:
    """Find features that are consistently active in both normal and random ranges."""
    normal_features = find_consistent_features(normal_counts, threshold)
    random_features = find_consistent_features(random_counts, threshold)
    return normal_features.intersection(random_features)

def analyze_operation_selectivity(feature_idx: int, 
                                add_counts: np.ndarray, random_add_counts: np.ndarray,
                                sub_counts: np.ndarray, random_sub_counts: np.ndarray) -> Tuple[float, float]:
    """Calculate how selective a feature is for addition vs subtraction operations."""
    total_examples = 10000
    
    # Average activation rate for addition (including random)
    add_rate = (add_counts[feature_idx] + random_add_counts[feature_idx]) / (2 * total_examples)
    
    # Average activation rate for subtraction (including random)
    sub_rate = (sub_counts[feature_idx] + random_sub_counts[feature_idx]) / (2 * total_examples)
    
    return add_rate, sub_rate

def analyze_layer(layer: int, threshold: float = 0.95) -> Dict:
    """Analyze range-agnostic features and their operation selectivity."""
    # Load counts for all datasets
    add_counts = load_counts_for_layer(layer, "addition")
    random_add_counts = load_counts_for_layer(layer, "random_addition")
    sub_counts = load_counts_for_layer(layer, "subtraction")
    random_sub_counts = load_counts_for_layer(layer, "random_subtraction")
    
    # Find range-agnostic features
    addition_agnostic = find_range_agnostic_features(add_counts, random_add_counts, threshold)
    subtraction_agnostic = find_range_agnostic_features(sub_counts, random_sub_counts, threshold)
    
    # Analyze operation selectivity for range-agnostic features
    selective_features = {
        "addition_selective": [],
        "subtraction_selective": [],
        "general_arithmetic": [],  # Features active in both operations
        "other": []  # Features that don't fit other categories
    }
    
    # Analyze all range-agnostic features
    all_agnostic = addition_agnostic.union(subtraction_agnostic)
    
    # More relaxed criteria for operation selectivity
    SENSITIVITY_THRESHOLD = 0.90  # 90% sensitivity
    SPECIFICITY_THRESHOLD = 0.80  # 80% specificity (meaning other operation should be < 20%)
    
    for feature_idx in all_agnostic:
        add_rate, sub_rate = analyze_operation_selectivity(
            feature_idx, add_counts, random_add_counts, sub_counts, random_sub_counts
        )
        
        # Classify feature based on selectivity with relaxed criteria
        if add_rate >= SENSITIVITY_THRESHOLD and sub_rate <= (1 - SPECIFICITY_THRESHOLD):
            selective_features["addition_selective"].append({
                "idx": feature_idx,
                "add_rate": add_rate,
                "sub_rate": sub_rate,
                "range_agnostic_for": "addition" if feature_idx in addition_agnostic else "both"
            })
        elif sub_rate >= SENSITIVITY_THRESHOLD and add_rate <= (1 - SPECIFICITY_THRESHOLD):
            selective_features["subtraction_selective"].append({
                "idx": feature_idx,
                "add_rate": add_rate,
                "sub_rate": sub_rate,
                "range_agnostic_for": "subtraction" if feature_idx in subtraction_agnostic else "both"
            })
        elif add_rate >= SENSITIVITY_THRESHOLD and sub_rate >= SENSITIVITY_THRESHOLD:
            selective_features["general_arithmetic"].append({
                "idx": feature_idx,
                "add_rate": add_rate,
                "sub_rate": sub_rate,
                "range_agnostic_for": "both" if feature_idx in addition_agnostic and feature_idx in subtraction_agnostic 
                                    else "addition" if feature_idx in addition_agnostic
                                    else "subtraction"
            })
        else:
            selective_features["other"].append({
                "idx": feature_idx,
                "add_rate": add_rate,
                "sub_rate": sub_rate,
                "range_agnostic_for": "both" if feature_idx in addition_agnostic and feature_idx in subtraction_agnostic 
                                    else "addition" if feature_idx in addition_agnostic
                                    else "subtraction"
            })
    
    return {
        "addition_agnostic": addition_agnostic,
        "subtraction_agnostic": subtraction_agnostic,
        "selective_features": selective_features,
        "counts": {
            "addition": add_counts,
            "random_addition": random_add_counts,
            "subtraction": sub_counts,
            "random_subtraction": random_sub_counts
        }
    }

def print_layer_results(layer: int, results: Dict):
    """Print analysis results for a layer."""
    print(f"\n=== Layer {layer} Analysis ===")
    print("Using criteria: sensitivity >= 90%, specificity >= 80%")
    
    # Print range-agnostic feature counts
    print("\nRange-Agnostic Features:")
    print(f"Addition: {len(results['addition_agnostic'])} features")
    print(f"Subtraction: {len(results['subtraction_agnostic'])} features")
    
    # Print operation-selective features
    print("\nFeature Analysis (among range-agnostic):")
    
    # Addition-selective
    add_selective = results['selective_features']['addition_selective']
    print(f"\nAddition-Selective (high add, low sub): {len(add_selective)} features")
    for i, feat in enumerate(add_selective[:10]):
        print(f"Feature {feat['idx']}: Add rate={feat['add_rate']:.3f}, "
              f"Sub rate={feat['sub_rate']:.3f}, Range-agnostic for: {feat['range_agnostic_for']}")
    
    # Subtraction-selective
    sub_selective = results['selective_features']['subtraction_selective']
    print(f"\nSubtraction-Selective (high sub, low add): {len(sub_selective)} features")
    for i, feat in enumerate(sub_selective[:10]):
        print(f"Feature {feat['idx']}: Add rate={feat['add_rate']:.3f}, "
              f"Sub rate={feat['sub_rate']:.3f}, Range-agnostic for: {feat['range_agnostic_for']}")
    
    # General arithmetic features
    general = results['selective_features']['general_arithmetic']
    print(f"\nGeneral Arithmetic Features (high add AND high sub): {len(general)} features")
    for i, feat in enumerate(general[:10]):
        print(f"Feature {feat['idx']}: Add rate={feat['add_rate']:.3f}, "
              f"Sub rate={feat['sub_rate']:.3f}, Range-agnostic for: {feat['range_agnostic_for']}")
    
    # Other features
    other = results['selective_features']['other']
    print(f"\nOther Features (not fitting above categories): {len(other)} features")
    for i, feat in enumerate(other[:10]):
        print(f"Feature {feat['idx']}: Add rate={feat['add_rate']:.3f}, "
              f"Sub rate={feat['sub_rate']:.3f}, Range-agnostic for: {feat['range_agnostic_for']}")

def main():
    parser = argparse.ArgumentParser(description='Analyze range-agnostic features across layers')
    parser.add_argument('layers', type=int, nargs='+', help='Layer numbers to analyze')
    parser.add_argument('--threshold', type=float, default=0.95, help='Activation threshold (default: 0.95)')
    args = parser.parse_args()
    
    for layer in args.layers:
        try:
            results = analyze_layer(layer, args.threshold)
            print_layer_results(layer, results)
            
            # Save results
            output_dir = os.path.join("range_agnostic_metrics", f"layer_{layer}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save feature indices
            np.save(os.path.join(output_dir, "addition_agnostic.npy"), 
                   np.array(list(results["addition_agnostic"])))
            np.save(os.path.join(output_dir, "subtraction_agnostic.npy"), 
                   np.array(list(results["subtraction_agnostic"])))
            
            # Save selective features data
            for category, features in results["selective_features"].items():
                if features:  # Only save if there are features in this category
                    feature_data = np.array([(f["idx"], f["add_rate"], f["sub_rate"]) 
                                          for f in features],
                                          dtype=[('idx', 'i4'), 
                                                ('add_rate', 'f4'), 
                                                ('sub_rate', 'f4')])
                    np.save(os.path.join(output_dir, f"{category}.npy"), feature_data)
            
        except FileNotFoundError as e:
            print(f"\nError analyzing layer {layer}: {e}")

if __name__ == "__main__":
    main() 