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

def analyze_correctness_selectivity(feature_idx: int, 
                                  correct_counts: np.ndarray,
                                  incorrect_counts: np.ndarray) -> Dict:
    """Calculate sensitivity, specificity, and selectivity for correct vs incorrect examples.
    
    Returns:
        Dict with metrics:
        - correct_rate: Activation rate in correct examples
        - incorrect_rate: Activation rate in incorrect examples
        - selectivity: Absolute difference between rates
        - specificity_for_correct: 1 - incorrect_rate (true negative rate for correct detection)
        - specificity_for_incorrect: 1 - correct_rate (true negative rate for incorrect detection)
    """
    total_examples = 20000  # Known dataset size
    
    # Calculate activation rates
    correct_rate = correct_counts[feature_idx] / total_examples
    incorrect_rate = incorrect_counts[feature_idx] / total_examples
    
    # Calculate metrics
    metrics = {
        "correct_rate": correct_rate,
        "incorrect_rate": incorrect_rate,
        "selectivity": abs(correct_rate - incorrect_rate),
        "specificity_for_correct": 1 - incorrect_rate,  # How specific is it to correct examples
        "specificity_for_incorrect": 1 - correct_rate   # How specific is it to incorrect examples
    }
    
    return metrics

def find_selective_features(correct_counts: np.ndarray, 
                          incorrect_counts: np.ndarray,
                          sensitivity_threshold: float = 0.90,
                          specificity_threshold: float = 0.80) -> Dict:
    """Find features that are selective for correct or incorrect examples."""
    num_features = len(correct_counts)
    selective_features = {
        "correct_selective": [],    # High activation for correct, low for incorrect
        "incorrect_selective": [],  # High activation for incorrect, low for correct
        "both_active": [],         # High activation for both
        "both_inactive": []        # Low activation for both
    }
    
    for feature_idx in range(num_features):
        metrics = analyze_correctness_selectivity(
            feature_idx, correct_counts, incorrect_counts
        )
        
        # Store feature data with all metrics
        feature_data = {
            "idx": feature_idx,
            **metrics  # Include all metrics
        }
        
        # Classify feature based on selectivity criteria
        if (metrics["correct_rate"] >= sensitivity_threshold and 
            metrics["specificity_for_correct"] >= specificity_threshold):
            selective_features["correct_selective"].append(feature_data)
        elif (metrics["incorrect_rate"] >= sensitivity_threshold and 
              metrics["specificity_for_incorrect"] >= specificity_threshold):
            selective_features["incorrect_selective"].append(feature_data)
        elif metrics["correct_rate"] >= sensitivity_threshold and metrics["incorrect_rate"] >= sensitivity_threshold:
            selective_features["both_active"].append(feature_data)
        elif metrics["correct_rate"] < sensitivity_threshold and metrics["incorrect_rate"] < sensitivity_threshold:
            selective_features["both_inactive"].append(feature_data)
    
    return selective_features

def analyze_layer(layer: int, 
                 sensitivity_threshold: float = 0.90,
                 specificity_threshold: float = 0.80) -> Dict:
    """Analyze feature selectivity for correct vs incorrect arithmetic."""
    # Load counts for correct and incorrect datasets
    correct_counts = load_counts_for_layer(layer, "correct_arithmetic")
    incorrect_counts = load_counts_for_layer(layer, "incorrect_arithmetic")
    
    # Find selective features
    selective_features = find_selective_features(
        correct_counts, incorrect_counts,
        sensitivity_threshold, specificity_threshold
    )
    
    return {
        "selective_features": selective_features,
        "counts": {
            "correct": correct_counts,
            "incorrect": incorrect_counts
        }
    }

def print_layer_results(layer: int, results: Dict):
    """Print analysis results for a layer."""
    print(f"\n=== Layer {layer} Correctness Selectivity Analysis ===")
    
    # Print selective feature counts and details
    for category in ["correct_selective", "incorrect_selective", "both_active", "both_inactive"]:
        features = results["selective_features"][category]
        print(f"\n{category.replace('_', ' ').title()} Features: {len(features)}")
        
        # Print details of top 10 features sorted by selectivity
        if features:
            if category in ["correct_selective", "both_active"]:
                # Sort by correct_rate descending
                sorted_features = sorted(features, 
                                      key=lambda x: x["selectivity"], 
                                      reverse=True)[:10]
            else:
                # Sort by incorrect_rate descending
                sorted_features = sorted(features, 
                                      key=lambda x: x["selectivity"], 
                                      reverse=True)[:10]
            
            for feat in sorted_features:
                print(f"Feature {feat['idx']}:")
                print(f"  Activation: Correct={feat['correct_rate']:.3f}, Incorrect={feat['incorrect_rate']:.3f}")
                print(f"  Selectivity: {feat['selectivity']:.3f}")
                print(f"  Specificity: For-Correct={feat['specificity_for_correct']:.3f}, For-Incorrect={feat['specificity_for_incorrect']:.3f}")
                print()

def save_results(layer: int, results: Dict):
    """Save analysis results to files."""
    output_dir = os.path.join("correctness_selective_metrics", f"layer_{layer}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save selective features data
    for category, features in results["selective_features"].items():
        if features:  # Only save if there are features in this category
            feature_data = np.array([(f["idx"], f["correct_rate"], f["incorrect_rate"]) 
                                   for f in features],
                                   dtype=[('idx', 'i4'), 
                                         ('correct_rate', 'f4'), 
                                         ('incorrect_rate', 'f4')])
            np.save(os.path.join(output_dir, f"{category}.npy"), feature_data)

def main():
    parser = argparse.ArgumentParser(description='Analyze correctness-selective features')
    parser.add_argument('layers', type=int, nargs='+', help='Layer numbers to analyze')
    parser.add_argument('--sensitivity', type=float, default=0.90,
                      help='Sensitivity threshold (default: 0.90)')
    parser.add_argument('--specificity', type=float, default=0.80,
                      help='Specificity threshold (default: 0.80)')
    args = parser.parse_args()
    
    for layer in args.layers:
        try:
            # Analyze layer
            results = analyze_layer(layer, args.sensitivity, args.specificity)
            
            # Print and save results
            print_layer_results(layer, results)
            save_results(layer, results)
            
        except FileNotFoundError as e:
            print(f"\nError analyzing layer {layer}: {e}")

if __name__ == "__main__":
    main() 