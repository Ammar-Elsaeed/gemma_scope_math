import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Tuple, Dict
from compare_activations import load_activation_frequencies, get_available_layers, get_active_feature_counts

def get_selective_feature_counts(freqs1: np.ndarray, freqs2: np.ndarray, high_threshold: float = 0.7, low_threshold: float = 0.2) -> Tuple[int, int]:
    """Get counts of features that are selective for each dataset.
    
    A feature is considered selective for dataset 1 if:
    - Its frequency in dataset 1 >= high_threshold
    - Its frequency in dataset 2 <= low_threshold
    And vice versa for dataset 2.
    
    Args:
        freqs1: Activation frequencies for first dataset
        freqs2: Activation frequencies for second dataset
        high_threshold: Threshold for high activation (default: 0.7)
        low_threshold: Threshold for low activation (default: 0.2)
        
    Returns:
        Tuple of (count_selective_ds1, count_selective_ds2)
    """
    selective_ds1 = (freqs1 >= high_threshold) & (freqs2 <= low_threshold)
    selective_ds2 = (freqs2 >= high_threshold) & (freqs1 <= low_threshold)
    
    return np.sum(selective_ds1), np.sum(selective_ds2)

def plot_activation_trends(
    dataset1: str,
    dataset2: str,
    layers: Optional[List[int]] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    sensitivity: float = 0.25,
    selective_high_threshold: float = 0.7,
    selective_low_threshold: float = 0.2
):
    """Create a line plot showing how feature counts change across layers.
    
    Args:
        dataset1: Name of first dataset
        dataset2: Name of second dataset
        layers: List of layers to plot. If None, plots all available layers.
        title: Custom title for the plot. If None, generates automatically.
        output_path: Path to save the plot. If None, displays instead.
        sensitivity: Threshold for considering a feature active
        selective_high_threshold: Threshold for high activation in selective features
        selective_low_threshold: Threshold for low activation in selective features
    """
    # Get available layers if not specified
    if layers is None:
        layers1 = get_available_layers(dataset1)
        layers2 = get_available_layers(dataset2)
        layers = sorted(list(set(layers1) & set(layers2)))
        
    if not layers:
        raise ValueError(f"No common layers found between {dataset1} and {dataset2}")
    
    # Initialize data storage
    valid_layers = []
    counts_ds1 = []
    counts_ds2 = []
    counts_intersection = []
    counts_selective_ds1 = []
    counts_selective_ds2 = []
    
    # Collect data for each layer
    for layer in layers:
        try:
            freqs1 = load_activation_frequencies(dataset1, layer)
            freqs2 = load_activation_frequencies(dataset2, layer)
            
            count1, count2, count_intersect = get_active_feature_counts(freqs1, freqs2, sensitivity)
            selective_count1, selective_count2 = get_selective_feature_counts(
                freqs1, freqs2, selective_high_threshold, selective_low_threshold)
            
            valid_layers.append(layer)
            counts_ds1.append(count1)
            counts_ds2.append(count2)
            counts_intersection.append(count_intersect)
            counts_selective_ds1.append(selective_count1)
            counts_selective_ds2.append(selective_count2)
            
        except FileNotFoundError as e:
            print(f"Skipping layer {layer}: {str(e)}")
            continue
    
    if not valid_layers:
        raise ValueError(f"No valid data found for any layers between {dataset1} and {dataset2}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot total feature counts
    plt.plot(valid_layers, counts_ds1, 'b-', marker='o', label=f'{dataset1} features')
    plt.plot(valid_layers, counts_ds2, 'r-', marker='s', label=f'{dataset2} features')
    plt.plot(valid_layers, counts_intersection, 'g-', marker='^', label='Common features')
    
    # Plot selective feature counts with dashed lines
    plt.plot(valid_layers, counts_selective_ds1, 'b--', marker='o', 
            label=f'{dataset1} selective (≥{selective_high_threshold:.1f}, ≤{selective_low_threshold:.1f})')
    plt.plot(valid_layers, counts_selective_ds2, 'r--', marker='s',
            label=f'{dataset2} selective (≥{selective_high_threshold:.1f}, ≤{selective_low_threshold:.1f})')
    
    # Fill between total and intersection
    plt.fill_between(valid_layers, counts_ds1, counts_intersection, alpha=0.2, color='blue')
    plt.fill_between(valid_layers, counts_ds2, counts_intersection, alpha=0.2, color='red')
    
    # Set labels and title
    plt.xlabel('Layer')
    plt.ylabel('Number of Features')
    if title is None:
        title = f'Feature Counts Across Layers\n{dataset1} vs {dataset2}'
    plt.title(title)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits
    plt.xlim(-0.5, 24.5)  # Show full range even if some layers are missing
    plt.ylim(0, 200)  # Fixed y-axis range
    
    # Set x-ticks to show all layer numbers
    plt.xticks(range(0, 26))  # Show all layer numbers 0-24
    
    # Add legend with thresholds
    plt.legend(title=f'Features (active ≥ {sensitivity:.2f})', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_all_trends(
    dataset_pairs: List[Tuple[str, str]],
    layers: List[int],
    output_dir: str = "figures/activation_trends",
    sensitivity: float = 0.25,
    selective_high_threshold: float = 0.7,
    selective_low_threshold: float = 0.2
):
    """Generate trend plots for all dataset pairs."""
    os.makedirs(output_dir, exist_ok=True)
    
    for ds1, ds2 in dataset_pairs:
        output_path = os.path.join(output_dir, f"{ds1}_vs_{ds2}_trends.png")
        plot_activation_trends(
            ds1,
            ds2,
            layers=layers,
            title=f"Feature Counts Across Layers\n{ds1} vs {ds2}",
            output_path=output_path,
            sensitivity=sensitivity,
            selective_high_threshold=selective_high_threshold,
            selective_low_threshold=selective_low_threshold
        )
        print(f"Saved trend plot to {output_path}")

if __name__ == "__main__":
    # Define dataset pairs to compare
    dataset_pairs = [
        ("addition", "random_addition"),
        ("subtraction", "random_subtraction"),
        ("addition", "subtraction"),
        ("random_addition", "random_subtraction"),
        ("correct_arithmetic", "incorrect_arithmetic")
    ]
    
    # Define layer range - all layers from 0 to 24
    layers = list(range(0, 26))
    
    # Generate all trend plots
    plot_all_trends(
        dataset_pairs, 
        layers,
        selective_high_threshold=0.7,
        selective_low_threshold=0.2
    ) 