import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Tuple
import colorsys

def load_activation_frequencies(dataset: str, layer: int) -> np.ndarray:
    """Load activation frequencies for a specific dataset and layer."""
    path = os.path.join("activ_freq", dataset, f"layer_{layer}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No activation frequencies found at {path}")
    return np.load(path)

def get_available_layers(dataset: str) -> List[int]:
    """Get list of available layers for a dataset."""
    activ_dir = os.path.join("activ_freq", dataset)
    if not os.path.exists(activ_dir):
        return []
    
    layers = []
    for file in os.listdir(activ_dir):
        if file.startswith("layer_") and file.endswith(".npy"):
            layer_num = int(file.split("_")[1].split(".")[0])
            layers.append(layer_num)
    return sorted(layers)

def create_distinct_colors(num_colors: int) -> List[str]:
    """Create a list of visually distinct colors for different layers.
    
    Args:
        num_colors: Number of colors needed
        
    Returns:
        List of color hex codes
    """
    # Use a mix of qualitative colors that are visually distinct
    base_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf',  # cyan
    ]
    
    # If we need more colors than base colors, create variations
    if num_colors <= len(base_colors):
        return base_colors[:num_colors]
    
    # For additional colors, cycle through base colors with different lightness
    colors = []
    for i in range(num_colors):
        base_color = base_colors[i % len(base_colors)]
        # Convert hex to RGB
        rgb = tuple(int(base_color[1:][i:i+2], 16) / 255 for i in (0, 2, 4))
        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        # Vary value (brightness) based on cycle
        cycle = i // len(base_colors)
        v = max(0.3, min(1.0, v * (1.0 - cycle * 0.2)))
        # Convert back to RGB
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    return colors

def get_active_feature_counts(freqs1: np.ndarray, freqs2: np.ndarray, sensitivity: float = 0.25) -> Tuple[int, int, int]:
    """Get counts of active features in each dataset and their intersection.
    
    Args:
        freqs1: Activation frequencies for first dataset
        freqs2: Activation frequencies for second dataset
        sensitivity: Threshold for considering a feature active
        
    Returns:
        Tuple of (count_ds1, count_ds2, count_intersection)
    """
    active1 = freqs1 >= sensitivity
    active2 = freqs2 >= sensitivity
    
    count_ds1 = np.sum(active1)
    count_ds2 = np.sum(active2)
    count_intersection = np.sum(active1 & active2)
    
    return count_ds1, count_ds2, count_intersection

def plot_activation_comparison(
    dataset1: str,
    dataset2: str,
    layers: Optional[List[int]] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    sensitivity: float = 0.25
):
    """Create a scatter plot comparing activation frequencies between two datasets.
    
    Args:
        dataset1: Name of first dataset
        dataset2: Name of second dataset
        layers: List of layers to plot. If None, plots all available layers.
        title: Custom title for the plot. If None, generates automatically.
        output_path: Path to save the plot. If None, displays instead.
        sensitivity: Threshold for considering a feature active
    """
    # Get available layers if not specified
    if layers is None:
        layers1 = get_available_layers(dataset1)
        layers2 = get_available_layers(dataset2)
        layers = sorted(list(set(layers1) & set(layers2)))
        
    if not layers:
        raise ValueError(f"No common layers found between {dataset1} and {dataset2}")
    
    # Create distinct colors
    colors = create_distinct_colors(len(layers))
    
    # Create plot
    plt.figure(figsize=(12, 10))  # Increased width for legend
    
    # Plot each layer
    for layer, color in zip(layers, colors):
        try:
            freqs1 = load_activation_frequencies(dataset1, layer)
            freqs2 = load_activation_frequencies(dataset2, layer)
            
            # Get feature counts
            count1, count2, count_intersect = get_active_feature_counts(freqs1, freqs2, sensitivity)
            
            # Create legend label with feature counts
            label = (f'Layer {layer} '
                    f'({count1:,d}, {count2:,d}, {count_intersect:,d})')
            
            plt.scatter(freqs1, freqs2, color=color, alpha=0.6,  # Slightly reduced alpha for better visibility
                       label=label, s=30)  # Slightly larger points
            
        except FileNotFoundError as e:
            print(f"Skipping layer {layer}: {str(e)}")
            continue
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')  # Reduced alpha for diagonal line
    
    # Set labels and title
    plt.xlabel(f'{dataset1} Activation Frequency')
    plt.ylabel(f'{dataset2} Activation Frequency')
    if title is None:
        title = f'Activation Frequency Comparison:\n{dataset1} vs {dataset2}'
    plt.title(title)
    
    # Add legend title explaining the counts
    legend_title = (f'Layer (Features with freq ≥ {sensitivity:.2f})\n'
                   f'(#feats_{dataset1}, #feats_{dataset2}, #intersection)')
    
    # Set axis limits
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Add legend with title
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=legend_title)
    
    # Make plot square
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add grid with light gray color
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Define dataset pairs to compare
    dataset_pairs = [
        ("addition", "random_addition"),
        ("subtraction", "random_subtraction"),
        ("addition", "subtraction"),
        ("random_addition", "random_subtraction"),
        ("correct_arithmetic", "incorrect_arithmetic")
    ]
    
    # Create output directory
    output_dir = "figures/activation_rate_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define layer range

    ranges = [list(range(0, 26)),   list(range(0, 9)), list(range(9, 17)), list(range(17, 26))]
    for layers in ranges:
        # Generate plots for each pair
        for ds1, ds2 in dataset_pairs:
            output_path = os.path.join(output_dir, f"layer_{layers[0]}-{layers[-1]}/{ds1}_vs_{ds2}_layer_{layers[0]}-{layers[-1]}.png")
            plot_activation_comparison(
                ds1,
                ds2,
                layers=layers,
                title=f"{ds1} vs {ds2} Activation Frequencies",
                output_path=output_path,
                sensitivity=0.25  # Consider features active if frequency >= 25%
            )
            print(f"Saved plot to {output_path}") 