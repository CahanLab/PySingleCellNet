import matplotlib.pyplot as plt
import numpy as np

def get_unique_colors(n_colors):
    """
    Generate a list of unique colors from the Tab20, Tab20b, and Tab20c colormaps.

    Parameters:
    - n_colors: The number of unique colors needed.

    Returns:
    - A list of unique colors.
    """
    # Get the colormaps
    tab20 = plt.get_cmap('tab20').colors
    tab20b = plt.get_cmap('tab20b').colors
    tab20c = plt.get_cmap('tab20c').colors
    
    # Combine the colors from the colormaps
    combined_colors = np.vstack([tab20, tab20b, tab20c])
    
    # Check if the requested number of colors exceeds the available unique colors
    if n_colors > len(combined_colors):
        raise ValueError(f"Requested number of colors ({n_colors}) exceeds the available unique colors ({len(combined_colors)}).")
    
    # Select the required number of unique colors
    selected_colors = combined_colors[:n_colors]
    return selected_colors

# where is this called?, In ontogeny_graph in dot. test and remove if not needed
def convert_color(color_array): 
    return tuple(color_array)