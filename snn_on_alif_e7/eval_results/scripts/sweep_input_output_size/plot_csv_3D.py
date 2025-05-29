import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

def load_csv_data(filepath):
    """
    Load CSV data where first row and column contain input values
    and remaining cells contain the output data.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath, header=0, index_col=0)
        
        # Remove any completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Get x and y values from column names and index
        x_values = np.array([float(col) for col in df.columns if str(col) != 'nan'])
        y_values = np.array([float(idx) for idx in df.index if str(idx) != 'nan'])
        
        # Get z values (the data matrix)
        z_values = df.iloc[:len(y_values), :len(x_values)].values.astype(float)
        
        return x_values, y_values, z_values
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def create_function_surface(func_str, x_range, y_range, resolution=50):
    """
    Create a surface from a mathematical function string.
    
    Args:
        func_str: String representation of function, e.g., "x**2 + y**2"
        x_range: tuple (min_x, max_x)
        y_range: tuple (min_y, max_y)
        resolution: number of points in each dimension
    """
    try:
        x_func = np.linspace(x_range[0], x_range[1], resolution)
        y_func = np.linspace(y_range[0], y_range[1], resolution)
        X_func, Y_func = np.meshgrid(x_func, y_func)
        
        # Create a safe environment for eval
        safe_dict = {
            "x": X_func, "y": Y_func,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            "pi": np.pi, "e": np.e,
            "abs": np.abs, "pow": np.power,
            "__builtins__": {}
        }
        
        Z_func = eval(func_str, safe_dict)
        return X_func, Y_func, Z_func
    
    except Exception as e:
        print(f"Error creating function surface: {e}")
        return None, None, None

def plot_3d_data(csv_file, function=None, title="3D Plot", save_file=None):
    """
    Create a 3D plot from CSV data with optional function overlay.
    
    Args:
        csv_file: path to CSV file
        function: mathematical function string (optional)
        title: plot title
        save_file: filename to save plot (optional)
    """
    # Load CSV data
    x_csv, y_csv, z_csv = load_csv_data(csv_file)
    
    # Create meshgrid for CSV data
    X_csv, Y_csv = np.meshgrid(x_csv, y_csv)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot CSV data as surface
    surface1 = ax.plot_surface(X_csv, Y_csv, z_csv, 
                              alpha=0.7, cmap='viridis', 
                              label='CSV Data')
    
    # Add function surface if provided
    if function:
        x_range = (np.min(x_csv), np.max(x_csv))
        y_range = (np.min(y_csv), np.max(y_csv))
        
        X_func, Y_func, Z_func = create_function_surface(function, x_range, y_range)
        
        if Z_func is not None:
            surface2 = ax.plot_surface(X_func, Y_func, Z_func, 
                                      alpha=0.5, cmap='plasma', 
                                      label=f'Function: {function}')
    
    # Customize the plot
    ax.set_xlabel('X Input')
    ax.set_ylabel('Y Input')
    ax.set_zlabel('Output Value')
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(surface1, ax=ax, shrink=0.5, aspect=5)
    
    # Add legend (workaround for 3D surface legend)
    legend_elements = ['CSV Data']
    if function:
        legend_elements.append(f'Function: {function}')
    
    # Add text box with legend
    textstr = '\n'.join([f"{i+1}. {elem}" for i, elem in enumerate(legend_elements)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
    
    # Save or show plot
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_file}")
    else:
        plt.show()

def main():
    """
    Main function to handle command line arguments and create plots.
    """
    parser = argparse.ArgumentParser(description='Plot 3D surface from CSV data with optional function overlay')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('-f', '--function', help='Mathematical function to overlay (e.g., "x**2 + y**2")')
    parser.add_argument('-t', '--title', default='3D Plot from CSV', help='Plot title')
    parser.add_argument('-s', '--save', help='Save plot to file (e.g., plot.png)')
    parser.add_argument('--show-examples', action='store_true', help='Show example usage')
    
    args = parser.parse_args()
    
    if args.show_examples:
        print("Example usage:")
        print("python script.py data.csv")
        print("python script.py data.csv -f 'x*y/100'")
        print("python script.py data.csv -f 'x**2 + y**2' -t 'Neural Network Weights'")
        print("python script.py data.csv -f 'sin(x/50) * cos(y/50) * 1000' -s 'plot.png'")
        print("\nSupported functions: sin, cos, tan, exp, log, sqrt, abs, pow")
        print("Constants: pi, e")
        print("Variables: x, y")
        return
    
    plot_3d_data(args.csv_file, args.function, args.title, args.save)

# Example usage function for interactive use
def plot_example():
    """
    Example function showing how to use the plotter programmatically.
    """
    # Example 1: Plot just CSV data
    print("Example 1: Plotting CSV data only")
    plot_3d_data('paste.txt', title='Neural Network Weight Matrix')
    
    # Example 2: Plot CSV data with function overlay
    print("Example 2: Plotting CSV data with function overlay")
    plot_3d_data('paste.txt', 
                 function='x*y/10', 
                 title='CSV Data vs Linear Function')
    
    # Example 3: Plot with more complex function
    print("Example 3: Plotting with trigonometric function")
    plot_3d_data('paste.txt', 
                 function='sin(x/100) * cos(y/100) * 500 + x*y/20', 
                 title='CSV Data vs Complex Function')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python script.py <csv_file> [options]")
        print("Use --help for more options or --show-examples for examples")
        print("\nFor interactive use, you can also call plot_example() function")
    else:
        main()