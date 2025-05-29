import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

def load_csv_data(filepath):
    """Load CSV data where first row and column contain input values."""
    try:
        df = pd.read_csv(filepath, header=0, index_col=0)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        x_values = np.array([float(col) for col in df.columns if str(col) != 'nan'])
        y_values = np.array([float(idx) for idx in df.index if str(idx) != 'nan'])
        z_values = df.iloc[:len(y_values), :len(x_values)].values.astype(float)
        
        return x_values, y_values, z_values
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def prepare_data_for_fitting(x_vals, y_vals, z_vals):
    """Convert 2D grid data to 1D arrays for curve fitting."""
    X, Y = np.meshgrid(x_vals, y_vals)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = z_vals.flatten()
    
    # Remove any NaN values
    mask = ~np.isnan(Z_flat)
    return X_flat[mask], Y_flat[mask], Z_flat[mask]

# Define various function models to fit
def linear_model(xy, a, b, c):
    """Linear: z = a*x + b*y + c"""
    x, y = xy
    return a*x + b*y + c

def quadratic_model(xy, a, b, c, d, e, f):
    """Quadratic: z = a*x² + b*y² + c*x*y + d*x + e*y + f"""
    x, y = xy
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

def power_model(xy, a, b, c, d):
    """Power: z = a*x^b * y^c + d"""
    x, y = xy
    return a * np.power(x, b) * np.power(y, c) + d

def exponential_model(xy, a, b, c, d):
    """Exponential: z = a*exp(b*x + c*y) + d"""
    x, y = xy
    return a * np.exp(b*x + c*y) + d

def logarithmic_model(xy, a, b, c, d):
    """Logarithmic: z = a*log(x) + b*log(y) + c*x*y + d"""
    x, y = xy
    # Avoid log of zero or negative values
    x_safe = np.maximum(x, 1e-10)
    y_safe = np.maximum(y, 1e-10)
    return a*np.log(x_safe) + b*np.log(y_safe) + c*x*y + d

def polynomial_3d_model(xy, a, b, c, d, e, f, g, h, i, j):
    """3rd degree polynomial: z = a*x³ + b*y³ + c*x²*y + d*x*y² + e*x² + f*y² + g*x*y + h*x + i*y + j"""
    x, y = xy
    return (a*x**3 + b*y**3 + c*x**2*y + d*x*y**2 + 
            e*x**2 + f*y**2 + g*x*y + h*x + i*y + j)

def rational_model(xy, a, b, c, d, e, f):
    """Rational: z = (a*x + b*y + c) / (d*x + e*y + f)"""
    x, y = xy
    denominator = d*x + e*y + f
    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    return (a*x + b*y + c) / denominator

def sinusoidal_model(xy, a, b, c, d, e, f, g):
    """Sinusoidal: z = a*sin(b*x + c) + d*cos(e*y + f) + g"""
    x, y = xy
    return a*np.sin(b*x + c) + d*np.cos(e*y + f) + g

# Dictionary of models
MODELS = {
    'linear': {
        'func': linear_model,
        'name': 'Linear (z = a*x + b*y + c)',
        'initial_guess': [1, 1, 0]
    },
    'quadratic': {
        'func': quadratic_model,
        'name': 'Quadratic (z = a*x² + b*y² + c*x*y + d*x + e*y + f)',
        'initial_guess': [0.1, 0.1, 0.1, 1, 1, 0]
    },
    'power': {
        'func': power_model,
        'name': 'Power (z = a*x^b * y^c + d)',
        'initial_guess': [1, 1, 1, 0]
    },
    'exponential': {
        'func': exponential_model,
        'name': 'Exponential (z = a*exp(b*x + c*y) + d)',
        'initial_guess': [1, 0.01, 0.01, 0]
    },
    'logarithmic': {
        'func': logarithmic_model,
        'name': 'Logarithmic (z = a*log(x) + b*log(y) + c*x*y + d)',
        'initial_guess': [1, 1, 0.01, 0]
    },
    'polynomial3d': {
        'func': polynomial_3d_model,
        'name': '3rd Degree Polynomial',
        'initial_guess': [0.001, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1, 1, 1, 0]
    },
    'rational': {
        'func': rational_model,
        'name': 'Rational (z = (a*x + b*y + c) / (d*x + e*y + f))',
        'initial_guess': [1, 1, 0, 0.01, 0.01, 1]
    },
    'sinusoidal': {
        'func': sinusoidal_model,
        'name': 'Sinusoidal (z = a*sin(b*x + c) + d*cos(e*y + f) + g)',
        'initial_guess': [1, 0.01, 0, 1, 0.01, 0, 0]
    }
}

def fit_function(X, Y, Z, model_name, max_iterations=10000):
    """Fit a specific function model to the data."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = MODELS[model_name]
    func = model['func']
    initial_guess = model['initial_guess']
    
    try:
        # Fit the function
        popt, pcov = curve_fit(func, (X, Y), Z, 
                              p0=initial_guess, 
                              maxfev=max_iterations,
                              method='lm')
        
        # Calculate predictions
        Z_pred = func((X, Y), *popt)
        
        # Calculate metrics
        r2 = r2_score(Z, Z_pred)
        rmse = np.sqrt(mean_squared_error(Z, Z_pred))
        mae = np.mean(np.abs(Z - Z_pred))
        
        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else None
        
        return {
            'parameters': popt,
            'parameter_errors': param_errors,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': Z_pred,
            'model_name': model_name,
            'success': True
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model_name': model_name
        }

def fit_all_models(csv_file, models_to_fit=None):
    """Fit all available models to the data and return results."""
    # Load data
    x_vals, y_vals, z_vals = load_csv_data(csv_file)
    X, Y, Z = prepare_data_for_fitting(x_vals, y_vals, z_vals)
    
    print(f"Loaded data: {len(X)} data points")
    print(f"X range: [{np.min(X):.1f}, {np.max(X):.1f}]")
    print(f"Y range: [{np.min(Y):.1f}, {np.max(Y):.1f}]")
    print(f"Z range: [{np.min(Z):.1f}, {np.max(Z):.1f}]")
    print("-" * 80)
    
    # Determine which models to fit
    if models_to_fit is None:
        models_to_fit = list(MODELS.keys())
    
    results = []
    
    for model_name in models_to_fit:
        print(f"Fitting {MODELS[model_name]['name']}...")
        result = fit_function(X, Y, Z, model_name)
        
        if result['success']:
            print(f"✓ R² = {result['r2_score']:.4f}, RMSE = {result['rmse']:.2f}")
            results.append(result)
        else:
            print(f"✗ Failed: {result['error']}")
        
    # Sort results by R² score
    results.sort(key=lambda x: x['r2_score'], reverse=True)
    
    return results, (x_vals, y_vals, z_vals, X, Y, Z)

def print_results(results):
    """Print detailed results of function fitting."""
    print("\n" + "="*80)
    print("FUNCTION FITTING RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {MODELS[result['model_name']]['name']}")
        print(f"   R² Score: {result['r2_score']:.6f}")
        print(f"   RMSE: {result['rmse']:.4f}")
        print(f"   MAE: {result['mae']:.4f}")
        
        print("   Parameters:")
        for j, param in enumerate(result['parameters']):
            error_str = ""
            if result['parameter_errors'] is not None:
                error_str = f" ± {result['parameter_errors'][j]:.4e}"
            print(f"     p{j}: {param:.6e}{error_str}")

def plot_best_fit(results, data_tuple, save_file=None):
    """Plot the original data and best fitting function."""
    if not results:
        print("No successful fits to plot!")
        return
    
    x_vals, y_vals, z_vals, X, Y, Z = data_tuple
    best_result = results[0]
    
    # Create prediction surface
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    func = MODELS[best_result['model_name']]['func']
    Z_pred_grid = func((X_grid, Y_grid), *best_result['parameters'])
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # Original data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_grid, Y_grid, z_vals, alpha=0.7, cmap='viridis')
    ax1.set_title('Original Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Best fit
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_grid, Y_grid, Z_pred_grid, alpha=0.7, cmap='plasma')
    ax2.set_title(f'Best Fit: {MODELS[best_result["model_name"]]["name"]}\nR² = {best_result["r2_score"]:.4f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Residuals
    ax3 = fig.add_subplot(133, projection='3d')
    residuals = z_vals - Z_pred_grid
    ax3.plot_surface(X_grid, Y_grid, residuals, alpha=0.7, cmap='coolwarm')
    ax3.set_title(f'Residuals\nRMSE = {best_result["rmse"]:.4f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Residual')
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_file}")
    else:
        plt.show()

def generate_function_code(result):
    """Generate Python code for the best fitting function."""
    model_name = result['model_name']
    params = result['parameters']
    
    if model_name == 'linear':
        return f"def fitted_function(x, y):\n    return {params[0]:.6e}*x + {params[1]:.6e}*y + {params[2]:.6e}"
    elif model_name == 'quadratic':
        return f"def fitted_function(x, y):\n    return {params[0]:.6e}*x**2 + {params[1]:.6e}*y**2 + {params[2]:.6e}*x*y + {params[3]:.6e}*x + {params[4]:.6e}*y + {params[5]:.6e}"
    elif model_name == 'power':
        return f"def fitted_function(x, y):\n    return {params[0]:.6e} * (x**{params[1]:.6e}) * (y**{params[2]:.6e}) + {params[3]:.6e}"
    # Add more cases as needed
    else:
        return f"# Parameters for {MODELS[model_name]['name']}:\n# {params}"

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Fit mathematical functions to 2D CSV data')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('-m', '--models', nargs='+', choices=list(MODELS.keys()),
                       help='Specific models to fit (default: all)')
    parser.add_argument('-p', '--plot', action='store_true', help='Show plot of best fit')
    parser.add_argument('-s', '--save', help='Save plot to file')
    parser.add_argument('--code', action='store_true', help='Generate Python code for best fit')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, model in MODELS.items():
            print(f"  {name}: {model['name']}")
        return
    
    # Fit functions
    results, data_tuple = fit_all_models(args.csv_file, args.models)
    
    if results:
        print_results(results)
        
        if args.code and results:
            print("\n" + "="*80)
            print("PYTHON CODE FOR BEST FIT FUNCTION:")
            print("="*80)
            print(generate_function_code(results[0]))
        
        if args.plot or args.save:
            plot_best_fit(results, data_tuple, args.save)
    else:
        print("No successful function fits found!")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python script.py <csv_file> [options]")
        print("Use --help for more options or --list-models to see available functions")
    else:
        main()