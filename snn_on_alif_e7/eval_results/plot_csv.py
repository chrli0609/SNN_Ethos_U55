import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path

def main():
    # Get the filepath from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_csv.py <csv_file_path>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)
    
    # Extract the session directory name from the CSV path
    # Expected structure: csv/session_name/file.csv
    if len(filepath.parts) >= 2 and filepath.parts[-2] != 'csv':
        session_name = filepath.parts[-2]  # Get the parent directory name
    else:
        # Fallback: use the CSV filename without extension
        session_name = filepath.stem
    
    # Create corresponding plot directory structure: plot/session_name/
    plot_session_dir = Path("plot") / session_name
    plot_session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {filepath}")
    print(f"Output directory: {plot_session_dir}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Extract neuron columns and units
    neuron_columns = [col for col in df.columns if col.startswith('neurons_')]
    
    if not neuron_columns:
        print("No neuron columns found in the CSV file")
        sys.exit(1)
    
    # Extract the unit from column names
    unit = extract_unit_from_columns(neuron_columns)
    
    # Extract neuron counts from column names
    neuron_counts = extract_neuron_counts(neuron_columns)
    
    if not neuron_counts:
        print("No valid neuron counts found in column names")
        sys.exit(1)
    
    # Calculate average values and create plot
    create_plot(df, neuron_columns, neuron_counts, unit, filepath, plot_session_dir)
    
    print(f"Plot saved to: {plot_session_dir}")

def extract_unit_from_columns(neuron_columns):
    """Extract unit information from column names"""
    # Try to extract unit from parentheses like (ms) or (cycles)
    unit_match = re.search(r'neurons_\d+_\((\w+)\)', neuron_columns[0])
    if unit_match:
        return unit_match.group(1)
    
    # Try alternative format without parentheses
    alt_match = re.search(r'neurons_\d+_(\w+)', neuron_columns[0])
    if alt_match:
        return alt_match.group(1)
    
    # Default fallback
    return "units"

def extract_neuron_counts(neuron_columns):
    """Extract neuron counts from column names"""
    neuron_counts = []
    for col in neuron_columns:
        match = re.search(r'neurons_(\d+)', col)
        if match:
            neuron_counts.append(int(match.group(1)))
    return neuron_counts

def create_plot(df, neuron_columns, neuron_counts, unit, original_filepath, output_dir):
    """Create and save the plot"""
    # Calculate the average value for each column
    avg_values = df[neuron_columns].mean()
    
    # Create a dictionary mapping neuron counts to their average values
    avg_dict = {count: value for count, value in zip(neuron_counts, avg_values)}
    
    # Sort by neuron count to ensure proper plotting order
    sorted_counts = sorted(neuron_counts)
    sorted_avgs = [avg_dict[count] for count in sorted_counts]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_counts, sorted_avgs, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create descriptive title
    data_type = original_filepath.stem
    plt.title(f'Average {data_type.replace("_", " ").title()} vs. Number of Output Neurons\n{original_filepath.parents[0].name}', 
              fontsize=16)
    plt.xlabel('Number of Output Neurons', fontsize=14)
    
    # Set appropriate y-axis label based on unit
    ylabel = get_ylabel_for_unit(unit)
    plt.ylabel(ylabel, fontsize=14)
    
    # Add tick marks for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Ensure x-axis starts from 0 to properly visualize the trend
    plt.xlim(0, max(sorted_counts) * 1.05)
    
    # Save the plot with the same filename as the CSV but as PNG
    plot_filename = original_filepath.stem + '.png'
    plot_path = output_dir / plot_filename
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print_statistics(sorted_counts, sorted_avgs, unit, data_type)

def get_ylabel_for_unit(unit):
    """Generate appropriate y-axis label based on unit"""
    unit_labels = {
        'ms': f"Average Execution Time ({unit})",
        #'cycles': f"Average Processing Cycles ({unit})",
        'cycles': f"Average Cycles",
        'operations': f"Average number of operations",
        'operations': f"Average number of events",
        'kb': f"Average Memory Usage ({unit.upper()})",
        'mb': f"Average Memory Usage ({unit.upper()})", 
        'gb': f"Average Memory Usage ({unit.upper()})",
        'us': f"Average Execution Time (μs)",
        'ns': f"Average Execution Time (ns)"
    }
    
    return unit_labels.get(unit.lower(), f"Average Value ({unit})")

def print_statistics(sorted_counts, sorted_avgs, unit, data_type):
    """Print summary statistics"""
    print(f"\n=== {data_type.replace('_', ' ').title()} Statistics ===")
    print(f"Data shows trend across {len(sorted_counts)} different neuron configurations.")
    print(f"Neuron counts: {sorted_counts}")
    
    min_idx = sorted_avgs.index(min(sorted_avgs))
    max_idx = sorted_avgs.index(max(sorted_avgs))
    
    print(f"Minimum average: {min(sorted_avgs):.2f} {unit} at {sorted_counts[min_idx]} neurons")
    print(f"Maximum average: {max(sorted_avgs):.2f} {unit} at {sorted_counts[max_idx]} neurons")
    
    # Calculate trend (simple linear approximation)
    if len(sorted_counts) > 1:
        trend = (sorted_avgs[-1] - sorted_avgs[0]) / (sorted_counts[-1] - sorted_counts[0])
        print(f"Overall trend: {trend:.4f} {unit} per neuron")

if __name__ == "__main__":
    main()