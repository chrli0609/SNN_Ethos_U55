import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path
import numpy as np

# Check if any arguments were provided
if len(sys.argv) < 2:
    print("Usage: python script.py <csv_file1> <csv_file2> ...")
    print("Please provide at least one CSV file path as an argument.")
    sys.exit(1)

# Get all filepaths from command line arguments
filepaths = [Path(arg) for arg in sys.argv[1:]]

plt.figure(figsize=(12, 8))
colors = plt.cm.tab10.colors  # Use a color cycle for different datasets
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P']  # Different markers for each dataset

all_stats = []

# Process each file
for i, filepath in enumerate(filepaths):
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Extract neuron counts from column names
        neuron_columns = [col for col in df.columns if col.startswith('neurons_')]
        if not neuron_columns:
            print(f"Warning: No neuron columns found in {filepath.name}. Skipping file.")
            continue
            
        neuron_counts = [int(re.search(r'neurons_(\d+)', col).group(1)) for col in neuron_columns]
        
        # Calculate the average value for each column
        avg_values = df[neuron_columns].mean()
        
        # Create a dictionary mapping neuron counts to their average values
        avg_dict = {count: value for count, value in zip(neuron_counts, avg_values)}
        
        # Sort by neuron count to ensure proper plotting order
        sorted_counts = sorted(neuron_counts)
        sorted_avgs = [avg_dict[count] for count in sorted_counts]
        
        # Plot this dataset with a unique color and marker
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        plt.plot(sorted_counts, sorted_avgs, 
                marker=markers[marker_idx], 
                linestyle='-', 
                linewidth=2, 
                markersize=6,
                color=colors[color_idx],
                label=f"{filepath.stem}")
        
        # Store statistics for later reporting
        all_stats.append({
            'name': filepath.stem,
            'num_configs': len(sorted_counts),
            'min_value': min(sorted_avgs),
            'min_at': sorted_counts[sorted_avgs.index(min(sorted_avgs))],
            'max_value': max(sorted_avgs),
            'max_at': sorted_counts[sorted_avgs.index(max(sorted_avgs))]
        })
        
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

# Check if any files were successfully processed
if not all_stats:
    print("No valid data to plot. Please check your CSV files.")
    sys.exit(1)

# Add grid and labels
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Average Value vs. Number of Neurons', fontsize=16)
plt.xlabel('Number of Neurons', fontsize=14)
plt.ylabel(r"Average Execution Time ($\mu$s)", fontsize=14)

# Add legend with custom positioning
plt.legend(loc='best', fontsize=12)

# Add tick marks for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set reasonable x-axis limits
all_counts = [stat['min_at'] for stat in all_stats] + [stat['max_at'] for stat in all_stats]
plt.xlim(0, max(all_counts) * 1.05)

# Save and show the plot
output_filename = "neuron_comparison.png"
if len(filepaths) == 1:
    output_filename = f"{filepaths[0].stem}.png"
    
plt.tight_layout()
plt.savefig(output_filename)
plt.show()

# Print statistics for each dataset
print(f"\n{'='*60}")
print(f"NEURAL NETWORK COMPARISON RESULTS")
print(f"{'='*60}")

for stat in all_stats:
    print(f"\nDataset: {stat['name']}")
    print(f"  Number of neuron configurations: {stat['num_configs']}")
    print(f"  Minimum average value: {stat['min_value']:.2f} at {stat['min_at']} neurons")
    print(f"  Maximum average value: {stat['max_value']:.2f} at {stat['max_at']} neurons")

# If multiple datasets were provided, show comparative analysis
if len(all_stats) > 1:
    print(f"\n{'-'*60}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'-'*60}")
    
    # Find the dataset with the lowest minimum value
    min_dataset = min(all_stats, key=lambda x: x['min_value'])
    print(f"Best performing configuration: {min_dataset['min_value']:.2f} μs with {min_dataset['min_at']} neurons in dataset '{min_dataset['name']}'")
    
    # Calculate average improvement percentages between datasets (if applicable)
    if len(all_stats) >= 2:
        print("\nPerformance comparisons:")
        for i in range(len(all_stats)):
            for j in range(i+1, len(all_stats)):
                dataset1 = all_stats[i]
                dataset2 = all_stats[j]
                
                # Find common neuron counts between datasets
                common_neurons = set([dataset1['min_at'], dataset1['max_at']]).intersection(
                                    set([dataset2['min_at'], dataset2['max_at']]))
                
                if common_neurons:
                    neuron = next(iter(common_neurons))
                    # We'd need the actual values at these points for a proper comparison
                    # This is just a placeholder for demonstration
                    print(f"  Detailed comparison between '{dataset1['name']}' and '{dataset2['name']}' available for common configurations")