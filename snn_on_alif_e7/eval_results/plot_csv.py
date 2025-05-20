import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path

# Ensure plot directory exists
plot_dir = Path("plot")
plot_dir.mkdir(exist_ok=True)

# Get the filepath from command line arguments
filepath = Path(sys.argv[1])

# Read the CSV file
df = pd.read_csv(filepath)

# Extract neuron counts and units from column names
neuron_columns = [col for col in df.columns if col.startswith('neurons_')]

# Extract the unit from column names if present
unit_match = re.search(r'neurons_\d+_\((\w+)\)', neuron_columns[0]) if neuron_columns else None
if unit_match:
    unit = unit_match.group(1)  # Extract unit from parentheses like (ms)
else:
    # Try alternative format without parentheses
    alt_match = re.search(r'neurons_\d+_(\w+)', neuron_columns[0]) if neuron_columns else None
    unit = alt_match.group(1) if alt_match else "μs"  # Default to microseconds if no unit found

# Extract just the neuron counts from column names
neuron_counts = []
for col in neuron_columns:
    match = re.search(r'neurons_(\d+)', col)
    if match:
        neuron_counts.append(int(match.group(1)))

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
plt.title('Average Value vs. Number of Neurons\n'+filepath.stem, fontsize=16)
plt.xlabel('Number of Neurons', fontsize=14)

# Use the extracted unit for the y-axis label
if unit.lower() == 'ms':
    plt.ylabel(f"Average Execution Time ({unit})", fontsize=14)
else:
    plt.ylabel(f"Average Execution Time ({unit})", fontsize=14)

# Add tick marks for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Ensure x-axis starts from 0 to properly visualize the trend
plt.xlim(0, max(sorted_counts) * 1.05)

# Save and show the plot
plt.tight_layout()
plt.savefig(plot_dir / Path(filepath.stem+'.png'))
plt.show()

# Print statistics
print(f"The data shows a trend across {len(sorted_counts)} different neuron configurations.")
print(f"Minimum average value: {min(sorted_avgs):.2f} {unit} at {sorted_counts[sorted_avgs.index(min(sorted_avgs))]} neurons")
print(f"Maximum average value: {max(sorted_avgs):.2f} {unit} at {sorted_counts[sorted_avgs.index(max(sorted_avgs))]} neurons")