from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (adjust path if needed)
csv_filepath = Path("../csv_results/repeating_models_exe_time_both.csv")
df = pd.read_csv(csv_filepath, sep=',')  # or use comma if comma-separated

print(df)
# Extract columns
x = df['Num hidden layers']
y1 = df['Layer-wise Update Enabled']
y2 = df['Layer-wise Update Disabled']

# Plot
plt.plot(x, y1, label='Enabled', color='blue')
plt.plot(x, y2, label='Disabled', color='orange')

# Label final values
plt.text(x.iloc[-1] - 13, y1.iloc[-1] + 50, f'{y1.iloc[-1]:.2f}', ha='left', va='top', color='blue')
plt.text(x.iloc[-1] - 13, y2.iloc[-1] + 20, f'{y2.iloc[-1]:.2f}', ha='left', va='top', color='orange')

# Padding on x-axis
plt.xlim(x.min(), x.max() + 1)

# Add legend and titles
plt.xlabel('Number of hidden layers')
plt.ylabel(r'Execution Time ($\mu$s)')
plt.title('The Effect of Layer-wise Update on Execution Time')
plt.legend()
plt.grid(True)

plt.savefig(Path("..") / Path("plots") / Path(csv_filepath.stem + ".png"), dpi=300)

# Show plot
plt.show()

