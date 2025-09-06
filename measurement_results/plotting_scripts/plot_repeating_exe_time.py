from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs
csv_filepath = Path("../csv_results/repeating_models_exe_time.csv")
df = pd.read_csv(csv_filepath, sep=',')

csv_filepath_update_rate = Path('../csv_results/repeating_models_update_expected_num_layers_to_compute.csv')
df_update_rate = pd.read_csv(csv_filepath_update_rate, sep=',')

# Extract data
x = df['Num hidden layers']
y1 = df['Layer-wise Update Enabled']
y2 = df['Layer-wise Update Disabled']
y1_update_rate = df_update_rate['Layer Wise Update Enabled']

# Plot original lines
plt.plot(x, y1, label='Layer-wise Update Enabled', color='blue')
plt.plot(x, y2, label='Layer-wise Update Disabled', color='orange')

# Fit and plot linear trendline for y2 vs x
coeffs_y2 = np.polyfit(x, y2, 1)
y2_trend = np.polyval(coeffs_y2, x)
plt.plot(x, y2_trend, '--', color='orange', label=f'Trendlline: {coeffs_y2[0]:.2f}x+{coeffs_y2[1]:.2f}')

# Fit and plot linear trendline for y1 vs y1_update_rate
#coeffs_y1 = np.polyfit(y1_update_rate, y1, 1)
#y1_trend = np.polyval(coeffs_y1, y1_update_rate)
y1_trend = np.polyval(coeffs_y2, y1_update_rate)
plt.plot(x, y1_trend, '--', color='blue', label=f'Trendline: {coeffs_y2[0]:.2f}(num_layers_to_compute)+{coeffs_y2[1]:.2f}')

# Label final values
plt.text(x.iloc[-1] - 13, y1.iloc[-1] + 50, f'{y1.iloc[-1]:.2f}', ha='left', va='top', color='blue')
plt.text(x.iloc[-1] - 13, y2.iloc[-1] + 20, f'{y2.iloc[-1]:.2f}', ha='left', va='top', color='orange')

# Formatting
plt.xlim(x.min(), x.max() + 1)
plt.xlabel('Number of hidden layers (48 neurons in each)')
plt.ylabel(r'Execution Time ($\mu$s)')
plt.title('The Effect of Layer-wise Update on Execution Time')
plt.legend()
plt.grid(True)

# Save and show
plt.savefig(Path("..") / "plots" / Path(f"repeating/{csv_filepath.stem}.png"), dpi=300)
plt.savefig(Path("..") / "plots" / Path(f"repeating/{csv_filepath.stem}.svg"))
plt.show()

