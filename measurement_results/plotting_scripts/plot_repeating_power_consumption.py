from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjust if needed: filename of your CSV file
filename = "../csv_results/repeating_models_power_consumption.csv"
plot_store_dir = Path("../plots/repeating/")



csv_filepath_update_rate = Path('../csv_results/repeating_models_update_expected_num_layers_to_compute.csv')
df_update_rate = pd.read_csv(csv_filepath_update_rate, sep=',')
y1_update_rate = df_update_rate['Layer Wise Update Enabled']



# Read the CSV file
# If your file is tab-separated (like your sample), use '\t' as separator
df = pd.read_csv(filename, sep=',')
print(df)
# Plotting
plt.figure(figsize=(8,6))

x = df['Num hidden layers']
y1 = df['Layer-wise Update Enabled']
y2 = df['Layer-wise Update Disabled']
y3 = df['WFE']

plt.plot(x, y1, label='Layer-wise Update Enabled', color='blue')
plt.plot(x, y2, label='Layer-wise Update Disabled', color='orange')
plt.plot(x, y3, label='WFE')


# Fit and plot linear trendline for y2 vs x
coeffs_y2 = np.polyfit(x, y2, 1)
y2_trend = np.polyval(coeffs_y2, x)
plt.plot(x, y2_trend, '--', color='orange', label=f'Trendlline: {coeffs_y2[0]:.2f}x+{coeffs_y2[1]:.2f}')

# Fit and plot linear trendline for y1 vs y1_update_rate
#coeffs_y1 = np.polyfit(y1_update_rate, y1, 1)
#y1_trend = np.polyval(coeffs_y1, y1_update_rate)
y1_trend = np.polyval(coeffs_y2, y1_update_rate)
plt.plot(x, y1_trend, '--', color='blue', label=f'Trendline: {coeffs_y2[0]:.2f}num_layers_to_compute+{coeffs_y2[1]:.2f}')



# Label final values
plt.text(x.iloc[-1], y1.iloc[-1], f'{y1.iloc[-1]:.2f}', ha='left', va='top', color='blue')
plt.text(x.iloc[-1], y2.iloc[-1], f'{y2.iloc[-1]:.2f}', ha='left', va='top', color='orange')
plt.text(x.iloc[-1], y3.iloc[-1], f'{y3.iloc[-1]:.2f}', ha='left', va='top')

#6 ms inference period
plt.xlabel('Number of Hidden Layers (48 neurons in each)')
plt.ylabel('Power Consumption (mW)')
plt.title('The Effect of Layer-wise Update on Power Consumption (6 ms inference period)')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig(plot_store_dir / Path("power_consumption.png"), dpi=300, bbox_inches='tight')
plt.savefig(plot_store_dir / Path("power_consumption.svg"))


# Show the plot
plt.show()

