from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Adjust if needed: filename of your CSV file
filename = "../csv_results/repeating_models_power_consumption.csv"
plot_store_dir = Path("../plots/repeating/")

# Read the CSV file
# If your file is tab-separated (like your sample), use '\t' as separator
df = pd.read_csv(filename, sep=',')
print(df)
# Plotting
plt.figure(figsize=(8,6))

x = df['Num hidden layers']
y1 = df['Layer-wise Update Enabled']
#y2 = df['Layer-wise Update Disabled']
y3 = df['WFE']

plt.plot(x, y1, label='Layer-wise Update Enabled', color='blue')
#plt.plot(x, y2, label='Layer-wise Update Disabled', color='orange')
plt.plot(x, y3, label='WFE')


# Label final values
plt.text(x.iloc[-1], y1.iloc[-1], f'{y1.iloc[-1]:.2f}', ha='left', va='top', color='blue')
#plt.text(x.iloc[-1], y2.iloc[-1], f'{y2.iloc[-1]:.2f}', ha='left', va='top', color='orange')
plt.text(x.iloc[-1], y3.iloc[-1], f'{y3.iloc[-1]:.2f}', ha='left', va='top')

plt.xlabel('Number of Hidden Layers')
plt.ylabel('Power Consumption (mW)')
plt.title('Power Consumption Scalability: 6 ms inference period')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig(plot_store_dir / Path("power_consumption.png"), dpi=300, bbox_inches='tight')


# Show the plot
plt.show()

