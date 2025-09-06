from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Where to store plots
#plot_store_dir = Path("../../../reports/status_update/measurement_data/")
plot_store_dir = Path("../plots/layer_update/")

# Model names
models = ['784x72x10', '784x64x64x10', '784x56x56x56x10', '784x48x48x48x48x10']
x = np.arange(len(models))  # the label locations

# Data
inference_time_enabled = [122.086222, 162.109333, 187.924444, 219.597333]
inference_time_disabled = [130.750222, 187.706667, 239.113778, 283.001778]

power_enabled = [98.9967, 100.287, 100.188, 100.6401]
power_disabled = [99.4323, 100.6962, 101.4123, 101.8479]

# Bar width
width = 0.35

# Plot 1: Inference Execution Time
fig1, ax1 = plt.subplots()
bars1 = ax1.bar(x - width/2, inference_time_enabled, width, label='Layer-wise Update Enabled', color='skyblue')
bars2 = ax1.bar(x + width/2, inference_time_disabled, width, label='Layer-wise Update Disabled', color='orange')

ax1.set_ylabel('Inference Time (μs)')
ax1.set_title('Inference Time Per Forward Pass by Model')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

fig1.tight_layout()
fig1.savefig(plot_store_dir / Path("inference_time.png"), dpi=300, bbox_inches='tight')
fig1.savefig(plot_store_dir / Path("inference_time.svg"))

# Plot 2: Power Consumption
fig2, ax2 = plt.subplots()
bars3 = ax2.bar(x - width/2, power_enabled, width, label='Layer-wise Update Enabled', color='skyblue')
bars4 = ax2.bar(x + width/2, power_disabled, width, label='Layer-wise Update Disabled', color='orange')

ax2.set_ylabel('Power Consumption (mW)')
ax2.set_title('Power Consumption by Model')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=15)
ax2.legend(loc="lower right")
ax2.grid(True, linestyle='--', alpha=0.5)


fig2.tight_layout()
fig2.savefig(plot_store_dir / Path("power_consumption.png"), dpi=300, bbox_inches='tight')
fig2.savefig(plot_store_dir / Path("power_consumption.svg"))



plt.show()



