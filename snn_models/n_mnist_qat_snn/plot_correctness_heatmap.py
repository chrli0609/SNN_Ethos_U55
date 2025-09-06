import importlib.util
import os
from pathlib import Path

class Result():
	def __init__(self, model_name, gpu, npu, target):
		self.model_name = model_name
		self.gpu = gpu
		self.gpu_acc = 0
		self.npu = npu
		self.npu_acc = 0
		self.target = target

	def compute_acc(self):
		if len(self.gpu) != len(self.target) or len(self.npu) != len(self.target):
			print(f"Error: GPU, NPU and Target should have the same length, but received {len(self.gpu)}, {len(self.npu)}, {len(self.target)}")
		num_gpu_correct = 0
		num_npu_correct = 0
		for i in range(len(self.gpu)):
			if self.gpu[i] == self.target[i]:
				num_gpu_correct+=1
			if self.npu[i] == self.target[i]:
				num_npu_correct+=1
		
		self.gpu_acc = num_gpu_correct/len(self.gpu)
		self.npu_acc = num_npu_correct/len(self.gpu)


	def print_acc(self):
		print(f"Model {self.model_name}:\tGPU Acc: {self.gpu_acc},\tNPU Acc: {self.npu_acc}")
			





model_list = []

model_names_list = ["784x72x10", "784x64x64x10", "784x56x56x56x10", "784x48x48x48x48x10"]

for model_name in model_names_list:

	#module = importlib.import_module(f"{model_name}/pred_vs_target")
	module_path = os.path.join(model_name, "pred_vs_target.py")
	module_name = f"{model_name}_pred_vs_target"  # give it a safe name

	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

    # now you can use functions/variables from the module
	results = Result(
			model_name=model_name,
			gpu=module.gpu,
			npu=module.npu,
			target=module.target
		)
	results.compute_acc()
	results.print_acc()
	model_list.append(results)


#model_list = [
#    Result(
#        model_name="784x72x10",
#        gpu = [6, 5, 1, 0, 8, 6, 0, 7, 1, 6, 4, 2, 3, 6, 5, 5, 8, 4, 8, 4, 6, 8, 3, 4, 5, 5, 2, 2, 8, 3, 4, 5, 4, 0, 4, 5, 4, 3, 3, 8, 3, 2, 6, 8, 3],
#        gpu_acc=91.111,
#        npu = [6, 5, 1, 0, 3, 6, 0, 7, 1, 6, 4, 2, 3, 6, 5, 5, 8, 4, 5, 4, 6, 8, 3, 4, 5, 5, 2, 2, 8, 3, 4, 5, 4, 0, 4, 5, 4, 3, 3, 2, 3, 2, 6, 5, 2],
#        npu_acc=84.4444,
#        target=[6, 5, 1, 0, 8, 6, 0, 7, 1, 0, 4, 2, 3, 6, 5, 5, 8, 4, 8, 4, 6, 8, 3, 4, 3, 5, 2, 2, 8, 8, 4, 5, 4, 0, 4, 5, 4, 3, 3, 8, 3, 2, 6, 8, 2]
#    ),
#
#    Result(
#        model_name="784x64x64x10",
#        gpu = [7, 2, 4, 1, 4, 3, 5, 6, 7, 2, 7, 2, 3, 5, 9, 9, 9, 1, 4, 9, 0, 2, 2, 4, 9, 7, 0, 9, 7, 7, 2, 8, 9, 7, 7, 8, 2, 1, 5, 9, 0, 0, 2, 2, 9],
#        gpu_acc=95.555555,
#        npu = [7, 2, 4, 1, 4, 3, 5, 6, 7, 3, 7, 2, 3, 5, 9, 9, 9, 1, 4, 4, 0, 4, 2, 4, 9, 3, 0, 9, 7, 7, 2, 8, 9, 7, 7, 8, 2, 1, 5, 9, 0, 0, 2, 2, 9],
#        npu_acc=93.3333,
#        target=[7, 2, 4, 1, 4, 3, 5, 6, 7, 5, 7, 2, 3, 5, 9, 9, 9, 1, 4, 4, 0, 2, 2, 4, 9, 7, 0, 9, 7, 7, 2, 8, 9, 7, 7, 8, 2, 1, 5, 9, 0, 0, 2, 2, 9],
#    ),
#
#    Result(
#        model_name="784x56x56x56x10",
#        gpu = [6, 5, 1, 0, 3, 6, 0, 7, 1, 6, 4, 2, 3, 6, 5, 5, 8, 4, 8, 4, 6, 8, 3, 4, 3, 5, 2, 2, 8, 3, 4, 5, 4, 0, 4, 5, 4, 3, 3, 8, 3, 2, 6, 8, 8],
#        gpu_acc=91.1111,
#        npu = [6, 5, 1, 0, 3, 6, 0, 7, 1, 6, 4, 2, 3, 6, 5, 5, 8, 4, 8, 4, 6, 8, 3, 4, 3, 5, 2, 2, 8, 3, 4, 5, 4, 0, 4, 5, 4, 3, 3, 8, 3, 2, 6, 8, 3],
#        npu_acc=91.11111,
#        target=[6, 5, 1, 0, 8, 6, 0, 7, 1, 0, 4, 2, 3, 6, 5, 5, 8, 4, 8, 4, 6, 8, 3, 4, 3, 5, 2, 2, 8, 8, 4, 5, 4, 0, 4, 5, 4, 3, 3, 8, 3, 2, 6, 8, 2],
#    ),
#
#
#    Result(
#        model_name="784x48x48x48x48x10",
#        gpu = [6, 3, 1, 0, 2, 1, 1, 8, 6, 8, 9, 4, 1, 2, 3, 7, 6, 5, 9, 9, 3, 8, 7, 2, 8, 9, 2, 2, 9, 9, 7, 6, 3, 6, 2, 9, 3, 7, 9, 0, 0, 0, 8, 6, 5],
#        gpu_acc=84.444444,
#        npu = [6, 3, 1, 0, 2, 7, 1, 8, 6, 8, 9, 4, 1, 6, 0, 7, 6, 5, 7, 9, 3, 8, 7, 2, 8, 9, 2, 2, 9, 7, 7, 0, 3, 6, 2, 1, 3, 7, 9, 0, 0, 0, 5, 6, 0],
#        npu_acc=82.22222,
#        target=[6, 3, 1, 0, 2, 9, 1, 5, 6, 8, 9, 4, 1, 6, 3, 7, 6, 5, 9, 9, 3, 8, 7, 2, 8, 9, 2, 2, 9, 9, 7, 5, 3, 6, 2, 8, 3, 7, 9, 0, 0, 0, 5, 6, 6]
#    )
#
#
#]





import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch



def analyze_results(result: Result):
		gpu_correct = np.array(result.gpu) == np.array(result.target)
		npu_correct = np.array(result.npu) == np.array(result.target)

		both_correct = np.sum(gpu_correct & npu_correct)
		both_wrong = np.sum(~gpu_correct & ~npu_correct)
		gpu_only = np.sum(gpu_correct & ~npu_correct)
		npu_only = np.sum(~gpu_correct & npu_correct)

		return gpu_correct, npu_correct, both_correct, both_wrong, gpu_only, npu_only




# Example with your model_list
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()
fig.suptitle(f"Correctness Heatmap GPU vs NPU", fontsize=14)


for idx, res in enumerate(model_list):
	gpu_correct, npu_correct, _, _, _, _ = analyze_results(res)
	heatmap = np.vstack([gpu_correct, npu_correct])

	# Two-color colormap: 0=red (wrong), 1=green (correct)
	im = axes[idx].imshow(heatmap, cmap=ListedColormap(['red', 'green']), aspect='auto')
	axes[idx].set_yticks([0, 1])
	axes[idx].set_yticklabels(["GPU", "NPU"])
	#axes[idx].set_xticks([])
	axes[idx].set_title(f"Model {res.model_name}")

	# Set x-axis to sample indices
	#axes[idx].set_xticks(np.arange(heatmap.shape[1]))
	#axes[idx].set_xticklabels(np.arange(1, heatmap.shape[1]+1), rotation=90, fontsize=8)
	axes[idx].set_xlabel("Sample Number")

# Manual legend for two colors
legend_elements = [Patch(facecolor='green', label='Correct'),
                   Patch(facecolor='red', label='Wrong')]
fig.legend(handles=legend_elements, loc='upper right', title="Prediction")

plt.tight_layout()

plt.savefig("correctness_heatmap.svg")
plt.show()




