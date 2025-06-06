from pathlib import Path


# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools




from model import Net, num_inputs, num_hidden, num_outputs 



#model_dir = Path("../../ethosu_compiler/gen_cms/spk_mnist_784x32x10/")
model_dir = Path(".")

#model_pt_path = model_dir / Path("model_params/save_model_dict.pt")
model_pt_path = Path("save_model_dict_784x32x10.pt")
test_input_path = model_dir / Path("test_patterns/test_input_0.npy")
test_target_path = model_dir / Path("test_patterns/test_target_0.npy")





# Load model
net = Net()
net.load_state_dict(torch.load(model_pt_path, weights_only=False))
net.eval()
print("model:\n", net)





from store_model_params import size_padding_list
  


# Get behaviour of first ten samples
torch.set_printoptions(threshold=float('inf'))

test_inputs_untransposed = np.load(test_input_path)
test_inputs = np.transpose(test_inputs_untransposed, (1, 0, 2))  # Rearrange axes


test_targets = np.load(test_target_path)

print("test_inputs", test_inputs.shape)
print("test_targets", test_targets)


max_v_mem_1 = 0
min_v_mem_1 = 0

max_v_mem_2 = 0
min_v_mem_2 = 0



total = 0
correct = 0


out_neuron_sum_list = []
with torch.no_grad():
  net.eval()

  #for i in range(len(test_inputs)):
  #for i in range(45):
  for i in range(1):
    # forward pass (get output spikes and lif2 membrane potential)
    #test_spk, test_v_mem_lif2 = net(data.view(data.size(0), -1))

    # Convert to PyTorch tensor with gradients enabled
    single_sample_test_input = torch.tensor(test_inputs[i], dtype=torch.float32, requires_grad=True)
    test_spk, test_v_mem_lif2 = net(single_sample_test_input, size_padding_list)


    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(0)
    if (predicted == test_targets[i]):
      correct += 1
    total += 1



    out_neuron_sum_list.append(torch.sum(test_spk, dim=0))
    print("out_neuron_sum of sample " + str(i) + ":\t", out_neuron_sum_list[i])




print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

