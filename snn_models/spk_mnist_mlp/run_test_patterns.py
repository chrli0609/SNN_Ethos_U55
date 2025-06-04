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




from model import Net



model_dir = Path("../../ethosu_compiler/gen_cms/spk_mnist_784x32x10/")

#model_pt_path = model_dir / Path("model_params/save_model_dict.pt")
model_pt_path = Path("save_model_dict_784x32x10.pt")
test_input_path = model_dir / Path("test_patterns/test_input_0.npy")
test_target_path = model_dir / Path("test_patterns/test_target_0.npy")



num_steps = 25


# Load model
net = Net()
net.load_state_dict(torch.load(model_pt_path, weights_only=False))
net.eval()
print("model:\n", net)





  


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
counter = 0
with torch.no_grad():
  net.eval()

  #for i in range(len(test_inputs)):
  for i in range(1):
    # forward pass (get output spikes and lif2 membrane potential)
    #test_spk, test_v_mem_lif2 = net(data.view(data.size(0), -1))

    # Convert to PyTorch tensor with gradients enabled
    single_sample_test_input = torch.tensor(test_inputs[i], dtype=torch.float32, requires_grad=True)
    test_spk, test_v_mem_lif2 = net(single_sample_test_input)


    #for time_step in range(num_steps):
      ##print("layer0->input\n", single_sample_test_input[time_step])



      ## Print the model behvaiour at every time step
      ##print("test_v_mem_lif2\n", test_v_mem_lif2[time_step])
      ##print("test_spk\n", test_spk[time_step])
      #write_C_style_output(single_sample_test_input[time_step], "Layer0->input")
      #write_C_style_output(test_v_mem_lif1[time_step], "Layer1->output")
      #write_C_style_output(test_v_mem_lif2[time_step], "Layer1->v_mem")
      #write_C_style_output(test_spk[time_step], "Layer1->output")

    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(0)
    #total += test_targets
    if (predicted == test_targets[i]):
      correct += 1
    #correct += (predicted == test_targets).sum().item()
    total += 1
    counter += 1



    out_neuron_sum_list.append(torch.sum(test_spk, dim=0))
    print("out_neuron_sum of sample " + str(i) + ":\t", out_neuron_sum_list[i])


    # get model prediction
    #predicted = test_spk.sum(dim=0)

    #print("prediction", predicted)
    #print("target", test_targets[i])



    #if (max_v_mem_1 < test_v_mem_lif1.max):
      #max_v_mem_1 = test_v_mem_lif1.max
    #if (min_v_mem_1 > test_v_mem_lif1.min):
      #min_v_mem_1 = test_v_mem_lif1.min

    #if (max_v_mem_2 < torch.max(test_v_mem_lif2)):
      #max_v_mem_2 = torch.max(test_v_mem_lif2)
    #if (min_v_mem_2 > torch.min(test_v_mem_lif2)):
      #min_v_mem_2 = torch.min(test_v_mem_lif2)


#print(f"lif 2 v_mem:\n\tmax: {max_v_mem_2}\n\tmin: {min_v_mem_2}")


print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

