from pathlib import Path

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




# Where to store
#model_param_dir = Path("../../ethosu_compiler/gen_cms/spk_mnist_784x32x10/model_params/")
model_param_dir = Path("../n_mnist_qat_snn/model_params/")


# Load model
net = Net()
#net.load_state_dict(torch.load("save_model_dict_784x32x10.pt", weights_only=False))
net.load_state_dict(torch.load("../n_mnist_qat_snn/model_state_dict.pkl", weights_only=False))
net.eval()

print("model", net)



# Extract weights and Biases (my own addition)

def print_max_min(tensor):
  max_val = torch.max(tensor)
  min_val = torch.min(tensor)
  print("Max:", max_val.item())
  print("Min:", min_val.item())


torch.set_printoptions(threshold=float('inf'))
print(net)
counter = 1
for layer in net.children():
    print("layer\n", layer)
    if isinstance(layer, nn.Linear):
        #print(layer.state_dict()['weight'])
        print("fc"+str(counter)+" weights")

        tmp_weight = layer.state_dict()['weight'].cpu()        
        weights_filepath = model_param_dir / Path("fc"+str(counter)+"_weights.npy")

        np.save(weights_filepath, tmp_weight.detach().numpy())
        print_max_min(tmp_weight)
        print("Stored to", weights_filepath)
        #print(tmp_weight)

        comp = np.load(weights_filepath)
        for i in range(comp.shape[0]):
            for j in range(comp.shape[1]):
                if (comp[i][j] != tmp_weight[i][j]):
                    print("Error: Weights mismatch for fc"+str(counter))
                    print(f"\tfrom script: {tmp_weight.shape}\n\tfrom stored in ethosu: {comp.shape}")
                    exit()


        #print(layer.state_dict()['bias'])
        print("fc"+str(counter)+" bias")

        tmp_bias = layer.state_dict()['bias'].cpu()        
        bias_filepath = model_param_dir / Path("fc"+str(counter)+"_bias.npy")

        np.save(bias_filepath, tmp_bias.detach().numpy())
        print_max_min(tmp_bias)
        #print(tmp_bias)
        print("Stored to", bias_filepath)
        #print(tmp_weight)

        comp = np.load(bias_filepath)
        for i in range(comp.shape[0]):
            if (comp[i] != tmp_bias[i]):
                print("Error: Bias mismatch for fc"+str(counter))
                exit()

        counter += 1
