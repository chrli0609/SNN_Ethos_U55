from pathlib import Path


import torch
import norse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tonic
from torch.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig, prepare_qat



#from model import Model, Net, net, snn, num_hid_layers, size_hid_layers, quant_aware, decode

import argparse
import importlib

# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model folder name, e.g. model_1 or model_2")
args = parser.parse_args()

# Dynamically import the model module
model_module = importlib.import_module(f"{args.model}.model")





#from model import Net, Model, decode, net, snn, n_time_bins, num_hid_layers, size_hid_layers, epochs, quant_aware, spike_factor, mean_weight_factor

# Explicitly extract attributes
Net = model_module.Net
Model = model_module.Model
decode = model_module.decode
net = model_module.net
snn = model_module.snn
n_time_bins = model_module.n_time_bins
num_hid_layers = model_module.num_hid_layers
size_hid_layers = model_module.size_hid_layers
epochs = model_module.epochs
quant_aware = model_module.quant_aware
spike_factor = model_module.spike_factor
mean_weight_factor = model_module.mean_weight_factor




# Get max min vals from quant params (qint8)
def quant_params_2_max_min_vals(scale, zero_point):
    
    q_max_val = 127
    q_min_val = -128

    max_val = scale * (q_max_val - zero_point)
    min_val = scale * (q_min_val - zero_point)

    return max_val, min_val




# Where to store
#model_dir = Path("../../ethosu_compiler/gen_cms/nmnist_784x64x64x10/")
#model_dir = Path("./")
model_dir = Path(args.model)
model_params_dir = Path(model_module.model_params_dir)
test_patterns_dir = Path(model_module.test_patterns_dir)



params_file_basename = "fc_lif_layer_"


# Load model
#net.load_state_dict(torch.load("save_model_dict_784x32x10.pt", weights_only=False))
snn.load_state_dict(torch.load(model_dir / Path("model_state_dict.pkl")))

snn.eval()
snn_quant = torch.quantization.convert(snn, inplace=False)

net = Model(snn=snn_quant, decoder=decode)
net.eval()


print("model", net)
params_filepath = model_dir / model_params_dir / Path("params_file.py")


# Save weights and biases of each fully connected (fc) layer
for idx, layer in enumerate(snn_quant.linear_layers):

    print(layer.weight().__class__)
    print(layer.weight().dtype)
    #print("weight", layer.weight().q_per_channel_scales())
    print("weight", layer.weight().q_scale())
    print("weight", layer.weight().q_zero_point())

    weight = layer.weight().dequantize().cpu().numpy()
    bias = layer.bias().data.detach().cpu().numpy()

    weight_filepath = model_dir / model_params_dir / Path(params_file_basename+str(idx)+"_weights.npy")
    bias_filepath = model_dir / model_params_dir / Path(params_file_basename+str(idx)+"_biases.npy")


    np.save(weight_filepath, weight)
    np.save(bias_filepath, bias)

    # save quantization parameter
    #print("layer", layer.__class__)
    weights_max_val, weights_min_val = quant_params_2_max_min_vals(layer.weight().q_scale(), layer.weight().q_zero_point())
    with open(params_filepath, "a") as f:
        f.write(params_file_basename+str(idx)+"_weights_max_val = " + str(weights_max_val) + "\n")
        f.write(params_file_basename+str(idx)+"_weights_min_val = " + str(weights_min_val) + "\n")

