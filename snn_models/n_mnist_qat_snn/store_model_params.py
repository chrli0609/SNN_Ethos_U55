from pathlib import Path


import torch
import norse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tonic
from torch.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig, prepare_qat



from model import Model, Net, num_hid_layers, size_hid_layers, quant_aware, decode




# Where to store
model_dir = Path("../../ethosu_compiler/gen_cms/nmnist_784x64x64x10/")
model_params_dir = Path("model_params")
test_patterns_dir = Path("test_patterns")




# Load model
snn = Net(input_size=784, 
          output_size=10, 
          num_hidden=num_hid_layers, 
          size_hidden=size_hid_layers, 
          quant_aware=quant_aware)
if quant_aware:
    snn.qconfig = get_default_qat_qconfig("fbgemm")  # or "qnnpack" for ARM
    prepare_qat(snn, inplace=True)
#net.load_state_dict(torch.load("save_model_dict_784x32x10.pt", weights_only=False))
snn.load_state_dict(torch.load("model_state_dict.pkl"))


net = Model(snn=snn, decoder=decode)
net.eval()

print("model", net)

# Save weights and biases of each fully connected (fc) layer
for idx, layer in enumerate(snn.linear_layers):
    weight = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()

    weight_filepath = model_dir / model_params_dir / Path("fc"+str(idx)+"_weights.npy")
    bias_filepath = model_dir / model_params_dir / Path("fc"+str(idx)+"_biases.npy")
    np.save(weight_filepath, weight)
    np.save(bias_filepath, bias)
