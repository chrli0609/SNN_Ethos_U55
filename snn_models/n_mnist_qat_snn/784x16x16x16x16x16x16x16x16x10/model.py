
from pathlib import Path



import torch
import norse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tonic
from torch.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig, prepare_qat

from norse.torch.functional.reset import reset_subtract




# Network architecture parameters
input_size = 28*28
output_size = 10
num_hid_layers = 8
size_hid_layers = [16, 16, 16, 16, 16, 16, 16, 16]

#Network training parameters
quant_aware = True
spike_factor = 1e-5
mean_weight_factor = 1e3
epochs = 60
n_time_bins = 25

# Where to store stuff
model_dir = Path("784x16x16x16x16x16x16x16x16x10")
model_params_dir = Path("model_params")
test_patterns_dir = Path("test_patterns")

#Create dirs if they dont already exist
(model_dir / model_params_dir).mkdir(parents=True, exist_ok=True)
(model_dir / test_patterns_dir).mkdir(parents=True, exist_ok=True)



class Net(torch.nn.Module):
    def __init__(self, input_size, output_size, num_hidden, size_hidden, quant_aware, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size_hidden.insert(0, input_size)
        size_hidden.append(output_size)
        self.layers_size = size_hidden
        self.num_hidden = num_hidden
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.layers_size[i], self.layers_size[i+1]) for i in range(num_hidden+1)]
        )
        self.temp_layers = torch.nn.ModuleList(
            [norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.05, reset_method=reset_subtract), dt=1) for _ in range(num_hidden+1)]
        )

        self.quant_aware = quant_aware
        if self.quant_aware:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()


    def forward(self, inp):
        states = [None for _ in range(self.num_hidden+1)]
        out_spikes = []
        layer_spikes = []
        if self.quant_aware: inp = self.quant(inp)

        inp = inp.squeeze(2)
        for x in inp:
            x = torch.flatten(x, 1)
            for idx, layer in enumerate(self.linear_layers):
                x = layer(x)
                x, states[idx] = self.temp_layers[idx](x, states[idx])
                layer_spikes.append(x.flatten())
            out_spikes.append(x)
        
        out = torch.stack(out_spikes)
        if self.quant_aware: out = self.dequant(out)  # Dequantize output
        return out, torch.cat(layer_spikes)



def decode(x):
    x = torch.sum(x, 0)

    log_p_y = torch.nn.functional.log_softmax(x, dim=1)

    return log_p_y


class Model(torch.nn.Module):
    def __init__(self, snn, decoder):
        super(Model, self).__init__()
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x, out_spikes = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y, out_spikes




snn = Net(input_size=input_size, 
          output_size=output_size, 
          num_hidden=num_hid_layers, 
          size_hidden=size_hid_layers, 
          quant_aware=quant_aware)



if quant_aware:


    #snn.qconfig = torch.quantization.QConfig(
    #    activation=torch.quantization.default_observer,
    #    weight=torch.quantization.default_per_tensor_weight_observer  # force per-tensor weight quantization
    #)
    #torch.quantization.prepare_qat(snn, inplace=True)
    #snn.qconfig = get_default_qat_qconfig("fbgemm")  # or "qnnpack" for ARM
    #prepare_qat(snn, inplace=True)

    from torch.ao.quantization import QConfig
    from torch.ao.quantization.observer import MinMaxObserver

    
    snn.qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
    )
    torch.backends.quantized.engine = 'fbgemm'  # or whichever you use
    torch.quantization.prepare_qat(snn, inplace=True)

net = Model(snn=snn, decoder=decode)
