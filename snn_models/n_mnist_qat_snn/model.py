
import torch
import norse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tonic
from torch.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig, prepare_qat
from norse.torch.functional.reset import reset_subtract


# Network architecture parameters
num_hid_layers = 2
size_hid_layers = [64, 64]

#Network training parameters
quant_aware = True
spike_factor = 1e-5
mean_weight_factor = 1e3
epochs = 8
n_time_bins = 25

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

        #print("pre squeeze:", inp.size())
        inp = inp.squeeze(2)
        #print("post squeeze:", inp.size())
        for x in inp:
            x = torch.flatten(x, 1)
            for idx, layer in enumerate(self.linear_layers):
                x = layer(x)
                x, states[idx] = self.temp_layers[idx](x, states[idx])
                layer_spikes.append(x.flatten())
            out_spikes.append(x)
        
        out = torch.stack(out_spikes)
        #print("out.size()", out.size())
        if self.quant_aware: out = self.dequant(out)  # Dequantize output
        return out, torch.cat(layer_spikes)



def decode(x):
    x = torch.sum(x, 0)

    #print("x.size()", x.size())
    #print("x", x)

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
