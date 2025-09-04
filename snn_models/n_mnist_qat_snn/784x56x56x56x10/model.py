from pathlib import Path



import torch
import norse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tonic
from torch.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig, prepare_qat

from norse.torch.functional.reset import reset_subtract


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to add common plotting functions
from plot import plot_neuron_mem_traces


# Network architecture parameters
input_size = 28*28
output_size = 10
num_hid_layers = 3
size_hid_layers = [56, 56, 56]

#Network training parameters
quant_aware = True
spike_factor = 1e-5
mean_weight_factor = 1e3
epochs = 10
n_time_bins = 25


# Where to store stuff
model_dir = Path("784x56x56x56x10")
model_params_dir = Path("model_params")
test_patterns_dir =  Path("test_patterns")

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


        # Initialize min/max tracking for each state
        self.state_mins = [None for _ in range(num_hidden+1)]
        self.state_maxs = [None for _ in range(num_hidden+1)]
        self.tracking_enabled = True  # Flag to enable/disable tracking


    def _update_state_minmax(self, idx, state_tensor):
        """Update min/max tracking for a specific state index"""
        if not self.tracking_enabled or state_tensor is None:
            return

        
        # Get current min/max values from the state tensor
        current_min = torch.min(state_tensor)
        current_max = torch.max(state_tensor)
        
        # Update global min
        if self.state_mins[idx] is None:
            self.state_mins[idx] = current_min.detach()
        else:
            self.state_mins[idx] = torch.min(self.state_mins[idx], current_min.detach())
        
        # Update global max
        if self.state_maxs[idx] is None:
            self.state_maxs[idx] = current_max.detach()
        else:
            self.state_maxs[idx] = torch.max(self.state_maxs[idx], current_max.detach())



    def forward(self, inp):
        states = [None for _ in range(self.num_hidden+1)]
        out_spikes = []
        layer_spikes = []
        if self.quant_aware: inp = self.quant(inp)

        #for getting membrane traces
        # We are looking for sample 0, neuron 2, in layer 1
        probe_sample = 7
        probe_layer = 0
        probe_neuron = 45

        in_curr_traces = []
        mem_traces = []
        out_spk_traces = []
        out_spk_sum = np.zeros(self.linear_layers[probe_layer].in_features)

        time_step = 0
        inp = inp.squeeze(2)
        for x in inp:
            x = torch.flatten(x, 1)
            for idx, layer in enumerate(self.linear_layers):
                x = layer(x)

                # Get in curr for plotting traces
                if idx == probe_layer:
                    in_curr_traces.append(x[probe_sample][probe_neuron].item())
                
                x, states[idx] = self.temp_layers[idx](x, states[idx])

                #print("scale", layer.scale)
                print("zp", layer.__dir__)



                if idx == probe_layer:
                    #print(f"time_step: {time_step} v_mem_nxt {(states[idx].v)[sample][neuron]}")
                    mem_traces.append(((states[idx].v)[probe_sample][probe_neuron]).item())

                    #import sys
                    #np.set_printoptions(threshold=sys.maxsize)
                    print("out_spk:")
                    #for i in range(x.size()[0]):
                    for j in range(x.size()[1]):
                            #print(f"{x[i][j].item()} ", end='')
                            #row_sum[i] += x[i][j].item()
                        out_spk_sum[j] += x[probe_sample][j].item()
                            
                    print("")
                    #print("row_sum", row_sum)

                    out_spk_traces.append(x[probe_sample][probe_neuron].item())
                # Track min/max values for the current state
                self._update_state_minmax(idx, states[idx].v)


                layer_spikes.append(x.flatten())
            out_spikes.append(x)

            time_step+=1
        
        print("in_curr_traces = ", in_curr_traces)
        print("mem_traces = ", mem_traces)
        print("out_spk_traces = ", out_spk_traces)
        print("out_spk_sum", out_spk_sum)
        print("out_spk_sum[21]", out_spk_sum[21])
        plot_neuron_mem_traces(in_curr_traces, mem_traces, out_spk_traces, time_step, 1, f"GPU: {model_dir}\n Activity of Neuron {probe_neuron}, Layer {probe_layer} (Test sample {probe_sample})")

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
        #weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    )
    torch.backends.quantized.engine = 'fbgemm'  # or whichever you use
    torch.quantization.prepare_qat(snn, inplace=True)

net = Model(snn=snn, decoder=decode)
