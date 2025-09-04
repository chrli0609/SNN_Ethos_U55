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
num_hid_layers = 2
size_hid_layers = [64, 64]

#Network training parameters
quant_aware = True
spike_factor = 1e-5
mean_weight_factor = 1e3
epochs = 10
n_time_bins = 25


# Where to store stuff
model_dir = Path("784x64x64x10")
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
        probe_neuron = 26

        in_curr_traces = []
        mem_traces = []
        out_spk_traces = []
        out_spk_sum = np.zeros(self.linear_layers[probe_layer].in_features) 

        inp = inp.squeeze(2)
        print("inp", inp.size())
        time_step = 0
        for x in inp:
            print("x pre flatten", x.size())
            x = torch.flatten(x, 1)
            print("x post flatten", x.size())
            for idx, layer in enumerate(self.linear_layers):


                print("idx", idx, "layer", layer)
                x = layer(x)
                print("x post layer(x)", x.size())
                if idx == probe_layer:
                    in_curr_traces.append(x[probe_sample][probe_neuron].item())
                x, states[idx] = self.temp_layers[idx](x, states[idx])

                print(f"states[{idx}] {states[idx].v.size()}")

                print(f"x: {x.size()}")

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
                #if idx == 1:
                    #print(f"time_step:{}")

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
        plot_neuron_mem_traces(in_curr_traces, mem_traces, out_spk_traces, time_step, 1, f"GPU: {model_dir}\nTest sample {probe_sample}, Layer {probe_layer}, Neuron {probe_neuron}")
        
        out = torch.stack(out_spikes)
        if self.quant_aware: out = self.dequant(out)  # Dequantize output
        return out, torch.cat(layer_spikes)

    #def plot_neuron_mem_traces(self, in_curr_traces, mem_traces, out_spk_traces, num_time_steps, thr_line=1, title=None, vline=False, slack=0.2):
        #gpu_in_curr_t = torch.from_numpy(np.array(in_curr_traces))
        #gpu_mem_t = torch.from_numpy(np.array(mem_traces))
        #gpu_out_spk_t = torch.from_numpy(np.array(out_spk_traces))

        ## Generate Plots
        #fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                            #gridspec_kw = {'height_ratios': [1, 1, 0.4]})

        ## Plot input current
        #ax[0].plot(gpu_in_curr_t, linestyle="dashed", marker=".", c="tab:orange", label="$I_{in}$")
        ##ax[0].plot(mem_prev, c="tab:blue", label="$U_{mem}$")
        ##sum_of_two = cur + mem_prev
        ##ax[0].plot(sum_of_two, c="tab:brown", label="$I_{in}")
        #ax[0].set_ylim([torch.min(gpu_in_curr_t).item()-slack, torch.max(gpu_in_curr_t).item()+slack])
        #ax[0].set_xlim([0, num_time_steps-1])
        #ax[0].set_ylabel("Input Current $I_{in}$")
        ##ax[0].set_ylabel("Before Neuron Update")
        #if title:
            #ax[0].set_title(title)
        ##ax[0].legend()



        ## Plot membrane potential
        #ax[1].plot(gpu_mem_t, linestyle="dashed", marker=".", c="tab:blue")
        #ax[1].set_ylim([torch.min(gpu_mem_t).item()-slack, torch.max(gpu_mem_t).item()+slack])
        #ax[1].set_ylabel("Membrane Potential $U_{mem}^{next}$\n(post reset)")
        ##if thr_line!=None:
        #ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        #plt.xlabel("Time step")

        ## Plot output spike using spikeplot
        #splt.raster(gpu_out_spk_t, ax[2], s=400, c="black", marker="|")
        #if vline:
            #ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
        #plt.ylabel("Output spikes")
        #plt.yticks([])


        #plt.savefig("v_mem_traces.svg", format="svg")


        #plt.show()





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
