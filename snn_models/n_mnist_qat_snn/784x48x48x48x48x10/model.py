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
num_hid_layers = 4
size_hid_layers = [48, 48, 48, 48]

#Network training parameters
quant_aware = True
spike_factor = 1e-5
mean_weight_factor = 1e3
epochs = 40
n_time_bins = 25


# Where to store stuff
model_dir = Path("784x48x48x48x48x10")
#model_dir = Path("./")
model_params_dir = Path("model_params")
test_patterns_dir = Path("test_patterns")

#Create dirs if they dont already exist
(model_dir / model_params_dir).mkdir(parents=True, exist_ok=True)
(model_dir / test_patterns_dir).mkdir(parents=True, exist_ok=True)



#def lif_op_sim(in_curr, v_mem):
#    beta = 0.95
#    vth = 1
#
#
#    print("v_mem_type", v_mem.size())
#    print("in_curr", in_curr.size())
#    decayed_mem = 0.95 * v_mem
#
#    decayed_mem_plus_v_mem = decayed_mem + in_curr
#
#
#    print("rgaraergerg", v_mem.size().__class__)
#    out_spk = torch.zeros([v_mem.size()[0], v_mem.size()[1]])
#
#    for sample in range(decayed_mem_plus_v_mem.size()[0]):
#        for neuron in range(decayed_mem_plus_v_mem.size()[1]):
#            #print("decayeD_mem_plus_v_mem", decayed_mem_plus_v_mem)
#            #print(f"sample: {sample},\tneuron: {neuron},\tdecayed_mem_plus_v_mem.size(): {decayed_mem_plus_v_mem.size()}")
#            if decayed_mem_plus_v_mem[sample][neuron] > 1:
#                out_spk[sample][neuron] = 1
#            else:
#                out_spk[sample][neuron] = 0
#    #if (decayed_mem_plus_v_mem > 1):
#    #    out_spk = 1
#    #else:
#    #    out_spk = 0
#
#
#        
#
#    v_mem = decayed_mem_plus_v_mem - out_spk * vth
#
#    return decayed_mem, decayed_mem_plus_v_mem, v_mem
#
#
#
#
#
#def get_max_min_from_tensor(tensor, curr_max_val, curr_min_val):
#    # Store max min val
#    max_val = tensor.max().item()
#    min_val = tensor.min().item()
#    if (max_val > curr_max_val):
#        curr_max_val = max_val
#    if (min_val < curr_min_val):
#        curr_min_val = min_val
#
#    return curr_max_val, curr_min_val
#
#class Net(torch.nn.Module):
#    def __init__(self, input_size, output_size, num_hidden, size_hidden, quant_aware, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        size_hidden.insert(0, input_size)
#        size_hidden.append(output_size)
#        self.layers_size = size_hidden
#        self.num_hidden = num_hidden
#        self.linear_layers = torch.nn.ModuleList(
#            [torch.nn.Linear(self.layers_size[i], self.layers_size[i+1]) for i in range(num_hidden+1)]
#        )
#        self.temp_layers = torch.nn.ModuleList(
#            [norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.05, reset_method=reset_subtract), dt=1) for _ in range(num_hidden+1)]
#        )
#
#        self.quant_aware = quant_aware
#        if self.quant_aware:
#            self.quant = QuantStub()
#            self.dequant = DeQuantStub()
#
#
#        self.max_in_curr = [0 for _ in range(self.num_hidden+1)]
#        self.min_in_curr = [0 for _ in range(self.num_hidden+1)]
#
#        self.max_v_mem = [0 for _ in range(self.num_hidden+1)]
#        self.min_v_mem = [0 for _ in range(self.num_hidden+1)]
#
#        
#        self.max_decayed_v_mem_plus_in_curr = [0 for _ in range(self.num_hidden+1)]
#        self.min_decayed_v_mem_plus_in_curr = [0 for _ in range(self.num_hidden+1)]
#
#        self.max_decayed_v_mem = [0 for _ in range(self.num_hidden+1)]
#        self.min_decayed_v_mem = [0 for _ in range(self.num_hidden+1)]
#
#
#    def forward(self, inp):
#        states = [None for _ in range(self.num_hidden+1)]
#        out_spikes = []
#        layer_spikes = []
#        if self.quant_aware: inp = self.quant(inp)
#
#
#        batch_size = inp.squeeze(2).size()[1]
#        #print("batch_size", batch_size)
#        v_mem = []
#        decayed_v_mem = []
#        decayed_v_mem_plus_in_curr = []
#        for i in range(self.num_hidden+1):
#            v_mem.append(torch.zeros([batch_size, self.layers_size[i+1]]))
#            decayed_v_mem.append(torch.zeros([batch_size, self.layers_size[i+1]]))
#            decayed_v_mem_plus_in_curr.append(torch.zeros([batch_size, self.layers_size[i+1]]))
#            
#        #v_mem = [None for _ in range(self.num_hidden+1)]
#        #decayed_v_mem = [None for _ in range(self.num_hidden+1)]
#        #decayed_v_mem_plus_v_mem = [None for _ in range(self.num_hidden+1)]
#
#
#        inp = inp.squeeze(2)
#        for x in inp:
#            x = torch.flatten(x, 1)
#            for idx, layer in enumerate(self.linear_layers):
#
#
#
#                # Compute FC layer
#                x = layer(x)
#
#            
#                ## Store max min in_curr val
#                #max_in_curr_val = x.max().item()
#                #min_in_curr_val = x.min().item()
#                #if (max_in_curr_val > self.max_in_curr[idx]):
#                #    self.max_in_curr[idx] = max_in_curr_val
#                #if (min_in_curr_val < self.min_in_curr[idx]):
#                #    self.min_in_curr[idx] = min_in_curr_val
#
#                print("x", x.size(), "\n", x)
#
#                self.max_in_curr[idx], self.min_in_curr[idx] = get_max_min_from_tensor(x, self.max_in_curr[idx], self.min_in_curr[idx])
#
#                # Simulate max min values
#                decayed_v_mem[idx], decayed_v_mem_plus_in_curr[idx], v_mem[idx] = lif_op_sim(x, v_mem[idx])
#
#                # Get max min vals
#                self.max_decayed_v_mem[idx], self.min_decayed_v_mem[idx] = get_max_min_from_tensor(decayed_v_mem[idx], self.max_decayed_v_mem[idx], self.min_decayed_v_mem[idx])
#                self.max_decayed_v_mem_plus_in_curr[idx], self.min_decayed_v_mem_plus_in_curr[idx] = get_max_min_from_tensor(decayed_v_mem_plus_in_curr[idx], self.max_decayed_v_mem_plus_in_curr[idx], self.min_decayed_v_mem_plus_in_curr[idx])
#                self.max_v_mem[idx], self.min_v_mem[idx] = get_max_min_from_tensor(v_mem[idx], self.max_v_mem[idx], self.min_v_mem[idx])
#
#
#                ## Store max min val
#                #max_v_mem_val = states[idx].v.max().item()
#                #min_v_mem_val = states[idx].v.min().item()
#                #if (max_v_mem_val > self.max_v_mem[idx]):
#                #    self.max_v_mem[idx] = max_v_mem_val
#                #if (min_v_mem_val < self.min_v_mem[idx]):
#                #    self.min_v_mem[idx] = min_v_mem_val
#
#
#                
#
#                # Compute LIF layer
#                x, states[idx] = self.temp_layers[idx](x, states[idx])
#                layer_spikes.append(x.flatten())
#
#                
#
#                '''
#                # Store max min val
#                max_v_mem_val = states[idx].v.max().item()
#                min_v_mem_val = states[idx].v.min().item()
#                if (max_v_mem_val > self.max_v_mem[idx]):
#                    self.max_v_mem[idx] = max_v_mem_val
#                if (min_v_mem_val < self.min_v_mem[idx]):
#                    self.min_v_mem[idx] = min_v_mem_val
#                '''
#
#                
#
#
#
#
#            out_spikes.append(x)
#        
#        out = torch.stack(out_spikes)
#        if self.quant_aware: out = self.dequant(out)  # Dequantize output
#        return out, torch.cat(layer_spikes)
#
#
#
#
#
#
#
#
#def decode(x):
#    x = torch.sum(x, 0)
#
#    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
#
#    return log_p_y
#
#
#class Model(torch.nn.Module):
#    def __init__(self, snn, decoder):
#        super(Model, self).__init__()
#        self.snn = snn
#        self.decoder = decoder
#
#    def forward(self, x):
#        x, out_spikes = self.snn(x)
#        log_p_y = self.decoder(x)
#
#
#        print("IN_CURR_MAX_VAL_LIST =", self.snn.max_in_curr)
#        print("IN_CURR_MIN_VAL_LIST =", self.snn.min_in_curr)
#
#        print("V_MEM_MAX_VAL_LIST =", self.snn.max_v_mem)
#        print("V_MEM_MIN_VAL_LIST =", self.snn.min_v_mem)
#
#
#        
#        print(f"{'max_decayed_v_mem_plus_in_curr_val_list'.upper()} = {self.snn.max_decayed_v_mem_plus_in_curr}")
#
#        print("DECAYED_MEM_PLUS_IN_CURR_VAL_LIST =", self.snn.min_decayed_v_mem_plus_in_curr)
#
#        print("DECAYED_MEM_MAX_VAL_LIST =", self.snn.max_decayed_v_mem)
#        print("DECAYED_MEM_MIN_VAL_LIST =", self.snn.min_decayed_v_mem)
#
#
#
#        return log_p_y, out_spikes
#
#
#
#
#snn = Net(input_size=input_size, 
#          output_size=output_size, 
#          num_hidden=num_hid_layers, 
#          size_hidden=size_hid_layers, 
#          quant_aware=quant_aware)
#
#
#
#if quant_aware:
#
#
#    #snn.qconfig = torch.quantization.QConfig(
#    #    activation=torch.quantization.default_observer,
#    #    weight=torch.quantization.default_per_tensor_weight_observer  # force per-tensor weight quantization
#    #)
#    #torch.quantization.prepare_qat(snn, inplace=True)
#    #snn.qconfig = get_default_qat_qconfig("fbgemm")  # or "qnnpack" for ARM
#    #prepare_qat(snn, inplace=True)
#
#    from torch.ao.quantization import QConfig
#    from torch.ao.quantization.observer import MinMaxObserver
#
#    
#    snn.qconfig = QConfig(
#        activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
#        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
#    )
#    torch.backends.quantized.engine = 'fbgemm'  # or whichever you use
#    torch.quantization.prepare_qat(snn, inplace=True)
#
#net = Model(snn=snn, decoder=decode)








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

        inp = inp.squeeze(2)
        for x in inp:
            x = torch.flatten(x, 1)
            for idx, layer in enumerate(self.linear_layers):
                x = layer(x)
                x, states[idx] = self.temp_layers[idx](x, states[idx])



                # Track min/max values for the current state
                self._update_state_minmax(idx, states[idx].v)


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
