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


# Where to load from (no storing in this script)
#model_dir = Path("../../ethosu_compiler/gen_cms/nmnist_784x64x64x10/")
#model_dir = Path("./")
model_dir = Path(args.model)
model_params_dir = Path(model_module.model_params_dir)
test_patterns_dir = Path(model_module.test_patterns_dir)




# Load model
#net.load_state_dict(torch.load("save_model_dict_784x32x10.pt", weights_only=False))
snn.load_state_dict(torch.load(model_dir / Path("model_state_dict.pkl")))


net = Model(snn=snn, decoder=decode)
net.eval()

print("model", net)










def test(net, test_data, test_target):

    print("data", test_data.shape)
    print("target", test_target.shape)

    # Only keep the first 45 samples (since thats the ones we have for the NPU)
    test_data = test_data[:, :45, ...]
    test_target = test_target[:45]

    print("data", test_data.shape)
    print("target", test_target.shape)

    net.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        #for i in range(len(test_target)):
        #data = torch.tensor(test_data[i], dtype=torch.float32, requires_grad=True)
        data = torch.tensor(test_data, dtype=torch.float32, requires_grad=True)
        #data, target = test_data, torch.LongTensor(test_target)
        target = torch.LongTensor(test_target)

        #target = test_target[i]
        output, spikes = net(data)


        #test_loss += torch.nn.functional.nll_loss(
            #output, target, reduction="sum"
        #).item()  # sum up batch loss
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        print("pred:")
        for val in pred:
	        print(str(val.item()) + " ", end='')
        print("")
        print("target:")
        for val in target:
            print(str(val.item()) + " ", end='')
        print("")

        #print("target:", target, "\tpred:", pred)


        # Store test data
        tmp_test_data = data.cpu()
        tmp_test_target = target.cpu()

        np.save(model_dir / test_patterns_dir / Path("test_input_"+ str(counter) + ".npy"), tmp_test_data)
        np.save(model_dir / test_patterns_dir / Path("test_target_" + str(counter) +".npy"), tmp_test_target)

        #print("spikes", spikes)

        counter += 1


    test_loss /= len(test_target)

    accuracy = 100.0 * correct / len(test_target)

    return test_loss, accuracy







def make_csv_rows(dicts, keys):
    rows = []
    rows.append(keys)
    for item in dicts:
        rows.append([item[key] for key in keys])
    return rows




## Do testing

test_data = np.load(model_dir / test_patterns_dir / Path("test_input_0.npy"))
target_data = np.load(model_dir / test_patterns_dir / Path("test_target_0.npy"))



test_loss, test_accuracy = test(net, test_data, target_data)
print("test_loss:", test_loss)
print("test_accuracy:", test_accuracy)


stats = []
for idx in range(net.snn.num_hidden+1):
    stats.append({
            'min': net.snn.state_mins[idx].item() if net.snn.state_mins[idx] is not None else None,
            'max': net.snn.state_maxs[idx].item() if net.snn.state_maxs[idx] is not None else None
    })

    print("max val", net.snn.state_maxs[idx].item())
    print("min val", net.snn.state_mins[idx].item())
    
    print(f"Layer 0,{net.snn.state_maxs[idx].item()},{net.snn.state_mins[idx].item()}")






import csv, sys

rows = make_csv_rows(stats, ('max', 'min'))

w = csv.writer(sys.stdout, csv.excel_tab)
w.writerows(rows)





for i in range(2):
    for j in range(len(stats)):
        if i == 0:
            if j == 0:
                print("max")

            print(stats[j]['max'])


        elif i == 1:
            if j == 0:
                print("min")

            print(stats[j]['min'])

