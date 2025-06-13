import torch
import norse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tonic
from torch.quantization import QuantStub, DeQuantStub, get_default_qat_qconfig, prepare_qat

"""
    Parameters for experiment control:
        - num_hid_layers: Number of hidden layers. Defaults to 2
        - size_hid_layers: Size of each hidden layer
        - quant_aware: Quintize aware training (QAT)(int8). Boolean if set true, the model trains applying the int8 quantization for it's parameters during inference and Quantize Aware gradient claculation during backprop
        - spike_factor: Factor to penalize the number of total spikes in the network (hidden layers + output layer)
        - mean_weight_factor: Factor to penalize distance of square of sum of network's weights and biases from 0.
        - epochs: epochs to train the network
        - n_time_bins: The total timesteps of each datapoint. Each digits (spikes generated from each digit) will be converted to a 3d tensor of (n_time_bins, 28, 28) "frames"

        I have isolated these hyperparameters in the beginning of the code as I found those interesting to investigate. Other hyperparameters (learning rate, optimizer, batch size) are not included here as I hypothesize that 
        they are not part of a potential "hyperparameter search" for this project (of course they can be changed in the code but would require a lot of up-down scrolling if done in the code :))

        The poplarity information is ignored (I took the liberty to follow this approach as I suppose that the rate encoding scheme you were using also dropped the polarity information). Let me know if you want to include it as well.

        I'd also suggest you use configuration files to set these hyperparameters if you want to scan many values rather than changing these in the script.
        
        Hope that helps :)
        FYI: this setting achieves ~92% acc with QAT on.

"""


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






import random
SEED = 50
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)




def spike_regular(spikes, alpha):
    return alpha*torch.sum(spikes)

def mean_weight_reg(net, alpha):
    val , count = 0, 0
    for mod in net.linear_layers:
        val += torch.sum(mod.weight.data) + torch.sum(mod.bias.data)
        count += mod.weight.numel() + mod.bias.numel()
    if count == 0:
        return 0.0
    mean = val / count
    return alpha * mean.pow(2)


def train(net, trainloader, optimizer):
    net.train()
    losses = []



    for (data, target) in tqdm(trainloader, leave=False):
        data, target = data, torch.LongTensor(target)
        optimizer.zero_grad()
        output, spikes = net(data)
        loss = torch.nn.functional.nll_loss(output, target) + spike_regular(spikes, alpha=spike_factor) + mean_weight_reg(net.snn, alpha=mean_weight_factor)

        # Print the effect of each loss term. Comment if not needed

        print(torch.nn.functional.nll_loss(output, target), spike_regular(spikes, alpha=spike_factor), mean_weight_reg(net.snn, alpha=mean_weight_factor))

       
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


from pathlib import Path
#from store_model_params import model_dir, test_patterns_dir

#model_dir = Path("../../ethosu_compiler/gen_cms/nmnist_784x64x64x10/")
#model_dir = Path("./")
model_dir = Path(args.model)

# Where to store stuff
model_params_dir = Path(model_module.model_params_dir)
test_patterns_dir = Path(model_module.test_patterns_dir)

print("model_dir", model_dir)
print("model_params_dir:", model_params_dir)
print("test_patterns_dir:", test_patterns_dir)
test_data_filepath = model_dir / test_patterns_dir / Path("test_input_"+ str(2) + ".npy")
print("test_data_filepath", test_data_filepath)
print("test_data_filepath", test_data_filepath.__class__)
print("test_data_filepath is dir", test_data_filepath.is_dir())
print("test_data_filepath is file", test_data_filepath.is_file())
print("test_data_filepath exists", test_data_filepath.exists())



def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data, torch.LongTensor(target)
            output, spikes = net(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()



            test_data_filepath = model_dir / test_patterns_dir / Path("test_input_"+ str(counter) + ".npy")
            test_target_filepath = model_dir / test_patterns_dir / Path("test_target_" + str(counter) +".npy")




            

            # Store test data
            tmp_test_data = data.cpu()
            tmp_test_target = target.cpu()

            np.save(test_data_filepath, tmp_test_data)
            np.save(test_target_filepath, tmp_test_target)

            #print("spikes", spikes)

            counter += 1


    test_loss /= len(testloader.dataset)

    accuracy = 100.0 * correct / len(testloader.dataset)

    return test_loss, accuracy







batch_size = 512
cropped_size = (28, 28)

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = tonic.transforms.Compose(
                [tonic.transforms.MergePolarities(),
                 tonic.transforms.CenterCrop(sensor_size=sensor_size, size=cropped_size),
                 tonic.transforms.ToFrame(sensor_size=(*cropped_size, 1),n_time_bins=n_time_bins,),
                 ])

trainset = tonic.datasets.NMNIST(save_to='../data', transform= frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='../data', transform=frame_transform, train=False)

trainloader = torch.utils.data.DataLoader(trainset, 
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=True
)

testloader = torch.utils.data.DataLoader(testset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=True
)




optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def main():
    for epoch in range(epochs):
        training_loss, mean_loss = train(net, trainloader, optimizer)
        print(test(net, testloader))

    # Save the model or the weights as you want. Example for saving the model
    #torch.save(net.snn, 'model.pkl')

    torch.save(net.snn.state_dict(), model_dir / Path('model_state_dict.pkl'))


if __name__ == '__main__':
    main()


