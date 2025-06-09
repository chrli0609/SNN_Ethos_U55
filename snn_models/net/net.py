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
# Network architecture parameters
num_hid_layers = 2
size_hid_layers = [64, 64]

#Network training parameters
quant_aware = True
spike_factor = 1e-5
mean_weight_factor = 1e3
epochs = 4
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
            [norse.torch.LIFBoxCell(norse.torch.LIFBoxParameters(tau_mem_inv=0.05), dt=1) for _ in range(num_hidden+1)]
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


def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
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


class Model(torch.nn.Module):
    def __init__(self, snn, decoder):
        super(Model, self).__init__()
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x, out_spikes = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y, out_spikes



snn = Net(input_size=784, 
          output_size=10, 
          num_hidden=num_hid_layers, 
          size_hidden=size_hid_layers, 
          quant_aware=quant_aware)


if quant_aware:
    snn.qconfig = get_default_qat_qconfig("fbgemm")  # or "qnnpack" for ARM
    prepare_qat(snn, inplace=True)


net = Model(snn=snn, decoder=decode)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


for epoch in range(epochs):
    training_loss, mean_loss = train(net, trainloader, optimizer)
    print(test(net, testloader))

# Save the model or the weights as you want. Example for saving the model
#torch.save(net.snn, 'model.pkl')
torch.save(net.snn.state_dict(), 'model_state_dict.pkl')