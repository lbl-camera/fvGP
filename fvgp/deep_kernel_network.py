import torch
from torch import nn


class Network(nn.Module):  # pragma: no cover
    def __init__(self, dim, layer_width):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.layer1 = nn.Linear(dim, layer_width)
        self.layer2 = nn.Linear(layer_width, layer_width)
        self.layer3 = nn.Linear(layer_width, dim)
        self.number_of_hps = int(2. * dim * layer_width + layer_width ** 2 + 2. * layer_width + dim)

    def forward(self, x):
        x = torch.Tensor(x)
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        return x.detach().numpy()

    def set_weights(self, w1, w2, w3):
        with torch.no_grad(): self.layer1.weight = nn.Parameter(torch.from_numpy(w1).float())
        with torch.no_grad(): self.layer2.weight = nn.Parameter(torch.from_numpy(w2).float())
        with torch.no_grad(): self.layer3.weight = nn.Parameter(torch.from_numpy(w3).float())

    def set_biases(self, b1, b2, b3):
        with torch.no_grad(): self.layer1.bias = nn.Parameter(torch.from_numpy(b1).float())
        with torch.no_grad(): self.layer2.bias = nn.Parameter(torch.from_numpy(b2).float())
        with torch.no_grad(): self.layer3.bias = nn.Parameter(torch.from_numpy(b3).float())

    def get_weights(self):
        return self.layer1.weight, self.layer2.weight, self.layer3.weight

    def get_biases(self):
        return self.layer1.bias, self.layer2.bias, self.layer3.bias
