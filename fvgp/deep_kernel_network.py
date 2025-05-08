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


import torch
import torch.nn as nn


# Define a simple neural network to warp a 3D space
class WarpNet(nn.Module):   # pragma: no cover
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super(WarpNet, self).__init__()

        # Define the architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through the layers
        x = self.relu(self.fc1(x))  # Input layer to hidden layer 1
        x = self.relu(self.fc2(x))  # Hidden layer 1 to hidden layer 2
        x = self.fc3(x)  # Hidden layer 2 to output layer
        return x


# Initialize the network
model = WarpNet()

# Example usage: Warping a 3D point (e.g., [x, y, z])
#points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Example input
#warped_points = model(points)  # Get warped points
#print("Warped Points: ", warped_points)

# Loss function and optimizer for training
#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy target points (assuming you know where the warped points should go)
#target_points = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])

# Training loop (single step for simplicity)
#optimizer.zero_grad()
#output = model(points)
#loss = criterion(output, target_points)  # Calculate loss
#loss.backward()  # Backpropagation
#optimizer.step()  # Update the weights

#print("Updated Warped Points: ", model(points))
