import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the 2-hidden-layer Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(PolicyNetwork, self).__init__()
        # Hidden Layer 1
        self.layer1 = nn.Linear(input_size, hidden_size)
        # Hidden Layer 2
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        # Output Layer
        self.output_layer = nn.Linear(hidden_size, num_actions)
        self.relu = nn.ReLU() # The activation function (makes the math non-linear)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        # Returns the raw scores for each action
        return self.output_layer(x) 

# Create the network
# Example: input is [x, y] (size 2), hidden layer has 64 neurons, 4 possible actions
net = PolicyNetwork(input_size=2, hidden_size=64, num_actions=4)

# 2. Define Weighted Cross Entropy Loss
# Let's say we have 4 actions, and we want to weight action index 2 to be twice as important
weights = torch.tensor([1.0, 1.0, 2.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=weights)

# 3. Optimizer (The tool that actually updates the network's weights to learn)
optimizer = optim.Adam(net.parameters(), lr=0.01)