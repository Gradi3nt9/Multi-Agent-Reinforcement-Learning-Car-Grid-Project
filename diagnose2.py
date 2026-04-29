import torch
import torch.nn as nn
import numpy as np
import math

# --- Copy of the network (must match training code exactly) ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions=4):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.softmax(self.output_layer(x))

# --- Load your saved weights ---
net = PolicyNetwork(input_size=11, hidden_size=96)
net.load_state_dict(torch.load("policy_network_best.pt", map_location="cpu"))
net.eval()

ACTION_NAMES = ["turn_left", "turn_right", "speed_up", "no_action"]

# --- Quick 5-state check ---
def make_state(x, y, angle_deg, speed, sensors):
    a = math.radians(angle_deg)
    return np.array([x, y, speed*math.cos(a), speed*math.sin(a)] + sensors, dtype=np.float32)

test_states = [
    ("Open field, all clear",     make_state(0.5, 0.5,   0, 0.05, [0.9]*7)),
    ("Wall straight ahead",       make_state(0.9, 0.5,   0, 0.05, [0.8,0.6,0.2,0.02,0.2,0.6,0.8])),
    ("Open left, wall right",     make_state(0.5, 0.5,   0, 0.05, [0.9,0.9,0.5,0.2,0.1,0.05,0.05])),
    ("Open right, wall left",     make_state(0.5, 0.5,   0, 0.05, [0.05,0.05,0.1,0.2,0.5,0.9,0.9])),
    ("Max speed, open field",     make_state(0.5, 0.5,   0, 0.10, [1.0]*7)),
]

print("State -> Action  (should vary based on situation!)")
print("-" * 60)
for desc, state in test_states:
    with torch.no_grad():
        probs = net(torch.tensor(state)).numpy()
    action = int(np.argmax(probs))
    print(f"{desc:<30} -> {ACTION_NAMES[action]:<12} {probs.round(3)}")