import torch
import numpy as np
from train_and_test import load_policy, CarEnv

def run_diagnostics():
    print("Loading original policy network...")
    # Based on your previous meta.json, your old model used k=9 and hidden_size=96
    try:
        net, meta = load_policy("policy_network_best.pt", input_size=11, hidden_size=96, num_actions=4)
    except Exception as e:
        print(f"Error loading model. Ensure 'policy_network_best.pt' is in the folder. Details: {e}")
        return

    # Create an environment with random starting positions
    env = CarEnv(num_sensors=7, random_starts=True)
    
    print("\n--- Testing 20 Random States ---")
    actions_chosen = []
    
    for i in range(20):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            probs = net(state_tensor).numpy()
            action = np.argmax(probs)
            actions_chosen.append(action)
            
        print(f"Test {i+1:02d} | Probs: {probs.round(3)} | Action Chosen: {action}")
        
    print("\n--- Diagnostic Summary ---")
    print(f"Total Left Turns (Action 0): {actions_chosen.count(0)}")
    print(f"Total Right Turns (Action 1): {actions_chosen.count(1)}")
    print(f"Total Speed Ups (Action 2): {actions_chosen.count(2)}")
    print(f"Total No Actions (Action 3): {actions_chosen.count(3)}")
    
    if actions_chosen.count(0) == 20:
        print("\nDIAGNOSIS CONFIRMED: Complete Mode Collapse. The car only turns left.")

if __name__ == "__main__":
    run_diagnostics()