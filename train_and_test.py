"""
Standalone training and testing script for autonomous vehicle policy.
All code inline - no module dependencies. Run: python train_and_test.py
"""

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# UTILITIES
# ============================================================================

def set_global_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_competition_config(k: int, hidden_size: int) -> None:
    """Validate core architecture constraints from competition instructions."""
    if k not in {1, 3, 5, 7, 9}:
        raise ValueError("Competition constraint: k must be one of {1,3,5,7,9}")
    if hidden_size < 1 or hidden_size > 100:
        raise ValueError("Competition constraint: hidden_size must be in [1, 100]")


# ============================================================================
# POLICY NETWORK
# ============================================================================

class PolicyNetwork(nn.Module):
    """Fully-connected policy with 2 hidden layers (ReLU)."""

    def __init__(self, input_size: int, hidden_size: int, num_actions: int = 4):
        super(PolicyNetwork, self).__init__()
        if hidden_size > 100:
            raise ValueError("Competition constraint: hidden_size must be <= 100")
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Competition-facing forward pass with explicit softmax outputs."""
        return self.softmax(self.forward_logits(x))


class ValueNetwork(nn.Module):
    """Value network (critic) for training only - NOT submitted to competition."""

    def __init__(self, input_size: int, hidden_size: int):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output_layer(x).squeeze(-1)


# ============================================================================
# CAR ENVIRONMENT
# ============================================================================

class CarEnv:
    """
    Continuous state-space car environment in [0,1]x[0,1] unit square.
    State: [x, y, vx, vy, s1, s2, ..., sk]
    Actions: 0=turn_left, 1=turn_right, 2=speed_up, 3=no_action
    """

    def __init__(self, num_sensors: int = 5, max_steps: int = 500, random_starts: bool = False):
        assert num_sensors in {1, 3, 5, 7, 9}, "num_sensors must be one of {1,3,5,7,9}"
        self.num_sensors = num_sensors
        self.max_steps = max_steps
        self.random_starts = random_starts

        # Competition parameters
        self.min_speed = 0.001
        self.max_speed = 0.1
        self.turn_delta = 0.01
        self.turn_speed_factor = 0.9875
        self.speedup_factor = 1.025
        self.crash_speed_factor = 0.125

        # Sensor fan in front of the car
        self.sensor_fov = math.pi / 2.0
        self.sensor_max_range = math.sqrt(2.0)

        self.reset()

    def reset(self) -> np.ndarray:
        if self.random_starts:
            self.x = float(np.random.uniform(0.2, 0.8))
            self.y = float(np.random.uniform(0.2, 0.8))
            self.angle = float(np.random.uniform(-math.pi, math.pi))
            self.speed = float(np.random.uniform(0.006, 0.02))
        else:
            self.x = 0.5
            self.y = 0.5
            self.angle = 0.0
            self.speed = 0.01
        self.step_count = 0
        self.crash_count = 0
        self.total_distance = 0.0
        return self._get_state()

    def _ray_distance_to_walls(self, ray_angle: float) -> float:
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)
        eps = 1e-12
        candidates = []

        if abs(dx) > eps:
            t = (0.0 - self.x) / dx
            if t >= 0:
                y_hit = self.y + t * dy
                if 0.0 <= y_hit <= 1.0:
                    candidates.append(t)
            t = (1.0 - self.x) / dx
            if t >= 0:
                y_hit = self.y + t * dy
                if 0.0 <= y_hit <= 1.0:
                    candidates.append(t)

        if abs(dy) > eps:
            t = (0.0 - self.y) / dy
            if t >= 0:
                x_hit = self.x + t * dx
                if 0.0 <= x_hit <= 1.0:
                    candidates.append(t)
            t = (1.0 - self.y) / dy
            if t >= 0:
                x_hit = self.x + t * dx
                if 0.0 <= x_hit <= 1.0:
                    candidates.append(t)

        if not candidates:
            return self.sensor_max_range
        return min(candidates)

    def _get_sensor_readings(self) -> List[float]:
        if self.num_sensors == 1:
            angles = [self.angle]
        else:
            left = self.angle - self.sensor_fov / 2.0
            step = self.sensor_fov / (self.num_sensors - 1)
            angles = [left + i * step for i in range(self.num_sensors)]

        readings = []
        for ray_angle in angles:
            d = self._ray_distance_to_walls(ray_angle)
            readings.append(min(d / self.sensor_max_range, 1.0))
        return readings

    def _get_state(self) -> np.ndarray:
        vx = self.speed * math.cos(self.angle)
        vy = self.speed * math.sin(self.angle)
        return np.array([self.x, self.y, vx, vy] + self._get_sensor_readings(), dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1

        # Apply action
        if action == 0:
            self.angle -= self.turn_delta
            self.speed *= self.turn_speed_factor
        elif action == 1:
            self.angle += self.turn_delta
            self.speed *= self.turn_speed_factor
        elif action == 2:
            self.speed *= self.speedup_factor
        elif action == 3:
            pass
        else:
            raise ValueError("action must be in {0,1,2,3}")

        self.speed = max(self.min_speed, min(self.speed, self.max_speed))

        old_x, old_y = self.x, self.y
        new_x = self.x + self.speed * math.cos(self.angle)
        new_y = self.y + self.speed * math.sin(self.angle)

        crashed = (new_x < 0.0 or new_x > 1.0 or new_y < 0.0 or new_y > 1.0)
        if crashed:
            self.crash_count += 1
            self.speed *= self.crash_speed_factor
            self.speed = max(self.min_speed, min(self.speed, self.max_speed))

        self.x = min(max(new_x, 0.0), 1.0)
        self.y = min(max(new_y, 0.0), 1.0)

        frame_distance = math.sqrt((self.x - old_x) ** 2 + (self.y - old_y) ** 2)
        self.total_distance += frame_distance

        done = self.step_count >= self.max_steps
        info = {
            "frame_distance": frame_distance,
            "crashed": crashed,
            "crash_count": self.crash_count,
            "total_distance": self.total_distance,
            "speed": self.speed,
        }
        return self._get_state(), frame_distance, done, info


def shape_reward(raw_reward: float, info: Dict, next_state: np.ndarray) -> float:
    """Shaped reward for training (evaluation still uses raw distance)."""
    # Base: scaled distance traveled this frame.
    reward = 10.0 * raw_reward

    # Crash penalty: crashes tank speed to 12.5%, hurting future distance.
    if info["crashed"]:
        reward -= 0.5

    # Speed bonus: higher speed = more distance over time.
    reward += 2.0 * info["speed"]

    # Wall proximity warning: discourage getting close to walls.
    sensors = next_state[4:]
    min_sensor = float(np.min(sensors))
    if min_sensor < 0.15:
        reward -= 0.1 * (0.15 - min_sensor) / 0.15

    return float(reward)


# ============================================================================
# TRAJECTORY GENERATION & TRAINING
# ============================================================================

def generate_trajectory(
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    env: CarEnv,
    temperature: float = 1.0,
) -> List[Dict]:
    """Collect one full trajectory using the policy and value network."""
    trajectory = []
    state = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits = policy.forward_logits(state_tensor) / temperature

        action_dist = torch.distributions.Categorical(logits=logits)
        action_tensor = action_dist.sample()
        action = int(action_tensor.item())
        log_prob = action_dist.log_prob(action_tensor)
        entropy = action_dist.entropy()

        with torch.no_grad():
            value = value_net(state_tensor)

        next_state, raw_reward, done, info = env.step(action)
        reward = shape_reward(raw_reward, info, next_state)

        trajectory.append({
            "state": state.copy(),
            "action": action,
            "reward": reward,
            "raw_reward": raw_reward,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": float(value.item()),
            "info": info,
        })
        state = next_state

    return trajectory


def heuristic_action_from_state(state: np.ndarray) -> int:
    """Simple expert: speed up when centered, turn away from closer wall."""
    sensors = state[4:]
    n = len(sensors)
    center_idx = n // 2
    center = float(sensors[center_idx])
    left = float(np.mean(sensors[:center_idx])) if center_idx > 0 else center
    right = float(np.mean(sensors[center_idx + 1:])) if center_idx > 0 else center

    # Keep accelerating when there is enough free space ahead.
    if center > 0.40:
        return 2  # speed_up

    # Turn toward the side with more clearance.
    if left > right:
        return 0  # turn_left
    if right > left:
        return 1  # turn_right
    return 3  # no_action


def warmstart_imitation(
    network: PolicyNetwork,
    env: CarEnv,
    optimizer: torch.optim.Optimizer,
    steps: int = 4000,
) -> None:
    """Brief supervised warm-start to avoid RL cold-start collapse."""
    losses = []
    criterion = nn.CrossEntropyLoss()
    network.train()
    for _ in range(steps):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        target_action = torch.tensor([heuristic_action_from_state(state)], dtype=torch.long)
        logits = network.forward_logits(state_tensor).unsqueeze(0)
        loss = criterion(logits, target_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    print(f"[OK] Warm-start imitation complete | steps={steps} | avg_loss={np.mean(losses):.4f}")


def compute_discounted_returns(rewards: List[float], gamma: float = 0.97) -> torch.Tensor:
    """Compute per-time-step discounted returns."""
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns-to-go."""
    T = len(rewards)
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.append(gae)
        next_value = values[t]
    advantages.reverse()
    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t = advantages_t + torch.tensor(values, dtype=torch.float32)
    return advantages_t, returns_t


def train_actor_critic(
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    trajectories: List[List[Dict]],
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    grad_clip: float = 0.5,
) -> Dict:
    """Execute one Actor-Critic training step with GAE."""
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()

    policy_losses = []
    value_losses = []
    entropies = []
    all_advantages: List[float] = []
    total_steps = 0

    for trajectory in trajectories:
        rewards = [step["reward"] for step in trajectory]
        values = [step["value"] for step in trajectory]
        states = [step["state"] for step in trajectory]

        advantages, returns = compute_gae(rewards, values, gamma=gamma, lam=lam)
        all_advantages.extend(advantages.tolist())
        total_steps += len(trajectory)

        # Normalize advantages per trajectory for stable gradients.
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if float(adv_std.item()) > 1e-6:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        for i, step in enumerate(trajectory):
            adv = float(advantages[i].item())
            ret = float(returns[i].item())

            # Policy loss (REINFORCE with baseline).
            policy_losses.append(-step["log_prob"] * adv)
            entropies.append(step["entropy"])

            # Value loss (MSE).
            state_tensor = torch.tensor(states[i], dtype=torch.float32)
            value_pred = value_net(state_tensor)
            value_losses.append((value_pred - ret) ** 2)

    if total_steps == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "adv_mean": 0.0, "adv_std": 0.0, "grad_norm": 0.0}

    avg_policy_loss = torch.stack(policy_losses).mean()
    avg_value_loss = torch.stack(value_losses).mean()
    avg_entropy = torch.stack(entropies).mean()

    policy_loss = avg_policy_loss - (entropy_coef * avg_entropy)
    value_loss = value_coef * avg_value_loss

    # Update policy.
    policy_loss.backward()
    policy_grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip).item())
    policy_optimizer.step()

    # Update value network.
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), grad_clip)
    value_optimizer.step()

    return {
        "loss": float((policy_loss + value_loss).item()),
        "policy_loss": float(avg_policy_loss.item()),
        "value_loss": float(avg_value_loss.item()),
        "entropy": float(avg_entropy.item()),
        "adv_mean": float(np.mean(all_advantages)) if all_advantages else 0.0,
        "adv_std": float(np.std(all_advantages)) if all_advantages else 0.0,
        "grad_norm": policy_grad_norm,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_policy(
    network: PolicyNetwork, env: CarEnv, episodes: int = 20, deterministic: bool = True
) -> Dict:
    """Evaluate policy over multiple episodes."""
    distance_scores = []
    crash_scores = []
    final_speed_scores = []

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = network(state_tensor)
                if deterministic:
                    action = int(torch.argmax(action_probs).item())
                else:
                    action = int(torch.distributions.Categorical(probs=action_probs).sample().item())
            state, _, done, _ = env.step(action)

        distance_scores.append(env.total_distance)
        crash_scores.append(env.crash_count)
        final_speed_scores.append(env.speed)

    return {
        "mode": "det" if deterministic else "stoch",
        "avg_distance": float(np.mean(distance_scores)),
        "avg_crashes": float(np.mean(crash_scores)),
        "crash_free_rate": float(np.mean([c == 0 for c in crash_scores])),
        "avg_final_speed": float(np.mean(final_speed_scores)),
    }


def evaluate_policy_bundle(network: PolicyNetwork, env: CarEnv, episodes: int = 30) -> Dict:
    """Evaluate policy in both deterministic and stochastic modes."""
    return {
        "det": evaluate_policy(network, env, episodes=episodes, deterministic=True),
        "stoch": evaluate_policy(network, env, episodes=episodes, deterministic=False),
    }


def score_for_selection(metrics: Dict) -> float:
    """Competition metric: maximize deterministic average distance."""
    det = metrics["det"]
    return float(det["avg_distance"])


# ============================================================================
# SAVE / LOAD
# ============================================================================

def save_policy(network: PolicyNetwork, path: str, metadata: Dict = None) -> None:
    """Save weights-only checkpoint plus optional metadata sidecar."""
    torch.save(network.state_dict(), path)
    print(f"[OK] Saved weights-only policy to {path}")
    if metadata:
        sidecar = Path(path).with_suffix(".meta.json")
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[OK] Saved metadata sidecar to {sidecar}")


def export_weights_txt(network: PolicyNetwork, path: str) -> None:
    """Export weights in competition text format.
    
    Format: Each block has weight.T rows first, then bias as last row.
    Block 1: (input_size + 1) rows x hidden_size cols  [input -> hidden1]
    Block 2: (hidden_size + 1) rows x hidden_size cols [hidden1 -> hidden2]
    Block 3: (hidden_size + 1) rows x num_actions cols [hidden2 -> output]
    """
    state = network.state_dict()
    lines = []

    # Layer 1: input -> hidden1
    # weight.T rows first (each row = one input to all hidden), then bias as last row
    w1 = state["layer1.weight"].numpy()  # (hidden, input)
    b1 = state["layer1.bias"].numpy()    # (hidden,)
    for row in w1.T:  # transpose: each row = weights from one input to all hidden
        lines.append(", ".join(f"{v:.8f}" for v in row))
    lines.append(", ".join(f"{v:.8f}" for v in b1))  # bias row last
    lines.append("-----")

    # Layer 2: hidden1 -> hidden2
    w2 = state["layer2.weight"].numpy()  # (hidden, hidden)
    b2 = state["layer2.bias"].numpy()    # (hidden,)
    for row in w2.T:  # transpose
        lines.append(", ".join(f"{v:.8f}" for v in row))
    lines.append(", ".join(f"{v:.8f}" for v in b2))  # bias row last
    lines.append("-----")

    # Output layer: hidden2 -> output
    wo = state["output_layer.weight"].numpy()  # (4, hidden)
    bo = state["output_layer.bias"].numpy()    # (4,)
    for row in wo.T:  # transpose: each row = weights from one hidden to all outputs
        lines.append(", ".join(f"{v:.8f}" for v in row))
    lines.append(", ".join(f"{v:.8f}" for v in bo))  # bias row last

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Exported weights to {path}")


def load_policy(
    path: str, input_size: int, hidden_size: int, num_actions: int = 4, map_location: str = "cpu"
) -> Tuple[PolicyNetwork, Dict]:
    """Load policy network and metadata."""
    payload = torch.load(path, map_location=map_location)
    model = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions)
    if isinstance(payload, dict) and "state_dict" in payload:
        # Backward compatibility with older bundled checkpoints.
        model.load_state_dict(payload["state_dict"])
        metadata = payload.get("metadata", {})
    else:
        model.load_state_dict(payload)
        sidecar = Path(path).with_suffix(".meta.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        else:
            metadata = {}
    model.eval()
    return model, metadata


def predict_action(network: PolicyNetwork, state: np.ndarray, deterministic: bool = True) -> int:
    """Predict deterministic action from state."""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        action_probs = network(state_tensor)
        if deterministic:
            return int(torch.argmax(action_probs).item())
        return int(torch.distributions.Categorical(probs=action_probs).sample().item())


# ============================================================================
# QUICK TEST
# ============================================================================

def quick_test():
    """Quick smoke test: environment and network sanity check."""
    print("\n" + "="*70)
    print("QUICK TEST: Environment & Network Sanity Check")
    print("="*70)

    SEED = 42
    set_global_seed(SEED)

    k = 7
    input_size = 4 + k
    hidden_size = 64
    num_actions = 4
    max_steps = 120

    train_env = CarEnv(num_sensors=k, max_steps=max_steps)
    eval_env = CarEnv(num_sensors=k, max_steps=max_steps)
    net = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions)

    # Test state shape
    state0 = train_env.reset()
    print(f"[OK] Initial state shape: {state0.shape} (expected {(input_size,)})")
    assert state0.shape == (input_size,), "State shape mismatch!"

    # Test network forward pass
    state_tensor = torch.tensor(state0, dtype=torch.float32)
    with torch.no_grad():
        probs = net(state_tensor)
    print(f"[OK] Network output shape: {probs.shape} (expected (4,))")
    assert probs.shape == (4,), "Output shape mismatch!"
    print(f"[OK] Action probabilities: {probs.numpy().round(3)}")
    assert abs(float(probs.sum()) - 1.0) < 1e-5, "Probabilities don't sum to 1!"

    # Test trajectory generation
    value_net = ValueNetwork(input_size=input_size, hidden_size=hidden_size)
    traj = generate_trajectory(net, value_net, train_env)
    traj_raw_reward = sum(step["raw_reward"] for step in traj)
    traj_crashes = traj[-1]["info"]["crash_count"] if len(traj) > 0 else 0
    print(f"[OK] Trajectory length: {len(traj)}")
    print(f"[OK] Trajectory raw distance: {traj_raw_reward:.4f}")
    print(f"[OK] Trajectory crashes: {traj_crashes}")

    # Test evaluation
    quick_bundle = evaluate_policy_bundle(net, eval_env, episodes=4)
    print(f"[OK] Deterministic metrics: {quick_bundle['det']}")
    print(f"[OK] Stochastic metrics: {quick_bundle['stoch']}")

    print("\n[OK] All quick tests passed!\n")


# ============================================================================
# FULL TRAINING
# ============================================================================

def train_policy(
    num_epochs: int = 200,
    k: int = 7,
    hidden_size: int = 96,
    max_steps: int = 200,
    lr: float = 0.003,
    games_per_epoch: int = 32,
    seed: int = 42,
    warmstart_steps: int = 0,
    rl_lr_scale: float = 1.0,
):
    """Train policy network and save best checkpoint."""
    validate_competition_config(k=k, hidden_size=hidden_size)
    print("\n" + "="*70)
    print(f"TRAINING: {num_epochs} epochs, k={k}, hidden_size={hidden_size}")
    print("="*70)

    set_global_seed(seed)

    input_size = 4 + k
    num_actions = 4

    train_env_random = CarEnv(num_sensors=k, max_steps=max_steps, random_starts=True)
    train_env_fixed = CarEnv(num_sensors=k, max_steps=max_steps, random_starts=False)
    eval_env = CarEnv(num_sensors=k, max_steps=max_steps, random_starts=False)

    # Policy network (submitted to competition).
    policy = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions)
    # Value network (training only, NOT submitted).
    value_net = ValueNetwork(input_size=input_size, hidden_size=hidden_size)

    policy_optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr * 2, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(policy_optimizer, T_max=max(num_epochs, 2), eta_min=lr * 0.1)

    print(f"State size: {input_size} | Hidden size: {hidden_size} | Max steps: {max_steps}")
    print(f"Policy optimizer: Adam(lr={lr})")
    print(f"Value optimizer: Adam(lr={lr * 2}) [training only, not submitted]")
    print()

    # Focused tweak: imitation warm-start before RL updates.
    if warmstart_steps > 0:
        warmstart_imitation(policy, train_env_random, policy_optimizer, steps=warmstart_steps)
        for group in policy_optimizer.param_groups:
            group["lr"] = max(1e-5, group["lr"] * rl_lr_scale)
        print(f"[OK] Reduced policy LR for RL fine-tuning: {policy_optimizer.param_groups[0]['lr']:.6f}")

        # Pre-train value network on warm-started policy trajectories.
        print("[..] Pre-training value network on warm-start policy...")
        value_pretrain_epochs = 10
        for ve in range(value_pretrain_epochs):
            pretrain_trajs = []
            for i in range(games_per_epoch):
                env = train_env_random if (i % 2 == 0) else train_env_fixed
                pretrain_trajs.append(generate_trajectory(policy, value_net, env, temperature=1.0))
            # Train value network only (no policy updates).
            value_optimizer.zero_grad()
            value_loss_total = torch.tensor(0.0)
            value_steps = 0
            for traj in pretrain_trajs:
                rewards = [step["reward"] for step in traj]
                values = [step["value"] for step in traj]
                _, returns = compute_gae(rewards, values, gamma=0.99, lam=0.95)
                for i, step in enumerate(traj):
                    state_tensor = torch.tensor(step["state"], dtype=torch.float32)
                    value_pred = value_net(state_tensor)
                    value_loss_total = value_loss_total + (value_pred - returns[i].detach()) ** 2
                    value_steps += 1
            if value_steps > 0:
                (value_loss_total / value_steps).backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                value_optimizer.step()
        print(f"[OK] Value network pre-trained for {value_pretrain_epochs} epochs")

    # Baseline
    baseline = evaluate_policy_bundle(policy, eval_env, episodes=12)
    baseline_distance = score_for_selection(baseline)
    print(f"Baseline (before training):")
    print(f"  Det distance: {baseline['det']['avg_distance']:.4f}")
    print(f"  Det crashes: {baseline['det']['avg_crashes']:.4f}")
    print(f"  Competition distance metric: {baseline_distance:.4f}\n")

    history = []
    best_distance = baseline_distance
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(policy.state_dict())

    gamma = 0.99
    lam = 0.95
    init_entropy_coef = 0.02
    min_entropy_coef = 0.002
    entropy_decay = 0.995
    init_temperature = 1.0
    min_temperature = 1.0
    temp_decay = 1.0
    eval_every = 2
    eval_episodes = 12
    early_stop_patience = 40

    for epoch in range(1, num_epochs + 1):
        entropy_coef = max(min_entropy_coef, init_entropy_coef * (entropy_decay ** (epoch - 1)))
        temperature = max(min_temperature, init_temperature * (temp_decay ** (epoch - 1)))

        # Collect trajectories with temperature-scaled exploration.
        batch_trajectories = []
        for i in range(games_per_epoch):
            env = train_env_random if (i % 2 == 0) else train_env_fixed
            batch_trajectories.append(generate_trajectory(policy, value_net, env, temperature=temperature))

        train_stats = train_actor_critic(
            policy,
            value_net,
            policy_optimizer,
            value_optimizer,
            batch_trajectories,
            gamma=gamma,
            lam=lam,
            entropy_coef=entropy_coef,
            value_coef=0.5,
            grad_clip=0.5,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(policy_optimizer.param_groups[0]["lr"]),
            "entropy_coef": entropy_coef,
            **train_stats,
        }

        # Periodic evaluation
        if epoch % eval_every == 0:
            eval_metrics = evaluate_policy_bundle(policy, eval_env, episodes=eval_episodes)
            distance_metric = score_for_selection(eval_metrics)
            row["det_avg_distance"] = eval_metrics["det"]["avg_distance"]
            row["det_avg_crashes"] = eval_metrics["det"]["avg_crashes"]
            row["det_crash_free_rate"] = eval_metrics["det"]["crash_free_rate"]
            row["competition_distance"] = distance_metric

            improved = distance_metric > (best_distance + 1e-6)
            if improved:
                best_distance = distance_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = copy.deepcopy(policy.state_dict())
                print(f"[OK] Epoch {epoch:3d}/{num_epochs} | distance={distance_metric:.4f} | "
                      f"dist={eval_metrics['det']['avg_distance']:.4f} | "
                      f"crash={eval_metrics['det']['avg_crashes']:.2f} | IMPROVED")
            else:
                epochs_without_improvement += 1
                print(f"  Epoch {epoch:3d}/{num_epochs} | distance={distance_metric:.4f} | "
                      f"dist={eval_metrics['det']['avg_distance']:.4f} | "
                      f"crash={eval_metrics['det']['avg_crashes']:.2f}")

            if epochs_without_improvement >= early_stop_patience:
                print(f"\n[WARN] Early stopping at epoch {epoch} (no improvement for {early_stop_patience} evals)")
                history.append(row)
                break

        history.append(row)

    # Restore best model
    policy.load_state_dict(best_state)

    # Final evaluation
    after = evaluate_policy_bundle(policy, eval_env, episodes=20)
    after_distance = score_for_selection(after)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best competition distance: {best_distance:.4f}")
    print(f"Final competition distance: {after_distance:.4f}")
    print(f"Final distance: {after['det']['avg_distance']:.4f} (d {after['det']['avg_distance'] - baseline['det']['avg_distance']:+.4f})")
    print(f"Final crashes: {after['det']['avg_crashes']:.2f} (d {after['det']['avg_crashes'] - baseline['det']['avg_crashes']:+.2f})")
    print(f"Final crash-free rate: {after['det']['crash_free_rate']:.2f} (d {after['det']['crash_free_rate'] - baseline['det']['crash_free_rate']:+.2f})")

    # Check if we should overwrite existing best (only if we improved).
    model_path = "policy_network_best.pt"
    txt_path = "policy_network_best.txt"
    meta_path = Path(model_path).with_suffix(".meta.json")

    should_save = True
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            old_best = old_meta.get("best_distance", 0.0)
            if best_distance <= old_best:
                print(f"[SKIP] This run ({best_distance:.4f}) did not beat previous best ({old_best:.4f})")
                print(f"[SKIP] Keeping existing weights in {txt_path}")
                should_save = False
            else:
                print(f"[NEW BEST] {best_distance:.4f} > previous {old_best:.4f}")
        except Exception:
            pass

    if should_save:
        save_policy(
            policy,
            model_path,
            metadata={
                "k": k,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_actions": num_actions,
                "max_steps": max_steps,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_distance": best_distance,
            },
        )
        export_weights_txt(policy, txt_path)

    # Quick load test (show current best)
    if Path(model_path).exists():
        loaded_net, loaded_meta = load_policy(model_path, input_size, hidden_size, num_actions)
        test_action = predict_action(loaded_net, eval_env.reset(), deterministic=True)
        print(f"\nCurrent best model metadata: {json.dumps(loaded_meta, indent=2)}")
        print(f"Sample predicted action: {test_action}\n")
    else:
        print("\n[WARN] No saved model found.\n")

    return policy, history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    def _positive_odd_sensors(value: str) -> int:
        parsed = int(value)
        if parsed not in {1, 3, 5, 7, 9}:
            raise argparse.ArgumentTypeError("k must be one of {1, 3, 5, 7, 9}")
        return parsed

    def _hidden_size_limit(value: str) -> int:
        parsed = int(value)
        if parsed < 1 or parsed > 100:
            raise argparse.ArgumentTypeError("hidden-size must be between 1 and 100")
        return parsed

    parser = argparse.ArgumentParser(description="Train and test autonomous vehicle policy")
    parser.add_argument("--quick-test", action="store_true", help="Run quick sanity check only")
    parser.add_argument("--train", action="store_true", help="Run full training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--k", type=_positive_odd_sensors, default=7, help="Number of sensors (must be odd in {1,3,5,7,9})")
    parser.add_argument("--hidden-size", type=_hidden_size_limit, default=96, help="Hidden layer size (competition max: 100)")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--games-per-epoch", type=int, default=32, help="Trajectories collected per training epoch")
    parser.add_argument("--warmstart-steps", type=int, default=0, help="Supervised warm-start steps before RL (0=pure RL)")
    parser.add_argument("--rl-lr-scale", type=float, default=1.0, help="Post-warmstart multiplier for RL learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not args.quick_test and not args.train:
        print("Usage:")
        print("  python train_and_test.py --quick-test          # Quick sanity check")
        print("  python train_and_test.py --train               # Train with defaults")
        print("  python train_and_test.py --train --epochs 100  # Train 100 epochs")
        print("  python train_and_test.py --train --k 5         # Train with 5 sensors")
        sys.exit(0)

    if args.quick_test:
        quick_test()

    if args.train:
        train_policy(
            num_epochs=args.epochs,
            k=args.k,
            hidden_size=args.hidden_size,
            max_steps=args.max_steps,
            lr=args.lr,
            games_per_epoch=args.games_per_epoch,
            warmstart_steps=args.warmstart_steps,
            rl_lr_scale=args.rl_lr_scale,
            seed=args.seed,
        )
