"""
Standalone training and testing script for autonomous vehicle policy.
All code inline—no module dependencies. Run: python train_and_test.py
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


# ============================================================================
# POLICY NETWORK
# ============================================================================

class PolicyNetwork(nn.Module):
    """Fully-connected policy with 2 hidden layers (ReLU)."""

    def __init__(self, input_size: int, hidden_size: int, num_actions: int = 4):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output_layer(x)


# ============================================================================
# CAR ENVIRONMENT
# ============================================================================

class CarEnv:
    """
    Continuous state-space car environment in [0,1]x[0,1] unit square.
    State: [x, y, vx, vy, s1, s2, ..., sk]
    Actions: 0=turn_left, 1=turn_right, 2=speed_up, 3=no_action
    """

    def __init__(self, num_sensors: int = 5, max_steps: int = 500):
        assert num_sensors in {1, 3, 5, 7, 9}, "num_sensors must be one of {1,3,5,7,9}"
        self.num_sensors = num_sensors
        self.max_steps = max_steps

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
    """Shape reward to encourage safe, fast travel."""
    min_sensor = float(np.min(next_state[4:]))
    wall_penalty = 0.04 * max(0.0, (0.25 - min_sensor) / 0.25)
    crash_penalty = 0.25 if info["crashed"] else 0.0
    speed_bonus = 0.03 * info["speed"]
    survival_bonus = 0.002
    shaped = (3.0 * raw_reward) + speed_bonus + survival_bonus - wall_penalty - crash_penalty
    return float(shaped)


# ============================================================================
# TRAJECTORY GENERATION & TRAINING
# ============================================================================

def generate_trajectory(network: PolicyNetwork, env: CarEnv) -> List[Dict]:
    """Collect one full trajectory using the policy."""
    trajectory = []
    state = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits = network(state_tensor)

        action_dist = torch.distributions.Categorical(logits=logits)
        action_tensor = action_dist.sample()
        action = int(action_tensor.item())
        log_prob = action_dist.log_prob(action_tensor)
        entropy = action_dist.entropy()

        next_state, raw_reward, done, info = env.step(action)
        reward = shape_reward(raw_reward, info, next_state)

        trajectory.append({
            "state": state.copy(),
            "action": action,
            "reward": reward,
            "raw_reward": raw_reward,
            "log_prob": log_prob,
            "entropy": entropy,
            "info": info,
        })
        state = next_state

    return trajectory


def compute_discounted_returns(rewards: List[float], gamma: float = 0.97) -> torch.Tensor:
    """Compute per-time-step discounted returns."""
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def train_reinforce(
    network: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    trajectories: List[List[Dict]],
    gamma: float = 0.97,
    normalize_advantages: bool = True,
    entropy_coef: float = 0.01,
    grad_clip: float = 1.0,
) -> Dict:
    """Execute one REINFORCE training step."""
    optimizer.zero_grad()

    total_policy_loss = torch.tensor(0.0)
    total_entropy = torch.tensor(0.0)
    total_steps = 0
    all_returns: List[float] = []

    for trajectory in trajectories:
        rewards = [step["reward"] for step in trajectory]
        returns = compute_discounted_returns(rewards, gamma=gamma)
        all_returns.extend(returns.tolist())

        advantages = returns - returns.mean()
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for step, adv in zip(trajectory, advantages):
            total_policy_loss = total_policy_loss + (-(step["log_prob"] * adv))
            total_entropy = total_entropy + step["entropy"]

        total_steps += len(trajectory)

    if total_steps == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "entropy": 0.0, "return_mean": 0.0, "return_std": 0.0, "grad_norm": 0.0}

    avg_policy_loss = total_policy_loss / total_steps
    avg_entropy = total_entropy / total_steps
    loss = avg_policy_loss - (entropy_coef * avg_entropy)

    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip).item())
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(avg_policy_loss.item()),
        "entropy": float(avg_entropy.item()),
        "return_mean": float(np.mean(all_returns)) if all_returns else 0.0,
        "return_std": float(np.std(all_returns)) if all_returns else 0.0,
        "grad_norm": grad_norm,
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
                logits = network(state_tensor)
                if deterministic:
                    action = int(torch.argmax(logits).item())
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
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
    """Compute selection score (optimize for distance, minimize crashes)."""
    det = metrics["det"]
    return (det["avg_distance"] * 3.0) - (det["avg_crashes"] * 1.0) + (det["crash_free_rate"] * 2.0)


# ============================================================================
# SAVE / LOAD
# ============================================================================

def save_policy(network: PolicyNetwork, path: str, metadata: Dict = None) -> None:
    """Save policy network with metadata."""
    payload = {
        "state_dict": network.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    print(f"✓ Saved policy to {path}")


def load_policy(
    path: str, input_size: int, hidden_size: int, num_actions: int = 4, map_location: str = "cpu"
) -> Tuple[PolicyNetwork, Dict]:
    """Load policy network and metadata."""
    payload = torch.load(path, map_location=map_location)
    model = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("metadata", {})


def predict_action(network: PolicyNetwork, state: np.ndarray, deterministic: bool = True) -> int:
    """Predict deterministic action from state."""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        logits = network(state_tensor)
        if deterministic:
            return int(torch.argmax(logits).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())


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
    print(f"✓ Initial state shape: {state0.shape} (expected {(input_size,)})")
    assert state0.shape == (input_size,), "State shape mismatch!"

    # Test network forward pass
    state_tensor = torch.tensor(state0, dtype=torch.float32)
    with torch.no_grad():
        logits = net(state_tensor)
        probs = torch.softmax(logits, dim=0)
    print(f"✓ Network output shape: {logits.shape} (expected (4,))")
    assert logits.shape == (4,), "Logits shape mismatch!"
    print(f"✓ Action probabilities: {probs.numpy().round(3)}")
    assert abs(float(probs.sum()) - 1.0) < 1e-5, "Probabilities don't sum to 1!"

    # Test trajectory generation
    traj = generate_trajectory(net, train_env)
    traj_raw_reward = sum(step["raw_reward"] for step in traj)
    traj_crashes = traj[-1]["info"]["crash_count"] if len(traj) > 0 else 0
    print(f"✓ Trajectory length: {len(traj)}")
    print(f"✓ Trajectory raw distance: {traj_raw_reward:.4f}")
    print(f"✓ Trajectory crashes: {traj_crashes}")

    # Test evaluation
    quick_bundle = evaluate_policy_bundle(net, eval_env, episodes=4)
    print(f"✓ Deterministic metrics: {quick_bundle['det']}")
    print(f"✓ Stochastic metrics: {quick_bundle['stoch']}")

    print("\n✓ All quick tests passed!\n")


# ============================================================================
# FULL TRAINING
# ============================================================================

def train_policy(
    num_epochs: int = 80,
    k: int = 7,
    hidden_size: int = 64,
    max_steps: int = 120,
    lr: float = 0.003,
    games_per_epoch: int = 16,
    seed: int = 42,
):
    """Train policy network and save best checkpoint."""
    print("\n" + "="*70)
    print(f"TRAINING: {num_epochs} epochs, k={k}, hidden_size={hidden_size}")
    print("="*70)

    set_global_seed(seed)

    input_size = 4 + k
    num_actions = 4

    train_env = CarEnv(num_sensors=k, max_steps=max_steps)
    eval_env = CarEnv(num_sensors=k, max_steps=max_steps)

    net = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    print(f"State size: {input_size} | Hidden size: {hidden_size} | Max steps: {max_steps}")
    print(f"Optimizer: Adam(lr={lr}, weight_decay=1e-5)")
    print()

    # Baseline
    baseline = evaluate_policy_bundle(net, eval_env, episodes=12)
    baseline_score = score_for_selection(baseline)
    print(f"Baseline (before training):")
    print(f"  Det distance: {baseline['det']['avg_distance']:.4f}")
    print(f"  Det crashes: {baseline['det']['avg_crashes']:.4f}")
    print(f"  Score: {baseline_score:.4f}\n")

    history = []
    best_score = baseline_score
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(net.state_dict())

    gamma = 0.97
    init_entropy_coef = 0.02
    min_entropy_coef = 0.002
    entropy_decay = 0.97
    eval_every = 5
    eval_episodes = 12
    early_stop_patience = 8

    for epoch in range(1, num_epochs + 1):
        entropy_coef = max(min_entropy_coef, init_entropy_coef * (entropy_decay ** (epoch - 1)))

        # Collect trajectories
        batch_trajectories = [generate_trajectory(net, train_env) for _ in range(games_per_epoch)]
        train_stats = train_reinforce(
            net,
            optimizer,
            batch_trajectories,
            gamma=gamma,
            normalize_advantages=True,
            entropy_coef=entropy_coef,
            grad_clip=1.0,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "entropy_coef": entropy_coef,
            **train_stats,
        }

        # Periodic evaluation
        if epoch % eval_every == 0:
            eval_metrics = evaluate_policy_bundle(net, eval_env, episodes=eval_episodes)
            score = score_for_selection(eval_metrics)
            row["det_avg_distance"] = eval_metrics["det"]["avg_distance"]
            row["det_avg_crashes"] = eval_metrics["det"]["avg_crashes"]
            row["det_crash_free_rate"] = eval_metrics["det"]["crash_free_rate"]
            row["score"] = score

            improved = score > (best_score + 1e-6)
            if improved:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = copy.deepcopy(net.state_dict())
                print(f"✓ Epoch {epoch:3d}/{num_epochs} | score={score:.4f} | "
                      f"dist={eval_metrics['det']['avg_distance']:.4f} | "
                      f"crash={eval_metrics['det']['avg_crashes']:.2f} | IMPROVED")
            else:
                epochs_without_improvement += 1
                print(f"  Epoch {epoch:3d}/{num_epochs} | score={score:.4f} | "
                      f"dist={eval_metrics['det']['avg_distance']:.4f} | "
                      f"crash={eval_metrics['det']['avg_crashes']:.2f}")

            if epochs_without_improvement >= early_stop_patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {early_stop_patience} evals)")
                history.append(row)
                break

        history.append(row)

    # Restore best model
    net.load_state_dict(best_state)

    # Final evaluation
    after = evaluate_policy_bundle(net, eval_env, episodes=20)
    after_score = score_for_selection(after)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.4f}")
    print(f"Final distance: {after['det']['avg_distance']:.4f} (Δ {after['det']['avg_distance'] - baseline['det']['avg_distance']:+.4f})")
    print(f"Final crashes: {after['det']['avg_crashes']:.2f} (Δ {after['det']['avg_crashes'] - baseline['det']['avg_crashes']:+.2f})")
    print(f"Final crash-free rate: {after['det']['crash_free_rate']:.2f} (Δ {after['det']['crash_free_rate'] - baseline['det']['crash_free_rate']:+.2f})")

    # Save
    model_path = "policy_network_best.pt"
    save_policy(
        net,
        model_path,
        metadata={
            "k": k,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_actions": num_actions,
            "max_steps": max_steps,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
    )

    # Quick load test
    loaded_net, loaded_meta = load_policy(model_path, input_size, hidden_size, num_actions)
    test_action = predict_action(loaded_net, eval_env.reset(), deterministic=True)
    print(f"\nLoaded model metadata: {json.dumps(loaded_meta, indent=2)}")
    print(f"Sample predicted action: {test_action}\n")

    return net, history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and test autonomous vehicle policy")
    parser.add_argument("--quick-test", action="store_true", help="Run quick sanity check only")
    parser.add_argument("--train", action="store_true", help="Run full training")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--k", type=int, default=7, help="Number of sensors (must be odd in {1,3,5,7,9})")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size (≤100)")
    parser.add_argument("--max-steps", type=int, default=120, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
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
            seed=args.seed,
        )
