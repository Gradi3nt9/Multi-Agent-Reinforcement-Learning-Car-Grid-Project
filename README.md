# Autonomous Vehicle Policy Learning (Reinforcement Learning)

**Author:** Vamsi Challa, Abdul Ghani
**Date:** 2026-02-28

------------------------------------------------------------------------

## 📌 Project Overview

This project implements a deterministic Markov policy network trained
using **Reinforcement Learning (RL)** to control an autonomous vehicle
in a continuous state-space environment.

The objective is to maximize total distance traveled within a fixed
number of frames while minimizing crashes and maintaining high velocity.

The trained policy outputs one of four discrete actions at each
timestep:

-   `Turn Left`
-   `Turn Right`
-   `Speed Up`
-   `No Action`

The policy network is implemented as a fully connected neural network
with:

-   Two hidden layers (ReLU activation)
-   Maximum 100 units per hidden layer
-   Output layer with 4 units (Softmax activation)

------------------------------------------------------------------------

## 🎯 Competition Objective

Your score is defined as:

> **Total Distance Traveled Across Frames**

Distance is computed as the sum of Euclidean distances between
consecutive vehicle positions.

### Important Environment Rules

-   The car operates within `[0, 1] x [0, 1]`
-   Speed is clamped between `0.001` and `0.1`
-   `Speed Up` increases speed by **2.5% per frame**
-   `Turn Left` / `Turn Right`:
    -   Decrease speed by **1.25% per frame**
    -   Change angle by `0.01` radians
-   Crashes reduce speed to **12.5% of current speed**
-   No direct crash penalty, only indirect via speed reduction

------------------------------------------------------------------------

## 🧠 State Representation

Each state includes:

-   Position: `(x, y)`
-   Velocity: `(vx, vy)`
-   Sensor distances: `(s1, s2, ..., sk)`

Where: - `k` = number of front-facing sensors - `k` must be an **odd
integer** - Sensors are ordered from left to right

Total input dimension:

    4 + k

------------------------------------------------------------------------

## 🏗️ Neural Network Architecture

Example architecture:

    Input Layer (4 + k)
            ↓
    Hidden Layer 1 (ReLU, ≤100 units)
            ↓
    Hidden Layer 2 (ReLU, ≤100 units)
            ↓
    Output Layer (4 units, Softmax)

The output corresponds to action probabilities. The deterministic policy
selects:

    argmax(softmax_output)

------------------------------------------------------------------------

## 🤖 Reinforcement Learning Approach

This implementation uses:

-   Policy-based learning (e.g., REINFORCE or PPO)
-   Deterministic inference policy
-   Reward shaping focused on:
    -   Positive reward proportional to distance traveled
    -   Implicit penalty via crash-induced speed reduction

### Reward Function Design

A suggested reward:

    r_t = distance_t

Optional shaping:

    r_t = distance_t - crash_penalty

Since crashes reduce speed, the agent naturally learns to avoid
collisions.

------------------------------------------------------------------------

## 🔁 Training Strategy

### 1️⃣ Environment Interaction

-   Collect trajectories via simulation
-   Store (state, action, reward, next_state)

### 2️⃣ Optimization

-   Compute discounted returns
-   Update policy using gradient ascent on expected return

### 3️⃣ Game-Theoretic Considerations

Since multiple students compete:

-   Train against different opponent policies
-   Randomize training seeds
-   Consider adversarial vehicle strategy (optional second policy)
-   Curriculum learning with increasing difficulty

------------------------------------------------------------------------

## 📊 Evaluation

Performance metrics:

-   Total distance traveled
-   Average speed
-   Crash frequency
-   Stability of trajectory

Model selection based on: - Highest validation environment score -
Robustness across random seeds

------------------------------------------------------------------------

## 📁 Repository Structure

    .
    ├── env/                    # Simulation environment
    ├── models/                 # Policy network implementations
    ├── training/               # Training loops and RL algorithms
    ├── evaluation/             # Evaluation scripts
    ├── utils/                  # Helper functions
    ├── checkpoints/            # Saved model weights
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 🚀 Running the Project

### Install Dependencies

    pip install -r requirements.txt

### Train

    python train.py

### Evaluate

    python evaluate.py --model checkpoints/best_model.pt

------------------------------------------------------------------------

## 🧪 Optional Second Policy (Adversarial Strategy)

The competition allows submission of two policies.

Possible second-policy strategy:

-   Aggressive steering
-   Intentional collision behavior
-   Blocking high-speed lanes

The maximum score between the two submissions determines ranking.

------------------------------------------------------------------------

## 🧮 Hyperparameter Suggestions

  Parameter           Suggested Value
  ------------------- -----------------
  Learning Rate       1e-3
  Gamma               0.99
  Batch Size          2048 steps
  Hidden Units        64--100
  Optimizer           Adam
  Training Episodes   10k+

------------------------------------------------------------------------

## 🔬 Future Improvements

-   PPO with clipping
-   Entropy regularization
-   Curriculum training
-   Self-play
-   Sensor noise robustness training
-   Ensemble policy selection

------------------------------------------------------------------------

## 📌 Final Submission Requirements

-   Deterministic policy network
-   Two hidden layers (≤100 units each)
-   Softmax output layer (4 units)
-   Input dimension = 4 + k

------------------------------------------------------------------------

## 🏁 Summary

This project applies Reinforcement Learning to learn a high-speed,
crash-minimizing autonomous driving policy in a competitive multi-agent
environment. Careful reward shaping, robust training, and game-theoretic
considerations are critical for achieving top performance.

------------------------------------------------------------------------

