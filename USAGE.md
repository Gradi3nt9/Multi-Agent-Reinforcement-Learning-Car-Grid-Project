# Training & Testing

## Prerequisites
Make sure you have `numpy` and `torch` installed:
```bash
pip install numpy torch
```

## Quick Test (Verify Setup - 30 seconds)
```bash
python train_and_test.py --quick-test
```
Expected output: All `[OK]` lines = everything works.

## Train Model (20-40 minutes)
```bash
python train_and_test.py --train
```
Saves best model to `policy_network_best.pt`

## Train with Custom Hyperparameters
```bash
python train_and_test.py --train --k 9 --hidden-size 96 --epochs 120 --lr 0.0007 --max-steps 200 --games-per-epoch 32
```

### Options:
- `--k`: Number of sensors (1, 3, 5, 7, 9) - more sensors = more info
- `--hidden-size`: Network hidden layer size (32-100) - bigger = more capacity  
- `--epochs`: Training epochs (30-200) - more = longer training
- `--lr`: Learning rate (0.0001-0.1) - affects convergence speed
- `--max-steps`: Steps per episode (50-300) - longer episodes = harder task
- `--games-per-epoch`: Number of trajectories per epoch (higher = stabler updates, slower runtime)
- `--warmstart-steps`: Imitation warm-start steps before RL (default 4000)
- `--rl-lr-scale`: Multiply LR after warm-start for RL fine-tuning stability (default 0.25)
- `--seed`: Random seed (for reproducibility)

## Competition Compliance Guardrails
- Hidden size is enforced to `<= 100` (competition maximum)
- Sensors are enforced to odd values in `{1, 3, 5, 7, 9}`
- Policy forward pass outputs softmax probabilities over 4 actions
- Inference uses deterministic `argmax` action selection
- Training checkpoint selection uses the competition metric (`avg_distance`) directly
- Training reward uses only per-frame distance (no extra shaping terms)
- Value network is used only during training (Actor-Critic) and NOT submitted

## What Gets Saved
- `policy_network_best.pt` - Weights-only PyTorch `state_dict` (submit this)
- `policy_network_best.meta.json` - Optional local metadata sidecar (do not need to submit)

## Typical Workflow
1. Run quick test to verify setup works
2. Train once with defaults: `python train_and_test.py --train`
3. Try improvements one at a time:
   - Try `--k 9`: `python train_and_test.py --train --k 9`
   - Try `--hidden-size 80`: `python train_and_test.py --train --hidden-size 80`
   - Try `--epochs 120`: `python train_and_test.py --train --epochs 120`
4. Keep changes that improve deterministic average distance, discard ones that don't
5. Submit best `policy_network_best.pt`

## Score Interpretation
- **Higher is better** - final distance traveled is the competition metric
- Competition distance is non-negative by definition
- Baseline (defaults): ~0.92-0.95
- With improvements: can reach ~1.0-1.05+
