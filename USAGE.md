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
Expected output: All ✓ marks = everything works.

## Train Model (20-40 minutes)
```bash
python train_and_test.py --train
```
Saves best model to `policy_network_best.pt`

## Train with Custom Hyperparameters
```bash
python train_and_test.py --train --k 9 --hidden-size 80 --epochs 120 --lr 0.001 --max-steps 200
```

### Options:
- `--k`: Number of sensors (1, 3, 5, 7, 9) - more sensors = more info
- `--hidden-size`: Network hidden layer size (32-100) - bigger = more capacity  
- `--epochs`: Training epochs (30-200) - more = longer training
- `--lr`: Learning rate (0.0001-0.1) - affects convergence speed
- `--max-steps`: Steps per episode (50-300) - longer episodes = harder task
- `--seed`: Random seed (for reproducibility)

## What Gets Saved
- `policy_network_best.pt` - Your trained model checkpoint (submit this)

## Typical Workflow
1. Run quick test to verify setup works
2. Train once with defaults: `python train_and_test.py --train`
3. Try improvements one at a time:
   - Try `--k 9`: `python train_and_test.py --train --k 9`
   - Try `--hidden-size 80`: `python train_and_test.py --train --hidden-size 80`
   - Try `--epochs 120`: `python train_and_test.py --train --epochs 120`
4. Keep changes that improve score, discard ones that don't
5. Submit best `policy_network_best.pt`

## Score Interpretation
- **Higher is better** - final distance traveled is the competition metric
- Baseline (defaults): ~0.92-0.95
- With improvements: can reach ~1.0-1.05+
