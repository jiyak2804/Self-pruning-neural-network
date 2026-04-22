# Self-Pruning Neural Network on CIFAR-10

> **Submission for Tredence Analytics — ML Engineering Challenge**  
> *A feed-forward network that learns to prune its own weights during training using learnable sigmoid gates and L1 sparsity regularization.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyak2804/Self-pruning-neural-network/blob/main/Trendence_task.ipynb)

---

## Problem Statement

Deploying large neural networks in production is constrained by memory and compute budgets. Standard pruning is a **post-training** step. This project implements a network that **prunes itself during training** — no separate pruning phase required.

---

## How It Works

### Core Idea — Learnable Gates

Every weight `w_ij` in the network has a paired learnable scalar `gate_score_ij`. During the forward pass:

```
gates        = sigmoid(gate_scores)        # squash to (0, 1)
pruned_weight = weight * gates             # element-wise gate
output        = pruned_weight @ x + bias   # standard linear op
```

When a gate collapses to `~0`, the corresponding weight is effectively **removed from the network**. Gradients flow through both `weight` and `gate_scores` via standard autograd.

### Loss Function

```
Total Loss = CrossEntropy(logits, labels) + λ × SparsityLoss
```

Where `SparsityLoss = Σ sigmoid(gate_scores)` — the L1 norm of all gate values across every layer.

### Why L1 Encourages Sparsity

The L1 penalty applies a **constant downward gradient** (`-λ`) to every gate, regardless of its current value. This is unlike L2, whose gradient shrinks near zero and never fully eliminates weights. L1's uniform pressure can drive gates to **exactly zero** — the same reason LASSO regression produces sparser solutions than Ridge. The classification loss pushes back, keeping important connections alive. The result: unimportant connections are pruned, important ones survive.

---

## Architecture

| Layer | Type | In → Out |
|-------|------|----------|
| fc1 | PrunableLinear + BN + ReLU | 3072 → 1024 |
| fc2 | PrunableLinear + BN + ReLU | 1024 → 512 |
| fc3 | PrunableLinear + BN + ReLU | 512 → 256 |
| fc4 | PrunableLinear (output) | 256 → 10 |

**Total prunable weights:** ~3.8M  
**Optimizer:** Adam (lr=1e-3) with StepLR scheduler  
**Dataset:** CIFAR-10 (50k train / 10k test)

---

## Results

Trained for 15 epochs across three values of λ to demonstrate the sparsity-accuracy trade-off:

| Lambda (λ) | Test Accuracy | Sparsity Level |
|------------|:------------:|:--------------:|
| 1e-5 (low) | ~52% | ~15% |
| 1e-4 (medium) | ~48% | ~55% |
| 1e-3 (high) | ~38% | ~90% |

**Key finding:** A higher λ aggressively prunes the network (up to 90% of weights removed) at the cost of accuracy. A medium λ offers the best trade-off — over half the weights pruned with only a modest accuracy drop.

### Gate Distribution (Best Model — λ = 1e-4)
A successful pruning run produces a **bimodal distribution** of gate values:
- Large spike near `0` → pruned (inactive) connections
- Cluster near `0.5–1.0` → active, important connections

---

## Project Structure

```
Self-pruning-neural-network/
├── Trendence_task.ipynb     # Full implementation — run end to end
├── README.md                # This file
├── requirements.txt         # Dependencies
└── .gitignore
```

---

## Implementation Highlights

- **`PrunableLinear`** — custom `nn.Module` replacing `nn.Linear`, with `gate_scores` registered as a learnable parameter
- **`sparsity_loss()`** — differentiable L1 penalty computed over all `PrunableLinear` layers
- **`compute_sparsity()`** — evaluation-only hard threshold (`gate < 0.01`) to report pruning percentage
- **Three λ experiments** — full training runs with curves, gate distributions, and trade-off plots

---

## How to Run

### Option 1 — Google Colab (Recommended)
Click the **Open in Colab** badge above. Run all cells top to bottom. CIFAR-10 downloads automatically.

### Option 2 — Local

```bash
git clone https://github.com/jiyak2804/Self-pruning-neural-network.git
cd Self-pruning-neural-network
pip install -r requirements.txt
jupyter notebook Trendence_task.ipynb
```

---

## Requirements

```
torch
torchvision
matplotlib
numpy
tqdm
```

---

## Evaluation Criteria Coverage

| Criterion | Where addressed |
|-----------|----------------|
| Correct `PrunableLinear` with gradient flow | Cell 3 — includes assertion check |
| Custom sparsity loss in training loop | Cell 5 & 6 |
| Results showing successful self-pruning | Cell 8 — results table + 3 plots |
| λ trade-off analysis | Cell 10 — report section |
| Clean, readable code | Docstrings and comments throughout |
