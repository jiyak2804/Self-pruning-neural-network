# Self-Pruning Neural Network on CIFAR-10

> **Submission for Tredence Analytics — ML Engineering Challenge**  
> A feed-forward neural network that learns to prune its own weights during training using learnable sigmoid gates and sparsity regularization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiyak2804/Self-pruning-neural-network/blob/main/self_pruning_nn.ipynb)

---

# Problem Statement

Modern neural networks often contain a large number of redundant parameters, making deployment difficult on memory and compute-constrained devices.

Traditional pruning methods are usually:
- performed after training,
- require multiple training stages,
- and involve separate fine-tuning steps.

This project implements a **self-pruning neural network** that learns which connections are important during training itself using learnable gates and sparsity regularization.

---

# Core Idea

Each weight matrix is paired with a learnable gate matrix.

Instead of directly using weights during forward propagation:

```python
gates = sigmoid(gate_scores)
effective_weights = weights * gates
```

The sigmoid gates control whether a connection remains active or becomes suppressed.

- Gate near `1` → important connection survives
- Gate near `0` → connection effectively pruned

Because the gates are differentiable, the network can learn pruning behavior through standard backpropagation.

---

# Sparsity Loss

The total training objective is:

```text
Total Loss = Classification Loss + λ × Sparsity Loss
```

Where:

```text
Sparsity Loss = Σ sigmoid(gate_scores)
```

The sparsity term penalizes open gates and encourages the network to remove unnecessary connections.

---

# Why L1-Style Sparsity Encourages Pruning

The gradient of the sparsity term with respect to each gate score is:

```text
∂L/∂g = λ × sigmoid(g) × (1 − sigmoid(g))
```

This gradient consistently pushes gate values downward toward zero.

Meanwhile:
- the classification loss tries to preserve useful connections,
- while the sparsity penalty tries to eliminate unnecessary ones.

This competition naturally separates:
- important connections,
- from redundant connections.

Unlike L2 regularization, sparsity-based regularization promotes highly sparse solutions where many gates collapse close to zero.

This behavior is conceptually similar to why:
- **LASSO** performs feature selection,
- while **Ridge regression** mainly shrinks weights.

---

# Architecture

| Layer | Type | Input → Output |
|---|---|---|
| fc1 | PrunableLinear + BN + ReLU | 3072 → 1024 |
| fc2 | PrunableLinear + BN + ReLU | 1024 → 512 |
| fc3 | PrunableLinear + BN + ReLU | 512 → 256 |
| fc4 | PrunableLinear | 256 → 10 |

### Training Details

- Dataset: CIFAR-10
- Optimizer: Adam
- Learning Rate: 1e-3
- Scheduler: StepLR
- Epochs: 15
- Framework: PyTorch

---

# Experimental Results

Three experiments were conducted with different sparsity strengths (`λ`) to analyze the sparsity-accuracy tradeoff.

| Lambda (λ) | Test Accuracy | Sparsity Level |
|---|---|---|
| 0.001 | 56.86% | 60.14% |
| 0.01 | 57.31% | 71.24% |
| 0.1 | 56.64% | 72.42% |

---

# Analysis

## λ = 0.001
- Moderate pruning pressure
- Around 60% of connections pruned
- Strong predictive performance maintained

The network successfully removed many redundant connections without significantly harming accuracy.

---

## λ = 0.01
- Higher sparsity pressure
- Over 71% sparsity achieved
- Best overall accuracy observed

This suggests moderate pruning improved generalization by suppressing noisy or less useful connections.

---

## λ = 0.1
- Aggressive pruning pressure
- Slight increase in sparsity
- Small accuracy degradation

At this point the model begins losing useful representational capacity because too many important pathways are weakened.

---

# Gate Distribution Interpretation

A successful pruning run produces a near-bimodal gate distribution:

- Large spike near `0`
  - inactive/pruned connections

- Cluster near `0.5–1`
  - active important connections

As λ increases:
- more gates collapse toward zero,
- network sparsity increases,
- and model capacity decreases.

---

# Project Structure

```text
Self-pruning-neural-network/
├── self_pruning_nn.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Key Components

## `PrunableLinear`
Custom PyTorch layer implementing:
- learnable gate scores,
- differentiable masking,
- and gated weight multiplication.

---

## `sparsity_loss()`
Computes sparsity regularization across all gated layers.

---

## `compute_sparsity()`
Calculates percentage of pruned connections using a threshold-based evaluation metric.

---

# How To Run

## Option 1 — Google Colab

Click the Colab badge above and run all cells sequentially.

---

## Option 2 — Local Setup

```bash
git clone https://github.com/jiyak2804/Self-pruning-neural-network.git
cd Self-pruning-neural-network
pip install -r requirements.txt
jupyter notebook self_pruning_nn.ipynb
```

---

# Requirements

```text
torch
torchvision
numpy
matplotlib
tqdm
```

---

# Evaluation Criteria Coverage

| Requirement | Implementation |
|---|---|
| Correct `PrunableLinear` layer | Implemented with differentiable gates |
| Gradient flow through gates | Verified during training |
| Sparsity regularization | Included in custom training loss |
| Successful self-pruning | Demonstrated experimentally |
| λ tradeoff analysis | Included in report and plots |
| Clean implementation | Modular PyTorch code with comments |

---

# Conclusion

This project demonstrates that neural networks can learn to prune themselves during training using differentiable gating mechanisms and sparsity regularization.

The best tradeoff occurred near:

```text
λ = 0.01
```

where the network achieved:
- 57.31% test accuracy
- 71.24% sparsity

This shows that a large fraction of network connections were redundant and could be removed without major loss in predictive performance.
