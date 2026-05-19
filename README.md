# Adversarial Active Learning

Code implementation of experiments from my co-authored paper (IEEE MLSP 2017).
This repository provides a clean Python/sklearn implementation of the adversarial
active learning experiments described in the paper.

## What This Reproduces

The paper studies active learning (AL) under adversarial attacks, where an attacker injects crafted samples near the classifier boundary to degrade learning. The paper proposes a **mixed sampling strategy** (MEU + uncertainty sampling) as a defense.

**Experiment 1 — Synthetic 2D** (Section III):
- Two Gaussian classes centered at (±2, 0)
- Bayes-optimal boundary: Y axis
- 10 trials × 25 AL queries

**Experiment 2 — MNIST 5 vs 6** (Section IV):
- Binary classification with linear SVM
- Oracle: SVM trained on full training set
- 10 trials × 50 AL queries

## Key Components

### Active Learning Strategies
| Strategy | Description |
|---|---|
| Uncertainty (p=0) | Always select sample nearest to boundary |
| MEU (p=1) | Always select sample maximizing expected utility |
| Mixed (p=0.25/0.5/0.75) | MEU with prob p, uncertainty sampling otherwise |
| Random | Uniform random selection (baseline) |

### Attacker
At each AL round, the attacker:
1. Projects all unlabeled + labeled samples onto the current SVM boundary
2. Injects the projection with **minimum expected utility** into the unlabeled pool

When the AL selects this adversarial sample, the oracle labels it correctly (Bayes-optimal rule), but the feature values still bias the classifier boundary.

### MEU Formula (Eq. 1)
```
U_i(θ) = Σ_{y_i} p_θ(y_i|x_i) · (1/N) · [
    Σ_{j∈L∪i} p_{θ+i}(y_j|x_j)   +
    Σ_{j∈U\i} Σ_y p_θ(y|x_j)·p_{θ+i}(y|x_j)
]
```
Where θ+i denotes model parameters after adding (x_i, y_i) to training.

## Results

**Expected findings (matching paper):**
- Uncertainty sampling is best without attack, but worst under attack
- MEU is robust to attack but converges to a suboptimal boundary (class-biased sampling)
- Mixed strategies (p=0.5–0.75) balance defense and accuracy
- Random sampling is robust to attack but slower to converge without attack

## Usage

```bash
pip install -r requirements.txt

# Fast: synthetic experiment only (~5 min)
python run_all.py synthetic

# Full: both experiments (~60 min for MNIST)
python run_all.py

# Or run individually
python synthetic_exp.py
python mnist_exp.py
```

Output figures are saved to `figures/`.

## File Structure

```
adversarial-active-learning/
├── utils.py          # SVM, MEU computation, attacker, AL loop
├── synthetic_exp.py  # Section III: 2D Gaussian experiment
├── mnist_exp.py      # Section IV: MNIST 5 vs 6 experiment
├── run_all.py        # Entry point
└── requirements.txt
```

## Notes on Runtime

- **Synthetic**: ~5 minutes (10 trials × 25 queries × ~190 Tu × 2 SVM retrains per MEU)
- **MNIST**: ~60 minutes (same structure but 784-dimensional features, 50 queries)

The bottleneck is MEU computation: for each AL query, every candidate in Tu requires 2 SVM retrains (one per putative label) to estimate its expected utility.
