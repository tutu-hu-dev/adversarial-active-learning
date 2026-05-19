"""
Reproduces Section III (Synthetic Data) of:
  Miller et al., "Adversarial Learning: A Critical Review and Active Learning Study"
  IEEE MLSP 2017.

Two Gaussian classes: Class 1 ~ N((2,0), I), Class 0 ~ N((-2,0), I).
Bayes-optimal boundary: Y axis (x[0] = 0).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import run_trial

N_PER_CLASS = 105   # training pool size per class (Tr)
N_TEST      = 200   # test set size per class
N_QUERIES   = 25
N_TRIALS    = 10
SEED        = 42

STRATEGIES = [
    ('uncertainty', 0.00, 'uncertainty (p=0)'),
    ('mixed',       0.25, 'mix (p=0.25)'),
    ('mixed',       0.50, 'mix (p=0.5)'),
    ('mixed',       0.75, 'mix (p=0.75)'),
    ('meu',         1.00, 'MEU (p=1)'),
    ('random',      0.00, 'random (baseline)'),
]

PLOT_STYLES = {
    'uncertainty (p=0)': dict(color='steelblue',   ls='-',  marker=None),
    'mix (p=0.25)':      dict(color='green',        ls='-',  marker='+',  ms=6),
    'mix (p=0.5)':       dict(color='darkorange',   ls='-',  marker='^',  ms=5),
    'mix (p=0.75)':      dict(color='purple',       ls='-',  marker='x',  ms=6),
    'MEU (p=1)':         dict(color='crimson',      ls='-',  marker=None),
    'random (baseline)': dict(color='black',        ls='--', marker=None),
}


def generate_data(seed=SEED):
    rng = np.random.RandomState(seed)
    c1 = rng.randn(N_PER_CLASS + N_TEST, 2) + [2.0, 0.0]
    c0 = rng.randn(N_PER_CLASS + N_TEST, 2) + [-2.0, 0.0]

    X_tr   = np.vstack([c1[:N_PER_CLASS],  c0[:N_PER_CLASS]])
    y_tr   = np.array([1] * N_PER_CLASS + [0] * N_PER_CLASS)
    X_test = np.vstack([c1[N_PER_CLASS:],  c0[N_PER_CLASS:]])
    y_test = np.array([1] * N_TEST + [0] * N_TEST)
    return X_tr, y_tr, X_test, y_test


def bayes_oracle(x):
    """Optimal Bayes rule for this dataset: y = 1 iff x[0] > 0."""
    return 1 if x[0] > 0 else 0


def run_experiment(use_attack):
    X_tr, y_tr, X_test, y_test = generate_data()
    results = {}
    label = 'with attack' if use_attack else 'without attack'

    for strat, p_mix, name in STRATEGIES:
        print(f'  [{label}] {name}', flush=True)
        trial_errors = []
        for t in range(N_TRIALS):
            rng = np.random.RandomState(t * 100 + 7)
            errs = run_trial(strat, p_mix, N_QUERIES,
                             X_tr, y_tr, X_test, y_test,
                             bayes_oracle, rng, use_attack)
            trial_errors.append(errs)
        results[name] = np.mean(trial_errors, axis=0)

    return results


def plot(res_attack, res_noattack, out='figures/synthetic_results.png'):
    import os; os.makedirs('figures', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(1, N_QUERIES + 1)

    for ax, results, title in [
        (axes[0], res_attack,   'With Attack'),
        (axes[1], res_noattack, 'Without Attack'),
    ]:
        for _, _, name in STRATEGIES:
            kw = {k: v for k, v in PLOT_STYLES[name].items()}
            ax.plot(xs, results[name], label=name, **kw)
        ax.set_xlabel('Number of queries')
        ax.set_ylabel('Test error')
        ax.set_title(f'Synthetic dataset — {title}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')
    plt.close()


def main():
    print('=== Synthetic Experiment ===')
    print('Running with attack...')
    res_atk = run_experiment(use_attack=True)
    print('Running without attack...')
    res_noatk = run_experiment(use_attack=False)
    plot(res_atk, res_noatk)
    return res_atk, res_noatk


if __name__ == '__main__':
    main()
