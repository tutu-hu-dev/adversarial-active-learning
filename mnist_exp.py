"""
Reproduces Section IV (MNIST Handwritten Digits) of:
  Miller et al., "Adversarial Learning: A Critical Review and Active Learning Study"
  IEEE MLSP 2017.

Task: binary classification of MNIST digits '5' vs '6'.
Oracle: SVM trained on the full training set (linearly separable).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from utils import run_trial
from synthetic_exp import STRATEGIES, PLOT_STYLES  # reuse config

N_PER_CLASS_TR = 105   # training pool per class
N_TEST_0       = 456   # '5' test samples (per paper)
N_TEST_1       = 462   # '6' test samples (per paper)
N_QUERIES      = 50
N_TRIALS       = 10


def load_mnist_56():
    """Load MNIST digits 5 and 6. Returns numpy arrays with labels 0 ('5') and 1 ('6')."""
    try:
        from torchvision import datasets as tvds
        train_ds = tvds.MNIST(root='./data', train=True,  download=True)
        test_ds  = tvds.MNIST(root='./data', train=False, download=True)

        def extract(ds):
            X = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
            y = ds.targets.numpy()
            mask = np.isin(y, [5, 6])
            return X[mask], (y[mask] == 6).astype(int)  # 5→0, 6→1

        return extract(train_ds), extract(test_ds)

    except ImportError:
        # Fallback: sklearn's fetch_openml
        from sklearn.datasets import fetch_openml
        print('  (torchvision not found, using fetch_openml — may be slow)')
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        mask = np.isin(y, [5, 6])
        X, y = X[mask], (y[mask] == 6).astype(int)
        split = 11791  # approximate train/test split for MNIST 5&6
        return (X[:split], y[:split]), (X[split:], y[split:])


def build_oracle(X_train, y_train):
    """Oracle: linear SVM trained on all available training data."""
    print('  Training oracle SVM on full training set...')
    clf = SVC(kernel='linear', C=1.0, random_state=0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_train, y_train)
    print(f'  Oracle train accuracy: {acc:.4f}')
    return lambda x: int(clf.predict(x.reshape(1, -1))[0])


def run_experiment(X_pool, y_pool, X_test, y_test, oracle_fn, use_attack):
    results = {}
    label = 'with attack' if use_attack else 'without attack'

    for strat, p_mix, name in STRATEGIES:
        print(f'  [{label}] {name}', flush=True)
        trial_errors = []
        for t in range(N_TRIALS):
            rng = np.random.RandomState(t * 100 + 7)
            errs = run_trial(strat, p_mix, N_QUERIES,
                             X_pool, y_pool, X_test, y_test,
                             oracle_fn, rng, use_attack)
            trial_errors.append(errs)
        results[name] = np.mean(trial_errors, axis=0)

    return results


def plot(res_attack, res_noattack, out='figures/mnist_results.png'):
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
        ax.set_title(f'MNIST (5 vs 6) — {title}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')
    plt.close()


def main():
    print('=== MNIST Experiment ===')
    print('Loading MNIST...')
    (X_train, y_train), (X_test_full, y_test_full) = load_mnist_56()

    # Training pool: 105 per class (as in paper)
    idx0 = np.where(y_train == 0)[0][:N_PER_CLASS_TR]
    idx1 = np.where(y_train == 1)[0][:N_PER_CLASS_TR]
    pool_idx = np.hstack([idx0, idx1])
    X_pool, y_pool = X_train[pool_idx], y_train[pool_idx]

    # Test set: 456 '5' and 462 '6' (per paper)
    idx_t0 = np.where(y_test_full == 0)[0][:N_TEST_0]
    idx_t1 = np.where(y_test_full == 1)[0][:N_TEST_1]
    X_test = np.vstack([X_test_full[idx_t0], X_test_full[idx_t1]])
    y_test = np.hstack([np.zeros(len(idx_t0), int), np.ones(len(idx_t1), int)])

    oracle_fn = build_oracle(X_train, y_train)

    print('Running with attack...')
    res_atk = run_experiment(X_pool, y_pool, X_test, y_test, oracle_fn, use_attack=True)
    print('Running without attack...')
    res_noatk = run_experiment(X_pool, y_pool, X_test, y_test, oracle_fn, use_attack=False)
    plot(res_atk, res_noatk)
    return res_atk, res_noatk


if __name__ == '__main__':
    main()
