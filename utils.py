"""
Shared utilities for reproducing Miller et al., IEEE MLSP 2017.
  - Linear SVM with Platt scaling
  - MEU (Max Expected Utility) sample selection
  - Uncertainty sampling
  - Adversarial sample injection
"""

import numpy as np
from sklearn.svm import SVC


def train_svm(X, y):
    return SVC(kernel='linear', probability=True, C=1.0, random_state=0).fit(X, y)


def compute_meu(clf, X_l, y_l, X_u):
    """
    Equation (1): U_i(theta) = sum_{y_i} p(y_i|x_i) * (1/N) *
        [ sum_{j in L+i} p_{+i}(y_j|x_j)  +  sum_{j in U minus i} sum_y p(y|x_j)*p_{+i}(y|x_j) ]

    For each candidate x_i in X_u, compute expected utility if added to training.
    Returns array of shape (len(X_u),).
    """
    N = len(X_l) + len(X_u)
    proba_u = clf.predict_proba(X_u)   # (|U|, 2): column c = p(class=c | x)
    utilities = np.zeros(len(X_u))

    for i in range(len(X_u)):
        xi = X_u[i]
        util = 0.0

        for c in [0, 1]:
            p_c = proba_u[i, c]           # p_theta(c | x_i)

            X_aug = np.vstack([X_l, xi.reshape(1, -1)])
            y_aug = np.hstack([y_l, [c]])
            if len(np.unique(y_aug)) < 2:
                continue

            clf2 = train_svm(X_aug, y_aug)

            # sum_{j in L+i} p_{+i}(true_label_j | x_j)
            proba_aug = clf2.predict_proba(X_aug)
            sum_l = float(np.sum(proba_aug[np.arange(len(y_aug)), y_aug]))

            # sum_{j in U\i} sum_y p_theta(y|x_j) * p_{+i}(y|x_j)
            mask = np.ones(len(X_u), bool)
            mask[i] = False
            sum_u = 0.0
            if mask.any():
                proba_rest_new = clf2.predict_proba(X_u[mask])
                sum_u = float(np.sum(proba_u[mask] * proba_rest_new))

            util += p_c * (sum_l + sum_u) / N

        utilities[i] = util

    return utilities


def uncertainty_idx(clf, X_u):
    """Index of the sample in X_u closest to the decision boundary."""
    proba = clf.predict_proba(X_u)
    return int(np.argmax(1.0 - np.max(proba, axis=1)))


def attacker_inject(clf, X_l, y_l, X_u):
    """
    Projects all samples in Tu ∪ Tl onto the current SVM boundary,
    then returns the projection with minimum expected utility (the most harmful
    candidate for the active learner).
    """
    w = clf.coef_[0]
    b = clf.intercept_[0]
    w_sq = float(np.dot(w, w))

    all_X = np.vstack([X_u, X_l])
    d = (all_X @ w + b) / w_sq          # signed distance scaled by ||w||^2
    X_proj = all_X - d[:, None] * w     # project onto hyperplane: w·x_proj + b = 0

    utils = compute_meu(clf, X_l, y_l, X_proj)
    return X_proj[int(np.argmin(utils))]


def run_trial(strategy, p_mix, n_queries, X_tr, y_tr, X_test, y_test,
              oracle_fn, rng, use_attack):
    """
    Run one trial of adversarial active learning.

    strategy : 'uncertainty' | 'meu' | 'random' | 'mixed'
    p_mix    : probability of choosing MEU when strategy='mixed'
    oracle_fn: callable x -> label (ground-truth label for a sample)
    rng      : numpy RandomState for reproducibility
    """
    # Initial split: 5 per class for Tl (paper also has V for Platt calibration;
    # sklearn handles calibration internally via cross-validation).
    idx0 = np.where(y_tr == 0)[0].copy(); rng.shuffle(idx0)
    idx1 = np.where(y_tr == 1)[0].copy(); rng.shuffle(idx1)

    n_init = 5   # per class
    tl_idx = np.hstack([idx0[:n_init], idx1[:n_init]])
    tu_idx = np.hstack([idx0[n_init * 2:], idx1[n_init * 2:]])

    X_l = X_tr[tl_idx].copy()
    y_l = y_tr[tl_idx].copy()
    X_u = X_tr[tu_idx].copy()

    errors = []
    for _ in range(n_queries):
        clf = train_svm(X_l, y_l)
        errors.append(1.0 - clf.score(X_test, y_test))

        if len(X_u) == 0:
            break

        # --- sample selection ---
        if strategy == 'uncertainty':
            idx = uncertainty_idx(clf, X_u)
        elif strategy == 'meu':
            idx = int(np.argmax(compute_meu(clf, X_l, y_l, X_u)))
        elif strategy == 'random':
            idx = rng.randint(len(X_u))
        else:  # mixed
            if rng.random() < p_mix:
                idx = int(np.argmax(compute_meu(clf, X_l, y_l, X_u)))
            else:
                idx = uncertainty_idx(clf, X_u)

        x_sel = X_u[idx].copy()
        y_sel = oracle_fn(x_sel)
        X_l = np.vstack([X_l, x_sel])
        y_l = np.hstack([y_l, [y_sel]])
        X_u = np.delete(X_u, idx, axis=0)

        # --- attacker injects one adversarial sample ---
        if use_attack:
            x_adv = attacker_inject(clf, X_l, y_l, X_u)
            X_u = np.vstack([X_u, x_adv])

    # pad if queries exhausted unlabeled pool early
    while len(errors) < n_queries:
        errors.append(errors[-1])
    return np.array(errors)
