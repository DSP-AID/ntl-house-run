"""Competition metric and diagnostics."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# small simulated pools may predict classes absent from y_true; recall is
# averaged over the true classes, which is what the competition metric does
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from .data import LABELS


def bacc(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def log_loss_safe(y_true, proba) -> float:
    p = np.clip(proba[np.arange(len(y_true)), y_true], 1e-9, 1)
    return float(-np.log(p).mean())


def per_class_recall(y_true, y_pred) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return {LABELS[i]: float(r) for i, r in enumerate(rec)}


def fold_spread(y_true, y_pred, folds) -> tuple[float, float]:
    """Mean and std of per-fold balanced accuracy (unseen-prompt variance)."""
    s = [bacc(y_true[folds == k], y_pred[folds == k]) for k in np.unique(folds)]
    return float(np.mean(s)), float(np.std(s))


def prompt_bootstrap(y, pred_a, prompts, pred_b=None, n: int = 2000, seed: int = 0):
    """Bootstrap over PROMPTS (the unit of independence: test prompts are unseen).
    Returns (mean, lo, hi) of balanced accuracy for pred_a, or of the paired
    delta bacc(pred_b) - bacc(pred_a) when pred_b is given (95% percentile CI)."""
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(prompts, return_inverse=True)
    rows = [np.where(inv == k)[0] for k in range(len(keys))]
    out = []
    for _ in range(n):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(keys), len(keys))])
        a = bacc(y[idx], pred_a[idx])
        out.append(a if pred_b is None else bacc(y[idx], pred_b[idx]) - a)
    out = np.array(out)
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
