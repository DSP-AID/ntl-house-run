"""Log-prob blending with nested-CV weight selection (weights chosen on 4
folds' OOF, scored on the 5th), so the reported blend score is not tuned on
the rows it is scored on."""
from __future__ import annotations

import itertools

import numpy as np

from .metrics import bacc


def blend(logps: list[np.ndarray], w: tuple) -> np.ndarray:
    return sum(wi * lp for wi, lp in zip(w, logps))


def grid(n_members: int, values=(0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    # first member is the anchor with weight 1
    for rest in itertools.product(values, repeat=n_members - 1):
        yield (1.0, *rest)


def fit_weights(logps, y, rows, values=(0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    best, bw = -1, None
    for w in grid(len(logps), values):
        s = bacc(y[rows], blend([lp[rows] for lp in logps], w).argmax(1))
        if s > best + 1e-9:
            best, bw = s, w
    return bw, best


def nested_score(logps, y, folds, values=(0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    """Return (outer-fold OOF predictions, per-fold chosen weights)."""
    pred = np.zeros(len(y), dtype=int)
    chosen = []
    for k in np.unique(folds):
        w, _ = fit_weights(logps, y, folds != k, values)
        va = folds == k
        pred[va] = blend([lp[va] for lp in logps], w).argmax(1)
        chosen.append(w)
    return pred, chosen
