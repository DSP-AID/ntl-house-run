"""Test-time post-processing on a pool of passages (no labels, no prompt_id)."""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax


def sinkhorn(logp: np.ndarray, alpha: float = 1.0, iters: int = 50) -> np.ndarray:
    """Partial class balancing. Column marginals are pushed toward
    (1-alpha)*predicted + alpha*uniform. alpha=0 is a no-op; alpha=1 forces
    equal predicted mass per class, which is wrong if the test mix is unequal,
    so alpha is chosen on imbalanced simulated pools."""
    if alpha <= 0 or len(logp) < 18:
        return logp
    P = np.exp(logp - logp.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    n, K = P.shape
    target = (1 - alpha) * P.sum(0) + alpha * np.full(K, n / K)
    for _ in range(iters):
        P *= target / np.maximum(P.sum(0), 1e-12)
        P /= P.sum(1, keepdims=True)
    return np.log(np.maximum(P, 1e-12))


def temperature(logp: np.ndarray, T: float) -> np.ndarray:
    return log_softmax(logp / T, axis=1)
