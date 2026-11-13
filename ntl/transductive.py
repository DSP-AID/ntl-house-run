"""Test-time prompt-structure handling.

Passages answering the same prompt share topic content, which dominates
embeddings. Subtracting a per-prompt mean ("centering") removes that topic
and leaves style. In train, prompt_id is known. At test it isn't, so we
estimate each passage's prompt neighbourhood from topic similarity.

All functions take an optional ``groups`` argument. In CV, validation
passages use *estimated* groups only, never the true prompt_id, so the CV
number reflects what happens on the hidden test.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def center_by_groups(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for g in np.unique(groups):
        m = groups == g
        Xc[m] -= X[m].mean(0)
    return Xc


def topic_similarity(texts: list[str]) -> np.ndarray:
    """Cosine similarity of content-word TF-IDF (stopwords removed): a proxy
    for 'answers the same prompt'."""
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1,
                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", lowercase=True)
    T = normalize(v.fit_transform(texts))
    return (T @ T.T).toarray()


def cluster_groups(sim: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Agglomerative (average linkage) clustering on topic similarity; the
    distance threshold decides group count, so variable group sizes are fine."""
    from sklearn.cluster import AgglomerativeClustering
    d = 1 - sim
    np.fill_diagonal(d, 0)
    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                 distance_threshold=1 - threshold)
    return ac.fit_predict(np.clip(d, 0, None))


def knn_center(X: np.ndarray, sim: np.ndarray, k: int = 10, min_sim: float = 0.15,
               alpha: float = 1.0) -> np.ndarray:
    """Soft, cluster-free centering: subtract alpha * weighted mean of each
    passage's topical neighbours (itself included), using only neighbours above
    ``min_sim``. A passage with no neighbours falls back to self + global mean,
    which degrades gracefully when a test prompt has few passages."""
    n = len(X)
    mu = X.mean(0)
    Xc = np.empty_like(X)
    for i in range(n):
        s = sim[i].copy()
        s[i] = 1.0
        nb = np.argsort(-s)[:k + 1]
        nb = nb[s[nb] >= min_sim]
        w = s[nb]
        local = (w[:, None] * X[nb]).sum(0) / w.sum()
        # shrink towards the global mean when the neighbourhood is small
        lam = len(nb) / (len(nb) + 2.0)
        Xc[i] = X[i] - alpha * (lam * local + (1 - lam) * mu)
    return Xc


def cluster_topic(texts: list[str], cut: float = 0.85) -> np.ndarray:
    """Average-linkage clustering on content-word TF-IDF cosine distance, cut at
    a fixed distance (the public notebooks' 0.85; tuned here on the simulator)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = np.clip(1 - topic_similarity(texts), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), "average"), cut, criterion="distance")


def centered_evidence(feats: np.ndarray, groups: np.ndarray, W: np.ndarray, b: np.ndarray,
                      min_size: int = 2, max_size: int = 18, loo: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Log-probs of a centered head on cluster-centered features, plus a mask of
    eligible rows (cluster size within [min_size, max_size])."""
    from scipy.special import log_softmax
    F = feats.astype(np.float32)
    Fc = np.zeros_like(F)
    size = np.bincount(groups)[groups]
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        # leave-one-out: x - mean(others) = n/(n-1) * (x - mean(all))
        Fc[m] = (F[m] - F[m].mean(0)) * (n / (n - 1) if loo and n > 1 else 1.0)
    ok = (size >= min_size) & (size <= max_size)
    return log_softmax(Fc @ W.T + b, axis=1), ok
