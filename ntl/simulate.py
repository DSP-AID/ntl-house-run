"""Test-pool simulator: the single scoring entry point for test-time logic.

The hidden test has ~4000 passages from unseen prompts. The number of
passages per prompt is unknown, and per-model counts are unequal and differ
between public and private. Any test-time step (grouping/centering, balancing,
assignment) depends on that structure, so we score it on many simulated pools
built from out-of-fold predictions:

  OOF predictions are all made by models that never saw that prompt, so a
  pool may mix rows from every fold (150 prompts, closer to the test's
  size than one fold's 30 prompts). For each scenario k and seed:
    * keep k random passages per held-out prompt ('mix' = k ~ U{1..18})
    * drop passages with a random per-model keep rate (imbalance)
    * run the post-processing on that pool *alone*, using only text and
      model outputs, never prompt_id or labels
    * balanced accuracy on the pool

A change is accepted only if it doesn't lose in any scenario (especially
k=1,2) and its mean gain over scenarios is above the seed/fold noise.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .metrics import bacc

SCENARIOS = ("1", "2", "3", "6", "18", "mix")

# post-processing: (texts, logp[n,18], idx_into_full) -> logp[n,18]
PostProc = Callable[[list[str], np.ndarray, np.ndarray], np.ndarray]


def sample_pool(df: pd.DataFrame, rows: np.ndarray, scenario: str, rng: np.random.Generator,
                keep_min: float = 0.3) -> np.ndarray:
    sub = df.iloc[rows]
    picked = []
    for _, g in sub.groupby("prompt_id"):
        k = rng.integers(1, 19) if scenario == "mix" else int(scenario)
        idx = g.index.values
        picked.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    picked = np.array(sorted(picked))
    # unequal per-model counts
    rates = rng.uniform(keep_min, 1.0, size=18)
    keep = rng.random(len(picked)) < rates[df.loc[picked, "y"].values]
    return picked[keep]


def evaluate(df: pd.DataFrame, logp: np.ndarray, post: PostProc | None = None,
             n_seeds: int = 12, scenarios=SCENARIOS, keep_min: float = 0.3) -> pd.DataFrame:
    """Balanced accuracy per (scenario, seed) on simulated pools.

    ``df`` must have a default RangeIndex aligned with the OOF ``logp`` rows.
    """
    y = df["y"].values
    texts = df["text"].tolist()
    rows = np.arange(len(df))
    rec = []
    for sc in scenarios:
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 * s + SCENARIOS.index(sc))
            pool = sample_pool(df, rows, sc, rng, keep_min)
            lp = logp[pool]
            if post is not None:
                lp = post([texts[i] for i in pool], lp, pool)
            rec.append({"scenario": sc, "seed": s, "n": len(pool),
                        "bacc": bacc(y[pool], lp.argmax(1))})
    return pd.DataFrame(rec)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    t = res.groupby("scenario")["bacc"].agg(["mean", "std"]).reindex(
        [s for s in SCENARIOS if s in set(res.scenario)])
    t.loc["avg"] = [t["mean"].mean(), np.nan]
    return t.round(4)


def compare(df, logp_a, logp_b, post_a: PostProc | None = None, post_b: PostProc | None = None,
            **kw) -> pd.DataFrame:
    """Paired comparison on identical pools: mean delta (b - a) and a 95% interval
    per scenario. Compares two models (logp_a vs logp_b), two post-processings,
    or both."""
    a = evaluate(df, logp_a, post_a, **kw)
    b = evaluate(df, logp_b, post_b, **kw)
    d = a[["scenario", "seed"]].copy()
    d["delta"] = b["bacc"].values - a["bacc"].values
    g = d.groupby("scenario")["delta"]
    out = pd.DataFrame({"a": a.groupby("scenario")["bacc"].mean(), "b": b.groupby("scenario")["bacc"].mean(),
                        "delta": g.mean(), "ci95": 1.96 * g.std() / np.sqrt(g.count())})
    return out.reindex([s for s in SCENARIOS if s in out.index]).round(4)


# --------------------------------------------------------------------------
# Feature-space test-time logic (centering) needs features from ONE model, so
# these pools are built per outer fold: that fold's held-out prompts plus
# ~2000 "distractor" passages from unrelated public-pool prompts, which are
# clustered alongside but never scored. The distractors give clustering a
# realistic number of foreign prompts (the test may hold thousands of prompts).

FeatPost = Callable[[list[str], np.ndarray, np.ndarray, dict], np.ndarray]


def evaluate_perfold(df: pd.DataFrame, fold_data: list[dict], post: FeatPost | None,
                     n_seeds: int = 6, scenarios=SCENARIOS, keep_min: float = 0.3,
                     n_distract: int = 1000) -> pd.DataFrame:
    """fold_data[k] has keys val_idx, logp, feats, dtexts, dlogp, dfeats, model
    (``model`` is passed through to ``post``, e.g. head weights)."""
    y = df["y"].values
    texts = df["text"].tolist()
    rec = []
    for sc in scenarios:
        for k, fd in enumerate(fold_data):
            pos = {r: i for i, r in enumerate(fd["val_idx"])}
            for s in range(n_seeds):
                rng = np.random.default_rng(1000 * s + 17 * k + SCENARIOS.index(sc))
                pool = sample_pool(df, fd["val_idx"], sc, rng, keep_min)
                li = np.array([pos[r] for r in pool])
                di = rng.choice(len(fd["dtexts"]), size=min(n_distract, len(fd["dtexts"])), replace=False)
                T = [texts[r] for r in pool] + [fd["dtexts"][j] for j in di]
                LP = np.concatenate([fd["logp"][li], fd["dlogp"][di]])
                FE = np.concatenate([fd["feats"][li], fd["dfeats"][di]])
                out = LP if post is None else post(T, LP, FE, fd["model"])
                rec.append({"scenario": sc, "fold": k, "seed": s, "n": len(pool),
                            "bacc": bacc(y[pool], out[:len(pool)].argmax(1))})
    return pd.DataFrame(rec)


def compare_perfold(df, fold_data, post_a, post_b, **kw) -> pd.DataFrame:
    a = evaluate_perfold(df, fold_data, post_a, **kw)
    b = evaluate_perfold(df, fold_data, post_b, **kw)
    d = a[["scenario"]].copy()
    d["delta"] = b["bacc"].values - a["bacc"].values
    g = d.groupby("scenario")["delta"]
    out = pd.DataFrame({"a": a.groupby("scenario")["bacc"].mean(), "b": b.groupby("scenario")["bacc"].mean(),
                        "delta": g.mean(), "ci95": 1.96 * g.std() / np.sqrt(g.count())})
    return out.reindex([s for s in SCENARIOS if s in out.index]).round(4)
