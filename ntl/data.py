"""Data loading and leak-free cross-validation splits.

Test prompts never appear in train, so every split here is grouped by
``prompt_id``: a validation fold contains only prompts unseen in training.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("NTL_RAW", ROOT / "data" / "raw"))  # Kaggle: competition input dir

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
LABEL2ID = {m: i for i, m in enumerate(LABELS)}
FAMILY = {m: m.split("-")[0].rstrip("0123456789.") for m in LABELS}


def load_train(path: Path | None = None) -> pd.DataFrame:
    df = _obj(pd.read_csv(path or RAW / "train.csv"))
    df["y"] = df["model"].map(LABEL2ID).astype(int)
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    return _obj(pd.read_csv(path or RAW / "test.csv"))


def _obj(df: pd.DataFrame) -> pd.DataFrame:
    # pandas>=3 uses arrow-backed strings whose .values don't support 2-D indexing
    return df.astype({c: object for c in df.columns if df[c].dtype != "int64"})


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Fold index per row, grouped by prompt (prompts shuffled with ``seed``)."""
    prompts = df["prompt_id"].unique()
    rng = np.random.default_rng(seed)
    perm = {p: i for i, p in enumerate(rng.permutation(prompts))}
    groups = df["prompt_id"].map(perm).values
    folds = np.full(len(df), -1)
    for k, (_, va) in enumerate(GroupKFold(n_splits).split(df, groups=groups)):
        folds[va] = k
    return folds


def subsample_like_test(df: pd.DataFrame, seed: int, keep_min: float = 0.5) -> np.ndarray:
    """Boolean mask that drops passages with a random per-model keep rate.

    The host says test counts per model are unequal and differ between public
    and private. Evaluating on such resamples checks that transductive tricks
    (e.g. one-model-per-prompt assignment) don't rely on a perfectly balanced
    test set.
    """
    rng = np.random.default_rng(seed)
    rates = rng.uniform(keep_min, 1.0, size=len(LABELS))
    return rng.random(len(df)) < rates[df["y"].values]
