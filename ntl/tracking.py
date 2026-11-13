"""Experiment tracking: every CV run appends one row to reports/experiments.csv
and saves its OOF probabilities to artifacts/oof/<run>.npy so runs can be
compared, ensembled and reported without retraining."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .data import ROOT
from .metrics import bacc, fold_spread, log_loss_safe

REPORT = ROOT / "reports" / "experiments.csv"
OOF_DIR = ROOT / "artifacts" / "oof"


def log_run(name: str, y: np.ndarray, oof: np.ndarray, folds: np.ndarray,
            params: dict | None = None, notes: str = "") -> dict:
    """Score OOF probabilities (n, 18), persist them and append a report row."""
    pred = oof.argmax(1)
    mean, std = fold_spread(y, pred, folds)
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M"),
        "run": name,
        "cv_bacc": round(bacc(y, pred), 4),
        "fold_mean": round(mean, 4),
        "fold_std": round(std, 4),
        "logloss": round(log_loss_safe(y, oof), 4),
        "params": json.dumps(params or {}, sort_keys=True),
        "notes": notes,
    }
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_DIR / f"{name}.npy", oof.astype(np.float32))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(REPORT, mode="a", header=not REPORT.exists(), index=False)
    print(f"[{name}] bacc={row['cv_bacc']} folds={mean:.4f}±{std:.4f} logloss={row['logloss']}")
    return row


def load_oof(name: str) -> np.ndarray:
    return np.load(OOF_DIR / f"{name}.npy")
