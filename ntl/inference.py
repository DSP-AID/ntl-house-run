"""End-to-end test-time pipeline (shared by local dry-runs and the Kaggle notebook).

Each member produces log-probs [n,18] (and optional features). Members are
combined with a weighted log-prob sum, then optional centered-head evidence
on topic clusters, then optional partial balancing. Every stage is wrapped so
that a failure falls back to the remaining members, and the TF-IDF member is
the last resort. Hidden reruns give no traceback, so a crash costs a
submission.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS
from .postproc import sinkhorn
from .transductive import centered_evidence, cluster_topic


@dataclass
class PipelineCfg:
    lora_members: list = field(default_factory=list)   # [{"path":..., "base":..., "w":1.0, "center_w":0.0}]
    enc_members: list = field(default_factory=list)    # [{"path":..., "w":...}] encoder_clf checkpoints (encoder/ + head.safetensors)
    tfidf_w: float = 0.0
    tfidf_C: float = 0.5
    tfidf_pool_w: float = 0.3      # per-row weight of a (class-balanced) pool row vs a train row
    cluster_cut: float = 0.8
    sinkhorn_alpha: float = 0.0
    center_loo: bool = True        # leave-one-out centering (sim: >= plain in every scenario)
    atlas: dict | None = None      # {"path":..., "base":..., "w":...}; T read from its meta.json
    batch_size: int = 8   # output_hidden_states on a 14B trunk is memory-hungry on T4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_lora_member(spec: dict, texts: list[str], batch_size: int) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """One output per adapter: the main checkpoint, then every fold adapter in
    spec["folds"] (same base, swapped in place -> the base loads once)."""
    import torch

    from .models.lora_clf import InferenceModel, extract
    meta = json.loads((Path(spec["path"]) / "meta.json").read_text())
    assert meta["labels"] == LABELS, "label order mismatch"
    m = InferenceModel(Path(spec["path"]), base=spec.get("base"))
    from collections import Counter
    log(f"  device_map: {dict(Counter(str(v) for v in getattr(m.trunk, 'hf_device_map', {}).values()))}")
    outs = []
    deadline = spec.get("fold_deadline")  # epoch s: no extra fold adapter may finish after this
    last = None
    for i, ck in enumerate([spec["path"]] + list(spec.get("folds", []))):
        if i > 0 and deadline is not None and time.time() + 1.2 * last > deadline:
            log(f"  time budget: skipping {len(spec['folds']) - i + 1} remaining fold adapter(s)")
            break
        if i > 0:
            m.swap(Path(ck))
        t0 = time.time()
        lp, feats = extract(m, texts, batch_size=batch_size)
        heads = {h: (getattr(m, h).weight.detach().float().cpu().numpy(),
                     getattr(m, h).bias.detach().float().cpu().numpy()) for h in ("plain", "centered")}
        outs.append((lp, feats, heads))
        last = time.time() - t0
        log(f"  adapter {Path(ck).name}: {torch.cuda.get_device_name(0)} dtype={m.dtype} rows={len(texts)} "
            f"extract={time.time() - t0:.0f}s fp32_fallback_batches={m.n_fallback}")
    del m
    import gc
    gc.collect()  # accelerate's dispatch hooks form reference cycles; without this the
    torch.cuda.empty_cache()  # previous member's weights stay on the GPU and the next one OOMs
    log("  after cleanup: " + ", ".join(f"cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.2f} GiB"
                                         for i in range(torch.cuda.device_count())))
    return outs


def predict(test: pd.DataFrame, cfg: PipelineCfg, train: pd.DataFrame | None = None,
            pool: pd.DataFrame | None = None, members_out: dict | None = None) -> np.ndarray:
    """Blend log-probs for ``test``. If ``members_out`` is a dict, each member's own
    plain log-probs are stored in it (for the commit-run sanity check)."""
    texts = test["text"].tolist()
    total = np.zeros((len(texts), len(LABELS)))
    used = 0.0
    groups = None
    try:
        groups = cluster_topic(texts, cfg.cluster_cut) if len(texts) > 1 else np.zeros(1, int)
        log(f"clusters: {len(np.unique(groups))} for {len(texts)} passages")
    except Exception:
        traceback.print_exc()
    for spec in cfg.lora_members:
        try:
            t0 = time.time()
            if spec.get("pub_l24"):  # public LoRA24 checkpoints (models/pub_l24.py): plain head, fold-averaged
                from .models.pub_l24 import predict_adapters
                lp = np.mean(predict_adapters(spec["base"], spec["adapters"], texts, spec["keep"], log=log), 0)
                if members_out is not None:
                    members_out[spec["path"]] = lp
                total += spec["w"] * lp
                used += spec["w"]
                log(f"member {spec['path']} ok in {time.time() - t0:.0f}s")
                continue
            outs = run_lora_member(spec, texts, cfg.batch_size)
            # average over adapters (full fit + fold adapters): plain log-probs and each
            # adapter's own centered evidence (each adapter has its own feature space)
            lp = np.mean([o[0] for o in outs], 0)
            if members_out is not None:
                members_out[spec["path"]] = lp
            ev = spec["w"] * lp
            if spec.get("center_w", 0) > 0 and groups is not None:
                cl = []
                for _, feats, heads in outs:
                    clp, ok = centered_evidence(feats, groups, *heads["centered"], loo=cfg.center_loo)
                    cl.append(clp)
                ev[ok] += spec["center_w"] * np.mean(cl, 0)[ok]
            total += ev
            used += spec["w"]
            log(f"member {spec['path']} ok in {time.time() - t0:.0f}s")
        except Exception:
            traceback.print_exc()
            log(f"member {spec['path']} FAILED - skipped")
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()  # a failed load must not starve the next member
    for spec in cfg.enc_members:
        try:
            t0 = time.time()
            import torch

            from .models.encoder_clf import EncCfg, EncClassifier
            from .models.encoder_clf import extract as enc_extract
            meta = json.loads((Path(spec["path"]) / "meta.json").read_text())
            assert meta["labels"] == LABELS, "label order mismatch"
            m = EncClassifier(EncCfg(**meta["cfg"]), init=Path(spec["path"]))
            lp, _ = enc_extract(m, texts, batch_size=spec.get("batch_size", 16))
            del m
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if members_out is not None:
                members_out[spec["path"]] = lp
            total += spec["w"] * lp
            used += spec["w"]
            log(f"enc member {Path(spec['path']).name} ok in {time.time() - t0:.0f}s")
        except Exception:
            traceback.print_exc()
            log(f"enc member {spec['path']} FAILED - skipped")
    if cfg.atlas:
        try:
            import torch
            from scipy.special import log_softmax

            from .models.atlas import AtlasInference, score
            t0 = time.time()
            meta = json.loads((Path(cfg.atlas["path"]) / "meta.json").read_text())
            m = AtlasInference(Path(cfg.atlas["path"]), base=cfg.atlas.get("base"))
            S = score(m, texts)
            del m
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            alp = log_softmax(S * float(meta.get("T", 30.0)), 1)
            if members_out is not None:
                members_out[cfg.atlas["path"]] = alp
            total += cfg.atlas["w"] * alp
            log(f"member {cfg.atlas['path']} ok in {time.time() - t0:.0f}s")
        except Exception:
            traceback.print_exc()
            log(f"member {cfg.atlas['path']} FAILED - skipped")
    if cfg.tfidf_w > 0 or used == 0:
        from .data import LABEL2ID
        from .features import tfidf
        X, y, w = train["text"], train["model"].map(LABEL2ID).values, np.ones(len(train))
        if pool is not None:
            X = pd.concat([X, pool["text"]])
            y = np.concatenate([y, pool["model"].map(LABEL2ID).values])
            cnt = np.bincount(y[len(train):], minlength=18)
            w = np.concatenate([w, cfg.tfidf_pool_w * len(pool) / 18 / cnt[y[len(train):]]])
        p = tfidf.fit_predict(X, y, texts, C=cfg.tfidf_C, model="svc", sample_weight=w, T=0.1)
        if members_out is not None:
            members_out["tfidf"] = np.log(np.maximum(p, 1e-9))
        total += (cfg.tfidf_w if used > 0 else 1.0) * np.log(np.maximum(p, 1e-9))
        log("tfidf member ok")
    if cfg.sinkhorn_alpha > 0:
        total = sinkhorn(total, cfg.sinkhorn_alpha)
    return total


def write_submission(test: pd.DataFrame, logp: np.ndarray, path: str = "submission.parquet") -> pd.DataFrame:
    sub = pd.DataFrame({"id": test["id"].astype(str).values,
                        "model": [LABELS[i] for i in logp.argmax(1)]})
    assert sub["id"].is_unique and sub["model"].isin(LABELS).all() and len(sub) == len(test)
    sub.to_parquet(path, index=False)
    return sub
