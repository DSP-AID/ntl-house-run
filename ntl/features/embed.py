"""Frozen decoder-LLM features: mean-pooled hidden states of selected layers.

Discussion + literature: intermediate layers carry more style signal than the
last layer, and concatenating several layers helps. We extract every
``layer_stride``-th layer once and cache it, so layer selection is a cheap
downstream experiment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..data import ROOT

CACHE = ROOT / "artifacts" / "emb"


@torch.no_grad()
def extract(texts: list[str], model_name: str, layers: list[int] | None = None,
            max_len: int = 512, batch_size: int = 8, dtype=torch.bfloat16) -> dict[int, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, device_map="cuda").eval()
    n_layers = model.config.num_hidden_layers
    layers = layers or list(range(0, n_layers + 1, 2))
    out = {l: [] for l in layers}
    # sort by length to minimise padding
    order = np.argsort([len(t) for t in texts])
    for i in tqdm(range(0, len(texts), batch_size), desc=f"embed {Path(model_name).name}"):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        for l in layers:
            pooled = (hs[l].float() * m).sum(1) / m.sum(1)
            out[l].append((idx, pooled.cpu().numpy()))
    res = {}
    for l, chunks in out.items():
        arr = np.zeros((len(texts), chunks[0][1].shape[1]), dtype=np.float32)
        for idx, v in chunks:
            arr[idx] = v
        res[l] = arr
    del model
    torch.cuda.empty_cache()
    return res


def cache_path(model_name: str, split: str) -> Path:
    return CACHE / f"{Path(model_name).name}__{split}.npz"


def get(texts: list[str], model_name: str, split: str, **kw) -> dict[int, np.ndarray]:
    p = cache_path(model_name, split)
    if p.exists():
        z = np.load(p)
        return {int(k): z[k] for k in z.files}
    res = extract(texts, model_name, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{str(k): v for k, v in res.items()})
    return res
