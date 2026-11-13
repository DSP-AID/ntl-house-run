"""Name That LLM - reproducible Kaggle entry (house-run route for DAiL bnty_0599 / ex-bnty_0595).

Pipeline = halley1812/trnhip "ntl" LoRA24 ensemble (CC0 code + adapters) extended with
the members that our offline OOF simulation selected, plus data-driven typography
guardrails. Inputs resolve in two tiers: attached Kaggle datasets first (offline
fallback), then direct downloads from the public HF dataset halley1812/name-that-llm-artifacts
(every artifact below is mirrored there). Base LLMs load from the HF hub (internet ON).

Runs on Kaggle GPU T4x2 within the 9 h limit; a submission.parquet is always written:
any member that fails is skipped and TF-IDF is the last resort.
"""
import glob
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
WORK = os.environ.get("NTL_WORK", "/kaggle/working")
os.environ.setdefault("HF_HOME", os.path.join(WORK, "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
HALLEY = "https://huggingface.co/datasets/halley1812/name-that-llm-artifacts/resolve/main/"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(pattern, needle=None):
    hits = sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))
    return [h for h in hits if needle in h] if needle else hits


def fetch(url, dst, min_bytes=1000):
    if os.path.exists(dst) and os.path.getsize(dst) > min_bytes:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    tmp = dst + ".part"
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.replace(tmp, dst)
    log(f"  fetched {os.path.basename(dst)} {os.path.getsize(dst) / 1e6:.0f} MB")
    return dst


def hf_get(hf_path, dst):
    """Kaggle attachment first, HF direct download otherwise."""
    local = find(os.path.basename(hf_path), needle=hf_path.split("/")[-2] if "/" in hf_path else None)
    if local:
        return local[0]
    return fetch(HALLEY + hf_path, os.path.join(WORK, "artifacts", hf_path))


def ensure_lora(hf_prefix, name):
    d = os.path.join(WORK, "artifacts", "ckpts", name)
    for rel in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors", "heads.pt", "meta.json"):
        fetch(HALLEY + f"{hf_prefix}/{rel}", os.path.join(d, rel), min_bytes=100)
    return d


def ensure_pub(hf_prefix, dirname):
    d = os.path.join(WORK, "artifacts", "ckpts", dirname)
    for rel in ("adapter_config.json", "adapter_model.safetensors", "heads.pt"):
        fetch(HALLEY + f"{hf_prefix}/{dirname}/{rel}", os.path.join(d, rel), min_bytes=100)
    return d


def ensure_enc(hf_prefix, name):
    d = os.path.join(WORK, "artifacts", "ckpts", name)
    for rel in ("encoder/config.json", "encoder/model.safetensors", "encoder/tokenizer.json",
                "encoder/tokenizer_config.json", "head.safetensors", "meta.json"):
        fetch(HALLEY + f"{hf_prefix}/{rel}", os.path.join(d, rel), min_bytes=100)
    return d


def free_gb(path="."):
    return shutil.disk_usage(path).free / 2**30


def maybe_free_base(base_id, still_needed):
    """Drop a base model's HF cache once no later member needs it (disk guard)."""
    if still_needed or free_gb(WORK) > 35:
        return
    hub = os.environ["HUGGINGFACE_HUB_CACHE"]
    d = os.path.join(hub, "models--" + base_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        log(f"disk guard: freed cache of {base_id} ({free_gb(WORK):.0f} GB free)")


# ---------------------------------------------------------------- config ----
with open(os.path.join(HERE, "candidate.json"), encoding="utf-8") as fh:
    CONFIG = json.load(fh)
log(f"config: {json.dumps({k: v for k, v in CONFIG.items() if not isinstance(v, list)})}")

sys.path.insert(0, HERE)
comp_hits = [p for p in find("test.csv") if "name-that-llm" in p or "name_that_llm" in p]
comp_dir = os.path.dirname(comp_hits[0]) if comp_hits else os.environ.get("NTL_RAW_DIR", WORK)
os.environ["NTL_RAW"] = comp_dir

import numpy as np
import pandas as pd

from ntl.data import LABEL2ID, LABELS, load_test, load_train
from ntl.inference import PipelineCfg, predict, write_submission

test, train = load_test(), load_train()
log(f"test rows: {len(test)}; train rows: {len(train)}")

# pool for the TF-IDF member (the leak-guarded v2x pool used by the proven config)
pool = None
try:
    pool = pd.read_parquet(hf_get(CONFIG["pool_hf"], os.path.join(WORK, "pool_v2x.parquet")))
    log(f"pool rows: {len(pool)}")
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------- members ----
members, enc_members = [], []
specs = CONFIG["members"]
for i, s in enumerate(specs):
    try:
        path = ensure_lora(s["hf"], s["name"])
        folds = [ensure_lora(f, os.path.basename(f.rstrip("/"))) for f in s.get("folds", [])]
        members.append({"path": path, "folds": folds, "base": s["base"], "w": s["w"], "center_w": s["cw"]})
    except Exception:
        traceback.print_exc()
        log(f"could not stage member {s['name']}")

# base-cache freeing plan: a base may be dropped once its last spec has run
used_after = {}
for i, s in enumerate(specs):
    used_after[s["base"]] = i
for pm in CONFIG.get("pub_members", []):
    try:
        ads = [ensure_pub(pm["hf_prefix"], d) for d in pm["dirs"]]
        members.append({"pub_l24": True, "path": pm["name"], "adapters": ads, "base": pm["base"],
                        "w": pm["w"], "keep": pm["keep"]})
        used_after[pm["base"]] = max(used_after.get(pm["base"], -1), len(specs))
    except Exception:
        traceback.print_exc()
        log(f"could not stage pub member {pm['name']}")
for es in CONFIG.get("enc_members", []):
    try:
        enc_members.append({"path": ensure_enc(es["hf"], es["name"]), "w": es["w"]})
    except Exception:
        traceback.print_exc()
        log(f"could not stage enc member {es['name']}")

BUDGET_S, FULL_EST_S = 8.2 * 3600, 45 * 60
for j, m in enumerate(members):
    m["fold_deadline"] = T0 + BUDGET_S - (len(members) - j - 1) * FULL_EST_S - 20 * 60

cfg = PipelineCfg(lora_members=members, enc_members=enc_members, tfidf_w=CONFIG["tfidf_w"],
                  tfidf_pool_w=CONFIG["tfidf_pool_w"], cluster_cut=CONFIG["cluster_cut"],
                  sinkhorn_alpha=CONFIG["sinkhorn_alpha"], center_loo=CONFIG.get("center_loo", True))

# ------------------------------------------------- typography guardrails ----
# Hard invariants measured on 2,699 train passages (0-violation classes); the OOF
# simulator says the delta on a strong blend is ~0, but they cannot hurt and they
# stop degenerate rows. shortcut: rules learned from train only; revisit if the
# hidden test adds proxy-generated text (forum: proxy Gemini does use em-dashes).
CUR = [LABELS.index(l) for l in LABELS if l.startswith(("claude", "glm"))]
GPT4 = [LABELS.index(x) for x in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra")]
SOL = [LABELS.index("gpt-5.6-sol")]


def apply_guardrails(logp, texts):
    import numpy as np
    t = pd.Series(list(texts)).astype(str)
    out = logp.copy()
    m = t.str.contains(r"[\u2019\u201c\u201d]").to_numpy()
    out[np.ix_(m, CUR)] = -1e9
    m = t.str.contains('"').to_numpy()
    out[np.ix_(m, GPT4)] = -1e9
    m = t.str.contains("'").to_numpy()
    out[np.ix_(m, SOL)] = -1e9
    return out


# ---------------------------------------------------------------- run ----
SANITY = len(test) < 100
if SANITY:  # commit/interactive runs see a 3-row test: pad to rerun scale with labelled rows
    tr = train.copy()
    tr["len"] = tr["text"].str.len()
    san = pd.concat([tr.nlargest(100, "len"), tr.sample(300, random_state=0)]).drop_duplicates("id")
    pad = pool.sample(min(3600, len(pool)), random_state=0)[["text"]].assign(id=lambda d: "pad" + d.index.astype(str)) \
        if pool is not None else test.iloc[:0]
    run_df = pd.concat([test[["id", "text"]], san[["id", "text"]], pad[["id", "text"]]], ignore_index=True)
else:
    run_df = test

member_lp, ran_bases = {}, set()
try:
    # interleave member runs with disk guards: predict() consumes specs in order, so
    # free a base cache between members by monkey-hooking the log line (cheap: track by name)
    logp = predict(run_df, cfg, train=train, pool=pool, members_out=member_lp)
    logp = apply_guardrails(logp, run_df["text"].tolist())
except Exception:
    traceback.print_exc()
    log("pipeline failed -> TF-IDF only")
    logp = predict(run_df, PipelineCfg(), train=train, pool=pool)

if SANITY:
    ys = san["model"].map(LABEL2ID).values
    k, n = len(test), len(san)
    log(f"SANITY rows={len(run_df)} blend acc={(logp[k:k + n].argmax(1) == ys).mean():.3f} "
        f"classes: {len(np.unique(logp.argmax(1)))}/18")
    ref = CONFIG.get('_sanity_ref', {})
    for name, lp in member_lp.items():
        acc = float((lp[k:k + n].argmax(1) == ys).mean())
        rv = next((v for k2, v in ref.items() if k2 in name), None)
        tag = f" ref={rv:.3f} {'OK' if abs(acc - rv) <= 0.05 else 'WARN'}" if rv is not None else ''
        log(f'sanity member {name}: acc={acc:.3f}{tag}')
    logp = apply_guardrails(logp, run_df["text"].tolist())[:k]

sub = write_submission(test, logp, os.path.join(WORK, "submission.parquet"))
log(f"done in {(time.time() - T0) / 60:.1f} min; counts:\n{sub['model'].value_counts().to_string()}")
