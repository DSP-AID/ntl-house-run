"""Public external pool (competitor-generated passages from the same 18 models).

Leak guard: some pool sources reuse prompts that also appear in train.csv.
Exact-text deduplication does not remove *other answers* to those prompts.
Training on them would let a validation fold see its own prompts and inflate
CV. So we drop every pool prompt-group whose topic matches any train prompt.
That makes the pool disjoint from all train prompts at once, so the pool is
fold-independent and stage-1 pretraining only needs to run once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import LABELS, ROOT, _obj

EXT = ROOT / "data" / "external"


def load_pool(prvsiyan: Path | None = None, nishant: Path | None = None) -> pd.DataFrame:
    """Union of the two public pools, de-duplicated on text.
    Kaggle: prvsiyan/ntl-public-in-style-pool-v1 and nishantdahal/ntl-two-stage-lora."""
    a = _obj(pd.read_parquet(prvsiyan or EXT / "prvsiyan" / "pool_v1.parquet"))
    b = _obj(pd.read_parquet(nishant or EXT / "nishant" / "pool_v1.parquet"))
    b = b.assign(prompt_id="nishant:" + b["prompt_key"].astype(str))[["prompt_id", "model", "text", "source"]]
    df = pd.concat([a, b], ignore_index=True)
    df["text"] = df["text"].str.strip()
    df = df.drop_duplicates("text").reset_index(drop=True)
    return df[df["model"].isin(LABELS)].reset_index(drop=True)


def prompt_match_scores(pool: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """For each pool prompt group: max cosine between its topic centroid and any
    train prompt's topic centroid (content-word TF-IDF)."""
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2,
                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    v.fit(pd.concat([train["text"], pool["text"]]))

    def centroids(df):
        # sparse group-mean: (G^T X) with a sparse indicator G; dense would be
        # n_groups x vocab (~18 GB for pool v2)
        from scipy.sparse import csr_matrix
        X = normalize(v.transform(df["text"]))
        g = df["prompt_id"].astype(str).values
        keys, inv = np.unique(g, return_inverse=True)
        G = csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(keys), len(inv)))
        return keys, normalize(G @ X)

    pk, PC = centroids(pool)
    _, TC = centroids(train)
    s = np.asarray((PC @ TC.T).max(1).todense()).ravel()
    return pd.Series(s, index=pk)


def clean_pool(pool: pd.DataFrame, train: pd.DataFrame, thr: float) -> tuple[pd.DataFrame, pd.Series]:
    s = prompt_match_scores(pool, train)
    bad = set(s.index[s >= thr])
    keep = ~pool["prompt_id"].astype(str).isin(bad)
    return pool[keep].reset_index(drop=True), s


# (label, source substring) slices whose rows disagree with the train-defined
# label far more than other sources do (scripts/audit_pool_labels.py).
SUSPECT = [("gemini-3.7-flash", "prvsiyans-extended"), ("gpt-5.6-luna", "prvsiyans-extended"),
           ("gemini-3.8-flash", "same-prompt-passages"), ("gpt-5.6-sol", "same-prompt-passages"),
           ("gpt-5.6-terra", "same-prompt-passages")]


def drop_suspect(pool: pd.DataFrame) -> pd.DataFrame:
    bad = np.zeros(len(pool), bool)
    for lab, src in SUSPECT:
        bad |= (pool["model"] == lab).values & pool["source"].astype(str).str.contains(src).values
    return pool[~bad].reset_index(drop=True)


# --------------------------------------------------------------------------
# Pool v2: v1 + datasets published 2026-10-20/21 (all public Kaggle datasets).
# Each entry: (relative file, label column, prompt column or None, served-model column or None)
NEW = EXT / "new"
SOURCES_V2 = [
    ("pukkei37_ntl-pool-v2/pool_v2.parquet", "model", "prompt_group", None),
    ("arxitekton_ntl-claude-gen-v1/passages.csv", "model", "prompt_id", None),
    ("arxitekton_ntl-claude-gen-v2/passages.csv", "model", "prompt_id", "served_model"),
    ("maniceet_name-that-llm-openrouter-openai-generations/ntl_openrouter_openai_generations.parquet", "model", None, "served_model"),
    ("drjcfmd_ntl-generated-v2/train_ext.csv", "model", "prompt_id", None),
    ("dedruck_ntl-proxy-gen/passages.csv", "model", "prompt_id", "model_seen"),
    ("abhinavsaipodugu_ntl-proxy-gpt-gemini-passages/ntl_generated_abhinavsaipodugu.csv", "model", "prompt_id", "model_seen"),
    ("illiadolenko_ntl-generated-corpus-v1/generated_clean.csv", "model", "prompt_id", None),
    ("tadeuhayashida_ntl-claude-fable-5-generations/gen001_claude_fable_5_filtered.parquet", "model", "prompt_id", "served_model"),
    ("bingli283_ntl-gen-pilot-v1/gen_pilot_v1.csv", "model", "prompt_id", "resolved_model"),
    ("aswathvanan_ntl-proxy-passages/passages.csv", "model", "prompt_group", None),
    ("abdullah1afify_ntl-generations/ntl_generations.csv", "label", "prompt_id", "model_seen"),
    ("lucasboesen_ntl-free-proxy-generations/gen_passages.csv", "model", "prompt_id", None),
]
# pool v3 = v2 + second sweep (2026-10-21)
SOURCES_V3_EXTRA = [
    ("jadkutbi_ntl-own-proxy-generations/passages.parquet", "label", "prompt_id", None),
    ("alperen5252525_ntl-public-passage-pool-v1/pool_v1.parquet", "model", "group", None),
    ("pukkei37_ntl-external-clean/external_clean.parquet", "model", None, None),
]
# pool v8 = v2 + public sweep of 2026-10-24 (datasets used by the 0.62-0.636 public notebooks that we lacked);
# michaelsemenoff merges several of these (exact duplicates are dropped downstream)
SOURCES_V8_EXTRA = [
    ("alperen5252525_ntl-vertex-gemini-proxy-gpt-generations/ntl_alp_generations.csv", "label", "prompt_id", "returned_model"),
    ("dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv", "label", "prompt_id", "model_seen"),
    ("haileycheng_name-that-llm-same-prompt-passages/passages.csv", "label", "prompt_id", "model_served"),
    ("kaplinskiy_ntl-gpt-proxy-passages/ntl_gpt_proxy_passages.csv", "label", "prompt_id", "model_seen"),
    ("keshavatearth_ntl-gpt-new-prompts-v1/accepted.csv", "model", "prompt_id", None),
    ("lxdlxdlxd_ntl-synth-generations-v1/generations.csv", "label", "pid", "returned_model"),
    ("michaelsemenoff_ntl-own-extra-v1/ntl_own_extra_v1.csv", "model", "group", None),
    ("pardheev_ntl-proxy-extra-passages/proxy_passages.csv", "label", "group", "model_served"),
    ("prvsiyan_ntl-own-prompt-generations-v1/generated_v1.csv", "label", "prompt_id", None),
    ("seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv", "model", "prompt_id", None),
    ("willrice_name-that-llm-openrouter-harvest/passages.csv", "label", "prompt_id", "served_model"),
    ("zakirkhanaleemi_name-that-llm-benchmarks-proxy-generations/ntl_proxy_generations.csv", "model", "prompt_id", None),
    ("zbigniewkirchner_ntl-extra-gpt-passages/ntl_gpt_passages.csv", "model", "prompt_id", None),
    ("ziyuzhou1207_ntl-zz-proxy-generations/passages.csv", "model", "prompt_id", None),
]
# pool v9 (26/10): new public sources. GLM/Kimi (danielandresmiranda), Gemini incl. 3.7-flash (harshitkmr),
# own-prompt bank (earthmaker) and four public aggregate pools (rows we already have drop out in the text dedup;
# philipvallen's is_train_prompt / train_regen rows are dropped, everything else goes through clean_pool's leak guard).
SOURCES_V9_EXTRA = [
    ("danielandresmiranda_ntl-openrouter-glm-kimi/ntl_openrouter_glm_kimi.csv", "model", "prompt_id", None),
    ("harshitkmr_ntl-google-gen-v1/google_gen_v1.csv", "model", "prompt_id", "api_model"),
    ("earthmaker_ntl-own-prompt-generations/passages.csv", "model", "prompt_id", None),
    ("sunny114514_ntl-mega-corpus/ntl_mega_corpus.parquet", "model", "prompt_id_or_group", None),
    ("philipvallen_ntl-pools/pool_broad.parquet", "label", "prompt_key", None),
    ("dominicjamil_ntl-pools/merged_pool_v2.parquet", "model", "prompt_id", None),
    ("mrralphie_ntl-pool-b2m/pool_b2m_public.parquet", "model", "prompt_id", None),
]
# pool v10 (26/10): + dhruvtibarewal/nlm-matched-clean (public aggregate pool v5 + matched set used by nlm-v27)
SOURCES_V10_EXTRA = [
    ("dhruvtibarewal_nlm-matched-clean/nlm_pool_v5.parquet", "model", "prompt_key", None),
    ("dhruvtibarewal_nlm-matched-clean/nlm_matched_clean.parquet", "model", None, None),
]
MARKS = {"emdash": "—", "endash": "–", "lquote": "“", "rquote": "”", "lsq": "‘",
         "rsq": "’", "ellipsis": "…"}


def _served_ok(label: str, served) -> bool:
    if not isinstance(served, str) or not served.strip():
        return True  # not recorded
    s = served.lower().split("/")[-1].split("@")[0]
    return s.startswith(label) or label in s


def style_invariants(train: pd.DataFrame) -> dict[str, set]:
    """Marks a model never uses in train (<1% rows). A pool row of that model
    containing one is dropped: off-style or mislabelled generation."""
    out = {}
    for m, g in train.groupby("model"):
        out[m] = {k for k, ch in MARKS.items() if g["text"].str.contains(ch, regex=False).mean() < 0.01}
    return out


def load_pool_v2(train: pd.DataFrame, root: Path | None = None, v3: bool = False,
                 own_gen: bool = False, own: str = "v4", v8: bool = False, v9: bool = False,
                 v10: bool = False) -> pd.DataFrame:
    v9 = v9 or v10
    root = root or NEW
    frames = [load_pool().assign(source="v1:" + load_pool()["source"].astype(str))]
    if own_gen:  # our Kaggle Benchmarks proxy generations (published as trnhip/ntl-proxy-gen)
        from .datagen import OWN_EXTRA, collect
        g = collect(extra=OWN_EXTRA[own])
        if len(g):
            frames.append(g[["prompt_id", "model", "text", "source"]])
    for rel, lab, pcol, scol in (SOURCES_V2 + (SOURCES_V3_EXTRA if v3 else []) + (SOURCES_V8_EXTRA if v8 or v9 else [])
                                 + (SOURCES_V9_EXTRA if v9 else []) + (SOURCES_V10_EXTRA if v10 else [])):
        p = root / rel
        if not p.exists():
            continue
        d = _obj(pd.read_parquet(p) if rel.endswith("parquet") else pd.read_csv(p))
        if scol and scol in d.columns:
            d = d[[_served_ok(str(a), b) for a, b in zip(d[lab], d[scol])]]
        if "split" in d.columns:  # drjcfmd: 'calib' rows are likely regenerated train prompts
            d = d[~d["split"].astype(str).isin(["calib", "train_regen"])]
        if "is_train_prompt" in d.columns:  # philipvallen: regenerations of train prompts
            d = d[~d["is_train_prompt"].astype(bool)]
        tag = rel.split("/")[0]
        pid = d[pcol].astype(str) if pcol and pcol in d.columns else d.index.astype(str)
        scol_ = "source" if "source" in d.columns else "src" if "src" in d.columns else None  # nlm_pool_v5 uses 'src'
        src = (tag + ":" + d[scol_].astype(str)).values if scol_ else tag
        frames.append(pd.DataFrame({"prompt_id": tag + ":" + pid.values, "model": d[lab].values,
                                    "text": d["text"].astype(str).values, "source": src}))
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].str.strip()
    df = df[df["model"].isin(LABELS)]
    nw = df["text"].str.split().str.len()
    md = df["text"].str.contains(r"\n|\*\*|^#|^\s*[-*] |`", regex=True)
    df = df[(nw >= 30) & (nw <= 300) & ~md]
    df = df[~df["text"].isin(set(train["text"]))].drop_duplicates("text")
    if v9:  # prvsiyan's train_regen rows answer the host's train prompts (topic 752810): drop them by text
        regen = EXT / "new" / "prvsiyan_extended" / "passages.csv"
        if regen.exists():
            r = pd.read_csv(regen, usecols=["text", "split"])
            df = df[~df["text"].isin(set(r.loc[r["split"] == "train_regen", "text"].astype(str).str.strip()))]
    if v9:  # aggregate pools re-label prompt ids, so hold val2 out by text as well
        from .datagen import collect_val2
        df = df[~df["text"].isin(set(collect_val2()["text"].str.strip()))]
    inv = style_invariants(train)
    bad = np.zeros(len(df), bool)
    for k, ch in MARKS.items():
        has = df["text"].str.contains(ch, regex=False).values
        bad |= has & df["model"].map(lambda m: k in inv[m]).values
    return df[~bad].reset_index(drop=True)


def pool_v2_clean(train: pd.DataFrame, thr: float = 0.3, rebuild: bool = False, v3: bool = False,
                  own_gen: bool = False, exclude: tuple = (), own: str | None = None,
                  v8: bool = False, v9: bool = False, v10: bool = False) -> pd.DataFrame:
    """Leak-guarded pool v2 (v3 = v2 + second sweep; v4 = v2 + own generations;
    v5 = v4 + bank-v2 trial; v6 = v4 + all bank-v2 files; v7 = v6 + bank v3), cached on disk.
    own: "v4".."v7" (datagen.OWN_EXTRA); own_gen=True alone means v4."""
    own = own or ("v4" if own_gen else None)
    own_gen = own is not None
    name = (f"pool_{own}{'p8' if v8 else ''}_clean.parquet" if own_gen else "pool_v10_clean.parquet" if v10 else "pool_v9_clean.parquet" if v9 else
            "pool_v8_clean.parquet" if v8 else
            "pool_v3_clean.parquet" if v3 else "pool_v2_clean.parquet")
    cache = EXT / name
    if own_gen:
        rebuild = True   # own generations keep arriving; the cache would go stale
    if cache.exists() and not rebuild:
        pool = _obj(pd.read_parquet(cache))
        if exclude:
            pool = pool[~pool["source"].astype(str).str.contains("|".join(exclude))].reset_index(drop=True)
        return pool
    # v4 = v2 + own generations (NOT v3: its extra sources hurt, journal 2026-10-21)
    pool, _ = clean_pool(load_pool_v2(train, v3=v3 and not own_gen, own_gen=own_gen, own=own or "v4", v8=v8, v9=v9, v10=v10), train, thr=thr)
    pool.to_parquet(cache)
    if exclude:
        pool = pool[~pool["source"].astype(str).str.contains("|".join(exclude))].reset_index(drop=True)
    return pool
