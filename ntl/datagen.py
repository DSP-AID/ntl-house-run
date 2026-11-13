"""In-style data generation through the Kaggle Benchmarks model proxy.

Protocol (matches the public competitor datasets, so outputs stay in-style):
  one user message, no system prompt, fresh chat:
  "<prompt> Write a single paragraph of about N words, with no title or headings."
  N ~ log-uniform[40, 250], seeded by sha256(prompt_id|label); GPT at
  reasoning_effort=low, Gemini/Claude at provider defaults; keep finish=stop.

Design: complete SAME-PROMPT groups across confusable siblings (e.g.
gpt-5.6 sol/terra/luna) on NOVEL prompts (low similarity to train AND to the
existing pool), so new rows add prompt diversity and sibling contrast.

Steps: build_prompt_bank -> make_task(batch) -> `kaggle b t push/run/download`
-> collect(batch) -> filters + leak guard via external.* -> A/B -> publish.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import ROOT, _obj

GEN = ROOT / "datagen"
PROMPTS = ROOT / "data" / "external" / "prompts"

# proxy slug -> competition label (12 labels the proxy serves)
SLUG_TO_LABEL = {
    "gpt-5.5-2026-04-23": "gpt-5.5", "gpt-5.6-luna": "gpt-5.6-luna", "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra", "gpt-6-astra": "gpt-6-astra",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview", "gemini-3.6-flash": "gemini-3.6-flash",
    "gemini-3.8-flash": "gemini-3.8-flash",
    "claude-opus-4-6-default": "claude-opus-4-6", "claude-opus-4-8-default": "claude-opus-4-8",
    "claude-opus-5-default": "claude-opus-5", "claude-sonnet-5-default": "claude-sonnet-5",
}
LABEL_TO_SLUG = {v: k for k, v in SLUG_TO_LABEL.items()}
SUFFIX = " Write a single paragraph of about {n} words, with no title or headings."


def target_words(prompt_id: str, label: str) -> int:
    h = int(hashlib.sha256(f"{prompt_id}|{label}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return int(round(np.exp(np.log(40) + h * (np.log(250) - np.log(40)))))


def _banks() -> pd.DataFrame:
    rows = []
    p = PROMPTS / "illiadolenko_ntl-prompt-bank-v1" / "prompts.csv"
    if p.exists():
        d = _obj(pd.read_csv(p))
        # only the curated g#### part; "oz_short" is a scraped mix (non-English,
        # dialogue fragments, harmful requests), not competition-style questions
        d = d[d["prompt_id"].astype(str).str.match(r"^g\d+$")]
        rows.append(pd.DataFrame({"bank_id": "ill:" + d["prompt_id"].astype(str), "prompt": d["prompt"]}))
    p = PROMPTS / "pardheev_ntl-proxy-extra-passages" / "fresh_prompts.csv"
    if p.exists():
        d = _obj(pd.read_csv(p))
        rows.append(pd.DataFrame({"bank_id": "par:" + d["prompt_id"].astype(str), "prompt": d["prompt"]}))
    df = pd.concat(rows, ignore_index=True)
    df["prompt"] = df["prompt"].astype(str).str.strip()
    # train-style prompts only: one question, no persona / template placeholders
    df = df[~df["prompt"].str.contains(r"\{|\}|You are a ", regex=True)]
    nw = df["prompt"].str.split().str.len()
    ascii_ok = df["prompt"].map(lambda t: sum(c.isascii() for c in t) / max(len(t), 1) > 0.97)
    imperative = df["prompt"].str.match(r"^(Explain|Describe|Narrate|Discuss|Write|Why|How|What|Argue|Imagine|"
                                        r"Tell|Summari[sz]e|Compare|Walk|Trace|Recount|Portray|Paint|Evoke|Make)")
    df = df[ascii_ok & (nw >= 8) & (nw <= 70) & imperative]
    return df.drop_duplicates("prompt").reset_index(drop=True)


def build_prompt_bank(train: pd.DataFrame, pool: pd.DataFrame, leak_thr: float = 0.25,
                      max_pool_sim: float = 0.35) -> pd.DataFrame:
    """Candidate prompts that are (a) not a train topic (leak guard) and (b)
    novel w.r.t. prompts already answered in the pool, ranked most novel first."""
    bank = _banks()
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2,
                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    v.fit(pd.concat([train["text"], pool["text"].sample(min(40000, len(pool)), random_state=0), bank["prompt"]]))

    def centroids(df, key):
        from scipy.sparse import csr_matrix
        X = normalize(v.transform(df["text"]))
        keys, inv = np.unique(df[key].astype(str).values, return_inverse=True)
        G = csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(keys), len(inv)))
        return normalize(G @ X)

    B = normalize(v.transform(bank["prompt"]))
    bank["train_sim"] = np.asarray((B @ centroids(train, "prompt_id").T).max(1).todense()).ravel()
    bank["pool_sim"] = np.asarray((B @ centroids(pool, "prompt_id").T).max(1).todense()).ravel()
    keep = (bank["train_sim"] < leak_thr) & (bank["pool_sim"] < max_pool_sim)
    # novel first, but deterministic shuffle within similarity bands to mix domains
    b = bank[keep].copy()
    b["band"] = (b["pool_sim"] * 10).astype(int)
    return b.sample(frac=1, random_state=0).sort_values("band", kind="stable").drop(columns="band").reset_index(drop=True)


TASK_TEMPLATE = r'''# Name That LLM in-style generation, batch {batch} (built by ntl.datagen.make_task; protocol in its docstring).
import json
import time

import kaggle_benchmarks as kbench

SLUG_TO_LABEL = {slug_to_label!r}
PROMPTS = json.loads({prompts!r})          # bank_id -> prompt
TARGETS = json.loads({targets!r})          # label -> {{bank_id: n_words}}
SUFFIX = {suffix!r}


def label_of(llm):
    model = str(getattr(llm, "model", None) or getattr(llm, "name", ""))
    slug = model.split("/")[-1].replace("@", "-")
    return model, SLUG_TO_LABEL.get(slug) or SLUG_TO_LABEL.get(model.split("/")[-1].split("@")[0])


@kbench.task(name={task!r}, description="In-style single-paragraph passages for the Name That LLM competition")
def generate(llm) -> int:
    model, label = label_of(llm)
    if not TARGETS.get(label):
        print(json.dumps({{"skipped": True, "model": model}}), flush=True)
        return 0
    client = llm.client.with_options(max_retries=2, timeout=300)
    extra = {{"reasoning_effort": "low"}} if label.startswith("gpt") else {{}}
    done = errors = 0
    spent = 0.0
    with open(f"gen_{{label}}.jsonl", "a") as out:
        for pid, n in TARGETS[label].items():
            t0 = time.time()
            try:
                r = client.chat.completions.create(model=model, max_tokens=16000, stream=False, **extra,
                        messages=[{{"role": "user", "content": PROMPTS[pid].replace("{{n}}", str(n)) + SUFFIX.format(n=n)}}])
            except Exception as e:
                errors += 1
                msg = f"{{type(e).__name__}}: {{getattr(e, 'status_code', '')}}"
                print(json.dumps({{"prompt_id": pid, "error": msg}}), flush=True)
                if "429" in msg or "quota" in str(e).lower() or errors >= 10:
                    print("stopping: quota / rate limit / too many errors", flush=True)
                    break
                time.sleep(5)
                continue
            m = r.choices[0].message if r.choices else None
            if m is None:
                errors += 1
                continue
            usage = r.usage.model_dump() if getattr(r, "usage", None) else {{}}
            cost = usage.get("cost") or {{}}
            c = sum(x for x in cost.values() if isinstance(x, (int, float))) / 1e9 if isinstance(cost, dict) else 0.0
            spent += c
            out.write(json.dumps({{"label": label, "prompt_id": pid, "target_words": n, "text": m.content or "",
                                  "finish_reason": r.choices[0].finish_reason, "model_seen": str(getattr(r, "model", model)),
                                  "cost_usd": c, "latency_s": round(time.time() - t0, 2)}}) + "\n")
            out.flush()
            done += 1
    print(f"{{label}}: completed {{done}}/{{len(TARGETS[label])}}, errors {{errors}}, cost ${{spent:.4f}}", flush=True)
    return done


generate.run(kbench.llm)
'''


# host-style banks (v2/v3): same per-prompt template choice as datagen/standalone/ntl_dify_gen.py
HOST_TEMPLATES = ("{prompt} Write a single paragraph of about {n} words, with no title or headings.",
                  "{prompt} Answer in a single paragraph of about {n} words.",
                  "{prompt}\n\nRespond with one self-contained paragraph (roughly {n} words). No headings, lists or markdown.")


def host_message(prompt_id: str, prompt: str) -> str:
    t = HOST_TEMPLATES[int(hashlib.sha256(prompt_id.encode()).hexdigest()[:8], 16) % 3]
    return t.replace("{prompt}", prompt)   # "{n}" is filled per label inside the task


def _cells(files) -> set:
    """(prompt_id, label) cells already generated (finish=stop) in these jsonl files."""
    out = set()
    for f in files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("finish_reason") == "stop" and r.get("text"):
                out.add((r["prompt_id"], r["label"]))
    return out


def make_task(batch: str, bank: pd.DataFrame, labels: list[str], n_prompts: int, offset: int = 0,
              host_bank: str | None = None) -> Path:
    """Write datagen/tasks/ntl-gen-<batch>.py for bank rows [offset, offset+n_prompts)
    x labels (every label answers the same prompts -> same-prompt sibling groups).
    host_bank="v3": host-style prompts with the Dify template per prompt; resumes from every
    file under api_out/v3 (incl. val2 cells, which collect() hides on purpose)."""
    rows = bank.iloc[offset:offset + n_prompts]
    if host_bank:
        prompts = {b: host_message(b, p) for b, p in zip(rows["bank_id"], rows["prompt"])}
        done = _cells((GEN / "api_out" / host_bank).rglob("gen_*.jsonl"))
    else:
        prompts = dict(zip(rows["bank_id"], rows["prompt"]))
        done = set()
        g = collect()
        if len(g):  # resumable: skip (prompt, label) cells already generated (proxy has a spend cap)
            done = set(zip(g["prompt_id"].str.replace("trnhip:", "", regex=False), g["model"]))
    targets = {lab: {pid: target_words(pid, lab) for pid in prompts if (pid, lab) not in done} for lab in labels}
    src = TASK_TEMPLATE.format(batch=batch, slug_to_label=SLUG_TO_LABEL, prompts=json.dumps(prompts),
                               targets=json.dumps(targets), suffix="" if host_bank else SUFFIX,
                               task=f"ntl-gen-{batch}")
    (GEN / "tasks").mkdir(parents=True, exist_ok=True)
    p = GEN / "tasks" / f"ntl-gen-{batch}.py"
    p.write_text(src)
    rows.assign(batch=batch).to_csv(GEN / "tasks" / f"ntl-gen-{batch}_prompts.csv", index=False)
    return p


VAL2_N = 500  # bank v3 ids v3:00000..v3:00499 (random order, all domains) = held-out val2


def is_val2(prompt_id: str) -> bool:
    return prompt_id.startswith("v3:") and int(prompt_id[3:]) < VAL2_N


def collect_val2() -> pd.DataFrame:
    """The held-out second validation set: own generations on bank-v3 prompts v3:00000-00499."""
    recs = []
    for f in sorted((GEN / "api_out").rglob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("finish_reason") == "stop" and r.get("text") and is_val2(r.get("prompt_id", "")) \
                    and r["label"] != "gemini-3.7-flash":
                recs.append({"prompt_id": r["prompt_id"], "model": r["label"], "text": strip_think(r["text"])})
    return pd.DataFrame(recs, columns=["prompt_id", "model", "text"]).drop_duplicates(["prompt_id", "model"])


def strip_think(text: str) -> str:
    """Drop a leading <think>...</think> reasoning block (Dify kimi-k3 / qwen3.8-max return it inline)."""
    i = text.rfind("</think>")
    return text[i + len("</think>"):].strip() if i >= 0 else text.strip()


# host-style banks per pool version, as globs under api_out/ (each version a FIXED file set,
# so an older pool stays reproducible while daily files keep arriving)
OWN_EXTRA = {"v4": (), "v5": ("v2/gen_v2_trial.jsonl",), "v6": ("v2/gen_*.jsonl",),
             "v7": ("v2/gen_*.jsonl", "v3/gen_*.jsonl", "v3/proxy/**/gen_*.jsonl")}


def collect(out_dir: Path | None = None, extra: tuple = ()) -> pd.DataFrame:
    """Parse every downloaded gen_*.jsonl into the pool schema (prompt_id, model, text, source).
    extra: globs under api_out/ for the host-style banks (OWN_EXTRA[pool version]); pilots and
    val2 prompts are always excluded."""
    out_dir = out_dir or GEN / "out"
    recs = []
    files = sorted(out_dir.rglob("gen_*.jsonl"))
    api = GEN / "api_out"   # direct-API generations (ntl.apigen), same schema; pilots excluded
    if out_dir == GEN / "out" and api.exists():
        files += sorted(f for f in api.glob("*.jsonl") if not f.name.startswith("pilot_"))
        for g in extra:
            files += sorted(f for f in api.glob(g) if not f.name.startswith("pilot_"))
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("finish_reason") == "stop" and r.get("text"):
                    # Dify "gemini-3.7-flash" outputs are indistinguishable from our 3.8 outputs
                    # (journal 2026-10-22): served by 3.8 -> excluded, never relabelled
                    if r["label"] == "gemini-3.7-flash" and "api_out" in str(f):
                        continue
                    if is_val2(r["prompt_id"]):  # second validation set: never in any pool
                        continue
                    recs.append({"prompt_id": "trnhip:" + r["prompt_id"], "model": r["label"], "text": strip_think(r["text"]),
                                 "source": ("trnhip/ntl-api-gen-v3" if "api_out/v3" in str(f) else
                                            "trnhip/ntl-api-gen-v2" if "api_out/v2" in str(f) else
                                            "trnhip/ntl-api-gen" if "api_out" in str(f) else "trnhip/ntl-proxy-gen"),
                                 "target_words": r["target_words"],
                                 "model_seen": r.get("model_seen", ""), "cost_usd": r.get("cost_usd", 0.0)})
    return pd.DataFrame(recs).drop_duplicates("text") if recs else pd.DataFrame()
