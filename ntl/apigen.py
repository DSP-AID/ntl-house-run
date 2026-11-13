"""Direct-API in-style generation (OpenAI, Gemini, Z.ai GLM) -- same protocol as
ntl.datagen (Kaggle proxy), so all generated rows are interchangeable.

Keys: .secrets/api.env (git-ignored), e.g.
    OPENAI_API_KEY=...
    GEMINI_API_KEY=...
    ZAI_API_KEY=...
All three are called through OpenAI-compatible chat endpoints.
Alternatively route every call through a Dify workflow (datagen/dify_ntl_generate.yml):
    NTL_GEN_VIA=dify  DIFY_BASE_URL=https://uat-dify-llm.torilab.ai  DIFY_APP_KEY=app-...
(the Dify server must accept requests from the machine running the generator).

Design (see docs/data_strategy.md):
  * complete SAME-PROMPT sibling groups: every label of a family answers the
    same prompt (GLM pair, GPT-5.x set, Gemini set);
  * novel train-style prompts (datagen.build_prompt_bank: leak-guarded vs train,
    novel vs the pool);
  * target length N ~ log-uniform[40,250] seeded by (prompt, label);
  * a pilot mode that tries several prompt TEMPLATES / settings and reports
    style statistics vs train per label, to pick the protocol closest to the
    host's (we do not know the host's exact prompt);
  * resumable jsonl output with the served model id, finish reason, usage.
Usage:
    python -m ntl.apigen pilot --labels glm-5.3,glm-5.3-flash --n 20
    python -m ntl.apigen run   --labels glm-5.3,glm-5.3-flash --batch g01 --n 500 --template t0
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import pandas as pd

from .data import ROOT
from .datagen import target_words

OUT = ROOT / "datagen" / "api_out"
ENV = ROOT / ".secrets" / "api.env"

# label -> (provider, api model id). Model ids are best guesses; verify with the
# provider's model list and fix here (the served id is recorded on every row).
MODELS = {
    "gpt-5.5": ("openai", "gpt-5.5"), "gpt-5.6-luna": ("openai", "gpt-5.6-luna"),
    "gpt-5.6-sol": ("openai", "gpt-5.6-sol"), "gpt-5.6-terra": ("openai", "gpt-5.6-terra"),
    "gpt-6-astra": ("openai", "gpt-6-astra"),
    "gemini-3.1-pro-preview": ("gemini", "gemini-3.1-pro-preview"),
    "gemini-3.6-flash": ("gemini", "gemini-3.6-flash"), "gemini-3.7-flash": ("gemini", "gemini-3.7-flash"),
    "gemini-3.8-flash": ("gemini", "gemini-3.8-flash"),
    "glm-5.3": ("zai", "glm-5.3"), "glm-5.3-flash": ("zai", "glm-5.3-flash"),
}
BASE_URL = {"openai": None, "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "zai": "https://api.z.ai/api/paas/v4/"}
KEY = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "zai": "ZAI_API_KEY"}

# Candidate user-message templates. t0 = the protocol of our proxy data and of most
# public competitor datasets.
TEMPLATES = {
    "t0": "{prompt} Write a single paragraph of about {n} words, with no title or headings.",
    "t1": "{prompt} Answer in a single paragraph of about {n} words.",
    "t2": "{prompt}\n\nRespond with one self-contained paragraph (roughly {n} words). No headings, lists or markdown.",
}


def _env():
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def client(provider: str):
    from openai import OpenAI
    _env()
    return OpenAI(api_key=os.environ[KEY[provider]], base_url=BASE_URL[provider], timeout=300, max_retries=3)


def call_dify(label: str, msg: str) -> dict:
    """Run the 'ntl-generate' Dify workflow (scripts/make_dify_dsl.py) for one message.
    Needs DIFY_BASE_URL (e.g. https://uat-dify-llm.torilab.ai) and DIFY_APP_KEY (app-...)."""
    import httpx
    _env()
    t0 = time.time()
    r = httpx.post(os.environ["DIFY_BASE_URL"].rstrip("/") + "/v1/workflows/run", timeout=300,
                   headers={"Authorization": f"Bearer {os.environ['DIFY_APP_KEY']}"},
                   json={"inputs": {"label": label, "message": msg}, "response_mode": "blocking", "user": "ntl-gen"})
    r.raise_for_status()
    d = r.json()["data"]
    ok = d.get("status") == "succeeded"
    return {"label": label, "text": (d.get("outputs") or {}).get("text", "").strip(),
            "finish_reason": "stop" if ok else d.get("status"), "model_seen": f"dify:{label}",
            "usage": {"total_tokens": d.get("total_tokens")}, "latency_s": round(time.time() - t0, 2)}


def call(label: str, prompt: str, n: int, template: str, settings: dict) -> dict:
    prov, model = MODELS[label]
    msg = TEMPLATES[template].format(prompt=prompt, n=n)
    if os.environ.get("NTL_GEN_VIA") == "dify":
        r = call_dify(label, msg)
        r.update(target_words=n, template=template, settings={"via": "dify"})
        return r
    kw = dict(settings)
    t0 = time.time()
    r = client(prov).chat.completions.create(model=model, messages=[{"role": "user", "content": msg}], **kw)
    ch = r.choices[0]
    return {"label": label, "text": (ch.message.content or "").strip(), "finish_reason": ch.finish_reason,
            "model_seen": getattr(r, "model", model), "target_words": n, "template": template,
            "settings": settings, "usage": r.usage.model_dump() if getattr(r, "usage", None) else {},
            "latency_s": round(time.time() - t0, 2)}


def default_settings(label: str) -> dict:
    # GPT reasoning models: low effort is what public sets used; others: provider defaults.
    return {"reasoning_effort": "low"} if label.startswith("gpt") else {}


def run(labels: list[str], prompts: pd.DataFrame, batch: str, template: str = "t0",
        workers: int = 8, settings: dict | None = None) -> Path:
    """Generate every (prompt, label) cell not already present in datagen/api_out/<batch>.jsonl."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{batch}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("finish_reason") == "stop" and r.get("text"):
                done.add((r["prompt_id"], r["label"]))
    jobs = [(pid, p, lab) for pid, p in zip(prompts["bank_id"], prompts["prompt"]) for lab in labels
            if (pid, lab) not in done]
    print(f"{len(jobs)} cells to generate ({len(done)} done)", flush=True)
    errors = 0
    with open(path, "a") as out, cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(call, lab, p, target_words(pid, lab), template,
                          settings if settings is not None else default_settings(lab)): (pid, lab)
                for pid, p, lab in jobs}
        for fu in cf.as_completed(futs):
            pid, lab = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                errors += 1
                print(json.dumps({"prompt_id": pid, "label": lab, "error": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
                if errors >= 25:
                    print("too many errors - stopping", flush=True)
                    ex.shutdown(cancel_futures=True)
                    break
                continue
            r["prompt_id"] = pid
            out.write(json.dumps(r) + "\n")
            out.flush()
    return path


def style_stats(texts: pd.Series) -> dict:
    nw = texts.str.split().str.len()
    return {"n": len(texts), "words_med": float(nw.median()),
            "emdash": float(texts.str.contains("—", regex=False).mean()),
            "curly": float(texts.str.contains("[“”’]", regex=True).mean()),
            "semicolon": float(texts.str.contains(";", regex=False).mean()),
            "multi_para": float(texts.str.contains("\n").mean()),
            "markdown": float(texts.str.contains(r"\*\*|^#|^\s*[-*] ", regex=True).mean())}


def pilot(labels: list[str], n_prompts: int = 20):
    """Generate n_prompts x labels x templates and compare style stats with train,
    per label; prints the template with the smallest distance to train."""
    from .data import load_train
    tr = load_train()
    bank = pd.read_csv(ROOT / "datagen" / "prompt_bank.csv").iloc[-n_prompts:]  # tail: unused by proxy batches
    res = {}
    for t in TEMPLATES:
        path = run(labels, bank, batch=f"pilot_{t}", template=t)
        g = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines()])
        for lab in labels:
            s = style_stats(g[g.label == lab].text)
            ref = style_stats(tr[tr.model == lab].text)
            dist = sum(abs(s[k] - ref[k]) for k in ("emdash", "curly", "semicolon", "multi_para", "markdown"))
            res[(lab, t)] = {**s, "dist_to_train": round(dist, 3), "served": g[g.label == lab].model_seen.unique().tolist()}
    print(pd.DataFrame(res).T.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "run"])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch", default="api01")
    ap.add_argument("--template", default="t0")
    a = ap.parse_args()
    labels = a.labels.split(",")
    if a.mode == "pilot":
        pilot(labels, a.n)
    else:
        bank = pd.read_csv(ROOT / "datagen" / "prompt_bank.csv").iloc[a.offset:a.offset + a.n]
        run(labels, bank, a.batch, a.template)
