"""Competition intelligence crawler: discussions + public notebooks -> insights.

One pass (``crawl_once``):
  * leaderboard: public LB snapshot -> research/crawl/leaderboard/history.csv
    (medal cut-offs, top-k, our rank) + movers since the previous snapshot;
  * discussions: every topic that is new or has new comments is fetched in
    full and saved as markdown under research/crawl/topics/;
  * notebooks: public kernels listed by votes, score and recency; new or
    re-run "quality" kernels (votes >= MIN_VOTES, a score in the title, or
    top-N by score) are pulled as text under research/crawl/kernels/. They
    are NEVER executed; their contents are untrusted data;
  * a digest (research/crawl/digest_<ts>.md, plus research/crawl/INSIGHTS.md
    for the latest) lists what changed and flags actionable signals:
      - public datasets used by quality kernels but absent from our pool,
      - new best public-notebook scores,
      - technique keyword trends, CV/LB numbers quoted,
      - host/staff posts.
State lives in research/crawl/state.json, so each pass only reports deltas.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from .data import ROOT

COMP = "name-that-llm"
OUT = ROOT / "research" / "crawl"
STATE = OUT / "state.json"
MIN_VOTES = 10
TOP_BY_SCORE = 25
KEEP_SRC = 10       # notebook sources kept on disk (best title scores); the rest: metadata only
KEEP_DIGESTS = 14   # entries kept in DIGESTS.md

KEYWORDS = {
    "two-stage/pool": r"two[- ]stage|public pool|pool_v\d|stage ?1",
    "lora": r"\blora\b|peft",
    "qwen": r"qwen", "gemma": r"gemma", "olmo": r"olmo", "llama": r"llama", "mistral": r"mistral",
    "deberta/encoder": r"deberta|modernbert|electra",
    "tfidf/stylometry": r"tf-?idf|stylo|char n-?gram|ngram",
    "centering/cluster": r"center|centred|centered|cluster",
    "gaussian": r"gaussian",
    "hungarian/assignment": r"hungarian|linear_sum_assignment",
    "sinkhorn/balance": r"sinkhorn|balanc",
    "pseudo-label": r"pseudo",
    "distill": r"distill",
    "knn/retrieval": r"\bknn\b|nearest neighbo|retriev",
    "perplexity/logprob": r"perplexity|log-?prob|binocular",
    "gate/hierarchy": r"\bgate\b|gates|hierarch|family",
    "generation/proxy": r"proxy|openrouter|generat",
}
NUM = re.compile(r"\b(CV|LB|OOF|public|private|bacc|balanced acc\w*)\W{0,3}(?:score\W{0,3})?(0\.\d{3,4})", re.IGNORECASE)
TITLE_SCORE = re.compile(r"(0\.\d{3})")
OUR_USERS = ("trnhip", "hieptran1812")
DEADLINE = "2027-01-14"
# where each crawler keyword stands in OUR pipeline (keeps the insight section actionable)
OURS = {"used": {"two-stage/pool", "lora", "qwen", "olmo", "tfidf/stylometry", "centering/cluster",
                 "generation/proxy", "llama"},
        "tested-rejected": {"sinkhorn/balance", "pseudo-label", "gate/hierarchy", "gemma"}}


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def _text(html: str) -> str:
    from bs4 import BeautifulSoup
    return BeautifulSoup(html or "", "html.parser").get_text("\n").strip()


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"topics": {}, "kernels": {}, "best_kernel_score": 0.0, "datasets_seen": []}


def _flatten(msgs, depth=0):
    for m in msgs or []:
        d = m if isinstance(m, dict) else m.to_dict()
        yield depth, d
        yield from _flatten(d.get("replies") or d.get("children") or [], depth + 1)


# --------------------------------------------------------------------- topics
def _store(name: str) -> dict:
    f = OUT / name
    return json.loads(f.read_text()) if f.exists() else {}


def crawl_topics(api, state: dict) -> list[dict]:
    """Full threads go to one file, research/crawl/topics.json ({topic id: markdown})."""
    changed, page = [], 1
    store = _store("topics.json")
    while True:
        r = api.competition_list_topics(COMP, sort_by="recent", page=page)
        topics = [t.to_dict() for t in (r.topics or [])]
        if not topics:
            break
        for t in topics:
            key = str(t["id"])
            sig = f"{t.get('commentCount')}|{t.get('lastCommentPostDate')}"
            if state["topics"].get(key) == sig:
                continue
            msgs = api.competition_list_topic_messages(COMP, int(t["id"]), page_size=-1).to_dict()
            lines = [f"# {t['title']}", f"votes={t.get('votes')} comments={t.get('commentCount')} "
                     f"url=https://www.kaggle.com{t.get('topicUrl')}", ""]
            host = False
            for depth, m in _flatten(msgs.get("messages")):
                author = m.get("author") or {}
                name = author.get("userName") or author.get("displayName") or "?"
                tier = str(author.get("tier") or author.get("performanceTier") or "")
                is_host = bool(m.get("isHost") or m.get("isCompetitionHost") or "staff" in tier.lower()
                               or "admin" in tier.lower())
                host |= is_host
                lines.append(f"{'  ' * depth}- **{name}**{' [HOST]' if is_host else ''} "
                             f"({m.get('postDate', '')[:16]}, votes {m.get('votes', 0)}): "
                             + _text(m.get("content", "")).replace("\n", " ")[:3000])
            body = "\n".join(lines)
            store[key] = body
            changed.append({"id": key, "title": t["title"], "votes": t.get("votes", 0),
                            "comments": t.get("commentCount", 0), "new": key not in state["topics"],
                            "host": host, "numbers": sorted(set(f"{a.upper()} {b}" for a, b in NUM.findall(body)))[:8],
                            "kw": _keywords(body)})
            state["topics"][key] = sig
        if len(topics) < 20:
            break
        page += 1
    (OUT / "topics.json").write_text(json.dumps(store, indent=0, ensure_ascii=False))
    return changed


# ---------------------------------------------------------------- leaderboard
def _medal_ranks(n: int) -> dict:
    """Kaggle progression rules (team count >= 1000): gold top 10 + 0.2%, silver 5%, bronze 10%."""
    if n >= 1000:
        return {"gold": 10 + int(0.002 * n), "silver": int(0.05 * n), "bronze": int(0.10 * n)}
    if n >= 250:
        return {"gold": 10, "silver": int(0.05 * n), "bronze": int(0.10 * n)}
    return {"gold": max(1, int(0.1 * n)), "silver": max(1, int(0.2 * n)), "bronze": max(1, int(0.4 * n))}


def crawl_leaderboard(state: dict) -> dict:
    import tempfile
    import zipfile

    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["kaggle", "competitions", "leaderboard", COMP, "-d", "-q", "-p", td],
                       capture_output=True, timeout=300, check=True)
        z = next(Path(td).glob("*.zip"))
        with zipfile.ZipFile(z) as f:
            name = next(n for n in f.namelist() if n.endswith(".csv"))
            lb = pd.read_csv(f.open(name), encoding="utf-8-sig")
    lb = lb[lb.Rank >= 1].sort_values("Rank").reset_index(drop=True)  # rank 0 = host baseline row
    # usernames of top-40 team members: their forum posts are flagged in the digest
    state["top_users"] = {u.strip().lower(): int(r.Rank) for r in lb.head(40).itertuples()
                          for u in str(r.TeamMemberUserNames or "").split(",") if u.strip()}
    ts = time.strftime("%Y-%m-%d %H:%M")
    n = len(lb)
    cut = {k: float(lb.Score.iloc[min(r, n) - 1]) for k, r in _medal_ranks(n).items()}
    ours = lb[lb.TeamMemberUserNames.fillna("").str.contains("|".join(OUR_USERS))]
    me = ours.iloc[0] if len(ours) else None
    row = {"ts": ts, "teams": n, "top1": float(lb.Score.iloc[0]), "top10": float(lb.Score.iloc[min(9, n - 1)]),
           **{f"{k}_cut": v for k, v in cut.items()},
           "our_rank": int(me.Rank) if me is not None else None, "our_score": float(me.Score) if me is not None else None}
    d = OUT / "leaderboard"
    d.mkdir(parents=True, exist_ok=True)
    hist = d / "history.csv"
    pd.DataFrame([row]).to_csv(hist, mode="a", header=not hist.exists(), index=False)
    top = lb.head(40)[["Rank", "TeamName", "Score", "SubmissionCount", "LastSubmissionDate"]]
    top.to_csv(d / "top40_latest.csv", index=False)
    prev = state.get("lb_top")
    movers = []
    for r in lb.head(40).itertuples():
        if prev is None:
            break  # first snapshot: nothing to compare
        old = prev.get(str(r.TeamId))
        if old is None:
            movers.append(f"NEW in top-40: {r.TeamName} #{r.Rank} {r.Score:.3f}")
        elif r.Score > old + 1e-9:
            movers.append(f"{r.TeamName}: {old:.3f} -> {r.Score:.3f} (#{r.Rank})")
    state["lb_top"] = {str(r.TeamId): float(r.Score) for r in lb.head(40).itertuples()}
    return {"row": row, "top": top, "movers": movers, "prev": state.get("lb_prev_row")}


# ------------------------------------------------------- top-team user posts
WEB = "https://www.kaggle.com/api/i/"


def _web():
    """Kaggle's site API (what the profile 'Discussion' tab uses). The public API
    returns no authors, so per-user activity needs this. Anonymous, read-only."""
    import httpx
    c = httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True, timeout=30)
    c.get("https://www.kaggle.com/")
    c.headers.update({"x-xsrf-token": c.cookies.get("XSRF-TOKEN", ""), "content-type": "application/json"})
    return c


def crawl_top_users(state: dict, per_user: int = 20) -> list[dict]:
    """New site-wide comments/topics by members of top-40 LB teams (any forum:
    they often discuss techniques in other competitions' threads too)."""
    users = state.get("top_users", {})
    if not users:
        return []
    c = _web()
    ids = state.setdefault("user_ids", {})
    seen = set(state.get("top_posts_seen", []))
    out = []
    for u, rank in sorted(users.items(), key=lambda x: x[1]):
        try:
            if u not in ids:
                ids[u] = c.post(WEB + "users.ProfileService/GetProfile", json={"userName": u}).json()["userId"]
            for dt in ("COMMENT", "TOPIC"):
                r = c.post(WEB + "search.SearchContentService/ListSearchContent",
                           json={"filters": {"ownerUserId": ids[u], "documentTypes": [dt]},
                                 "canonicalOrderBy": "LIST_SEARCH_CONTENT_ORDER_BY_DATE_CREATED",
                                 "pageSize": per_user})
                if r.status_code != 200:
                    continue
                for d in r.json().get("documents", []):
                    if d["documentId"] in seen:
                        continue
                    dd = d.get("discussionDocument") or {}
                    out.append({"user": u, "rank": rank, "date": d["createTime"][:16], "type": dt.lower(),
                                "doc": d["documentId"], "title": d.get("title", ""),
                                "ours": COMP in (dd.get("forumUrl") or ""), "forum": dd.get("forumName", ""),
                                "url": "https://www.kaggle.com" + (dd.get("newCommentUrl") or (d.get("enrichedInfo") or {}).get("url", "")),
                                "text": (dd.get("messageStripped") or "")[:700]})
            time.sleep(0.5)
        except Exception as e:  # one user failing must not stop the pass
            print("top user", u, e)
    state["top_posts_seen"] = sorted(seen | {p["doc"] for p in out})
    # other competitions' threads only if recent (old ones are rarely relevant here)
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 14 * 86400))
    out = [p for p in out if p["ours"] or p["date"] >= cutoff]
    store = _store("top_posts.json")
    for p in out:
        store[p["doc"]] = p
    (OUT / "top_posts.json").write_text(json.dumps(store, indent=0, ensure_ascii=False))
    return out


# -------------------------------------------------------------------- kernels
def _keywords(text: str) -> list[str]:
    low = text.lower()
    return [k for k, rx in KEYWORDS.items() if re.search(rx, low)]


def _list_kernels(api, sort_by: str, n: int):
    ks = api.kernels_list(competition=COMP, sort_by=sort_by, page_size=n) or []
    return [k if isinstance(k, dict) else k.to_dict() for k in ks]


def crawl_kernels(api, state: dict) -> list[dict]:
    pool = {}
    for sort_by, n in (("voteCount", 100), ("scoreDescending", TOP_BY_SCORE), ("dateRun", 60)):
        for i, k in enumerate(_list_kernels(api, sort_by, n)):
            k["_rank_score"] = i if sort_by == "scoreDescending" else None
            pool.setdefault(k["ref"], k)
    changed = []
    for ref, k in pool.items():
        votes = int(k.get("totalVotes") or 0)
        ts = TITLE_SCORE.findall(k.get("title", ""))
        tscore = max(map(float, ts)) if ts else None
        quality = votes >= MIN_VOTES or tscore is not None or k.get("_rank_score") is not None
        sig = str(k.get("lastRunTime"))
        if not quality or state["kernels"].get(ref) == sig:
            continue
        d = OUT / "kernels" / ref.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "pull", ref, "-p", str(d), "-m"], capture_output=True, timeout=300)
        src = ""
        for f in d.glob("*"):
            if f.suffix in (".ipynb", ".py", ".r", ".R"):
                src += f.read_text(errors="ignore")
        meta = {}
        if (d / "kernel-metadata.json").exists():
            meta = json.loads((d / "kernel-metadata.json").read_text())
        changed.append({"ref": ref, "title": k.get("title"), "votes": votes, "title_score": tscore,
                        "new": ref not in state["kernels"], "datasets": meta.get("dataset_sources", []),
                        "models": meta.get("model_sources", []), "kw": _keywords(src),
                        "numbers": sorted(set(f"{a.upper()} {b}" for a, b in NUM.findall(src)))[:8]})
        state["kernels"][ref] = sig
    idx = _store("kernels.json")
    for k in changed:
        idx[k["ref"]] = {kk: k[kk] for kk in ("title", "votes", "title_score", "datasets", "models", "kw", "numbers")}
    keep = {r for r, _ in sorted(((r, v["title_score"]) for r, v in idx.items() if v.get("title_score")),
                                 key=lambda x: -x[1])[:KEEP_SRC]}
    import shutil
    for d in (OUT / "kernels").glob("*"):
        ref = d.name.replace("__", "/", 1)
        if ref in idx and ref not in keep:  # never delete a source the index does not know
            shutil.rmtree(d, ignore_errors=True)
    (OUT / "kernels.json").write_text(json.dumps(idx, indent=0, ensure_ascii=False))
    return changed


# -------------------------------------------------------------------- digest
def _our_sources() -> set[str]:
    from .external import SOURCES_V2
    v1 = {"seyitkaangunes/ntl-extra-api-passages", "willrice/name-that-llm-openrouter-harvest",
          "lxdlxdlxd/ntl-synth-generations-v1", "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
          "prvsiyan/ntl-own-prompt-generations-v1", "alperen5252525/ntl-vertex-gemini-proxy-gpt-generations",
          "haileycheng/name-that-llm-same-prompt-passages", "kaplinskiy/ntl-gpt-proxy-passages",
          "lucasboesen/ntl-free-proxy-generations", "abdullah1afify/ntl-generations",
          "dophuchung/ntl-proxy-generations", "ziyuzhou1207/ntl-zz-proxy-generations",
          "keshavatearth/ntl-gpt-new-prompts-v1", "michaelsemenoff/ntl-own-extra-v1",
          "zbigniewkirchner/ntl-extra-gpt-passages", "prvsiyan/ntl-public-in-style-pool-v1",
          "nishantdahal/ntl-two-stage-lora"}
    v2 = {s[0].split("/")[0].replace("_", "/", 1) for s in SOURCES_V2}
    return v1 | v2


def _insights(topics, kernels, state, lb, top_posts) -> list[str]:
    """Rule-based 'so what' for us, at the top of each digest."""
    import pandas as pd
    out = ["## Insights for us", ""]
    if lb:
        r = lb["row"]
        h = pd.read_csv(OUT / "leaderboard" / "history.csv", parse_dates=["ts"])
        days_left = (pd.Timestamp(DEADLINE) - pd.Timestamp.now()).days
        line = f"- Gold cut {r['gold_cut']:.3f}; we are {r['our_score']} (#{r['our_rank']}), gap {r['gold_cut'] - (r['our_score'] or 0):+.3f}; {days_left} days left."
        span = (h.ts.iloc[-1] - h.ts.iloc[0]).total_seconds() / 86400
        if span >= 0.5:
            v = (h.gold_cut.iloc[-1] - h.gold_cut.iloc[0]) / span
            line += f" Gold cut moving {v * 7:+.3f}/week -> projected at deadline ~{r['gold_cut'] + v * days_left:.3f} (linear, upper bound)."
        out.append(line)
    ntl = [m for m in top_posts if m["ours"]]
    if top_posts:
        out.append(f"- {len(top_posts)} new post(s) by top-40 LB members ({len(ntl)} in this competition): "
                   + "; ".join(f"{m['user']} (#{m['rank']}): {m['title'][:50]}" for m in sorted(ntl, key=lambda m: m['rank'])[:5]))
    hosts = [t["title"] for t in topics if t["host"]]
    if hosts:
        out.append(f"- Host/staff activity: {'; '.join(hosts[:5])} (rules/data changes?).")
    our = (lb or {}).get("row", {}).get("our_score") or 0
    better = [k for k in kernels if (k["title_score"] or 0) > our]
    if better:
        kw = Counter(x for k in better for x in k["kw"])
        new_kw = [f"{k} ({c})" for k, c in kw.most_common() if k not in OURS["used"] | OURS["tested-rejected"]]
        rej = [f"{k} ({c})" for k, c in kw.most_common() if k in OURS["tested-rejected"]]
        out.append(f"- {len(better)} changed public notebook(s) score above our LB ({our}): "
                   + ", ".join(f"{k['title_score']:.3f} {k['ref']}" for k in sorted(better, key=lambda k: -k["title_score"])[:5]))
        if new_kw:
            out.append(f"  - techniques there we have NOT tried: {', '.join(new_kw)}")
        if rej:
            out.append(f"  - techniques there we tested and rejected (re-check their variant): {', '.join(rej)}")
    ours = _our_sources()
    nd = {d for k in kernels for d in k["datasets"] if d and d not in ours and d not in state["datasets_seen"]}
    if nd:
        out.append(f"- {len(nd)} new public dataset(s) referenced by quality notebooks (see below; check licence + leak guard).")
    if len(out) == 2:
        out.append("- nothing actionable since the previous pass.")
    return out + [""]


def write_digest(topics: list[dict], kernels: list[dict], state: dict, lb: dict | None = None,
                 top_posts: list[dict] | None = None) -> Path:
    ts = time.strftime("%Y%m%d_%H%M")
    L = [f"# Competition crawl digest {ts}", "",
         f"{len(topics)} new/updated topics, {len(kernels)} new/re-run quality notebooks.", ""]
    L += _insights(topics, kernels, state, lb, top_posts or [])
    if lb:
        r, pr = lb["row"], lb["prev"] or {}
        dl = lambda k: f" ({r[k] - pr[k]:+.3f})" if pr.get(k) is not None and r.get(k) is not None else ""
        L += ["## Leaderboard (public)", "",
              f"- teams {r['teams']} | top1 {r['top1']:.3f}{dl('top1')} | top10 {r['top10']:.3f}{dl('top10')}",
              f"- cut-offs: gold {r['gold_cut']:.3f}{dl('gold_cut')} | silver {r['silver_cut']:.3f}{dl('silver_cut')} "
              f"| bronze {r['bronze_cut']:.3f}{dl('bronze_cut')}",
              f"- **us: rank {r['our_rank']} score {r['our_score']}** | gap to gold "
              + (f"{r['gold_cut'] - r['our_score']:+.3f}" if r["our_score"] is not None else "n/a"),
              "", "Top-15:", ""]
        L += [f"  {t.Rank}. {t.TeamName} {t.Score:.3f} ({t.SubmissionCount} subs, last {str(t.LastSubmissionDate)[:16]})"
              for t in lb["top"].head(15).itertuples()]
        L += ["", "Movers in top-40 since last pass:", ""] + ([f"- {m}" for m in lb["movers"][:20]] or ["- none"]) + [""]
        state["lb_prev_row"] = r
    # actionable 1: new data sources
    ours = _our_sources()
    ds = Counter(d for k in kernels for d in k["datasets"])
    new_ds = [(d, c) for d, c in ds.most_common() if d and d not in ours and d not in state["datasets_seen"]]
    weights = re.compile(r"lora|weight|model|head|adapter|safetensors|deberta|bert|qwen|glm|olmo|asset|code|atlas", re.IGNORECASE)
    data_like = [(d, c) for d, c in new_ds if not weights.search(d)]
    L += ["## Actionable: new public DATA sources used by quality notebooks (not in our pool)", ""]
    L += [f"- `{d}` (used by {c} notebook(s))" for d, c in data_like] or ["- none"]
    L += ["", "## New model/weight datasets used by quality notebooks", ""]
    L += [f"- `{d}` ({c})" for d, c in new_ds if weights.search(d)] or ["- none"]
    state["datasets_seen"] = sorted(set(state["datasets_seen"]) | {d for d, _ in new_ds})
    # actionable 2: score records
    scored = sorted([k for k in kernels if k["title_score"]], key=lambda k: -k["title_score"])
    best = scored[0]["title_score"] if scored else 0
    L += ["", "## Public-notebook scores (from titles)", ""]
    if best > state["best_kernel_score"]:
        L.append(f"**NEW BEST public notebook: {best:.3f}** (previous {state['best_kernel_score']:.3f})")
        state["best_kernel_score"] = best
    L += [f"- {k['title_score']:.3f} | {k['votes']} votes | [{k['title']}](https://www.kaggle.com/code/{k['ref']}) "
          f"| kw: {', '.join(k['kw'])} | {'; '.join(k['numbers'])}" for k in scored[:15]] or ["- none"]
    # trends
    kw = Counter(x for k in kernels for x in k["kw"]) + Counter(x for t in topics for x in t["kw"])
    L += ["", "## Technique mentions in changed items", "", ", ".join(f"{k} ({c})" for k, c in kw.most_common())]
    tp = top_posts or []
    L += ["", "## Posts by top-40 leaderboard members (new; this competition first)", ""]
    L += [f"- {'**[NTL]** ' if m['ours'] else ''}**{m['user']}** (LB #{m['rank']}, {m['date']}, {m['type']}) "
          f"*{m['title'][:80]}* ({m['forum']}) {m['url']}\n  > {m['text'][:400]}"
          for m in sorted(tp, key=lambda m: (not m["ours"], m["rank"]))[:40]] or ["- none"]
    L += ["", "## Discussions (new / updated)", ""]
    for t in sorted(topics, key=lambda t: (-t["host"], -t["votes"])):
        L.append(f"- {'**[HOST]** ' if t['host'] else ''}{'NEW ' if t['new'] else ''}{t['title']} "
                 f"(votes {t['votes']}, comments {t['comments']}) -> topics.json[{t['id']}]"
                 + (f" | numbers: {'; '.join(t['numbers'])}" if t["numbers"] else ""))
    L += ["", "## Other changed notebooks", ""]
    L += [f"- {k['votes']} votes | {k['title']} | models: {', '.join(k['models'])[:120]}"
          for k in kernels if not k["title_score"]][:30]
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L)
    (OUT / "INSIGHTS.md").write_text(text)
    roll = OUT / "DIGESTS.md"
    old = roll.read_text().split("\n<!-- digest -->\n") if roll.exists() else []
    old = [x for x in old if x.strip()]
    roll.write_text("\n<!-- digest -->\n".join([text] + old[:KEEP_DIGESTS - 1]))
    return OUT / "INSIGHTS.md"


def crawl_once() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    api = _api()
    lb = None
    try:
        lb = crawl_leaderboard(state)
    except Exception as e:  # LB download must not block the topic/kernel pass
        print("leaderboard failed:", e)
    topics = crawl_topics(api, state)
    kernels = crawl_kernels(api, state)
    top = []
    try:
        top = crawl_top_users(state)
    except Exception as e:
        print("top users failed:", e)
    p = write_digest(topics, kernels, state, lb, top)
    STATE.write_text(json.dumps(state, indent=1))
    return p
