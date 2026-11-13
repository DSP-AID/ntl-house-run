"""CPU-only smoke tests for the house-run repo (no torch required)."""
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_candidate_json_schema():
    cfg = json.loads((REPO / "candidate.json").read_text(encoding="utf-8"))
    assert cfg["pool_hf"].endswith("pool_v2x.parquet")
    assert len(cfg["members"]) == 7 and len(cfg["pub_members"]) == 1 and len(cfg["enc_members"]) == 1
    assert len(cfg["pub_members"][0]["dirs"]) == 6
    for m in cfg["members"]:
        assert {"name", "hf", "base", "w", "cw"} <= set(m) and m["w"] > 0
    bases = {m["base"] for m in cfg["members"]} | {cfg["pub_members"][0]["base"]}
    assert bases == {"Qwen/Qwen3-14B-Base", "allenai/OLMo-2-1124-13B"}
    for k in ("tfidf_w", "tfidf_pool_w", "cluster_cut"):
        assert cfg[k] > 0
    assert cfg["sinkhorn_alpha"] == 0.0 and cfg["center_loo"] is True


def test_ntl_data_roundtrip():
    os.environ["NTL_RAW"] = str(REPO / "comp")
    from ntl.data import LABELS, load_test, load_train, make_folds, subsample_like_test
    assert len(LABELS) == 18
    tr = load_train()
    assert len(tr) == 2699 and set(tr["model"]) <= set(LABELS)
    te = load_test()
    assert {"id", "text"} <= set(te.columns)
    folds = make_folds(tr)
    assert set(folds) <= {0, 1, 2, 3, 4, -1} and (folds >= 0).sum() > 2000
    tr["y"] = tr["model"].map({m: i for i, m in enumerate(LABELS)})
    assert subsample_like_test(tr, seed=0).sum() > 1000


def test_postproc_and_cfg_imports():
    import numpy as np

    from ntl.inference import PipelineCfg
    from ntl.postproc import sinkhorn
    cfg = PipelineCfg()
    assert cfg.tfidf_w == 0.0 and cfg.sinkhorn_alpha == 0.0
    lp = np.random.default_rng(0).normal(size=(18, 18))
    balanced = sinkhorn(lp.copy(), alpha=1.0)
    counts = balanced.argmax(1)
    assert len(set(counts.tolist())) >= 12  # assignment becomes near-uniform
