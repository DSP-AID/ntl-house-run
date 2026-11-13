# Name That LLM — house-run reproducible notebook (DAiL bnty_0599, re-listing of bnty_0595)

Deterministic pipeline for the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm):
given 26 text samples per prompt, identify which of 18 LLMs generated them (balanced accuracy).

**Ready-to-run kernel:** [`kaggle/ntl_house_run.ipynb`](kaggle/ntl_house_run.ipynb) with
`kernel-metadata.json` (ref `dspaid/name-that-llm-house-run`). Push via
`kaggle kernels push` or the web UI, then commit "full run" (internet ON, GPU T4 x2 or
better, Save & Submit). Kaggle credentials are not held by this agent, so the kernel is
delivered as reproducible source + exact run instructions.

## Expected score

Honest expected LB band **0.62–0.70** (evidence below; 18-scenario mean 0.657, std ≈ 0.009).
The published 0.790 threshold is above the current leaderboard leader (0.769), so this
submission is best-effort reproducible science, not a guaranteed pass — see
[`evidence/oof_validation.md`](evidence/oof_validation.md).

## How to reproduce (house run)

1. Upload `kaggle/ntl_house_run.ipynb` to Kaggle (or `kaggle kernels push` from the
   `kaggle/` folder) → **Commit** (Save & Run All). The notebook fetches every artifact
   by direct URL from `halley1812/name-that-llm-artifacts` + HF hub; the optional
   `trnhip/*` dataset attachments (this repo's code snapshot `ntl-code` v3, `ntl-pool` v2,
   `ntl-models` v2, all CC0) only speed things up and are not required.
2. Or attach datasets and run `run_submission.py` as a script:
   - Kaggle datasets: `trnhip/ntl-code`, `trnhip/ntl-pool`, `trnhip/ntl-models` (CC0);
     if they are ever removed, the script falls back to fetching the same artifacts from
     `halley1812/name-that-llm-artifacts` on Hugging Face (every URL pre-verified);
   - base weights stream from HF hub: `Qwen/Qwen3-14B-Base`, `allenai/OLMo-2-1124-13B`
     (internet required; ~60 GB peak disk, ~47 min on T4 x2);
   - output: `submission.parquet` (columns `id`, `model`).
3. On short/commit runs the script pads the 3-row sanity test with labelled train rows
   and prints per-member accuracies with a self-check against `_sanity_ref` in
   `candidate.json` (trnhip's published numbers, tolerance ±0.05) before writing
   the submission.

## Pipeline

- 7 pool-v2x adapter members: Qwen3-14B × {p2x, p4, e2, p9, p10}, OLMo-2-1124-13B × {p4, p5};
- public member `q14v5`: dhruvtibarewal nlm-l24-adapters-v5 × 6 adapters on Qwen3-14B
  (pool layers 2..27, `keep_last_n_layers=27`);
- DeBERTa-v3 encoder member (27 hidden states, halley1812 artifacts);
- TF-IDF char/word ridge + cluster-balanced prior correction + Sinkhorn label-balanced
  assignment (all validated on nested CV, weights kept at proven values);
- typography guardrails: hard zero-evidence rules (e.g. `}`/`</thinking>` ⇒ never GLM/GPT;
  straight quotes ⇒ never Claude/GLM for GPT-* classes), applied before Sinkhorn.

## Evidence (this repo)

| artifact | content |
|---|---|
| `candidate.json` | exact final blend config consumed by `run_submission.py` |
| `evidence/oof_report.json` | headline OOF numbers and rejected-trick verdicts |
| `evidence/oof_sim2.log` | full simulation log: per-member BA table, greedy trace, postproc deltas |
| `evidence/oof_validation.md` | nested-CV protocol (0.651), 18-scenario forecast (0.657 ± 0.009), rejected tricks (Sinkhorn α, Hungarian, group DP, yauwes npz misalignment) |

Local OOF numbers: proven trnhip core alone = 0.652; full candidate under nested CV = 0.651;
scenario forecast mean 0.657. Independent community re-runs of public notebooks top out at
0.684 — consistent with our band, not with 0.790.

## Tests

`pytest tests/` runs CPU-only smoke tests (no torch needed): candidate.json schema,
`ntl.data` roundtrip on the bundled `comp/` data, Sinkhorn/PipelineCfg sanity.

## Provenance & licensing

Code vendored from [trnhip/ntl-code](https://www.kaggle.com/datasets/trnhip/ntl-code) (CC0),
plus one additive patch in `ntl/inference.py` (`enc_members` support for the DeBERTa member).
Adapters: [trnhip/ntl-models](https://www.kaggle.com/datasets/trnhip/ntl-models) (CC0),
[dhruvtibarewal/nlm-l24-adapters-v5](https://huggingface.co/datasets/dhruvtibarewal/nlm-l24-adapters-v5),
[halley1812/name-that-llm-artifacts](https://huggingface.co/datasets/halley1812/name-that-llm-artifacts).
This repo: MIT-style, attribution as above.
