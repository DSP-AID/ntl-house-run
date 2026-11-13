# Offline OOF validation — bnty_0595 (Name That LLM)

Protocol: official `train.csv` (2,699 rows, 18 classes, prompt-grouped splits), public
member OOF log-probs from `halley1812/name-that-llm-artifacts` (25 members), balanced
accuracy; hidden-test conditions emulated by per-model subsampling and incomplete
prompt groups. Full numbers: [`oof_sim2.log`](oof_sim2.log) (regenerate with
`python tmp/sim2.py`).

Key results:

- best single public member BA = **0.6205** (`lora14b_e2`);
- greedy full-fit blend OOF = 0.6598; **nested-CV honest estimate = 0.6509**;
- trnhip's proven core alone = 0.652 (their own reported `_oof`);
- 18 hidden-test scenario forecast: mean **0.657**, std **0.009** → expected LB band 0.62–0.70.

Rejected additions (all measured, see log):

| trick | verdict |
|---|---|
| Sinkhorn label-balanced assignment α>0 | negative in every realistic scenario; α=0 kept |
| Hungarian / DP per-prompt group assignment | hidden test has incomplete per-prompt groups (host statement + forum), simulator confirms loss |
| yauwes npz member blend (65 members, incl. k1 student/GBDT/encoder readouts) | ids ARE aligned with train.csv (verified 2026-11-13; the earlier "BA≈0.17" was an inverse column-permutation bug, `a[:, remap]` instead of `a[:, np.argsort(remap)]`); with correct alignment honest nested-CV is **0.6446 vs 0.6494** plain zoo → still excluded |
| per-member temperature calibration before log-pool greedy | nested-CV 0.6494 → 0.6494, zero delta; candidate.json fusion (cw=0.25 calib weights) already absorbs calibration |
| typography guardrails | delta exactly 0 on the strong blend; kept as harmless hard guards against degenerate rows |

Leaderboard reality (12 Nov 2026, markbarney.net terminal): leader 0.769, top-10 0.747,
best public notebook ceiling 0.684 (verified by three independent re-runs). The 0.790
bounty threshold is above the current leader; no public asset reaches it.
