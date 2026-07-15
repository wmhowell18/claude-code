# SCORING — BenchPR definition + XG PR mapping

> Phase 0 placeholder. See PLAN.md §4 for the authoritative spec; fill in as
> `harness/scoring.py` is implemented and calibrated.

## Primary metric — equity loss

For each decision, `error = best_equity - chosen_equity` (>= 0), read from stored
rollout data (PLAN.md §1.3, §4.4). The equities come from **GNU BG rollouts —
the authoritative ground-truth engine (permanent, per the 2026-07-15 decision).**
Averaged over positions, optionally per-tier and per-track.

## BenchPR

`BenchPR = 500 x mean(equity_loss)` on the same money-equity units XG uses, so
models and humans sit on the same axis. Lower is better:

- PR ~0 = flawless · PR 2-4 = world-class · PR 5-8 = strong expert ·
  PR 10-15 = intermediate · PR 20+ = beginner.

Human north-star reference lines: PR 2 / 4 / 8 (PLAN.md §4.5).

## Open calibration item

XG's exact PR constant and its cube-vs-checker handling must be matched for the
numbers to be literally comparable. Until validated against >=3 XG-analyzed
reference games (`scripts/validate_pr.py`), label the metric
**"BenchPR (PR-calibrated)"** (PLAN.md §4.4, §9 item 2). This is a *metric-scale*
calibration only — it keeps BenchPR on the same human PR axis as XG's published
scale. It does **not** make XG a ground-truth engine; the equities are always
GNU BG rollouts (PLAN.md §1.3, §9.1).
