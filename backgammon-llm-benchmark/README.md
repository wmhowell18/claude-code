# Backgammon LLM Benchmark

A benchmark that measures how well large language models play backgammon —
**checker plays** *and* **cube decisions** — scored against eXtreme Gammon (XG)
rollout ground truth, on a single human-comparable error metric.

> **Status: Phase 0 — scaffolding.** This repo currently contains only the
> project skeleton (directory tree, schema stubs, docstring-only module stubs).
> No dataset, no working harness, no results yet. See
> [`PLAN.md`](PLAN.md) for the full founding design doc and roadmap.

## What it is

We evaluate single decisions from fixed positions (that is how backgammon
strength is actually measured), not full-game self-play. Each position ships in
multiple redundant representations so one source of truth can serve every track.

## Key design points

- **One position, many representations.** Every position carries `xgid`,
  `gnubg_id`, structured `board_json`, a fixed-width `ascii` render, and a
  rendered `image_png` (+ source `image_svg`). Text and image tracks are served
  from the same records. (PLAN.md §1.1–1.2)
- **XG rollouts are ground truth.** Truth is a full rollout per position (not a
  static eval), with the settings blob recorded. GNU BG rollouts are the interim
  engine until XG automation is solved. (PLAN.md §1.3, §7, §9)
- **Contamination avoidance.** Positions are *generated* via strong-bot
  self-play (never harvested), deduped against public collections, and
  spot-checked for web-findability. The authoritative set is a **private
  held-out split** — only SHA-256 hashes and canaries are published. (PLAN.md §2)
- **Difficulty tiers T1–T4.** Defined by objective rollout-derived criteria,
  primarily the best-vs-second equity gap (blunder margin) plus phase. (PLAN.md §3)
- **BenchPR metric with a human north star.** `BenchPR = 500 × mean(equity_loss)`,
  computed like XG's Performance Rating so models and humans sit on the same
  axis. The leaderboard draws PR 2 / 4 / 8 human reference lines. (PLAN.md §4)
- **OpenRouter harness.** One OpenAI-compatible endpoint, hundreds of models,
  and per-request cost reporting for the cost axis. Async, cached, resumable.
  (PLAN.md §5)
- **Fixed-$-budget track.** A second leaderboard where each model gets a fixed
  budget (default $10) and spends it however it plays best; ranked by best
  BenchPR within budget. (PLAN.md §5.3)
- **Static leaderboard site.** Generated from `results/*.json`, hosted on
  GitHub Pages: skill-vs-cost scatter, per-tier bars, text-vs-image comparison.
  (PLAN.md §6)

## Layout

See [`PLAN.md` §8](PLAN.md) for the authoritative repository tree. Documentation
lives in [`docs/`](docs/):

- [`docs/SCORING.md`](docs/SCORING.md) — BenchPR definition + XG PR mapping.
- [`docs/DATASET.md`](docs/DATASET.md) — schema, tiers, splits, provenance.
- [`docs/CONTAMINATION.md`](docs/CONTAMINATION.md) — anti-contamination policy.
- [`docs/HARNESS.md`](docs/HARNESS.md) — how to run models, budget track.

## Licensing

Code and dataset licenses are **TBD** — see `LICENSE`, `DATA_LICENSE`, and
Open Questions §8 in `PLAN.md`.
