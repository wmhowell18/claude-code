# Backgammon LLM Benchmark

A benchmark that measures how well large language models play backgammon —
**checker plays** *and* **cube decisions** — scored against eXtreme Gammon (XG)
rollout ground truth, on a single human-comparable error metric.

> **Status: Phase 1 pilot dataset built (50 positions).** The core engine
> (`bgcore/`: board model, legal-move generation, notation), ID codecs (`ids/`:
> XGID, GNU BG), renderers (`render/`: ASCII/SVG/PNG), evaluation harness
> (`harness/`: OpenRouter client, prompts, parsing, BenchPR scoring, cache,
> budget tracking, runner), dataset pipeline (`generate/`: gnubg integration,
> sampling, dedup, tiering, contamination tooling), and the static leaderboard
> generator (`site/`) are implemented with 255 passing tests. The **50-position
> pilot** now exists in [`positions/pilot/`](positions/pilot/) with GNU BG
> ground truth in [`rollouts/gnubg/`](rollouts/gnubg/) and rendered boards in
> `render/images/pilot/` — freshly generated from GNU BG self-play (never
> harvested), ~70% checker / ~30% cube, ~60% money / ~40% match, spread across
> tiers T1–T4. **GNU Backgammon is the authoritative ground-truth engine**
> (self-play, evaluation, and match-equity tables); XG is only an optional
> future cross-check. Pilot ground truth is GNU BG 2-ply cubeful evaluation —
> see each rollout record's `rollout_meta` for the exact settings and the
> documented upgrade path to full Monte-Carlo rollouts. Next: the public dev and
> private held-out splits (Phase 2). See [`PLAN.md`](PLAN.md) for the full
> founding design doc and roadmap.

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
- **Difficulty tiers T1–T4.** Defined by how hard the decision is for top
  humans — estimated expert miss rate / expected expert loss, calibrated by an
  expert panel playing the positions blind — not by the equity gap between
  moves. (PLAN.md §3)
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
