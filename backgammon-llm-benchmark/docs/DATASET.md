# DATASET — schema, tiers, splits, provenance

> Phase 0 placeholder. See PLAN.md §1-3 for the authoritative spec.

## Representations

Each position ships as one JSON record with `xgid` (canonical ID), `gnubg_id`,
`board_json`, `ascii`, `image_png`, and `image_svg` (PLAN.md §1.1). Schema:
`schema/position.schema.json`. Ground truth: `schema/rollout.schema.json`.

## Tiers (T1-T4)

Tiers measure **difficulty for top humans**, estimated via a known-hard
taxonomy prior, a human-error model trained on analyzed public matches, and
expert-panel calibration (PLAN.md §3.1). Stored per position as
`expected_expert_loss` (EEL, millipoints for a reference PR-3 human) and
`expert_miss_rate`. Equity gap is a filter/scoring input only — not the tier
axis. Provisional thresholds (re-fit after panel calibration):

| Tier | Human difficulty | Intuition |
|------|------------------|-----------|
| T1 | miss rate < 2%, EEL < 1 mpt | any decent club player finds it |
| T2 | miss rate 2-10% | club players err, experts rarely |
| T3 | miss rate 10-30% | experts err at a real rate |
| T4 | miss rate > 30% or EEL > 10 mpt | even world-class players err often |

## Splits

Target ~1,500 positions: pilot 50 / public dev 150 / private held-out ~1,300.
Held-out per-tier target T1 25% / T2 30% / T3 30% / T4 15%. Within each tier keep
~70/30 checker/cube and ~60/40 money/match (PLAN.md §1.4, §3.3).

## Provenance

Generated fresh via strong-bot self-play (never harvested); each artifact records
a creation date, content hash, and canary. See CONTAMINATION.md and CANARY.md.
