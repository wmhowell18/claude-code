# DATASET — schema, tiers, splits, provenance

> Phase 0 placeholder. See PLAN.md §1-3 for the authoritative spec.

## Representations

Each position ships as one JSON record with `xgid` (canonical ID), `gnubg_id`,
`board_json`, `ascii`, `image_png`, and `image_svg` (PLAN.md §1.1). Schema:
`schema/position.schema.json`. Ground truth: `schema/rollout.schema.json`.

## Tiers (T1-T4)

Objective, rollout-derived; primary axis is the best-vs-second equity gap
(blunder margin) plus phase (PLAN.md §3.1):

| Tier | Gap | Intuition |
|------|-----|-----------|
| T1 | >= 0.080 | any decent player finds it |
| T2 | 0.030-0.080 | solid intermediate |
| T3 | 0.008-0.030 | strong-player territory |
| T4 | < 0.008 or rare phase | reference-grade |

## Splits

Target ~1,500 positions: pilot 50 / public dev 150 / private held-out ~1,300.
Held-out per-tier target T1 25% / T2 30% / T3 30% / T4 15%. Within each tier keep
~70/30 checker/cube and ~60/40 money/match (PLAN.md §1.4, §3.2).

## Provenance

Generated fresh via strong-bot self-play (never harvested); each artifact records
a creation date, content hash, and canary. See CONTAMINATION.md and CANARY.md.
