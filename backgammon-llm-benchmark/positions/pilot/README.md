# positions/pilot/

Phase 1 pilot set (~50 positions), **public**. Used for wiring and calibration
(PLAN.md §3.2, §7). Each file is a position record validating against
`schema/position.schema.json`. Empty in Phase 0.

## Quality gate — 8 positions need replacement

Eight of the 50 pilot positions fail the quiz-eligibility gate
(`generate/quality.py`) because they pose **no real decision**, so the human quiz
(`scripts/build_human_benchmark.py`) presents only the remaining **42** (27
checker + 15 cube). The dataset files are left in place — regenerating them needs
a GNU BG rollout, and no `gnubg` binary is available in this environment.

| # (display) | position_id | reason |
|---|---|---|
| 5  | `bg-0d3cf1acba7e656d` | trivial: every scored move has 0.0 mpt error (take-two-off bear-off) |
| 11 | `bg-2dff31791e5c29a8` | forced: only one legal move in the display frame |
| 19 | `bg-4474e236bf67b9b0` | forced: only one legal move |
| 21 | `bg-6211351d8ca6977d` | forced: only one legal move |
| 25 | `bg-6aa50863df140c18` | forced: only one legal move (rollout scored a single forced bar entry) |
| 26 | `bg-77105a01ad87534f` | trivial: every scored move has 0.0 mpt error |
| 37 | `bg-c57b3f4707c48e9b` | trivial: every scored move has 0.0 mpt error |
| 40 | `bg-c853084b55fcc7b5` | forced: only one legal move in the display frame (rollout scored a single bear-off) |

The gate requires, for a checker decision: ≥2 legal moves in the display frame,
≥2 scored rollout moves, and a best-to-worst loss spread ≥ 10 mpt (0.01 equity
points — an order of magnitude above rollout noise); for a cube decision, all
three actions scored. Non-trivial bear-offs (a real slotting/ordering choice with
a >10 mpt gap) pass; only the degenerate "take two off" races fail.

**To restore the pilot to 50:** replace these 8 with fresh candidates and run
`generate/gnubg.py` rollouts in an environment that has the `gnubg` binary, then
re-run the audit — `python3 -m pytest tests/test_quality.py` will confirm zero
exclusions once the replacements pass the gate.
