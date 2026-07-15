# rollouts/xg/ — reserved, unused in v1

**Not an authoritative source.** As of the 2026-07-15 decision, GNU Backgammon
(gnubg) is the permanent ground-truth engine (PLAN.md §1.3, §9.1); see
`rollouts/gnubg/`. This directory is **reserved for an optional future XG
cross-check** on a handful of marquee positions only — it is never the source of
truth and is empty/unused in v1. Any records placed here would validate against
`schema/rollout.schema.json` (`engine: "xg"`) but would be treated as a
comparison aid, not as scoring input.
