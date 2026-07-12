# results/

Model-run outputs (`results/*.json`) consumed by the static site (PLAN.md §5.2,
§6). Each file validates against `schema/results.schema.json` and is committed
(published). For the private held-out set, results are aggregate-only; per-decision
detail is published for the public dev set. Empty in Phase 0.
