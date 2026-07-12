# HARNESS — running models, adding a model, budget track

> Phase 0 placeholder. See PLAN.md §5 for the authoritative spec.

## Gateway

OpenRouter (one key, OpenAI-compatible endpoint, hundreds of models,
per-request usage/cost reporting) — PLAN.md §5.1. Set the API key via `.env`
(never commit it).

## Running a run

Edit a config under `runs/*.yaml` (model list, track, sampling, budget) and run
`scripts/run_benchmark.py`. Each run writes `results/*.json` plus a
reproducibility manifest (dataset hash, prompt_version, timestamp). Runs are
cached (content-addressed) so they are idempotent and resumable — PLAN.md §5.2.

## Adding a model

Add its OpenRouter slug to the run config's `models:` list. No code change.

## Fixed-$ budget track

Each model gets a fixed budget (default $10) and may spend it however it plays
best (reasoning, retries, self-consistency, best-of-N). Ranked by best BenchPR
within budget; unanswered positions at budget-exhaustion score worst-case. Cost
source is OpenRouter's authoritative per-request cost — PLAN.md §5.3.
