# HARNESS — running models, adding a model, budget track

> See PLAN.md §5 for the authoritative spec.

## Gateway

OpenRouter (one key, OpenAI-compatible endpoint, hundreds of models,
per-request usage/cost reporting) — PLAN.md §5.1. Set the API key via `.env`
(never commit it).

## Run your first benchmark

1. **Set your key.** One environment variable is all the harness needs:

   ```sh
   export OPENROUTER_API_KEY='sk-or-...'
   ```

   (A real run fails fast with a readable message if this is unset.)

2. **Pick a config.** `runs/pilot.yaml` (or its `runs/pilot.json` twin) runs the
   50-position pilot set with gnubg rollouts, text track, greedy decoding
   (temperature 0), and the quality gate on. Edit the `models:` list to your
   OpenRouter slugs — one per line, no code change.

3. **Dry-run first (no network).** Estimate request counts and inspect a prompt:

   ```sh
   python3 scripts/run_benchmark.py --config runs/pilot.yaml --dry-run
   python3 scripts/run_benchmark.py --config runs/pilot.yaml --dry-run --show-prompt
   ```

   The pilot reports **42 positions** (50 minus the 8 gated out; see below) and
   `estimated_requests = models × 42`.

4. **Run for real.** Drop `--dry-run`:

   ```sh
   python3 scripts/run_benchmark.py --config runs/pilot.yaml
   ```

   Each run writes `results/<run_id>__<model>__<track>.json` plus a per-run
   working tree under `runs/<run_id>/` (raw responses, parsed scores, and a
   content-addressed cache) — so runs are idempotent, resumable, and reproducible
   (PLAN.md §5.2). Results carry a manifest (dataset hash, prompt/render versions,
   timestamp) and a `quality_gate` record of which positions were excluded.

5. **Build the leaderboard.** Turn `results/*.json` into the static site:

   ```sh
   python3 site/build.py          # -> site/public/ (gitignored, regenerable)
   ```

## Quality gate

By default (`quality_gate: true` in a run config) the runner applies the shared
quiz-eligibility gate (`generate/quality.py`, PLAN §3.2), so LLM runs face the
exact same eligible positions the human quiz does — forced, unscoreable, and
trivial positions (no real decision) are dropped. Excluded positions are logged
to stdout and recorded under `manifest.quality_gate` in the results JSON. Set
`quality_gate: false` to score the raw set.

## Adding a model

Add its OpenRouter slug to the run config's `models:` list. No code change.

## Fixed-$ budget track

Each model gets a fixed budget (default $10) and may spend it however it plays
best (reasoning, retries, self-consistency, best-of-N). Ranked by best BenchPR
within budget; unanswered positions at budget-exhaustion score worst-case. Cost
source is OpenRouter's authoritative per-request cost — PLAN.md §5.3.
