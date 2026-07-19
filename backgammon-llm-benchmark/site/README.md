# site/ — static leaderboard (GitHub Pages)

The v1 results website (PLAN.md §6). A small build step (`build.py`) reads
`results/*.json` and renders static HTML — no backend, no database, one-way data
flow (results JSON -> build -> static HTML). The build is stdlib-only.

## Human-benchmark quiz (`human-benchmark-pilot.html`)

`scripts/build_human_benchmark.py` generates a **single self-contained HTML quiz**
that lets human panelists sit the exact same 50-position pilot set the LLMs are
scored on, blind, producing a directly comparable **BenchPR**. Everything is
inlined (all 50 board SVGs, CSS, JS, and the rollout answer key as a JSON blob) —
zero network requests — so the file can be emailed and opened locally.

```bash
# Generate site/public/human-benchmark-pilot.html (stdlib only):
python3 scripts/build_human_benchmark.py

# Options:
#   --out PATH          output file (default: site/public/human-benchmark-pilot.html)
#   --timestamp "..."   fixed ISO manifest timestamp (default: now, UTC)
```

### What the panelist sees

- An intro screen (instructions + a name/identifier field + a blind-test notice).
- One position at a time with a `Position N / 50` progress bar, deterministic
  order (sorted by `position_id`). Every diagram is shown from the on-roll
  player's perspective — **the panelist is always `X`** (the light checkers).
  The generator normalises each record to the mover frame before rendering: a
  record stored with `turn == "o"` is the color-flipped (opponent's-view) form
  (`bgcore.board.flip`), so it is flipped back — the board SVG is re-rendered and
  the pips / score / cube-owner are recomputed — so the displayed `X` is always
  the player the rollout move list is for. Shown: the board SVG, dice (checker
  decisions only — cube records carry no dice), cube value/owner, money-vs-match
  + score, and pip counts. Tier, phase, expected-loss and other difficulty/answer
  hints are **not** shown.
- Checker decisions take a free-text move in standard notation (a JS port of
  `bgcore/notation.py` normalises it — reordered plays, `*` hits, `(n)` repeats,
  `bar/`/`/off`). A move that reaches the same position spelled differently (e.g.
  naming a single checker's intermediate point, `13/10/9` for `13/9`) also matches
  via a pre-computed endpoint map (mirrors `bgcore.moves.moves_equivalent`).
  Unrecognised input warns once, then may be submitted anyway.
- Cube decisions are buttons for exactly the actions the record poses
  (`No double` / `Double, Take` / `Double, Pass`).
- No going back; answers persist to `localStorage` (keyed by `position_id`), so a
  tab close doesn't lose progress.
- A results screen reveals ground truth **only at the end**: total BenchPR, mean
  equity loss, best-move accuracy, per-tier (T1–T4) and per-decision-type
  breakdowns, a per-position review (answer vs. best move + equity loss), and a
  **Download results JSON** button.

### Scoring (mirrors `harness/scoring.py`)

`BenchPR = 500 × mean(equity_loss)`, `equity_loss = error_mp / 1000` (equity
points). A checker answer is matched against the rollout move list by canonical
notation, then (if no direct hit) by resulting-position equivalence via the
endpoint map; `is_best` iff the matched `error_mp` is zero. Unmatched or
unparseable answers are scored as the **worst listed move** (PLAN §4.6). A cube
answer reads the chosen action's `error_mp` directly.

### Workflow

1. **Generate** the file (above) and verify it opens.
2. **Send** `site/public/human-benchmark-pilot.html` to panelists (email/attach).
3. Each panelist completes it and clicks **Download results JSON**
   (`human-panel_<name>_text.json`, shaped like `tests/fixtures/human-panel_text.json`:
   `kind: "human"`, `model: "human-panel/<name>"`, `track: "text"`).
4. **Collect** the returned JSONs and drop them into `results/`. They are picked
   up by `build.py` as measured human-panel baselines (`kind: "human"`) —
   excluded from the ranked model list, badged `human`, and plotted as the
   north-star reference points.

## Build

```bash
# Dev preview from the synthetic fixtures in tests/fixtures/ (clearly labelled):
python3 site/build.py --fixtures

# Real build from results/*.json:
python3 site/build.py

# Options:
#   --results-dir DIR   directory of results/*.json (default: results/)
#   --out DIR           output directory        (default: site/public)
#   --fixtures          preview from tests/fixtures/ (synthetic, banner-labelled)
#   --timestamp "..."   inject a fixed "generated" stamp (deterministic output)
```

## What gets emitted

Into `--out` (default `site/public/`):

- `index.html` — one fully self-contained page: **inline CSS + inline JS +
  inline SVG**. No external CDNs, fonts, or network requests. Theme-aware
  (light default, dark via `prefers-color-scheme` and a manual toggle).
- `leaderboard.json` — machine-readable leaderboard (`leaderboard`, `humans`,
  `budget_track`, `meta`, `generated`), models ranked by BenchPR ascending.

## Page sections

- **Header** — dataset hash, prompt / ASCII / image render versions, generation
  timestamp, and a plain-English explanation of BenchPR (lower is better;
  PR 2–4 = world-class) and the human north-star lines.
- **Leaderboard table** — rank, model, BenchPR (with CI when present),
  best-move accuracy, per-track columns (text / image), total cost, cost per
  position. Sortable by any column (inline JS).
- **Skill-vs-cost scatter** (SVG) — x = total cost (log), y = BenchPR
  (lower better), one point per model/track, with north-star reference lines at
  PR 2 / 4 / 8.
- **Per-tier bars** (SVG) — grouped BenchPR bars by tier T1–T4 per model.
- **Text-vs-image dumbbell** (SVG) — paired text/image BenchPR per model.
- **Fixed-budget track** — BenchPR @ budget table, shown when a run carries
  `manifest.budget_usd` / `aggregate.benchpr_at_budget`.
- **Graceful empty state** — with zero results files the page still builds and
  says "no runs yet".

Charts are deterministic, server-side inline SVG computed in pure Python (axes,
ticks, labels, legends). Given identical inputs (and an injected timestamp) the
output is byte-for-byte identical.

## Input contract

`schema/results.schema.json` is the frozen input shape. Unknown extra fields are
tolerated. A result entry flagged `"kind": "human"` at the top level is treated
as a **measured human-panel baseline** (PLAN.md §4.5): it is excluded from the
ranked model list, shown with a `human` badge, and plotted as a distinct diamond
marker on the scatter.

## Layout

- `build.py` — the generator (loading, aggregation, SVG charting, orchestration).
- `templates/` — the page shell (CSS + sort/theme JS) and HTML builders
  (`string`-free f-string builders; no jinja2).
- `public/` — built output (**gitignored**; regenerated from results).

## Testing / fixtures

`tests/fixtures/*.json` are **synthetic** result files (obviously-fake model
names like `example/synthetic-model-a`, a `"synthetic": true` marker) used for
tests and the `--fixtures` dev preview — they are not benchmark data. Tests live
in `tests/test_site_*.py`; run them with:

```bash
python3 -m pytest tests/test_site_*.py -q
```
