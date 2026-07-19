# site/ — static leaderboard (GitHub Pages)

The v1 results website (PLAN.md §6). A small build step (`build.py`) reads
`results/*.json` and renders static HTML — no backend, no database, one-way data
flow (results JSON -> build -> static HTML). The build is stdlib-only.

## Human-benchmark quiz (`human-benchmark-pilot.html`)

`scripts/build_human_benchmark.py` generates a **single self-contained HTML quiz**
that lets human panelists sit the exact same 50-position pilot set the LLMs are
scored on, producing a directly comparable **BenchPR**. Everything is inlined
(CSS, JS, and the rollout answer key as a JSON blob) — zero network requests — so
the file can be emailed and opened locally. There are **no pre-baked SVGs**: a
single runtime JS board engine (a faithful port of `render/svg.py`) draws every
position from structured data and re-renders after each click.

```bash
# Generate site/public/human-benchmark-pilot.html (stdlib only):
python3 scripts/build_human_benchmark.py

# Options:
#   --out PATH          output file (default: site/public/human-benchmark-pilot.html)
#   --timestamp "..."   fixed ISO manifest timestamp (default: now, UTC)
```

### Run modes

The intro screen offers two modes (persisted in the saved state and recorded in
the exported results JSON as a top-level `mode` field):

- **Practice — feedback after each answer** (default). After every submission the
  quiz shows a feedback panel *before advancing*: whether the answer was best, the
  engine's best play/action, the panelist's equity loss in millipoints, the rank
  of their choice among the rollout's listed moves/actions, and — for checker
  decisions — the board redrawn with the best play applied. A **Next position**
  button advances. Good for learning.
- **Blind panel run — results only at the end**. The original protocol: no engine
  ground truth is shown until all 50 positions are done. Use this for a clean
  benchmark run. (The mode is locked once a run has any answers; "Start over"
  clears it.)

Both modes end on the same results screen and export the same results JSON shape.

### What the panelist sees

- An intro screen (instructions + a run-mode chooser + a name/identifier field).
- One position at a time with a `Position N / 50` progress bar, deterministic
  order (sorted by `position_id`). Every diagram is shown from the on-roll
  player's perspective — **the panelist always plays the White checkers**
  (opponent is Black). `board_json` is authoritative and already mover-relative
  (positive/"x" = on roll, confirmed by both `ids/xgid.py` and `ids/gnubg_id.py`),
  so its cube owner / score / pips are read as-is — **cube decisions are never
  color-flipped** (this is what keeps "you own the cube" on the right side). A
  handful of checker rollouts were computed in the color-mirror frame; those
  positions are presented as `flip(board_json)` — a legal, symmetric equivalent
  where the on-roll player is still White — chosen per position by which
  orientation makes the rollout's moves legal, so the diagram matches the answer
  key. Checker fills are pushed to true white/black so the UI can say "White"/
  "Black". Shown: the board SVG, dice (checker decisions only — cube records
  carry no dice), cube value/owner, money-vs-match + score, and pip counts. Tier,
  phase, expected-loss and other difficulty/answer hints are **not** shown.
- Checker decisions are composed by **clicking checkers** on the live board: the
  engine (a JS port of `bgcore/moves.py`'s single-die hop rules) highlights legal
  destinations and only enables **Submit play** once the composed sequence reaches
  a legal full-move signature enumerated at build time (`Undo` / `Reset`
  available). A collapsed secondary **"Prefer to type the move?"** panel still
  accepts free-text notation (JS port of `bgcore/notation.py` — reordered plays,
  `*` hits, `(n)` repeats, `bar/`/`/off`) with live legality validation; a move
  that reaches the same position spelled differently (e.g. a single checker's
  intermediate point, `10/4/3` for `10/3`) matches via a pre-computed endpoint map
  and resulting-position fallback (mirrors `bgcore.moves.moves_equivalent`).
  Unrecognised typed input warns once, then may be submitted anyway.
- Cube decisions are buttons for exactly the actions the record poses
  (`No double` / `Double, Take` / `Double, Pass`).
- No going back; answers persist to `localStorage`, so a tab close doesn't lose
  progress. The saved state is **version-stamped**: on a schema mismatch (older
  quiz build) or corrupt storage the page offers a clean "start fresh" instead of
  crashing or mixing schemas. Boot is wrapped in an error boundary that surfaces a
  readable error box rather than a blank page, and double-clicking a submit never
  double-records or double-advances. Reloading mid-quiz resumes at the right
  question in both modes.
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
   `kind: "human"`, `model: "human-panel/<name>"`, `track: "text"`, plus an
   additive `mode: "practice" | "blind"` recording how the run was taken).
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
