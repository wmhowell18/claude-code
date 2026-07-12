# site/ — static leaderboard (GitHub Pages)

The v1 results website (PLAN.md §6). A small build step (`build.py`) reads
`results/*.json` and renders static HTML — no backend, no database, one-way data
flow (results JSON -> build -> static HTML). The build is stdlib-only.

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
