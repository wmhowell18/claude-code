# site/ — static leaderboard (GitHub Pages)

The v1 results website (PLAN.md §6). A small build step (`build.py`) reads
`results/*.json` and renders static HTML — no backend, no database, one-way data
flow (results JSON -> build -> static HTML).

Planned views:

- Leaderboard table: model, BenchPR (text), BenchPR (image), best-move
  accuracy, $/run, BenchPR @ $10.
- Skill-vs-cost scatter: BenchPR (y, lower is better) vs. $/run (x, log), with
  human north-star lines at PR 2 / 4 / 8.
- Per-tier bars (T1-T4) and text-vs-image comparison.
- Position explorer (public dev set only).

Layout:

- `build.py` — the generator.
- `templates/` — page + chart templates.
- `public/` — built output (gitignored; regenerated from results).

Charts follow the repo's dataviz conventions: light/dark, theme-aware,
self-contained. Keep v1 dead simple (one page + a couple of charts).
