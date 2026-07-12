"""Static leaderboard build: results/*.json -> static HTML (PLAN.md §6).

One-way data flow (results JSON -> build -> static HTML, no backend). Renders the
leaderboard table, skill-vs-cost scatter with human north-star lines (PR 2/4/8),
per-tier bars, text-vs-image comparison, and the public-dev position explorer.
Charts follow the repo dataviz conventions (theme-aware, self-contained).
"""
