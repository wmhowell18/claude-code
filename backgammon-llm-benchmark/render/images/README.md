# render/images/

Generated board images (PNG + source SVG), output of `render/svg.py` and
`render/raster.py`.

- Images for the **pilot** and **dev** splits are committed here.
- Images for the **held-out** split live under `render/images/heldout/` and are
  **gitignored** — they would leak the private positions (PLAN.md §2.5, §8).

See PLAN.md §1.2 for the rendering spec (house style, required on-board state,
render versions).
