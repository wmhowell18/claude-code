# site/templates/

Page and chart scaffolding consumed by `site/build.py` (PLAN.md §6).

`__init__.py` holds the dead-simple templating (no jinja2):

- `CSS` / `JS` — the inline stylesheet (theme-aware, light + dark) and the
  dependency-free behaviour (column sorting + light/dark toggle).
- `page(title, body)` — assembles the full self-contained HTML document.
- `esc(text)` — HTML-escape helper.

The SVG charts and all dynamic HTML are built in `build.py`; this package only
provides the outer shell and escaping so the build stays brace-escape-free.
