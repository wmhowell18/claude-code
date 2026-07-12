"""Dead-simple templating for the static leaderboard (PLAN.md §6).

No jinja2 — just Python string builders. The page shell (CSS + sort JS) lives
here as plain string constants so ``build.py`` can concatenate the dynamic body
without brace-escaping gymnastics. All output is fully self-contained: inline
CSS, inline JS, inline SVG. No external CDNs, fonts, or network requests.
"""

from __future__ import annotations

import html as _html

# --------------------------------------------------------------------------
# Style — one clean default, theme-aware (light default + dark via media query
# and an explicit data-theme override).
# --------------------------------------------------------------------------

CSS = """
:root {
  --bg: #ffffff;
  --panel: #f6f7f9;
  --fg: #1a1d21;
  --muted: #5b6572;
  --border: #d9dee5;
  --accent: #2563eb;
  --text-track: #2563eb;
  --image-track: #d97706;
  --human: #059669;
  --ref: #b91c1c;
  --row-hover: #eef2f7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216;
    --panel: #171b21;
    --fg: #e7ebf0;
    --muted: #9aa4b2;
    --border: #2a313b;
    --accent: #60a5fa;
    --text-track: #60a5fa;
    --image-track: #fbbf24;
    --human: #34d399;
    --ref: #f87171;
    --row-hover: #1d232b;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1216; --panel: #171b21; --fg: #e7ebf0; --muted: #9aa4b2;
  --border: #2a313b; --accent: #60a5fa; --text-track: #60a5fa;
  --image-track: #fbbf24; --human: #34d399; --ref: #f87171; --row-hover: #1d232b;
}
:root[data-theme="light"] {
  --bg: #ffffff; --panel: #f6f7f9; --fg: #1a1d21; --muted: #5b6572;
  --border: #d9dee5; --accent: #2563eb; --text-track: #2563eb;
  --image-track: #d97706; --human: #059669; --ref: #b91c1c; --row-hover: #eef2f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 20px 80px; }
header.site { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 8px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 40px 0 8px; }
.sub { color: var(--muted); margin: 0 0 12px; }
.meta { color: var(--muted); font-size: 13px; margin: 8px 0 0; }
.meta code { background: var(--panel); padding: 1px 5px; border-radius: 4px; }
.explain {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; margin: 16px 0; font-size: 14px;
}
.explain b { color: var(--fg); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--muted); margin: 6px 0 2px; }
.legend .k { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 8px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child, th.left, td.left { text-align: left; }
thead th { background: var(--panel); cursor: pointer; user-select: none; position: relative; }
thead th:hover { color: var(--accent); }
thead th .arrow { color: var(--accent); font-size: 11px; }
tbody tr:hover { background: var(--row-hover); }
tbody tr:last-child td { border-bottom: none; }
.badge {
  display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px;
  background: var(--human); color: #fff; vertical-align: middle;
}
.chart { border: 1px solid var(--border); border-radius: 8px; background: var(--panel); padding: 8px; margin: 8px 0; overflow-x: auto; }
svg { display: block; max-width: 100%; height: auto; }
svg text { fill: var(--fg); }
svg .axis, svg .tick { stroke: var(--border); }
svg .grid { stroke: var(--border); stroke-opacity: 0.5; }
svg .muted { fill: var(--muted); }
.empty { padding: 60px 20px; text-align: center; color: var(--muted); font-size: 18px; }
.foot { margin-top: 48px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); padding-top: 12px; }
.themetoggle {
  float: right; font-size: 12px; cursor: pointer; color: var(--muted);
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
}
"""

# --------------------------------------------------------------------------
# Behaviour — column sorting + light/dark toggle. Plain, dependency-free JS.
# --------------------------------------------------------------------------

JS = """
(function () {
  function cellValue(row, idx) {
    var cell = row.children[idx];
    if (!cell) return "";
    var v = cell.getAttribute("data-sort");
    if (v === null) v = cell.textContent.trim();
    var n = parseFloat(v);
    return isNaN(n) ? v.toLowerCase() : n;
  }
  function makeSortable(table) {
    var heads = table.tHead ? table.tHead.rows[0].cells : [];
    for (var i = 0; i < heads.length; i++) {
      (function (idx, th) {
        th.addEventListener("click", function () {
          var body = table.tBodies[0];
          var rows = Array.prototype.slice.call(body.rows);
          var asc = th.getAttribute("data-dir") !== "asc";
          rows.sort(function (a, b) {
            var av = cellValue(a, idx), bv = cellValue(b, idx);
            if (av < bv) return asc ? -1 : 1;
            if (av > bv) return asc ? 1 : -1;
            return 0;
          });
          for (var h = 0; h < heads.length; h++) {
            heads[h].removeAttribute("data-dir");
            var ar = heads[h].querySelector(".arrow");
            if (ar) ar.textContent = "";
          }
          th.setAttribute("data-dir", asc ? "asc" : "desc");
          var arrow = th.querySelector(".arrow");
          if (arrow) arrow.textContent = asc ? " \\u25B2" : " \\u25BC";
          rows.forEach(function (r) { body.appendChild(r); });
        });
      })(i, heads[i]);
    }
  }
  var tables = document.querySelectorAll("table[data-sortable]");
  for (var t = 0; t < tables.length; t++) makeSortable(tables[t]);

  var toggle = document.getElementById("themetoggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var root = document.documentElement;
      var cur = root.getAttribute("data-theme");
      if (!cur) {
        cur = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      }
      root.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
    });
  }
})();
"""


def esc(text) -> str:
    """HTML-escape a value (stringifies first)."""
    return _html.escape("" if text is None else str(text))


def page(title: str, body: str) -> str:
    """Assemble the full self-contained HTML document."""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        '<div class="wrap">\n'
        f"{body}\n"
        "</div>\n"
        f"<script>{JS}</script>\n"
        "</body>\n</html>\n"
    )
