"""Dead-simple templating for the GammonBench static leaderboard (PLAN.md §6).

No jinja2 — just Python string builders. The page shell (CSS + sort/theme JS)
lives here as plain string constants so ``build.py`` can concatenate the dynamic
body without brace-escaping gymnastics. All output is fully self-contained:
inline CSS, inline JS, inline SVG. No external CDNs, fonts, or network requests.

Design tokens follow the ``dataviz`` skill's validated default palette (blue /
orange / green categorical hues, a blue ordinal ramp for difficulty tiers),
exposed as CSS custom properties so light and dark swap in one place. The page is
theme-aware: light default, dark via ``prefers-color-scheme`` and a persisted
manual toggle.
"""

from __future__ import annotations

import html as _html

# --------------------------------------------------------------------------
# Style — token-level custom properties, theme-aware (light default + dark via
# media query and an explicit data-theme override that always wins).
# Palette sourced from the dataviz skill reference (validated for CVD/contrast).
# --------------------------------------------------------------------------

CSS = """
:root {
  color-scheme: light;
  --plane: #f4f5f3;
  --surface: #ffffff;
  --surface-2: #f7f8f6;
  --fg: #14161a;
  --fg2: #4b5563;
  --muted: #7c8593;
  --border: #e3e5e2;
  --grid: #e6e8e4;
  --axis: #c6cabf;
  --accent: #2a78d6;
  --text-track: #2a78d6;
  --image-track: #eb6834;
  --human: #008300;
  --checker: #1baf7a;
  --cube: #6d5ae0;
  --ref: #6b7280;
  --good: #0ca30c;
  --row-hover: #f1f4f8;
  --tier-1: #86b6ef;
  --tier-2: #5598e7;
  --tier-3: #2a78d6;
  --tier-4: #184f95;
  --gold: #c99700;
  --silver: #8a94a3;
  --bronze: #b06a2c;
  --shadow: 0 1px 2px rgba(15,20,30,0.05), 0 1px 3px rgba(15,20,30,0.04);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0c0e12;
    --surface: #14181e;
    --surface-2: #1a1f27;
    --fg: #eef1f5;
    --fg2: #b3bcc8;
    --muted: #8b95a3;
    --border: #29303a;
    --grid: #262d37;
    --axis: #3a424e;
    --accent: #3987e5;
    --text-track: #3987e5;
    --image-track: #d95926;
    --human: #29a329;
    --checker: #199e70;
    --cube: #9085e9;
    --ref: #9aa4b2;
    --good: #29c229;
    --row-hover: #1b212a;
    --tier-1: #b7d3f6;
    --tier-2: #6da7ec;
    --tier-3: #3987e5;
    --tier-4: #256abf;
    --gold: #e2b53a;
    --silver: #aab3c0;
    --bronze: #cf894f;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 1px 3px rgba(0,0,0,0.3);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #0c0e12; --surface: #14181e; --surface-2: #1a1f27; --fg: #eef1f5;
  --fg2: #b3bcc8; --muted: #8b95a3; --border: #29303a; --grid: #262d37;
  --axis: #3a424e; --accent: #3987e5; --text-track: #3987e5; --image-track: #d95926;
  --human: #29a329; --checker: #199e70; --cube: #9085e9; --ref: #9aa4b2;
  --good: #29c229; --row-hover: #1b212a; --tier-1: #b7d3f6; --tier-2: #6da7ec;
  --tier-3: #3987e5; --tier-4: #256abf; --gold: #e2b53a; --silver: #aab3c0;
  --bronze: #cf894f; --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 1px 3px rgba(0,0,0,0.3);
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--plane);
  color: var(--fg);
  font: 15px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1120px; margin: 0 auto; padding: 20px 20px 96px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.86em;
}

/* ---- top bar / wordmark ---- */
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 4px 0 18px; }
.wordmark { display: inline-flex; align-items: baseline; gap: 8px; font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
.wordmark .die { font-size: 22px; line-height: 1; transform: translateY(1px); color: var(--accent); }
.wordmark .b1 { color: var(--fg); }
.wordmark .b2 { color: var(--accent); }
.wordmark .tag {
  font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); border: 1px solid var(--border); border-radius: 999px;
  padding: 2px 7px; transform: translateY(-2px);
}
.themetoggle {
  font: inherit; font-size: 12px; cursor: pointer; color: var(--fg2);
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 11px; display: inline-flex; align-items: center; gap: 6px;
}
.themetoggle:hover { border-color: var(--axis); color: var(--fg); }

/* ---- hero ---- */
.hero { margin: 2px 0 6px; }
.hero h1 { font-size: clamp(26px, 4vw, 38px); line-height: 1.1; margin: 0 0 8px; letter-spacing: -0.02em; }
.hero .lede { color: var(--fg2); font-size: clamp(15px, 2vw, 18px); max-width: 60ch; margin: 0 0 16px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 4px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--fg2);
  background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 4px 11px;
}
.chip b { color: var(--fg); font-weight: 600; }
.chip code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; color: var(--fg2); }

/* ---- stat tiles ---- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 22px 0 8px; }
.tile {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 16px 18px; box-shadow: var(--shadow); position: relative; overflow: hidden;
}
.tile .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 8px; font-weight: 600; }
.tile .value { font-size: 30px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
.tile .value .unit { font-size: 15px; font-weight: 600; color: var(--muted); margin-left: 3px; }
.tile .sub2 { font-size: 13px; color: var(--fg2); margin-top: 7px; }
.tile.accent { border-color: color-mix(in srgb, var(--accent) 45%, var(--border)); }
.tile.accent::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.tile.human-tile::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--human); }

/* ---- sections ---- */
section { margin-top: 44px; }
.sec-head { margin: 0 0 4px; }
h2 { font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--fg2); margin: 0 0 14px; max-width: 72ch; font-size: 14px; }

/* ---- table ---- */
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 14px; background: var(--surface); box-shadow: var(--shadow); }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 11px 14px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--border); }
td { font-variant-numeric: tabular-nums; }
th:first-child, td:first-child, th.left, td.left { text-align: left; }
thead th {
  background: var(--surface-2); cursor: pointer; user-select: none; position: sticky; top: 0;
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); font-weight: 600;
}
thead th:hover { color: var(--accent); }
thead th.sorted { color: var(--fg); }
thead th .arrow { color: var(--accent); font-size: 10px; }
tbody tr:hover { background: var(--row-hover); }
tbody tr:last-child td { border-bottom: none; }
tbody td.primary { font-weight: 700; color: var(--fg); font-size: 15px; }
.modelcell { display: inline-flex; align-items: center; gap: 8px; }
.modelname { font-weight: 600; }
.rankchip {
  display: inline-flex; align-items: center; justify-content: center; min-width: 26px; height: 24px;
  border-radius: 7px; font-weight: 700; font-size: 13px; padding: 0 6px;
  background: var(--surface-2); color: var(--fg2); border: 1px solid var(--border);
}
.rankchip.r1 { background: color-mix(in srgb, var(--gold) 22%, var(--surface)); color: var(--gold); border-color: color-mix(in srgb, var(--gold) 45%, var(--border)); }
.rankchip.r2 { background: color-mix(in srgb, var(--silver) 22%, var(--surface)); color: var(--silver); border-color: color-mix(in srgb, var(--silver) 45%, var(--border)); }
.rankchip.r3 { background: color-mix(in srgb, var(--bronze) 22%, var(--surface)); color: var(--bronze); border-color: color-mix(in srgb, var(--bronze) 45%, var(--border)); }
tr.human-row { background: color-mix(in srgb, var(--human) 6%, var(--surface)); }
tr.human-row:hover { background: color-mix(in srgb, var(--human) 11%, var(--surface)); }
tr.human-row td { border-top: 1px dashed color-mix(in srgb, var(--human) 40%, var(--border)); }
.badge {
  display: inline-block; font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
  background: color-mix(in srgb, var(--human) 16%, var(--surface)); color: var(--human);
  border: 1px solid color-mix(in srgb, var(--human) 40%, var(--border));
  letter-spacing: 0.03em; text-transform: uppercase; vertical-align: middle;
}
.ci { color: var(--muted); font-weight: 400; font-size: 12px; }
.dim { color: var(--muted); }
.split { display: inline-flex; align-items: center; gap: 5px; }
.split .k { display: inline-flex; align-items: center; gap: 3px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.checker { background: var(--checker); }
.dot.cube { background: var(--cube); }

/* ---- charts ---- */
.chart {
  border: 1px solid var(--border); border-radius: 14px; background: var(--surface);
  padding: 10px 12px; margin: 6px 0 0; overflow-x: auto; box-shadow: var(--shadow);
}
svg { display: block; max-width: 100%; height: auto; margin: 0 auto; }
svg text { fill: var(--fg); }
svg .axis { stroke: var(--axis); }
svg .grid { stroke: var(--grid); }
svg .muted { fill: var(--muted); }
svg .lbl { fill: var(--fg2); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--fg2); margin: 10px 2px 2px; }
.legend .k { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.swatch.diamond { border-radius: 2px; transform: rotate(45deg); width: 10px; height: 10px; }

/* ---- explainer / methodology ---- */
.explain {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 18px; margin: 16px 0; font-size: 14px; color: var(--fg2); box-shadow: var(--shadow);
}
.explain b { color: var(--fg); }
.scale { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.scale .band { font-size: 12px; padding: 3px 9px; border-radius: 7px; color: #fff; font-weight: 600; white-space: nowrap; }
.methods { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.methods a {
  font-size: 13px; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface-2); color: var(--fg2);
}
.methods a:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }

.banner {
  background: color-mix(in srgb, var(--image-track) 12%, var(--surface));
  border: 1px solid color-mix(in srgb, var(--image-track) 45%, var(--border));
  color: var(--fg); border-radius: 12px; padding: 11px 16px; margin: 14px 0 0; font-size: 13.5px;
}
.banner b { color: var(--image-track); }
.empty { padding: 72px 20px; text-align: center; color: var(--muted); font-size: 18px; }
.foot { margin-top: 56px; color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--border); padding-top: 16px; line-height: 1.6; }
.foot .fnote { margin: 0 0 8px; }
.foot b { color: var(--fg2); }

@media (max-width: 560px) {
  .wrap { padding: 16px 14px 72px; }
  .tile .value { font-size: 26px; }
  th, td { padding: 9px 11px; }
}
@media (prefers-reduced-motion: no-preference) {
  .themetoggle, .methods a, thead th { transition: color .12s ease, border-color .12s ease, background .12s ease; }
}
"""

# --------------------------------------------------------------------------
# Behaviour — column sorting + persisted light/dark toggle. Dependency-free JS.
# --------------------------------------------------------------------------

# A tiny head script sets the stored theme before first paint (no flash).
HEAD_JS = """
(function(){try{var t=localStorage.getItem('gb-theme');if(t==='dark'||t==='light'){document.documentElement.setAttribute('data-theme',t);}}catch(e){}})();
"""

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
            heads[h].classList.remove("sorted");
            var ar = heads[h].querySelector(".arrow");
            if (ar) ar.textContent = "";
          }
          th.setAttribute("data-dir", asc ? "asc" : "desc");
          th.classList.add("sorted");
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
      var next = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("gb-theme", next); } catch (e) {}
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
        f"<script>{HEAD_JS}</script>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        '<div class="wrap">\n'
        f"{body}\n"
        "</div>\n"
        f"<script>{JS}</script>\n"
        "</body>\n</html>\n"
    )
