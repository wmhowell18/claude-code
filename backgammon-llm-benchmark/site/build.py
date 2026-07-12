"""Static leaderboard build: results/*.json -> static HTML (PLAN.md §6).

One-way data flow (results JSON -> build -> static HTML, no backend). Renders the
leaderboard table, skill-vs-cost scatter with human north-star lines (PR 2/4/8),
per-tier bars, text-vs-image comparison, and the fixed-budget track section.
Charts are deterministic, server-side inline SVG computed in pure Python. The
page is fully self-contained: inline CSS + inline JS + inline SVG, no external
CDNs, fonts, or network requests.

Stdlib only. Run ``python3 site/build.py --fixtures`` for a dev preview built
from ``tests/fixtures/`` (clearly-labelled synthetic data). ``site/public/`` is
gitignored and regenerated from ``results/*.json``.

The result-file shape is ``schema/results.schema.json`` (the frozen input
contract). Unknown extra fields are tolerated. A result entry flagged
``"kind": "human"`` (top level) is treated as a measured human-panel baseline
rather than a ranked model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

# Import the sibling ``templates`` package. ``site`` collides with a stdlib
# module name, so we put this directory on sys.path and import ``templates``
# directly instead of ``site.templates``.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import templates  # noqa: E402
from templates import esc  # noqa: E402

REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DEFAULT_FIXTURES_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")
DEFAULT_OUT_DIR = os.path.join(_HERE, "public")

# Human north-star reference PRs (PLAN.md §4.5).
REF_LINES = [(2.0, "PR 2 · world-class"), (4.0, "PR 4 · elite"), (8.0, "PR 8 · strong club")]
TIERS = ["T1", "T2", "T3", "T4"]


# ==========================================================================
# Loading + aggregation
# ==========================================================================


def load_results(results_dir):
    """Read every ``*.json`` in ``results_dir`` (sorted for determinism)."""
    out = []
    if not os.path.isdir(results_dir):
        return out
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def _ts(run):
    return (run.get("manifest") or {}).get("timestamp") or ""


def _num(value):
    """Coerce to float, tolerating None/garbage."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_runs(results):
    """Latest run per (model, track), keyed by timestamp (PLAN.md §5.2)."""
    by_key = {}
    for run in results:
        key = (run.get("model"), run.get("track"))
        if key not in by_key or _ts(run) >= _ts(by_key[key]):
            by_key[key] = run
    return by_key


def _agg(run):
    return run.get("aggregate") or {}


def _benchpr(run):
    return _num(_agg(run).get("benchpr"))


def _benchpr_ci(run):
    """Optional CI as [low, high]; tolerated extra field, several spellings."""
    a = _agg(run)
    ci = a.get("benchpr_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo, hi = _num(ci[0]), _num(ci[1])
        if lo is not None and hi is not None:
            return [lo, hi]
    lo, hi = _num(a.get("benchpr_ci_low")), _num(a.get("benchpr_ci_high"))
    if lo is not None and hi is not None:
        return [lo, hi]
    return None


def _cost(run):
    return _num(_agg(run).get("cost_usd"))


def _n_positions(run):
    decisions = run.get("decisions")
    if isinstance(decisions, list) and decisions:
        return len(decisions)
    per_tier = _agg(run).get("per_tier") or {}
    total = 0
    seen = False
    for stats in per_tier.values():
        n = _num((stats or {}).get("n"))
        if n is not None:
            total += int(n)
            seen = True
    return total if seen else None


def _is_human(run):
    return run.get("kind") == "human"


def build_models(latest):
    """Group latest (model, track) runs into per-model entries.

    Returns a list of dicts with combined text/image track metrics. Human-panel
    entries (``kind == "human"``) are flagged so callers can rank models and
    plot humans separately.
    """
    order = []
    models = {}
    for (model, track), run in latest.items():
        entry = models.get(model)
        if entry is None:
            entry = {"model": model, "tracks": {}, "human": False}
            models[model] = entry
            order.append(model)
        entry["tracks"][track] = run
        if _is_human(run):
            entry["human"] = True

    rows = []
    for model in order:
        entry = models[model]
        tracks = entry["tracks"]
        # Primary track for the headline BenchPR: text is the primary track
        # (PLAN.md §4.1); fall back to image, then anything present.
        primary = None
        for key in ("text", "image", "text+image"):
            if key in tracks:
                primary = tracks[key]
                break
        if primary is None and tracks:
            primary = next(iter(tracks.values()))

        text_run = tracks.get("text")
        image_run = tracks.get("image")
        cost_total = 0.0
        have_cost = False
        for run in tracks.values():
            c = _cost(run)
            if c is not None:
                cost_total += c
                have_cost = True
        npos = _n_positions(primary) if primary else None
        cost_per_pos = (cost_total / npos) if (have_cost and npos) else None

        rows.append(
            {
                "model": model,
                "human": entry["human"],
                "benchpr": _benchpr(primary) if primary else None,
                "benchpr_ci": _benchpr_ci(primary) if primary else None,
                "benchpr_text": _benchpr(text_run) if text_run else None,
                "benchpr_image": _benchpr(image_run) if image_run else None,
                "best_move_accuracy": _num(_agg(primary).get("best_move_accuracy")) if primary else None,
                "cost_usd": cost_total if have_cost else None,
                "cost_per_position": cost_per_pos,
                "benchpr_at_budget": _num(_agg(primary).get("benchpr_at_budget")) if primary else None,
                "per_tier": (_agg(primary).get("per_tier") or {}) if primary else {},
                "n_positions": npos,
            }
        )
    return rows


def _sort_key(row):
    b = row["benchpr"]
    return (b is None, b if b is not None else 0.0, row["model"])


def rank_models(rows):
    """Split into ranked models (kind != human) and human baselines.

    Models are ranked by BenchPR ascending (lower is better, PLAN.md §4.4).
    """
    models = sorted((r for r in rows if not r["human"]), key=_sort_key)
    for i, r in enumerate(models, start=1):
        r["rank"] = i
    humans = sorted((r for r in rows if r["human"]), key=_sort_key)
    for r in humans:
        r["rank"] = None
    return models, humans


def budget_rows(latest):
    """Runs carrying a fixed-budget result (PLAN.md §5.3)."""
    out = []
    for (model, track), run in latest.items():
        a = _agg(run)
        manifest = run.get("manifest") or {}
        budget = _num(manifest.get("budget_usd"))
        at_budget = _num(a.get("benchpr_at_budget"))
        if budget is not None or at_budget is not None:
            out.append(
                {
                    "model": model,
                    "track": track,
                    "human": _is_human(run),
                    "budget_usd": budget,
                    "benchpr_at_budget": at_budget,
                    "benchpr": _benchpr(run),
                }
            )
    out.sort(key=lambda r: (r["benchpr_at_budget"] is None, r["benchpr_at_budget"] or 0.0, r["model"]))
    return out


def site_meta(results):
    """Header metadata from the most recent run's manifest."""
    if not results:
        return {}
    latest = max(results, key=_ts)
    m = latest.get("manifest") or {}
    return {
        "dataset_hash": m.get("dataset_hash"),
        "prompt_version": m.get("prompt_version"),
        "ascii_render_version": m.get("ascii_render_version"),
        "image_render_version": m.get("image_render_version"),
        "latest_run_timestamp": m.get("timestamp"),
    }


# ==========================================================================
# Number formatting
# ==========================================================================


def fmt_pr(v):
    return "—" if v is None else f"{v:.2f}"


def fmt_pct(v):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.1f}%"


def fmt_usd(v):
    if v is None:
        return "—"
    if v < 1:
        return f"${v:.3f}"
    return f"${v:,.2f}"


def fmt_usd_small(v):
    if v is None:
        return "—"
    return f"${v:.4f}"


# ==========================================================================
# SVG charting (deterministic, computed in Python)
# ==========================================================================


def _svg_open(w, h):
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="12">'
    )


def _nice_ceiling(v):
    """A pleasant round upper bound >= v for a linear axis."""
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10 ** exp
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * base:
            return m * base
    return 10 * base


def _y_ticks(maxv, count=5):
    step = _nice_ceiling(maxv / count)
    ticks = []
    t = 0.0
    while t <= maxv + 1e-9:
        ticks.append(round(t, 6))
        t += step
    if ticks[-1] < maxv:
        ticks.append(round(ticks[-1] + step, 6))
    return ticks


def _fmt_num(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def scatter_svg(points, humans):
    """Skill-vs-cost scatter: x=cost (log), y=BenchPR (lower better).

    ``points``  : list of dicts {label, cost, benchpr, track}.
    ``humans``  : list of dicts {label, cost, benchpr} plotted as distinct
                  diamond markers (measured human-panel baselines, PLAN.md §4.5).
    Horizontal reference lines are drawn at PR 2 / 4 / 8.
    """
    W, H = 720, 420
    ML, MR, MT, MB = 60, 150, 24, 52
    pw, ph = W - ML - MR, H - MT - MB

    all_pr = [p["benchpr"] for p in points if p.get("benchpr") is not None]
    all_pr += [h["benchpr"] for h in humans if h.get("benchpr") is not None]
    all_pr += [r[0] for r in REF_LINES]
    max_pr = _nice_ceiling(max(all_pr) if all_pr else 10.0)

    costs = [p["cost"] for p in points if p.get("cost") not in (None, 0)]
    if costs:
        lo, hi = min(costs), max(costs)
    else:
        lo, hi = 0.01, 1.0
    lo_e = math.floor(math.log10(lo))
    hi_e = math.ceil(math.log10(hi))
    if hi_e <= lo_e:
        hi_e = lo_e + 1

    def x_of(cost):
        if cost is None or cost <= 0:
            return ML  # humans / free entries pinned to the left lane
        frac = (math.log10(cost) - lo_e) / (hi_e - lo_e)
        return ML + frac * pw

    def y_of(pr):
        return MT + (pr / max_pr) * ph  # lower PR -> higher on screen (top)

    s = [_svg_open(W, H)]
    s.append(f'<text x="{ML}" y="16" class="muted">Skill vs. cost — lower BenchPR is better</text>')

    for t in _y_ticks(max_pr):
        y = y_of(t)
        s.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{ML + pw}" y2="{y:.1f}" />')
        s.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end" class="muted">{_fmt_num(t)}</text>')
    s.append(
        f'<text x="16" y="{MT + ph / 2:.0f}" transform="rotate(-90 16 {MT + ph / 2:.0f})" '
        f'text-anchor="middle" class="muted">BenchPR</text>'
    )

    e = lo_e
    while e <= hi_e:
        x = x_of(10 ** e)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + ph}" />')
        label = f"${_fmt_num(10 ** e)}" if e >= 0 else f"${10 ** e:g}"
        s.append(f'<text x="{x:.1f}" y="{MT + ph + 16}" text-anchor="middle" class="muted">{label}</text>')
        e += 1
    s.append(
        f'<text x="{ML + pw / 2:.0f}" y="{H - 6}" text-anchor="middle" class="muted">'
        "Total cost (USD, log scale)</text>"
    )

    s.append(f'<line class="axis" x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + ph}" />')
    s.append(f'<line class="axis" x1="{ML}" y1="{MT + ph}" x2="{ML + pw}" y2="{MT + ph}" />')

    for pr, label in REF_LINES:
        y = y_of(pr)
        s.append(
            f'<line class="refline" data-pr="{_fmt_num(pr)}" x1="{ML}" y1="{y:.1f}" '
            f'x2="{ML + pw}" y2="{y:.1f}" stroke="var(--ref)" stroke-width="1" '
            f'stroke-dasharray="5 4" />'
        )
        s.append(f'<text x="{ML + pw + 6}" y="{y + 4:.1f}" fill="var(--ref)">{esc(label)}</text>')

    for p in points:
        pr = p.get("benchpr")
        if pr is None:
            continue
        x, y = x_of(p.get("cost")), y_of(pr)
        color = "var(--image-track)" if p.get("track") == "image" else "var(--text-track)"
        s.append(
            f'<circle class="pt" data-track="{esc(p.get("track"))}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="5" fill="{color}" fill-opacity="0.85" stroke="var(--bg)" stroke-width="1" />'
        )

    for h in humans:
        pr = h.get("benchpr")
        if pr is None:
            continue
        x, y = x_of(h.get("cost")), y_of(pr)
        d = 6
        s.append(
            f'<polygon class="human-marker" points="{x:.1f},{y - d:.1f} {x + d:.1f},{y:.1f} '
            f'{x:.1f},{y + d:.1f} {x - d:.1f},{y:.1f}" fill="var(--human)" '
            f'stroke="var(--bg)" stroke-width="1" />'
        )
        s.append(f'<text x="{x + d + 3:.1f}" y="{y + 4:.1f}" fill="var(--human)">{esc(h.get("label"))}</text>')

    lx, ly = ML + pw + 6, MT + 20
    legend = [("var(--text-track)", "circle", "text track"), ("var(--image-track)", "circle", "image track")]
    if humans:
        legend.append(("var(--human)", "diamond", "human panel"))
    for i, (color, shape, label) in enumerate(legend):
        yy = ly + i * 18
        if shape == "diamond":
            s.append(f'<polygon points="{lx + 5},{yy - 5} {lx + 10},{yy} {lx + 5},{yy + 5} {lx},{yy} " fill="{color}" />')
        else:
            s.append(f'<circle cx="{lx + 5}" cy="{yy}" r="5" fill="{color}" />')
        s.append(f'<text x="{lx + 16}" y="{yy + 4}" class="muted">{esc(label)}</text>')

    s.append("</svg>")
    return "".join(s)


def tier_bars_svg(models):
    """Grouped BenchPR bars by tier T1–T4, one group per model (PLAN.md §6)."""
    ranked = [m for m in models if m.get("per_tier")]
    if not ranked:
        return '<p class="sub">No per-tier data in the current results.</p>'

    ML, MR, MT, MB = 46, 16, 28, 64
    group_w = 108
    W = max(720, ML + MR + group_w * len(ranked))
    H = 320
    pw = W - ML - MR
    ph = H - MT - MB

    values = []
    for m in ranked:
        for t in TIERS:
            v = _num((m["per_tier"].get(t) or {}).get("benchpr"))
            if v is not None:
                values.append(v)
    max_pr = _nice_ceiling(max(values) if values else 10.0)

    tier_colors = {"T1": "#22c55e", "T2": "#3b82f6", "T3": "#f59e0b", "T4": "#ef4444"}

    s = [_svg_open(W, H)]
    s.append(f'<text x="{ML}" y="16" class="muted">BenchPR by difficulty tier (lower is better)</text>')

    def y_of(pr):
        return MT + ph - (pr / max_pr) * ph

    for t in _y_ticks(max_pr):
        y = y_of(t)
        s.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" />')
        s.append(f'<text x="{ML - 6}" y="{y + 4:.1f}" text-anchor="end" class="muted">{_fmt_num(t)}</text>')

    s.append(f'<line class="axis" x1="{ML}" y1="{MT + ph}" x2="{W - MR}" y2="{MT + ph}" />')

    gw = pw / len(ranked)
    bar_area = gw * 0.7
    bw = bar_area / len(TIERS)
    for gi, m in enumerate(ranked):
        gx = ML + gi * gw + (gw - bar_area) / 2
        for ti, t in enumerate(TIERS):
            v = _num((m["per_tier"].get(t) or {}).get("benchpr"))
            x = gx + ti * bw
            if v is None:
                continue
            bh = (v / max_pr) * ph
            y = MT + ph - bh
            s.append(
                f'<rect class="tier-bar" data-tier="{t}" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bw - 2:.1f}" height="{bh:.1f}" fill="{tier_colors[t]}" />'
            )
        label = m["model"].split("/")[-1]
        cx = ML + gi * gw + gw / 2
        s.append(
            f'<text x="{cx:.0f}" y="{MT + ph + 16}" text-anchor="end" class="muted" '
            f'transform="rotate(-25 {cx:.0f} {MT + ph + 16})">{esc(label)}</text>'
        )

    for ti, t in enumerate(TIERS):
        x = ML + ti * 70
        s.append(f'<rect x="{x}" y="{H - 20}" width="11" height="11" fill="{tier_colors[t]}" />')
        s.append(f'<text x="{x + 15}" y="{H - 10}" class="muted">{t}</text>')

    s.append("</svg>")
    return "".join(s)


def dumbbell_svg(models):
    """Text-vs-image dumbbell: one row per model, connecting the two tracks."""
    rows = [m for m in models if m.get("benchpr_text") is not None or m.get("benchpr_image") is not None]
    if not rows:
        return '<p class="sub">No text/image comparison data in the current results.</p>'

    W = 720
    ML, MR, MT, MB = 150, 130, 28, 40
    row_h = 34
    H = MT + MB + row_h * len(rows)
    pw = W - ML - MR

    vals = []
    for m in rows:
        for k in ("benchpr_text", "benchpr_image"):
            if m[k] is not None:
                vals.append(m[k])
    max_pr = _nice_ceiling(max(vals) if vals else 10.0)

    def x_of(pr):
        return ML + (pr / max_pr) * pw

    s = [_svg_open(W, H)]
    s.append(f'<text x="{ML}" y="16" class="muted">Text vs. image BenchPR per model (lower is better)</text>')

    for t in _y_ticks(max_pr):
        x = x_of(t)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + row_h * len(rows)}" />')
        s.append(f'<text x="{x:.1f}" y="{H - 12}" text-anchor="middle" class="muted">{_fmt_num(t)}</text>')

    for i, m in enumerate(rows):
        cy = MT + i * row_h + row_h / 2
        label = m["model"].split("/")[-1]
        s.append(f'<text x="{ML - 8}" y="{cy + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        tv, iv = m["benchpr_text"], m["benchpr_image"]
        if tv is not None and iv is not None:
            s.append(
                f'<line x1="{x_of(tv):.1f}" y1="{cy:.1f}" x2="{x_of(iv):.1f}" y2="{cy:.1f}" '
                f'stroke="var(--border)" stroke-width="2" />'
            )
        if tv is not None:
            s.append(f'<circle class="dot-text" cx="{x_of(tv):.1f}" cy="{cy:.1f}" r="5.5" fill="var(--text-track)" />')
        if iv is not None:
            s.append(f'<circle class="dot-image" cx="{x_of(iv):.1f}" cy="{cy:.1f}" r="5.5" fill="var(--image-track)" />')

    lx = W - MR + 6
    s.append(
        f'<circle cx="{lx + 5}" cy="{MT + 8}" r="5" fill="var(--text-track)" />'
        f'<text x="{lx + 16}" y="{MT + 12}" class="muted">text</text>'
    )
    s.append(
        f'<circle cx="{lx + 5}" cy="{MT + 26}" r="5" fill="var(--image-track)" />'
        f'<text x="{lx + 16}" y="{MT + 30}" class="muted">image</text>'
    )
    s.append("</svg>")
    return "".join(s)


# ==========================================================================
# HTML sections
# ==========================================================================


def _th(label, key, numeric=True, left=False):
    cls = ' class="left"' if left else ""
    ds = ' data-numeric="1"' if numeric else ""
    return f'<th{cls}{ds} data-key="{esc(key)}">{esc(label)}<span class="arrow"></span></th>'


def leaderboard_table_html(models, humans):
    heads = [
        _th("#", "rank", left=True),
        _th("Model", "model", numeric=False, left=True),
        _th("BenchPR", "benchpr"),
        _th("Best-move acc.", "acc"),
        _th("Text", "text"),
        _th("Image", "image"),
        _th("Total cost", "cost"),
        _th("Cost / pos", "costpos"),
    ]
    body = []
    for row in list(models) + list(humans):
        rank = row.get("rank")
        rank_cell = str(rank) if rank else '<span class="badge">human</span>'
        rank_sort = rank if rank else 9999
        b = row["benchpr"]
        ci = row.get("benchpr_ci")
        b_disp = fmt_pr(b)
        if ci:
            b_disp += f' <span class="muted">[{ci[0]:.2f}–{ci[1]:.2f}]</span>'
        name = esc(row["model"])
        if row.get("human"):
            name += ' <span class="badge">human</span>'
        acc = row["best_move_accuracy"]
        body.append(
            "<tr>"
            f'<td class="left" data-sort="{rank_sort}">{rank_cell}</td>'
            f'<td class="left" data-sort="{esc(row["model"])}">{name}</td>'
            f'<td data-sort="{b if b is not None else 1e9}">{b_disp}</td>'
            f'<td data-sort="{acc if acc is not None else -1}">{fmt_pct(acc)}</td>'
            f'<td data-sort="{row["benchpr_text"] if row["benchpr_text"] is not None else 1e9}">{fmt_pr(row["benchpr_text"])}</td>'
            f'<td data-sort="{row["benchpr_image"] if row["benchpr_image"] is not None else 1e9}">{fmt_pr(row["benchpr_image"])}</td>'
            f'<td data-sort="{row["cost_usd"] if row["cost_usd"] is not None else 1e9}">{fmt_usd(row["cost_usd"])}</td>'
            f'<td data-sort="{row["cost_per_position"] if row["cost_per_position"] is not None else 1e9}">{fmt_usd_small(row["cost_per_position"])}</td>'
            "</tr>"
        )
    return (
        '<div class="tablewrap"><table data-sortable>'
        f'<thead><tr>{"".join(heads)}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def budget_table_html(rows):
    if not rows:
        return ""
    body = []
    for r in rows:
        name = esc(r["model"])
        if r.get("human"):
            name += ' <span class="badge">human</span>'
        budget_label = fmt_usd(r["budget_usd"]) if r["budget_usd"] is not None else "—"
        body.append(
            "<tr>"
            f'<td class="left">{name}</td>'
            f'<td class="left">{esc(r["track"])}</td>'
            f'<td>{budget_label}</td>'
            f'<td>{fmt_pr(r["benchpr_at_budget"])}</td>'
            f'<td>{fmt_pr(r["benchpr"])}</td>'
            "</tr>"
        )
    return (
        '<h2 id="budget">Fixed-budget track</h2>'
        '<p class="sub">Best BenchPR each model reaches within a fixed dollar budget '
        "(PLAN.md §5.3). Lower is better.</p>"
        '<div class="tablewrap"><table data-sortable>'
        "<thead><tr>"
        f"{_th('Model', 'model', numeric=False, left=True)}"
        f"{_th('Track', 'track', numeric=False, left=True)}"
        f"{_th('Budget', 'budget')}"
        f"{_th('BenchPR @ budget', 'atbudget')}"
        f"{_th('BenchPR (full)', 'full')}"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def header_html(meta, generated, synthetic):
    banner = ""
    if synthetic:
        banner = (
            '<div class="explain" style="border-color:var(--image-track)">'
            "<b>SYNTHETIC PREVIEW.</b> Built from <code>tests/fixtures/</code> — "
            "fake models, fabricated numbers. Not benchmark results.</div>"
        )
    bits = []
    if meta.get("dataset_hash"):
        bits.append(f'dataset <code>{esc(str(meta["dataset_hash"])[:16])}</code>')
    if meta.get("prompt_version"):
        bits.append(f'prompt <code>{esc(meta["prompt_version"])}</code>')
    if meta.get("ascii_render_version"):
        bits.append(f'ascii render <code>{esc(meta["ascii_render_version"])}</code>')
    if meta.get("image_render_version"):
        bits.append(f'image render <code>{esc(meta["image_render_version"])}</code>')
    meta_line = " · ".join(bits) if bits else "no run metadata"
    return (
        '<header class="site">'
        '<span class="themetoggle" id="themetoggle">light / dark</span>'
        "<h1>Backgammon LLM Benchmark</h1>"
        '<p class="sub">How well do LLMs play backgammon? Scored against XG rollout '
        "ground truth as <b>BenchPR</b> — an XG-Performance-Rating-like error score.</p>"
        f"{banner}"
        '<div class="explain">'
        "<b>BenchPR = 500 × mean equity loss per decision. Lower is better.</b> "
        "It sits on the same axis as a human XG Performance Rating, so the north-star "
        "lines below read directly: <b>PR 2–4 is world-class</b>, PR 5–8 strong expert, "
        "PR 10–15 intermediate, PR 20+ beginner. A model at PR 2 plays like a top human."
        "</div>"
        f'<p class="meta">{meta_line}</p>'
        f'<p class="meta">Generated {esc(generated)}'
        + (f' · latest run {esc(meta["latest_run_timestamp"])}' if meta.get("latest_run_timestamp") else "")
        + "</p>"
        "</header>"
    )


def build_html(results, generated, synthetic=False):
    """Assemble the full page HTML from loaded result dicts."""
    meta = site_meta(results)
    header = header_html(meta, generated, synthetic)

    if not results:
        body = header + (
            '<div class="empty">No runs yet — once <code>results/*.json</code> '
            "exists, the leaderboard appears here.</div>"
        )
        return templates.page("Backgammon LLM Benchmark", body)

    latest = latest_runs(results)
    all_rows = build_models(latest)
    models, humans = rank_models(all_rows)
    budgets = budget_rows(latest)

    points = []
    human_pts = []
    for (model, track), run in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        pr = _benchpr(run)
        if pr is None:
            continue
        rec = {"label": model.split("/")[-1], "cost": _cost(run), "benchpr": pr, "track": track}
        if _is_human(run):
            human_pts.append(rec)
        else:
            points.append(rec)

    sections = [header]
    sections.append("<h2>Leaderboard</h2>")
    sections.append('<p class="sub">Ranked by BenchPR (lower is better). Click a column header to sort.</p>')
    sections.append(leaderboard_table_html(models, humans))

    sections.append("<h2>Skill vs. cost</h2>")
    sections.append(
        '<p class="sub">BenchPR against total dollar cost (log). Dashed red lines are the '
        "human north star (PR 2 / 4 / 8).</p>"
    )
    sections.append(f'<div class="chart">{scatter_svg(points, human_pts)}</div>')

    sections.append("<h2>Per-tier breakdown</h2>")
    sections.append('<p class="sub">Where models fall apart: BenchPR by difficulty tier T1–T4.</p>')
    sections.append(f'<div class="chart">{tier_bars_svg(models + humans)}</div>')

    sections.append("<h2>Text vs. image</h2>")
    sections.append('<p class="sub">Does seeing the board (image) help or hurt vs. text?</p>')
    sections.append(f'<div class="chart">{dumbbell_svg(models + humans)}</div>')

    budget_html = budget_table_html(budgets)
    if budget_html:
        sections.append(budget_html)

    sections.append(
        '<div class="foot">Static leaderboard · one-way data flow (results JSON → build → HTML) · '
        "self-contained, no external requests. See PLAN.md §6.</div>"
    )
    return templates.page("Backgammon LLM Benchmark", "\n".join(sections))


def build_leaderboard_json(results, generated):
    """Machine-readable leaderboard (PLAN.md §6)."""
    meta = site_meta(results)
    if not results:
        return {"generated": generated, "meta": meta, "leaderboard": [], "humans": [], "budget_track": []}
    latest = latest_runs(results)
    all_rows = build_models(latest)
    models, humans = rank_models(all_rows)
    budgets = budget_rows(latest)

    def _row(r):
        return {
            "rank": r.get("rank"),
            "model": r["model"],
            "benchpr": r["benchpr"],
            "benchpr_ci": r.get("benchpr_ci"),
            "benchpr_text": r["benchpr_text"],
            "benchpr_image": r["benchpr_image"],
            "best_move_accuracy": r["best_move_accuracy"],
            "cost_usd": r["cost_usd"],
            "cost_per_position": r["cost_per_position"],
            "benchpr_at_budget": r["benchpr_at_budget"],
            "n_positions": r["n_positions"],
        }

    return {
        "generated": generated,
        "meta": meta,
        "leaderboard": [_row(r) for r in models],
        "humans": [_row(r) for r in humans],
        "budget_track": budgets,
    }


# ==========================================================================
# Orchestration
# ==========================================================================


def build_site(results_dir, out_dir, generated=None, synthetic=False):
    """Read results, render, and write index.html + leaderboard.json.

    Returns ``(html, leaderboard_dict)``. ``generated`` may be injected for
    deterministic output in tests.
    """
    if generated is None:
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    results = load_results(results_dir)
    html_out = build_html(results, generated, synthetic=synthetic)
    board = build_leaderboard_json(results, generated)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html_out)
    with open(os.path.join(out_dir, "leaderboard.json"), "w", encoding="utf-8") as fh:
        json.dump(board, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return html_out, board


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the static backgammon-LLM leaderboard.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory of results/*.json (default: results/).")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory (default: site/public).")
    parser.add_argument("--fixtures", action="store_true", help="Dev preview from tests/fixtures/ (synthetic, labelled).")
    parser.add_argument("--timestamp", default=None, help="Inject a fixed 'generated' timestamp (determinism).")
    args = parser.parse_args(argv)

    results_dir = DEFAULT_FIXTURES_DIR if args.fixtures else args.results_dir
    _, board = build_site(results_dir, args.out, generated=args.timestamp, synthetic=args.fixtures)
    n = len(board["leaderboard"])
    h = len(board["humans"])
    print(f"Built {args.out}/index.html and leaderboard.json — {n} model(s), {h} human baseline(s), from {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
