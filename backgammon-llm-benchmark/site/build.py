"""GammonBench static leaderboard build: results/*.json -> static HTML (PLAN.md §6).

One-way data flow (results JSON -> build -> static HTML, no backend). Renders the
hero/stat tiles, ranked leaderboard table, skill-vs-cost scatter with human
north-star lines (PR 2/4/8), per-tier bars, checker-vs-cube and text-vs-image
comparisons, and the fixed-budget track section.
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


def _mean_equity_loss(run):
    return _num(_agg(run).get("mean_equity_loss"))


def _tokens(run):
    """Aggregate token count. Prefer an aggregate ``tokens`` field; else sum the
    per-decision usage (tolerated optional fields, schema §decisions.usage)."""
    a = _agg(run)
    for key in ("tokens", "total_tokens"):
        v = _num(a.get(key))
        if v is not None:
            return v
    decisions = run.get("decisions")
    if isinstance(decisions, list):
        total = 0.0
        seen = False
        for d in decisions:
            usage = (d or {}).get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                n = _num(usage.get(key))
                if n is not None:
                    total += n
                    seen = True
        if seen:
            return total
    return None


def _pdt_benchpr(run, kind):
    """BenchPR for a decision type (``checker`` / ``cube``) if present."""
    pdt = _agg(run).get("per_decision_type") or {}
    return _num((pdt.get(kind) or {}).get("benchpr"))


def _dataset_positions(run):
    """Number of scored positions, preferring the per-tier roll-up totals (the
    full scored set) over a possibly-truncated ``decisions`` list."""
    per_tier = _agg(run).get("per_tier") or {}
    total = 0
    seen = False
    for stats in per_tier.values():
        n = _num((stats or {}).get("n"))
        if n is not None:
            total += int(n)
            seen = True
    if seen:
        return total
    return _n_positions(run)


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
        tokens_total = 0.0
        have_tokens = False
        for run in tracks.values():
            c = _cost(run)
            if c is not None:
                cost_total += c
                have_cost = True
            tk = _tokens(run)
            if tk is not None:
                tokens_total += tk
                have_tokens = True
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
                "mean_equity_loss": _mean_equity_loss(primary) if primary else None,
                "benchpr_checker": _pdt_benchpr(primary, "checker") if primary else None,
                "benchpr_cube": _pdt_benchpr(primary, "cube") if primary else None,
                "cost_usd": cost_total if have_cost else None,
                "cost_per_position": cost_per_pos,
                "tokens": tokens_total if have_tokens else None,
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
    positions = None
    for run in results:
        p = _dataset_positions(run)
        if p:
            positions = max(positions or 0, p)
    return {
        "dataset_hash": m.get("dataset_hash"),
        "prompt_version": m.get("prompt_version"),
        "ascii_render_version": m.get("ascii_render_version"),
        "image_render_version": m.get("image_render_version"),
        "latest_run_timestamp": m.get("timestamp"),
        "positions": positions,
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


def fmt_tokens(v):
    """Compact token count: 1,284 / 12.9K / 4.2M."""
    if v is None:
        return "—"
    v = float(v)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 10_000:
        return f"{v / 1_000:.0f}K"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{int(round(v)):,}"


def fmt_eq(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def fmt_int(v):
    if v is None:
        return "—"
    return f"{int(round(v)):,}"


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
    W, H = 940, 460
    ML, MR, MT, MB = 54, 140, 20, 56
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

    # y grid + ticks (BenchPR)
    for t in _y_ticks(max_pr):
        y = y_of(t)
        s.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{ML + pw}" y2="{y:.1f}" stroke-width="1" />')
        s.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end" class="muted">{_fmt_num(t)}</text>')
    s.append(
        f'<text x="14" y="{MT + ph / 2:.0f}" transform="rotate(-90 14 {MT + ph / 2:.0f})" '
        f'text-anchor="middle" class="lbl">BenchPR (lower is better)</text>'
    )

    # x grid + ticks (log cost)
    e = lo_e
    while e <= hi_e:
        x = x_of(10 ** e)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + ph}" stroke-width="1" />')
        label = f"${_fmt_num(10 ** e)}" if e >= 0 else f"${10 ** e:g}"
        s.append(f'<text x="{x:.1f}" y="{MT + ph + 18}" text-anchor="middle" class="muted">{label}</text>')
        e += 1
    s.append(
        f'<text x="{ML + pw / 2:.0f}" y="{H - 8}" text-anchor="middle" class="lbl">'
        "Total cost per run (USD, log scale)</text>"
    )

    s.append(f'<line class="axis" x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + ph}" stroke-width="1" />')
    s.append(f'<line class="axis" x1="{ML}" y1="{MT + ph}" x2="{ML + pw}" y2="{MT + ph}" stroke-width="1" />')

    # human north-star reference lines (annotation, not data)
    for pr, label in REF_LINES:
        y = y_of(pr)
        s.append(
            f'<line class="refline" data-pr="{_fmt_num(pr)}" x1="{ML}" y1="{y:.1f}" '
            f'x2="{ML + pw}" y2="{y:.1f}" stroke="var(--ref)" stroke-width="1" '
            f'stroke-dasharray="4 4" />'
        )
        s.append(f'<text x="{ML + pw + 8}" y="{y + 4:.1f}" class="lbl" font-size="11">{esc(label)}</text>')

    # model/track points with a surface ring + direct label
    for p in points:
        pr = p.get("benchpr")
        if pr is None:
            continue
        x, y = x_of(p.get("cost")), y_of(pr)
        color = "var(--image-track)" if p.get("track") == "image" else "var(--text-track)"
        s.append(
            f'<circle class="pt" data-track="{esc(p.get("track"))}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="6" fill="{color}" stroke="var(--surface)" stroke-width="2" />'
        )
        s.append(
            f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" class="lbl" font-size="11">{esc(p.get("label"))}</text>'
        )

    # measured human panel — distinct diamond marker
    for h in humans:
        pr = h.get("benchpr")
        if pr is None:
            continue
        x, y = x_of(h.get("cost")), y_of(pr)
        d = 7
        s.append(
            f'<polygon class="human-marker" points="{x:.1f},{y - d:.1f} {x + d:.1f},{y:.1f} '
            f'{x:.1f},{y + d:.1f} {x - d:.1f},{y:.1f}" fill="var(--human)" '
            f'stroke="var(--surface)" stroke-width="2" />'
        )
        s.append(
            f'<text x="{x + d + 5:.1f}" y="{y + 4:.1f}" fill="var(--human)" font-size="11" '
            f'font-weight="600">{esc(h.get("label"))}</text>'
        )

    s.append("</svg>")
    return "".join(s)


TIER_VARS = {"T1": "var(--tier-1)", "T2": "var(--tier-2)", "T3": "var(--tier-3)", "T4": "var(--tier-4)"}


def tier_bars_svg(models):
    """Grouped BenchPR bars by tier T1–T4, one group per model (PLAN.md §6).

    Tiers use a single-hue ordinal ramp (light T1 → dark T4) — difficulty is an
    ordered magnitude, not an identity, so a rainbow would misencode it.
    """
    ranked = [m for m in models if m.get("per_tier")]
    if not ranked:
        return '<p class="sub">No per-tier data in the current results.</p>'

    ML, MR, MT, MB = 44, 16, 16, 70
    group_w = 116
    W = max(960, ML + MR + group_w * len(ranked))
    H = 330
    pw = W - ML - MR
    ph = H - MT - MB

    values = []
    for m in ranked:
        for t in TIERS:
            v = _num((m["per_tier"].get(t) or {}).get("benchpr"))
            if v is not None:
                values.append(v)
    max_pr = _nice_ceiling(max(values) if values else 10.0)

    s = [_svg_open(W, H)]

    def y_of(pr):
        return MT + ph - (pr / max_pr) * ph

    for t in _y_ticks(max_pr):
        y = y_of(t)
        s.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" stroke-width="1" />')
        s.append(f'<text x="{ML - 6}" y="{y + 4:.1f}" text-anchor="end" class="muted">{_fmt_num(t)}</text>')

    s.append(f'<line class="axis" x1="{ML}" y1="{MT + ph}" x2="{W - MR}" y2="{MT + ph}" stroke-width="1" />')

    gw = pw / len(ranked)
    bar_area = min(gw * 0.74, 22 * len(TIERS) + 2 * (len(TIERS) - 1))
    slot = bar_area / len(TIERS)
    bw = min(slot - 2, 22)
    for gi, m in enumerate(ranked):
        gx = ML + gi * gw + (gw - bar_area) / 2
        for ti, t in enumerate(TIERS):
            v = _num((m["per_tier"].get(t) or {}).get("benchpr"))
            x = gx + ti * slot
            if v is None:
                continue
            bh = (v / max_pr) * ph
            y = MT + ph - bh
            s.append(
                f'<rect class="tier-bar" data-tier="{t}" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{TIER_VARS[t]}" />'
            )
        label = m["model"].split("/")[-1]
        cx = ML + gi * gw + gw / 2
        s.append(
            f'<text x="{cx:.0f}" y="{MT + ph + 18}" text-anchor="middle" class="lbl" '
            f'font-size="11">{esc(label[:22])}</text>'
        )

    # tier ramp legend, low → high difficulty
    lx = ML
    for ti, t in enumerate(TIERS):
        x = lx + ti * 84
        s.append(f'<rect x="{x}" y="{H - 16}" width="12" height="12" rx="2" fill="{TIER_VARS[t]}" />')
        cap = {"T1": "T1 easiest", "T4": "T4 hardest"}.get(t, t)
        s.append(f'<text x="{x + 16}" y="{H - 6}" class="muted" font-size="11">{cap}</text>')

    s.append("</svg>")
    return "".join(s)


def decision_type_bars_svg(models):
    """Grouped BenchPR bars, checker vs. cube, one group per model.

    Two categorical hues (aqua = checker, violet = cube), well separated for CVD.
    """
    rows = [m for m in models if m.get("benchpr_checker") is not None or m.get("benchpr_cube") is not None]
    if not rows:
        return '<p class="sub">No checker/cube split in the current results.</p>'

    ML, MR, MT, MB = 44, 16, 16, 58
    group_w = 96
    W = max(900, ML + MR + group_w * len(rows))
    H = 300
    pw = W - ML - MR
    ph = H - MT - MB

    vals = []
    for m in rows:
        for k in ("benchpr_checker", "benchpr_cube"):
            if m.get(k) is not None:
                vals.append(m[k])
    max_pr = _nice_ceiling(max(vals) if vals else 10.0)

    s = [_svg_open(W, H)]

    def y_of(pr):
        return MT + ph - (pr / max_pr) * ph

    for t in _y_ticks(max_pr):
        y = y_of(t)
        s.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" stroke-width="1" />')
        s.append(f'<text x="{ML - 6}" y="{y + 4:.1f}" text-anchor="end" class="muted">{_fmt_num(t)}</text>')
    s.append(f'<line class="axis" x1="{ML}" y1="{MT + ph}" x2="{W - MR}" y2="{MT + ph}" stroke-width="1" />')

    kinds = [("benchpr_checker", "checker", "var(--checker)"), ("benchpr_cube", "cube", "var(--cube)")]
    gw = pw / len(rows)
    bar_area = min(gw * 0.62, 24 * len(kinds) + 4)
    slot = bar_area / len(kinds)
    bw = min(slot - 2, 24)
    for gi, m in enumerate(rows):
        gx = ML + gi * gw + (gw - bar_area) / 2
        for ki, (key, kind, color) in enumerate(kinds):
            v = m.get(key)
            x = gx + ki * slot
            if v is None:
                continue
            bh = (v / max_pr) * ph
            y = MT + ph - bh
            s.append(
                f'<rect class="dtype-bar" data-dtype="{kind}" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}" />'
            )
        label = m["model"].split("/")[-1]
        cx = ML + gi * gw + gw / 2
        s.append(
            f'<text x="{cx:.0f}" y="{MT + ph + 18}" text-anchor="middle" class="lbl" '
            f'font-size="11">{esc(label[:22])}</text>'
        )

    s.append("</svg>")
    return "".join(s)


def dumbbell_svg(models):
    """Text-vs-image dumbbell: one row per model, connecting the two tracks."""
    rows = [m for m in models if m.get("benchpr_text") is not None or m.get("benchpr_image") is not None]
    if not rows:
        return '<p class="sub">No text/image comparison data in the current results.</p>'

    W = 960
    ML, MR, MT, MB = 160, 40, 20, 40
    row_h = 40
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

    for t in _y_ticks(max_pr):
        x = x_of(t)
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + row_h * len(rows)}" stroke-width="1" />')
        s.append(f'<text x="{x:.1f}" y="{H - 12}" text-anchor="middle" class="muted">{_fmt_num(t)}</text>')
    s.append(
        f'<text x="{ML + pw / 2:.0f}" y="{H - 24}" text-anchor="middle" class="lbl" font-size="11">'
        "BenchPR (lower is better)</text>"
    )

    for i, m in enumerate(rows):
        cy = MT + i * row_h + row_h / 2
        label = m["model"].split("/")[-1]
        s.append(f'<text x="{ML - 10}" y="{cy + 4:.1f}" text-anchor="end" class="lbl">{esc(label)}</text>')
        tv, iv = m["benchpr_text"], m["benchpr_image"]
        if tv is not None and iv is not None:
            s.append(
                f'<line x1="{x_of(tv):.1f}" y1="{cy:.1f}" x2="{x_of(iv):.1f}" y2="{cy:.1f}" '
                f'stroke="var(--axis)" stroke-width="2" stroke-linecap="round" />'
            )
        if tv is not None:
            s.append(f'<circle class="dot-text" cx="{x_of(tv):.1f}" cy="{cy:.1f}" r="6" fill="var(--text-track)" stroke="var(--surface)" stroke-width="2" />')
        if iv is not None:
            s.append(f'<circle class="dot-image" cx="{x_of(iv):.1f}" cy="{cy:.1f}" r="6" fill="var(--image-track)" stroke="var(--surface)" stroke-width="2" />')

    s.append("</svg>")
    return "".join(s)


# ==========================================================================
# HTML sections
# ==========================================================================


def _th(label, key, numeric=True, left=False):
    cls = ' class="left"' if left else ""
    ds = ' data-numeric="1"' if numeric else ""
    return f'<th{cls}{ds} data-key="{esc(key)}">{esc(label)}<span class="arrow"></span></th>'


def _rank_chip(rank):
    if not rank:
        return '<span class="rankchip" title="baseline">—</span>'
    cls = f" r{rank}" if rank <= 3 else ""
    return f'<span class="rankchip{cls}">{rank}</span>'


def _split_cell(row):
    ch, cu = row.get("benchpr_checker"), row.get("benchpr_cube")
    if ch is None and cu is None:
        return '<td class="dim" data-sort="1e9">—</td>'
    sort = ch if ch is not None else (cu if cu is not None else 1e9)
    inner = (
        '<span class="split">'
        f'<span class="k"><span class="dot checker"></span>{fmt_pr(ch)}</span>'
        f'<span class="dim">/</span>'
        f'<span class="k"><span class="dot cube"></span>{fmt_pr(cu)}</span>'
        "</span>"
    )
    return f'<td data-sort="{sort}">{inner}</td>'


def leaderboard_table_html(models, humans):
    heads = [
        _th("#", "rank", left=True),
        _th("Model", "model", numeric=False, left=True),
        _th("BenchPR", "benchpr"),
        _th("Best move", "acc"),
        _th("Eq. loss", "eqloss"),
        _th("Checker / cube", "split"),
        _th("Cost / run", "cost"),
        _th("Tokens", "tokens"),
    ]
    body = []
    for row in list(models) + list(humans):
        rank = row.get("rank")
        is_human = bool(row.get("human"))
        rank_sort = rank if rank else 9999
        b = row["benchpr"]
        ci = row.get("benchpr_ci")
        b_disp = fmt_pr(b)
        if ci:
            b_disp += f' <span class="ci">[{ci[0]:.2f}–{ci[1]:.2f}]</span>'
        short = esc(row["model"].split("/")[-1])
        name = f'<span class="modelname">{short}</span>'
        if is_human:
            name += ' <span class="badge">human</span>'
        acc = row["best_move_accuracy"]
        eq = row.get("mean_equity_loss")
        tr_cls = ' class="human-row"' if is_human else ""
        body.append(
            f"<tr{tr_cls}>"
            f'<td class="left" data-sort="{rank_sort}">{_rank_chip(rank)}</td>'
            f'<td class="left" data-sort="{esc(row["model"])}"><span class="modelcell">{name}</span></td>'
            f'<td class="primary" data-sort="{b if b is not None else 1e9}">{b_disp}</td>'
            f'<td data-sort="{acc if acc is not None else -1}">{fmt_pct(acc)}</td>'
            f'<td data-sort="{eq if eq is not None else 1e9}">{fmt_eq(eq)}</td>'
            f'{_split_cell(row)}'
            f'<td data-sort="{row["cost_usd"] if row["cost_usd"] is not None else 1e9}">{fmt_usd(row["cost_usd"])}</td>'
            f'<td data-sort="{row["tokens"] if row["tokens"] is not None else 1e9}">{fmt_tokens(row["tokens"])}</td>'
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


def _wordmark():
    return (
        '<span class="wordmark">'
        '<span class="die">⚄</span>'
        '<span><span class="b1">Gammon</span><span class="b2">Bench</span></span>'
        '<span class="tag">beta</span>'
        "</span>"
    )


def header_html(meta, generated, synthetic):
    """Top bar (wordmark + theme toggle), hero, metadata chips, synthetic banner."""
    banner = ""
    if synthetic:
        banner = (
            '<div class="banner">'
            "<b>Synthetic preview.</b> Built from <code>tests/fixtures/</code> — "
            "fake models, fabricated numbers. Not real benchmark results.</div>"
        )

    chips = []
    if meta.get("positions"):
        chips.append(f'<span class="chip"><b>{fmt_int(meta["positions"])}</b> positions</span>')
    if meta.get("dataset_hash"):
        chips.append(f'<span class="chip">dataset <code>{esc(str(meta["dataset_hash"])[:16])}</code></span>')
    if meta.get("prompt_version"):
        chips.append(f'<span class="chip">prompt <code>{esc(meta["prompt_version"])}</code></span>')
    if meta.get("latest_run_timestamp"):
        chips.append(f'<span class="chip">updated <b>{esc(str(meta["latest_run_timestamp"])[:10])}</b></span>')
    chips_html = f'<div class="chips">{"".join(chips)}</div>' if chips else ""

    return (
        '<div class="topbar">'
        f"{_wordmark()}"
        '<button class="themetoggle" id="themetoggle" type="button" aria-label="Toggle light/dark theme">'
        '<span aria-hidden="true">◐</span> Theme</button>'
        "</div>"
        '<header class="hero">'
        "<h1>How well do LLMs play backgammon?</h1>"
        '<p class="lede">GammonBench scores language models on real backgammon decisions — '
        "checker plays and cube decisions — against GNU Backgammon rollouts, on a single "
        "human-comparable error metric, <b>BenchPR</b>.</p>"
        f"{chips_html}"
        f"{banner}"
        "</header>"
    )


def stat_tiles_html(models, humans, meta):
    """Headline KPI tiles: leader, human baseline, dataset size, field size."""
    tiles = []
    ranked = [m for m in models if m.get("benchpr") is not None]
    if ranked:
        leader = ranked[0]
        acc = leader.get("best_move_accuracy")
        acc_line = f"{fmt_pct(acc)} best-move accuracy" if acc is not None else ""
        tiles.append(
            '<div class="tile accent">'
            '<p class="label">Leader · BenchPR</p>'
            f'<div class="value">{fmt_pr(leader["benchpr"])}</div>'
            f'<p class="sub2"><b>{esc(leader["model"].split("/")[-1])}</b>'
            + (f" · {acc_line}" if acc_line else "")
            + "</p>"
            "</div>"
        )
    if humans:
        h = humans[0]
        tiles.append(
            '<div class="tile human-tile">'
            '<p class="label">Human panel · BenchPR</p>'
            f'<div class="value">{fmt_pr(h["benchpr"])}</div>'
            f'<p class="sub2">Measured baseline · {esc(h["model"].split("/")[-1])}</p>'
            "</div>"
        )
    if meta.get("positions"):
        tiles.append(
            '<div class="tile">'
            '<p class="label">Scored positions</p>'
            f'<div class="value">{fmt_int(meta["positions"])}</div>'
            '<p class="sub2">Checker &amp; cube decisions, tiers T1–T4</p>'
            "</div>"
        )
    tiles.append(
        '<div class="tile">'
        '<p class="label">Models ranked</p>'
        f'<div class="value">{len(ranked)}</div>'
        '<p class="sub2">Scored vs. GNU Backgammon rollouts</p>'
        "</div>"
    )
    return f'<div class="tiles">{"".join(tiles)}</div>'


def methodology_html():
    scale = [
        ("PR 0–2", "flawless", "var(--human)"),
        ("2–4", "world-class", "var(--tier-3)"),
        ("5–8", "strong expert", "var(--tier-2)"),
        ("10–15", "intermediate", "var(--image-track)"),
        ("20+", "beginner", "var(--cube)"),
    ]
    bands = "".join(
        f'<span class="band" style="background:{c}">{esc(a)} · {esc(b)}</span>' for a, b, c in scale
    )
    return (
        '<div class="explain">'
        "<b>What is BenchPR?</b> BenchPR = 500 × mean equity loss per decision "
        "(lower is better). Equity loss is how much money-game equity a move gives up "
        "versus the engine's best play, read from GNU Backgammon rollouts. The scale is "
        "calibrated to sit on the same axis as a human Performance Rating, so the "
        "north-star lines read directly:"
        f'<div class="scale">{bands}</div>'
        '<div class="methods">'
        '<a href="../../docs/SCORING.md">Scoring &amp; BenchPR</a>'
        '<a href="../../docs/DATASET.md">Dataset &amp; tiers</a>'
        '<a href="../../docs/HARNESS.md">Harness &amp; budget track</a>'
        '<a href="../../docs/CONTAMINATION.md">Contamination policy</a>'
        "</div>"
        "</div>"
    )


def _legend_html(items):
    """Small HTML legend row. ``items`` = list of (css_color, shape, label)."""
    keys = []
    for color, shape, label in items:
        cls = "swatch diamond" if shape == "diamond" else "swatch"
        keys.append(
            f'<span class="k"><span class="{cls}" style="background:{color}"></span>{esc(label)}</span>'
        )
    return f'<div class="legend">{"".join(keys)}</div>'


SITE_TITLE = "GammonBench — Backgammon LLM Benchmark"


def build_html(results, generated, synthetic=False):
    """Assemble the full page HTML from loaded result dicts."""
    meta = site_meta(results)
    header = header_html(meta, generated, synthetic)

    if not results:
        body = header + (
            '<div class="empty">No runs yet — once <code>results/*.json</code> '
            "exists, the leaderboard appears here.</div>"
        )
        return templates.page(SITE_TITLE, body)

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

    scatter_legend = [
        ("var(--text-track)", "circle", "text track"),
        ("var(--image-track)", "circle", "image track"),
    ]
    if human_pts:
        scatter_legend.append(("var(--human)", "diamond", "human panel"))

    sections = [header]
    sections.append(stat_tiles_html(models, humans, meta))
    sections.append(methodology_html())

    sections.append('<section id="leaderboard">')
    sections.append('<div class="sec-head"><h2>Leaderboard</h2></div>')
    sections.append(
        '<p class="sub">Ranked by BenchPR (lower is better). Human-panel rows are baselines, '
        "not ranked competitors. Click any column header to re-sort.</p>"
    )
    sections.append(leaderboard_table_html(models, humans))
    sections.append("</section>")

    sections.append('<section id="skill-cost">')
    sections.append('<div class="sec-head"><h2>Skill vs. cost</h2></div>')
    sections.append(
        '<p class="sub">BenchPR against total dollar cost per run (log x-axis). Dashed lines are '
        "the human north star (PR 2 / 4 / 8).</p>"
    )
    sections.append(f'<div class="chart">{scatter_svg(points, human_pts)}</div>')
    sections.append(_legend_html(scatter_legend))
    sections.append("</section>")

    sections.append('<section id="tiers">')
    sections.append('<div class="sec-head"><h2>Difficulty tiers</h2></div>')
    sections.append(
        '<p class="sub">Where models fall apart: BenchPR by difficulty tier T1 (easiest) '
        "to T4 (hardest for top humans).</p>"
    )
    sections.append(f'<div class="chart">{tier_bars_svg(models + humans)}</div>')
    sections.append("</section>")

    sections.append('<section id="decision-type">')
    sections.append('<div class="sec-head"><h2>Checker vs. cube</h2></div>')
    sections.append(
        '<p class="sub">BenchPR split by decision type — checker plays vs. doubling-cube '
        "decisions.</p>"
    )
    sections.append(f'<div class="chart">{decision_type_bars_svg(models + humans)}</div>')
    sections.append(
        _legend_html([("var(--checker)", "circle", "checker"), ("var(--cube)", "circle", "cube")])
    )
    sections.append("</section>")

    sections.append('<section id="text-image">')
    sections.append('<div class="sec-head"><h2>Text vs. image</h2></div>')
    sections.append('<p class="sub">Does seeing the rendered board (image) help or hurt vs. plain text?</p>')
    sections.append(f'<div class="chart">{dumbbell_svg(models + humans)}</div>')
    sections.append(
        _legend_html([("var(--text-track)", "circle", "text"), ("var(--image-track)", "circle", "image")])
    )
    sections.append("</section>")

    budget_html = budget_table_html(budgets)
    if budget_html:
        sections.append('<section id="budget-track">' + budget_html + "</section>")

    sections.append(
        '<div class="foot">'
        '<p class="fnote"><b>BenchPR</b> = 500 × mean equity loss per decision, versus GNU '
        "Backgammon rollout ground truth. Lower is better; the scale mirrors a human "
        "Performance Rating.</p>"
        "<p>GammonBench · static leaderboard, one-way data flow (results JSON → build → HTML) · "
        "self-contained, no external requests. See PLAN.md §6.</p>"
        "</div>"
    )
    return templates.page(SITE_TITLE, "\n".join(sections))


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
            "benchpr_checker": r.get("benchpr_checker"),
            "benchpr_cube": r.get("benchpr_cube"),
            "best_move_accuracy": r["best_move_accuracy"],
            "mean_equity_loss": r.get("mean_equity_loss"),
            "cost_usd": r["cost_usd"],
            "cost_per_position": r["cost_per_position"],
            "tokens": r.get("tokens"),
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
