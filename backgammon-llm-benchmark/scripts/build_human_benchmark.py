#!/usr/bin/env python3
"""Build a single self-contained human-benchmark quiz HTML for the pilot set.

Reads the 50 pilot position records (``positions/pilot/bg-*.json``) and their GNU
BG rollouts (``rollouts/gnubg/bg-*.json``), emits each position's board as
structured data (points/bar/off/dice/cube/pips in the mover frame, plus its XGID
for reference) and the enumerated legal-move set, and writes ONE fully
self-contained HTML file at ``site/public/human-benchmark-pilot.html`` — no
pre-baked SVG, all CSS/JS inline, zero network requests — so it can be emailed to
panelists and opened locally. The page carries a *single* JavaScript board engine
that renders any position from its data (a faithful port of ``render/svg.py``'s
geometry) and re-renders after each interactive click-to-move. ``render/svg.py``
itself is untouched and still authors the SVGs used in the LLM image prompts.

The page presents each position blind (no engine ground truth until the end),
one at a time, from the on-roll player's perspective. The panelist always plays
the **White** checkers (the opponent is Black). ``board_json`` is authoritative
and already mover-relative — positive ("x") checkers are the player on roll,
confirmed by both id codecs (``ids/xgid.py``, ``ids/gnubg_id.py``) — so its cube
owner / score / pips are read as-is (see :func:`_display_board`). Cube decisions
are never color-flipped. A subset of the pilot's checker rollouts were computed
in the color-mirror frame; those positions are presented as ``flip(board_json)``
(a legal, symmetric equivalent, on-roll player still White) so the diagram stays
consistent with the answer key. White/black is drawn by pushing render/svg.py's
mover/opponent checker fills to true white/black.

Answers are scored like ``harness/scoring.py``:

* ``BenchPR = 500 * mean(equity_loss)``; ``equity_loss = error_mp / 1000``.
* Checker: normalise the typed move (JS port of ``bgcore/notation.py``) and match
  it against the rollout move list, first by canonical notation and then — for a
  play that reaches the same position spelled differently — by the pre-computed
  endpoint map (mirrors ``bgcore.moves.moves_equivalent``). Unmatched / unparseable
  answers score the worst listed move (PLAN §4.6). ``is_best`` iff the matched
  move has zero error.
* Cube: the record poses exactly {No double, Double-Take, Double-Pass}; the chosen
  action's ``error_mp`` is the loss.

Stdlib only (imports the repo's ``bgcore`` / ``render`` packages at build time).
Run ``python3 scripts/build_human_benchmark.py`` from the repo root.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Authoritative move-notation canonicaliser (used to pre-canonicalise the
# rollout move strings at build time; the browser ports the same logic to JS).
from bgcore import notation  # noqa: E402
from bgcore import moves as bgmoves  # noqa: E402
from bgcore.board import Board, flip, pip_counts  # noqa: E402
from ids.xgid import board_to_xgid  # noqa: E402

POS_DIR = os.path.join(REPO_ROOT, "positions", "pilot")
ROLL_DIR = os.path.join(REPO_ROOT, "rollouts", "gnubg")
DEFAULT_OUT = os.path.join(REPO_ROOT, "site", "public", "human-benchmark-pilot.html")

PROMPT_VERSION = "human-benchmark-1"


def _canonical(move: str) -> str:
    """Canonical form of a rollout move string via the authoritative parser."""
    try:
        return notation.format_move(notation.parse_move(move))
    except notation.NotationError:
        return move.strip()


def _display_board(rec: dict, rollout: dict) -> Board:
    """Return the board to render/score, in the on-roll player's frame.

    ``board_json`` is authoritative and already mover-relative: positive ("x")
    checkers are the player on roll (confirmed by both the XGID and gnubg-id
    codecs — ``ids/xgid.py`` and ``ids/gnubg_id.py`` decode board_json's points
    with the on-roll player positive, regardless of the ``turn`` *seat* label).
    So the cube owner / score / pips in ``board_json`` are ALREADY correct from
    the on-roll player's view and must NOT be color-flipped.

    Cube decisions therefore never flip. For checker decisions, a subset of the
    pilot's ``rollouts/gnubg`` move lists were computed in the color-mirror frame
    (a data quirk: the moves are legal only on ``flip(board_json)``). For those we
    present ``flip(board_json)`` — a legal, color-symmetric equivalent position in
    which the on-roll player is again "x" — so the diagram the panelist plays is
    consistent with the answer key. The frame is chosen per position by which
    orientation actually makes the rollout's moves legal (board_json wins ties).
    """
    b = Board.from_json(rec["board_json"])
    if rec["decision_type"] == "cube":
        return b
    moves = (rollout.get("checker") or {}).get("moves") or []

    def n_legal(bd: Board) -> int:
        c = 0
        for m in moves:
            mv = m.get("move")
            if not mv:
                continue
            try:
                if bgmoves.is_legal(bd, mv):
                    c += 1
            except Exception:  # noqa: BLE001
                pass
        return c

    fb = flip(b)
    return fb if n_legal(fb) > n_legal(b) else b


def _sig(points, bar_x, bar_o, off_x) -> str:
    """Resulting-layout signature string — MUST match the JS ``sigOf`` port.

    Identifies a full legal move by the position it reaches (board-index points
    array plus the mover bar / opponent bar / mover off counts). The interactive
    board engine reaches a move iff the live state's signature equals one of the
    embedded legal signatures, so the format has to be byte-identical either side.
    """
    pts = ",".join(str(int(p)) for p in points)
    return f"{pts}|{int(bar_x)},{int(bar_o)},{int(off_x)}"


def _legal_embed(mover: Board) -> list[dict]:
    """Compact legal-move set for the display frame: ``[{"n": notation, "s": sig}]``.

    Enumerated by the authoritative engine (``bgcore.moves.generate_moves``) in the
    exact frame the board is drawn in, so the JS click-to-move engine (which ports
    ``_legal_hops``/``_apply_hop`` and searches for these signatures) stays in lock
    step with Python. Empty list ⇒ the mover has no legal play ("Cannot Move").
    """
    out = []
    for lm in bgmoves.generate_moves(mover):
        out.append({"n": lm.notation, "s": _sig(lm.points, lm.bar["x"], lm.bar["o"], lm.off["x"])})
    return out


def _best_after(mover: Board, best_move: str | None) -> dict | None:
    """Board layout (``points``/``bar``/``off``) after applying ``best_move``.

    Used by the practice-mode feedback panel to redraw the board with the engine's
    best play applied. Returns ``None`` if the move can't be resolved/applied so
    the panel degrades gracefully (names the best move without animating it).
    """
    if not best_move:
        return None
    try:
        after = bgmoves.apply_move(mover, best_move)
    except Exception:  # noqa: BLE001
        return None
    return {
        "points": [int(v) for v in after.points],
        "bar": {"x": int(after.bar["x"]), "o": int(after.bar["o"])},
        "off": {"x": int(after.off["x"]), "o": int(after.off["o"])},
    }


def _endpoint_key(cnt) -> str:
    """Stable key for an endpoint (start,end) multiset — must match the JS port."""
    def s(x):
        return "bar" if x == "bar" else ("off" if x == "off" else str(int(x)))
    items = sorted(s(a) + ">" + s(b) for (a, b), n in cnt.items() for _ in range(n))
    return ",".join(items)


def _checker_epmap(mover: Board, rollout_moves: list[dict], worst_mp: float) -> dict:
    """Map endpoint-multiset key -> resolved score, for position-equivalent matching.

    Mirrors ``harness.scoring.score_checker`` / ``bgcore.moves.moves_equivalent``:
    two move strings score the same iff they reach the same resulting position.
    A human may spell a single checker's two-die play with or without its
    intermediate point ("13/9" vs "13/10/9"); both denote the same legal move.
    We enumerate the legal moves once (authoritative engine), attach each
    rollout error to the legal move it resolves to, and key by the (start,end)
    endpoint multiset — exactly what the engine uses to identify a play. Endpoint
    keys shared by two distinct legal moves (a rare hit-vs-non-hit ambiguity) are
    omitted so the fallback can never mis-score; those fall through to the direct
    string match / worst-legal penalty, matching prior behaviour.
    """
    legal = bgmoves.generate_moves(mover)
    sig_err: dict = {}
    for m in rollout_moves:
        mv = m.get("move")
        if not mv:
            continue
        try:
            lm = bgmoves._match(mover, mv)
        except Exception:  # noqa: BLE001
            lm = None
        if lm not in (None, "cannot-move"):
            sig_err[lm.signature()] = (abs(float(m.get("error_mp", 0.0))), mv)
    groups: dict = {}
    for lm in legal:
        groups.setdefault(_endpoint_key(lm.endpoint_multiset), []).append(lm)
    epmap: dict = {}
    for key, lms in groups.items():
        if len({lm.signature() for lm in lms}) != 1:
            continue  # ambiguous endpoint key -> omit (safe)
        lm = lms[0]
        if lm.signature() in sig_err:
            err, disp = sig_err[lm.signature()]
        else:
            err, disp = worst_mp, lm.notation  # legal but absent from rollout list
        epmap[key] = {"error_mp": err, "is_best": err <= 1e-6, "move": disp}
    return epmap


def load_positions() -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(POS_DIR, "bg-*.json"))):
        with open(path, encoding="utf-8") as fh:
            records.append(json.load(fh))
    records.sort(key=lambda d: d["position_id"])
    return records


def build_data(records: list[dict]) -> list[dict]:
    """Assemble the per-position data blob consumed by the page's JS."""
    out = []
    for rec in records:
        pid = rec["position_id"]
        bj = rec["board_json"]
        with open(os.path.join(ROLL_DIR, pid + ".json"), encoding="utf-8") as fh:
            roll = json.load(fh)

        # Score/draw in the on-roll player's frame (board_json authoritative;
        # only color-mirrored for the checker positions whose rollout demands it).
        mover = _display_board(rec, roll)
        pip_x, pip_o = pip_counts(mover)
        cube = {
            "value": int(mover.cube.get("value", 1)),
            "owner": mover.cube.get("owner", "center"),
        }
        score = {
            "x": int(mover.score.get("x", 0)),
            "o": int(mover.score.get("o", 0)),
            "length": int(mover.score.get("length", 0)),
            "crawford": bool(mover.score.get("crawford", False)),
        }
        dice = [int(d) for d in mover.dice]

        entry: dict = {
            "position_id": pid,
            "decision_type": rec["decision_type"],
            "tier": rec["tier"],
            "play_mode": rec["play_mode"],
            "score": score,
            "cube": cube,
            "dice": dice,
            "pip": {"x": pip_x, "o": pip_o},
            # Structured board state consumed by the single JS board engine, in the
            # display (mover) frame. Positive point counts are the White player on
            # roll, negative are Black — matching board.py's mover-relative layout.
            "board": {
                "points": [int(v) for v in mover.points],
                "bar": {"x": int(mover.bar["x"]), "o": int(mover.bar["o"])},
                "off": {"x": int(mover.off["x"]), "o": int(mover.off["o"])},
                "dice": dice,
                "cube": cube,
                "score": score,
                "pip": {"x": pip_x, "o": pip_o},
            },
            # The display-frame XGID (may differ from rec["xgid"] when the position
            # is color-mirrored for the answer key); kept for reference / debugging.
            "xgid": board_to_xgid(mover),
        }

        if rec["decision_type"] == "checker":
            moves = list((roll.get("checker") or {}).get("moves") or [])
            entry["moves"] = [
                {
                    "move": m["move"],
                    "canonical": _canonical(m["move"]),
                    "error_mp": abs(float(m.get("error_mp", 0.0))),
                    "rank": m.get("rank"),
                }
                for m in moves
            ]
            entry["best_move"] = roll.get("best_move")
            entry["worst_error_mp"] = max(
                (mm["error_mp"] for mm in entry["moves"]), default=0.0
            )
            entry["epmap"] = _checker_epmap(mover, moves, entry["worst_error_mp"])
            # Legal-move set for click-to-move (display frame). The JS engine plays
            # hops on a live board and only lets the panelist submit a sequence that
            # reaches one of these signatures.
            entry["legal"] = _legal_embed(mover)
            # Board layout AFTER the engine's best move, so the practice-mode
            # feedback panel can re-draw the board with the best play applied. None
            # when the best move can't be resolved/applied (degrade gracefully — the
            # panel then just names the best move without animating it).
            entry["best_after"] = _best_after(mover, entry.get("best_move"))
        else:  # cube
            cube = roll.get("cube") or {}
            errmap = {k: abs(float(v)) for k, v in (cube.get("error_mp") or {}).items()}
            # Preserve the canonical option order the record poses.
            order = ["No double", "Double, Take", "Double, Pass"]
            entry["options"] = [k for k in order if k in errmap] + [
                k for k in errmap if k not in order
            ]
            entry["error_mp"] = errmap
            entry["best_action"] = cube.get("best_action")
            entry["worst_error_mp"] = max(errmap.values(), default=0.0)

        out.append(entry)
    return out


def dataset_hash(records: list[dict]) -> str:
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda d: d["position_id"]):
        h.update(rec["position_id"].encode())
        h.update(b"\0")
        h.update(rec.get("xgid", "").encode())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def build_manifest(records: list[dict], timestamp: str) -> dict:
    versions = {r.get("ascii_render_version", "ascii-1") for r in records}
    img_versions = {r.get("image_render_version", "svg-1") for r in records}
    return {
        "dataset_hash": dataset_hash(records),
        "prompt_version": PROMPT_VERSION,
        "ascii_render_version": sorted(versions)[0] if versions else "ascii-1",
        "image_render_version": sorted(img_versions)[0] if img_versions else "svg-1",
        "timestamp": timestamp,
    }


def _json_for_script(obj) -> str:
    """JSON safe to embed inside a <script> element (escape </ and unicode line seps)."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("</", "<\\/")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def render_html(data: list[dict], manifest: dict) -> str:
    data_json = _json_for_script(data)
    manifest_json = _json_for_script(manifest)
    counts = {
        "n": len(data),
        "checker": sum(1 for d in data if d["decision_type"] == "checker"),
        "cube": sum(1 for d in data if d["decision_type"] == "cube"),
    }
    return _PAGE_TEMPLATE.format(
        data_json=data_json,
        manifest_json=manifest_json,
        total=counts["n"],
        css=_CSS,
        js=_JS,
    )


# --------------------------------------------------------------------------
# Page template (CSS + JS are inlined; data blobs injected via .format()).
# The literal braces in CSS/JS are handled by injecting them as fields so the
# template itself has no stray ``{}`` for str.format.
# --------------------------------------------------------------------------

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backgammon Human Benchmark — Pilot (50 positions)</title>
<style>
{css}
</style>
</head>
<body>
<main id="app" class="app"></main>
<script id="bench-data" type="application/json">{data_json}</script>
<script id="bench-manifest" type="application/json">{manifest_json}</script>
<script>
{js}
</script>
</body>
</html>
"""

_CSS = r"""
:root {
  --bg: #f6f4ef; --panel: #ffffff; --ink: #1c1a17; --muted: #5f5a52;
  --line: #d9d3c8; --accent: #6b4f2e; --accent-ink: #ffffff;
  --ok: #2f6d3a; --warn: #8a5a12; --bad: #9a3324; --chip: #efeae0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14110f; --panel: #201b17; --ink: #ece6da; --muted: #a39a8b;
    --line: #38302a; --accent: #cba36a; --accent-ink: #191510;
    --ok: #7fce8b; --warn: #e0b568; --bad: #e6947f; --chip: #2a231d;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5; -webkit-text-size-adjust: 100%;
}
.app { max-width: 860px; margin: 0 auto; padding: 24px 18px 64px; }
h1 { font-size: 1.5rem; margin: 0 0 .4rem; }
h2 { font-size: 1.15rem; margin: 1.4rem 0 .5rem; }
p { margin: .5rem 0; }
.muted { color: var(--muted); }
.small { font-size: .85rem; }
.panel {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 20px; margin: 14px 0;
}
.board-wrap { width: 100%; overflow-x: auto; margin: 10px 0 16px; }
.board-wrap svg { width: 100%; height: auto; max-width: 100%; display: block;
  border: 1px solid var(--line); border-radius: 8px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 10px; margin: 10px 0; }
.chip {
  background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
  padding: 3px 12px; font-size: .85rem; white-space: nowrap;
}
.chip b { font-weight: 700; }
label { display: block; font-weight: 600; margin: 12px 0 4px; }
input[type=text] {
  width: 100%; padding: 11px 12px; font-size: 1.05rem;
  border: 1px solid var(--line); border-radius: 8px;
  background: var(--bg); color: var(--ink); font-family: inherit;
}
input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.hint { font-size: .82rem; color: var(--muted); margin-top: 4px; }
.btn {
  display: inline-block; border: 1px solid var(--accent);
  background: var(--accent); color: var(--accent-ink);
  padding: 11px 20px; border-radius: 8px; font-size: 1rem; font-weight: 600;
  cursor: pointer; font-family: inherit;
}
.btn:hover { filter: brightness(1.06); }
.btn.secondary { background: transparent; color: var(--accent); }
.btn:disabled { opacity: .5; cursor: not-allowed; }
.btn-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.cube-opts { display: flex; flex-direction: column; gap: 10px; margin: 14px 0; }
.cube-opts .btn { width: 100%; text-align: left; }
.progress { font-size: .9rem; color: var(--muted); margin-bottom: 6px;
  display: flex; justify-content: space-between; align-items: baseline; }
.bar { height: 6px; background: var(--chip); border-radius: 4px; overflow: hidden; margin-bottom: 14px; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.warn { color: var(--warn); font-weight: 600; margin: 10px 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: .9rem; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tag-ok { color: var(--ok); font-weight: 700; }
.tag-bad { color: var(--bad); }
.headline { display: flex; flex-wrap: wrap; gap: 18px; margin: 12px 0 6px; }
.stat { background: var(--chip); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 16px; min-width: 140px; }
.stat .n { font-size: 1.8rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat .l { font-size: .8rem; color: var(--muted); }
code { background: var(--chip); padding: 1px 5px; border-radius: 4px; font-size: .9em; }
.divider { border: none; border-top: 1px solid var(--line); margin: 18px 0; }

/* interactive board engine */
.bgboard.interactive .hit { cursor: pointer; }
.bgboard .destdot { pointer-events: none; }
.cb-controls { margin: 6px 0 2px; }
.cb-status { margin: 2px 0 6px; }
.cb-move { margin: 6px 0 10px; font-size: 1.05rem; }
.cb-notation { font-variant-numeric: tabular-nums; }
.cb-notation.done { color: var(--ok); }
.text-entry { margin-top: 4px; }
.text-entry summary { cursor: pointer; color: var(--accent); font-weight: 600; }
.live-valid { min-height: 1.1em; font-weight: 600; }
.live-valid.ok { color: var(--ok); }
.live-valid.bad { color: var(--bad); }

/* run-mode chooser */
.mode-opts { display: flex; flex-direction: column; gap: 10px; margin: 6px 0 4px; }
.mode-card { display: grid; grid-template-columns: auto 1fr; grid-gap: 2px 10px;
  align-items: start; border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; cursor: pointer; font-weight: 400; margin: 0; }
.mode-card:hover { border-color: var(--accent); }
.mode-card input { grid-row: span 2; margin-top: 3px; }
.mode-card b { font-weight: 700; }
.mode-card span { grid-column: 2; }

/* per-answer feedback */
.verdict { font-size: 1.25rem; font-weight: 800; margin: 10px 0 6px; }
.verdict.ok { color: var(--ok); }
.verdict.bad { color: var(--bad); }
.fb-grid { display: flex; flex-direction: column; gap: 6px; margin: 8px 0 4px; }
.fb-row { display: flex; flex-wrap: wrap; gap: 4px 12px; border-bottom: 1px solid var(--line); padding-bottom: 5px; }
.fb-k { color: var(--muted); min-width: 150px; font-size: .9rem; }
.fb-v { font-weight: 600; font-variant-numeric: tabular-nums; }
.errbox { background: var(--chip); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px; overflow-x: auto; white-space: pre-wrap; font-size: .82rem; color: var(--bad); }
"""

_JS = r"""
"use strict";
/* ---- data ---- */
var DATA = JSON.parse(document.getElementById("bench-data").textContent);
var MANIFEST = JSON.parse(document.getElementById("bench-manifest").textContent);
var TOTAL = DATA.length;
var STORE_KEY = "bg-human-bench-pilot-v1";
/* Bump when the saved-state shape changes so stale localStorage is detected on
   load (we offer "start fresh" rather than silently mixing schemas). */
var STATE_VERSION = 2;
var MODES = { practice: "practice", blind: "blind" };

/* ---- notation port (faithful JS port of bgcore/notation.py) ---- */
var CANNOT_MOVE = {"cannot move":1,"cannotmove":1,"cant move":1,"no move":1,"no play":1,"dance":1};
var POINT_RE = /^(bar|off|\d{1,2})$/i;
var TOKEN_RE = /^([^()]+?)(?:\((\d+)\))?$/;

function NotationError(msg){ this.message = msg; this.name = "NotationError"; }

function parsePoint(tok){
  var m = POINT_RE.exec(tok);
  if(!m) throw new NotationError("bad point " + tok);
  var low = tok.toLowerCase();
  if(low === "bar" || low === "off") return low;
  var n = parseInt(tok, 10);
  if(!(n >= 1 && n <= 24)) throw new NotationError("point out of range");
  return n;
}

function parseMove(s){
  if(s === null || s === undefined) throw new NotationError("empty move string");
  var text = String(s).trim();
  if(text === "") throw new NotationError("empty move string");
  var norm = text.replace(/[.\s]+/g, " ").trim().toLowerCase();
  if(CANNOT_MOVE[norm]) return null;
  var tokens = [];
  var raws = text.split(/\s+/);
  for(var r = 0; r < raws.length; r++){
    var raw = raws[r].trim();
    if(raw === "") continue;
    var m = TOKEN_RE.exec(raw);
    if(!m) throw new NotationError("bad move token " + raw);
    var chainStr = m[1];
    var count = m[2] ? parseInt(m[2], 10) : 1;
    var parts = chainStr.split("/");
    if(parts.length < 2) throw new NotationError("move token needs a '/'");
    var chain = [];
    var chainHits = [];
    for(var i = 0; i < parts.length; i++){
      var part = parts[i];
      var hit = part.charAt(part.length - 1) === "*";
      var clean = hit ? part.slice(0, -1) : part;
      var pt = parsePoint(clean);
      chain.push(pt);
      if(i === 0){
        if(hit) throw new NotationError("'*' cannot mark the first point");
      } else {
        chainHits.push(hit);
      }
    }
    var token = { chain: chain, chainHits: chainHits };
    for(var c = 0; c < count; c++){
      tokens.push({ chain: chain.slice(), chainHits: chainHits.slice() });
    }
  }
  return tokens;
}

function renderChain(chain, chainHits){
  var out = String(chain[0]);
  for(var i = 1; i < chain.length; i++){
    out += "/" + String(chain[i]) + (chainHits[i - 1] ? "*" : "");
  }
  return out;
}

function tokSortKey(tok){
  var head = tok.split("/", 1)[0];
  var start = head === "bar" ? 25 : parseInt(head, 10);
  return start;
}

function formatMove(tokens){
  if(tokens === null || tokens.length === 0) return "Cannot Move";
  var rendered = [];
  for(var i = 0; i < tokens.length; i++){
    rendered.push(renderChain(tokens[i].chain, tokens[i].chainHits));
  }
  var cnt = {};
  for(var j = 0; j < rendered.length; j++){ cnt[rendered[j]] = (cnt[rendered[j]] || 0) + 1; }
  var uniq = Object.keys(cnt);
  uniq.sort(function(a, b){
    var sa = tokSortKey(a), sb = tokSortKey(b);
    if(sa !== sb) return sb - sa;          /* -start ascending == start descending */
    return a < b ? -1 : (a > b ? 1 : 0);   /* then token string ascending */
  });
  var parts = [];
  for(var k = 0; k < uniq.length; k++){
    var n = cnt[uniq[k]];
    parts.push(n > 1 ? (uniq[k] + "(" + n + ")") : uniq[k]);
  }
  return parts.join(" ");
}

/* canonicalise typed input; returns null if it cannot be parsed */
function canonicalizeMove(s){
  try { return formatMove(parseMove(s)); }
  catch(e){ return null; }
}

/* endpoint (start,end) multiset key of a parsed move — mirrors the engine's
   move identity (bgcore.moves): a play is identified by which checkers go from
   where to where, so intermediate points / die order do not matter. Must match
   the Python _endpoint_key used to build pos.epmap. */
function endpointKey(tokens){
  var parts = [];
  for(var i = 0; i < tokens.length; i++){
    var ch = tokens[i].chain;
    var a = ch[0], b = ch[ch.length - 1];
    var sa = (a === "bar") ? "bar" : (a === "off" ? "off" : String(a));
    var sb = (b === "bar") ? "bar" : (b === "off" ? "off" : String(b));
    parts.push(sa + ">" + sb);
  }
  parts.sort();
  return parts.join(",");
}

/* True iff a typed checker move resolves to any listed / position-equivalent move. */
function checkerMatches(pos, answer){
  var canon = canonicalizeMove(answer);
  if(canon === null) return false;
  for(var i = 0; i < pos.moves.length; i++){ if(pos.moves[i].canonical === canon) return true; }
  if(pos.epmap){
    var tokens = null;
    try { tokens = parseMove(answer); } catch(e){ tokens = null; }
    if(tokens && pos.epmap[endpointKey(tokens)]) return true;
  }
  return false;
}

/* ---- scoring (mirror of harness/scoring.py) ---- */
var MP_PER_POINT = 1000.0;
var BENCHPR_CONSTANT = 500.0;
var EPS_MP = 1e-6; /* err_mp/1000 <= 1e-9 tol  ->  err_mp <= 1e-6 */

/* Score a checker answer. Returns {matched, chosen, equity_loss, is_best, parse_failed} */
function scoreChecker(pos, answer){
  var canon = canonicalizeMove(answer);
  if(canon === null){
    return { chosen: answer, matched: null, equity_loss: pos.worst_error_mp / MP_PER_POINT,
             is_best: false, parse_failed: true };
  }
  /* 1) direct match against the (canonicalised) rollout move list */
  for(var i = 0; i < pos.moves.length; i++){
    var mv = pos.moves[i];
    if(mv.canonical === canon){
      var isBest = mv.error_mp <= EPS_MP;
      return { chosen: mv.move, matched: mv.move,
               equity_loss: isBest ? 0.0 : mv.error_mp / MP_PER_POINT,
               is_best: isBest, parse_failed: false };
    }
  }
  /* 2) position-equivalent match: same resulting position, spelled differently
     (e.g. intermediate point named, or a legal die reordering). Mirrors
     bgcore.moves.moves_equivalent via the pre-computed endpoint map. */
  if(pos.epmap){
    var tokens = null;
    try { tokens = parseMove(answer); } catch(e){ tokens = null; }
    if(tokens){
      var hit = pos.epmap[endpointKey(tokens)];
      if(hit){
        return { chosen: hit.move, matched: hit.move,
                 equity_loss: hit.is_best ? 0.0 : hit.error_mp / MP_PER_POINT,
                 is_best: !!hit.is_best, parse_failed: false };
      }
    }
  }
  /* 3) legal-looking but not a listed / reproducible move -> worst-legal penalty */
  return { chosen: canon, matched: null, equity_loss: pos.worst_error_mp / MP_PER_POINT,
           is_best: false, parse_failed: true };
}

function scoreCube(pos, label){
  var err = pos.error_mp[label];
  if(err === undefined){
    return { chosen: label, matched: pos.best_action,
             equity_loss: pos.worst_error_mp / MP_PER_POINT, is_best: false, parse_failed: false };
  }
  var isBest = err <= EPS_MP || label === pos.best_action;
  return { chosen: label, matched: pos.best_action,
           equity_loss: isBest ? 0.0 : err / MP_PER_POINT, is_best: isBest, parse_failed: false };
}

/* ---- persistence ---- */
/* Read raw saved state. Returns {state, stale}: `stale` is true when there is a
   non-empty prior run whose version does not match STATE_VERSION (schema drift) —
   the boot code then offers a clean restart instead of mixing schemas. */
function loadState(){
  var raw;
  try { raw = localStorage.getItem(STORE_KEY); }
  catch(e){ raw = null; }
  if(!raw) return { state: freshState(), stale: false };
  var st;
  try { st = JSON.parse(raw); }
  catch(e){ return { state: freshState(), stale: false, corrupt: true }; }
  if(!st || typeof st !== "object") return { state: freshState(), stale: false, corrupt: true };
  if(!st.answers || typeof st.answers !== "object") st.answers = {};
  var hasProgress = st.name || Object.keys(st.answers).length > 0;
  if(hasProgress && st.v !== STATE_VERSION){ return { state: st, stale: true }; }
  if(st.v !== STATE_VERSION) st.v = STATE_VERSION;
  if(st.mode !== MODES.practice && st.mode !== MODES.blind) st.mode = MODES.practice;
  return { state: st, stale: false };
}
function freshState(){ return { v: STATE_VERSION, mode: MODES.practice, answers: {} }; }
function saveState(st){
  try { localStorage.setItem(STORE_KEY, JSON.stringify(st)); } catch(e){}
}
function clearState(){ try { localStorage.removeItem(STORE_KEY); } catch(e){} }
/* STATE = { v, mode, name, started, answers: { position_id: {chosen,is_best,equity_loss,matched,parse_failed} } } */
var _loaded = loadState();
var STATE = _loaded.state;
var STATE_STALE = _loaded.stale;
if(!STATE.answers) STATE.answers = {};

/* ---- helpers ---- */
var app = document.getElementById("app");
function el(tag, attrs, kids){
  var e = document.createElement(tag);
  if(attrs){ for(var k in attrs){
    if(k === "class") e.className = attrs[k];
    else if(k === "html") e.innerHTML = attrs[k];
    else if(k === "text") e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }}
  if(kids){ for(var i = 0; i < kids.length; i++){
    var c = kids[i]; if(c === null || c === undefined) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }}
  return e;
}
function clear(){ app.innerHTML = ""; }
function esc(s){ return String(s).replace(/[&<>]/g, function(c){
  return c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;"; }); }
function fmt(n, d){ return Number(n).toFixed(d === undefined ? 2 : d); }

/* first unanswered index (deterministic resume) */
function firstUnanswered(){
  for(var i = 0; i < TOTAL; i++){ if(!STATE.answers[DATA[i].position_id]) return i; }
  return TOTAL;
}

/* ====================================================================
   SINGLE BOARD ENGINE
   A faithful port of render/svg.py: one pure function turns a position
   object into an SVG string, re-rendered after every interactive move.
   Below it, a click-to-move controller ports bgcore/moves.py's single-die
   hop rules so the panelist composes plays by clicking checkers.
   ==================================================================== */

/* -- geometry (1:1 with render/svg.py) -- */
var BW = 1000, BH = 700, BX0 = 30, BCOLW = 60, BBARW = 40, BTRAYW = 60;
var BYTOP = 90, BYBOT = 610, BPTH = 210, BCR = 24;
/* -- theme (White = mover/you, Black = opponent) -- */
var C_BG = "#14110f", C_BOARD = "#3a2c22", C_PT_A = "#c9a06a", C_PT_B = "#7a5a3c", C_BAR = "#241a12";
var C_X = "#ffffff", C_X_EDGE = "#8a8a8a", C_O = "#000000", C_O_EDGE = "#5a5a5a";
var C_TEXT = "#f2ead9", C_DIE = "#f4efe6", C_PIP = "#1a1a1a";
var C_SEL = "#f2c14e", C_DEST = "#5bbf6a";
var BTOP = [13,14,15,16,17,18,19,20,21,22,23,24];
var BBOT = [12,11,10,9,8,7,6,5,4,3,2,1];
var _PIP_LAYOUT = { 1:[[1,1]], 2:[[0,0],[2,2]], 3:[[0,0],[1,1],[2,2]],
  4:[[0,0],[2,0],[0,2],[2,2]], 5:[[0,0],[2,0],[1,1],[0,2],[2,2]],
  6:[[0,0],[2,0],[0,1],[2,1],[0,2],[2,2]] };

function bf(v){ var s = v.toFixed(1); return s.replace(/\.0$/, "").replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, ""); }
function barLeft(){ return BX0 + 6 * BCOLW; }
function colX(idx){ return idx < 6 ? BX0 + idx * BCOLW : barLeft() + BBARW + (idx - 6) * BCOLW; }
function pointColumn(n){ var i = BTOP.indexOf(n); if(i >= 0) return [i, true]; return [BBOT.indexOf(n), false]; }
function svgEsc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function pipCounts(points, barX, barO){
  var px = 25 * barX, po = 25 * barO;
  for(var p = 1; p < 25; p++){ var v = points[p]; if(v > 0) px += v * (25 - p); else if(v < 0) po += (-v) * p; }
  return [px, po];
}

function svgTriangle(col, top, light){
  var x = colX(col), cx = x + BCOLW / 2, fill = light ? C_PT_A : C_PT_B;
  var pts = top
    ? bf(x)+","+BYTOP+" "+bf(x+BCOLW)+","+BYTOP+" "+bf(cx)+","+bf(BYTOP+BPTH)
    : bf(x)+","+BYBOT+" "+bf(x+BCOLW)+","+BYBOT+" "+bf(cx)+","+bf(BYBOT-BPTH);
  return '<polygon points="'+pts+'" fill="'+fill+'" stroke="#241a12" stroke-width="1"/>';
}
function svgChecker(cx, cy, side){
  var fill = side === "X" ? C_X : C_O, edge = side === "X" ? C_X_EDGE : C_O_EDGE;
  return '<circle cx="'+bf(cx)+'" cy="'+bf(cy)+'" r="'+BCR+'" fill="'+fill+'" stroke="'+edge+'" stroke-width="2"/>';
}
function stackY(top, i){ return top ? BYTOP + BCR + i * (2 * BCR + 1) : BYBOT - BCR - i * (2 * BCR + 1); }
function svgStack(col, top, side, count){
  var out = [], cx = colX(col) + BCOLW / 2, shown = Math.min(count, 5);
  for(var i = 0; i < shown; i++){ out.push(svgChecker(cx, stackY(top, i), side)); }
  if(count > 5){
    var cy = stackY(top, 4), lab = side === "X" ? C_PIP : C_TEXT;
    out.push('<text x="'+bf(cx)+'" y="'+bf(cy+5)+'" text-anchor="middle" font-family="monospace" font-size="20" fill="'+lab+'">'+count+'</text>');
  }
  return out.join("");
}
function svgDie(x, y, value, size){
  size = size || 54;
  var out = ['<rect x="'+bf(x)+'" y="'+bf(y)+'" width="'+bf(size)+'" height="'+bf(size)+'" rx="8" fill="'+C_DIE+'" stroke="#000" stroke-width="1.5"/>'];
  var step = size / 3, r = size / 12, layout = _PIP_LAYOUT[value] || [];
  for(var i = 0; i < layout.length; i++){
    var cx = x + step * (layout[i][0] + 0.5), cy = y + step * (layout[i][1] + 0.5);
    out.push('<circle cx="'+bf(cx)+'" cy="'+bf(cy)+'" r="'+bf(r)+'" fill="'+C_PIP+'"/>');
  }
  return out.join("");
}
function svgText(x, y, s, size, anchor, weight){
  return '<text x="'+bf(x)+'" y="'+bf(y)+'" text-anchor="'+(anchor||"start")+'" font-family="monospace" font-size="'+(size||20)+'" font-weight="'+(weight||"normal")+'" fill="'+C_TEXT+'">'+svgEsc(s)+'</text>';
}

/* Render a board (display frame) to an SVG string. `view` is optional and adds
   the interactive layer: clickable point/bar/tray hit regions (data-bi), the
   selected-source ring, destination markers, and used/remaining dice shading. */
function renderBoardSVG(bd, view){
  view = view || {};
  var points = bd.points, barX = bd.bar.x, barO = bd.bar.o, offX = bd.off.x, offO = bd.off.o;
  var pc = pipCounts(points, barX, barO), px = pc[0], po = pc[1];
  var playRight = barLeft() + BBARW + 6 * BCOLW;
  var s = [];
  s.push('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+BW+' '+BH+'" width="'+BW+'" height="'+BH+'" class="bgboard'+(view.interactive?" interactive":"")+'">');
  s.push('<rect x="0" y="0" width="'+BW+'" height="'+BH+'" fill="'+C_BG+'"/>');
  s.push('<rect x="'+BX0+'" y="'+BYTOP+'" width="'+bf(playRight-BX0)+'" height="'+bf(BYBOT-BYTOP)+'" fill="'+C_BOARD+'" stroke="#000" stroke-width="2"/>');
  s.push('<rect x="'+bf(barLeft())+'" y="'+BYTOP+'" width="'+BBARW+'" height="'+bf(BYBOT-BYTOP)+'" fill="'+C_BAR+'"/>');
  s.push('<rect x="'+bf(playRight+10)+'" y="'+BYTOP+'" width="'+BTRAYW+'" height="'+bf(BYBOT-BYTOP)+'" fill="'+C_BAR+'" stroke="#000" stroke-width="2"/>');

  var n, col, top, light, i;
  for(i = 0; i < 24; i++){ n = i < 12 ? BTOP[i] : BBOT[i-12]; var r = pointColumn(n); col = r[0]; top = r[1];
    light = (col + (top ? 0 : 1)) % 2 === 0; s.push(svgTriangle(col, top, light)); }
  for(i = 0; i < 12; i++){ n = BTOP[i]; col = pointColumn(n)[0]; s.push(svgText(colX(col)+BCOLW/2, BYTOP-8, String(n), 16, "middle")); }
  for(i = 0; i < 12; i++){ n = BBOT[i]; col = pointColumn(n)[0]; s.push(svgText(colX(col)+BCOLW/2, BYBOT+22, String(n), 16, "middle")); }

  /* checkers */
  for(i = 0; i < 24; i++){ n = i < 12 ? BTOP[i] : BBOT[i-12]; var v = points[25-n]; if(v === 0) continue;
    var rr = pointColumn(n); s.push(svgStack(rr[0], rr[1], v > 0 ? "X" : "O", Math.abs(v))); }
  var barcx = barLeft() + BBARW / 2;
  for(i = 0; i < barO; i++){ s.push(svgChecker(barcx, BYTOP + 40 + i * (2*BCR+1), "O")); }
  for(i = 0; i < barX; i++){ s.push(svgChecker(barcx, BYBOT - 40 - i * (2*BCR+1), "X")); }
  var trayx = playRight + 10 + BTRAYW / 2;
  for(i = 0; i < offO; i++){ s.push(svgChecker(trayx, BYTOP + 20 + i * 14, "O")); }
  for(i = 0; i < offX; i++){ s.push(svgChecker(trayx, BYBOT - 20 - i * 14, "X")); }

  /* cube */
  var cubeVal = bd.cube.value, owner = bd.cube.owner;
  var cubeY = owner === "o" ? BYTOP + 6 : (owner === "x" ? BYBOT - 60 : (BYTOP+BYBOT)/2 - 27);
  s.push('<rect x="'+bf(BX0-6)+'" y="'+bf(cubeY)+'" width="54" height="54" rx="8" fill="#e8e2d2" stroke="#000" stroke-width="2"/>');
  s.push('<text x="'+bf(BX0-6+27)+'" y="'+bf(cubeY+36)+'" text-anchor="middle" font-family="monospace" font-size="26" fill="#1a1a1a">'+cubeVal+'</text>');

  /* dice or cube marker */
  if(bd.dice && bd.dice.length === 2){
    var dx = barLeft() + BBARW + 3 * BCOLW, dy = (BYTOP+BYBOT)/2 - 27;
    var rem = view.remaining, d0 = bd.dice[0], d1 = bd.dice[1], usedMask = [false, false];
    if(rem){
      if(d0 === d1){ usedMask = [rem.length < 2, rem.length < 1]; }
      else { var pool = rem.slice(); function take(v){ var k = pool.indexOf(v); if(k>=0){ pool.splice(k,1); return true;} return false; }
             usedMask[0] = !take(d0); usedMask[1] = !take(d1); }
    }
    s.push('<g'+(usedMask[0]?' opacity="0.32"':'')+'>'+svgDie(dx, dy, d0)+'</g>');
    s.push('<g'+(usedMask[1]?' opacity="0.32"':'')+'>'+svgDie(dx+70, dy, d1)+'</g>');
    if(d0 === d1 && rem){ s.push(svgText(dx+62, dy-10, "x"+rem.length, 18, "middle", "bold")); }
  } else {
    var lbl = (view.diceLabel !== undefined) ? view.diceLabel : "[cube decision]";
    if(lbl) s.push(svgText(barLeft() + BBARW + 3 * BCOLW, (BYTOP+BYBOT)/2, lbl, 20));
  }

  /* header text */
  var ctx;
  if(bd.score && bd.score.length){ ctx = "Match to " + bd.score.length + (bd.score.crawford ? " Crawford" : "") + "  White " + bd.score.x + "-" + bd.score.o + " Black"; }
  else { ctx = "Money game"; }
  s.push(svgText(BX0, 40, ctx, 24, "start", "bold"));
  s.push(svgText(BX0, 66, "Cube: " + cubeVal + " (" + (owner === "center" ? "centered" : owner === "x" ? "White" : "Black") + ")", 18));
  s.push(svgText(playRight, 40, "Black pip " + po, 20, "end"));
  s.push(svgText(playRight, 66, "White pip " + px + "  (White on roll)", 20, "end"));
  s.push('<polygon points="'+bf(BX0-20)+','+bf(BYBOT-20)+' '+bf(BX0-20)+','+bf(BYBOT)+' '+bf(BX0-4)+','+bf(BYBOT-10)+'" fill="'+C_X+'"/>');

  /* -- interactive overlay -- */
  if(view.interactive){
    /* selected-source ring */
    if(view.selected !== null && view.selected !== undefined){
      if(view.selected === "bar"){ s.push('<circle cx="'+bf(barcx)+'" cy="'+bf(BYBOT-40)+'" r="'+(BCR+5)+'" fill="none" stroke="'+C_SEL+'" stroke-width="4"/>'); }
      else { var sn = 25 - view.selected, sr = pointColumn(sn); var scx = colX(sr[0]) + BCOLW/2;
        var scount = Math.abs(points[view.selected]); var syi = Math.min(scount, 5) - 1; if(syi < 0) syi = 0;
        s.push('<circle cx="'+bf(scx)+'" cy="'+bf(stackY(sr[1], syi))+'" r="'+(BCR+5)+'" fill="none" stroke="'+C_SEL+'" stroke-width="4"/>'); }
    }
    /* destination markers (visual only; pointer-events off so the hit layer wins) */
    var hl = view.highlights || {};
    for(var key in hl){ if(!hl.hasOwnProperty(key)) continue;
      if(key === "off"){ s.push('<circle class="destdot" cx="'+bf(trayx)+'" cy="'+bf(BYBOT-20-offX*14)+'" r="14" fill="'+C_DEST+'" fill-opacity="0.85" stroke="#fff" stroke-width="2"/>'); }
      else { var dn = 25 - parseInt(key,10), dr = pointColumn(dn); var dcx = colX(dr[0]) + BCOLW/2;
        var dv = points[parseInt(key,10)]; var slot = dv > 0 ? dv : 0;
        s.push('<circle class="destdot" cx="'+bf(dcx)+'" cy="'+bf(stackY(dr[1], slot))+'" r="'+(BCR-2)+'" fill="'+C_DEST+'" fill-opacity="0.55" stroke="#fff" stroke-width="3"/>'); }
    }
    /* clickable hit regions LAST so they sit above every visual layer. Top and
       bottom points share a column, so each gets only its own half-height band. */
    var midY = (BYTOP + BYBOT) / 2, halfH = midY - BYTOP;
    for(i = 0; i < 24; i++){ n = i < 12 ? BTOP[i] : BBOT[i-12]; var cr = pointColumn(n); var cx0 = colX(cr[0]);
      var hy = cr[1] ? BYTOP : midY;
      s.push('<rect class="hit" data-bi="'+(25-n)+'" x="'+bf(cx0)+'" y="'+bf(hy)+'" width="'+BCOLW+'" height="'+bf(halfH)+'" fill="transparent"/>'); }
    s.push('<rect class="hit" data-bi="bar" x="'+bf(barLeft())+'" y="'+BYTOP+'" width="'+BBARW+'" height="'+bf(BYBOT-BYTOP)+'" fill="transparent"/>');
    s.push('<rect class="hit" data-bi="off" x="'+bf(playRight+10)+'" y="'+BYTOP+'" width="'+BTRAYW+'" height="'+bf(BYBOT-BYTOP)+'" fill="transparent"/>');
  }

  s.push("</svg>");
  return s.join("");
}

/* -- move engine: single-die hops in the board-index frame (ports bgcore/moves) -- */
function expandDice(dice){ if(dice.length === 2 && dice[0] === dice[1]) return [dice[0], dice[0], dice[0], dice[0]]; return dice.slice(); }
function allHomeJS(points, barX){ if(barX > 0) return false; for(var i = 1; i < 19; i++){ if(points[i] > 0) return false; } return true; }
function legalHops(st, d){
  var points = st.points, barX = st.barX, hops = [];
  if(barX > 0){ var j = d; if(points[j] >= -1) hops.push({frm:"bar", to:j, hit:points[j] === -1, die:d}); return hops; }
  var home = allHomeJS(points, barX), minHome = null;
  if(home){ for(var k = 19; k <= 24; k++){ if(points[k] > 0){ minHome = k; break; } } }
  for(var p = 1; p < 25; p++){ if(points[p] <= 0) continue; var dest = p + d;
    if(dest <= 24){ if(points[dest] >= -1) hops.push({frm:p, to:dest, hit:points[dest] === -1, die:d}); }
    else if(home){ if(dest === 25) hops.push({frm:p, to:"off", hit:false, die:d});
      else if(minHome !== null && p === minHome) hops.push({frm:p, to:"off", hit:false, die:d}); } }
  return hops;
}
function applyHop(st, h){
  var points = st.points.slice(), barX = st.barX, barO = st.barO, offX = st.offX, offO = st.offO;
  if(h.frm === "bar") barX -= 1; else points[h.frm] -= 1;
  if(h.to === "off") offX += 1;
  else { var j = h.to; if(points[j] < 0){ points[j] = 0; barO += 1; } points[j] += 1; }
  return {points:points, barX:barX, barO:barO, offX:offX, offO:offO};
}
function sigOf(st){ return st.points.join(",") + "|" + st.barX + "," + st.barO + "," + st.offX; }
/* DFS: can we reach one of `targets` (sig->1) from st using some of `remaining`? */
function reachTarget(st, remaining, targets){
  if(targets[sigOf(st)]) return true;
  if(remaining.length === 0) return false;
  var tried = {};
  for(var i = 0; i < remaining.length; i++){ var d = remaining[i]; if(tried[d]) continue; tried[d] = 1;
    var hops = legalHops(st, d);
    for(var k = 0; k < hops.length; k++){ var ns = applyHop(st, hops[k]);
      var rem = remaining.slice(0, i).concat(remaining.slice(i + 1));
      if(reachTarget(ns, rem, targets)) return true; } }
  return false;
}
/* notation coord + hop -> chain notation (ports bgcore.moves.format_hops) */
function ptNota(x){ return x === "bar" ? "bar" : (x === "off" ? "off" : (25 - x)); }
function chainsFromHops(hops){
  var chains = [];
  for(var i = 0; i < hops.length; i++){ var h = hops[i], a = ptNota(h.frm), b = ptNota(h.to), attached = false;
    if(a !== "bar"){ for(var c = 0; c < chains.length; c++){ var ch = chains[c];
      if(ch.pts[ch.pts.length - 1] === a){ ch.pts.push(b); ch.hits.push(h.hit); attached = true; break; } } }
    if(!attached) chains.push({pts:[a, b], hits:[h.hit]}); }
  return chains;
}
function renderChainObj(ch){ var out = String(ch.pts[0]); for(var i = 1; i < ch.pts.length; i++){ out += "/" + String(ch.pts[i]) + (ch.hits[i-1] ? "*" : ""); } return out; }
function formatHopsJS(hops){
  if(!hops.length) return "Cannot Move";
  var toks = chainsFromHops(hops).map(renderChainObj), cnt = {};
  for(var i = 0; i < toks.length; i++){ cnt[toks[i]] = (cnt[toks[i]] || 0) + 1; }
  var uniq = Object.keys(cnt);
  uniq.sort(function(a, b){ var sa = tokSortKey(a), sb = tokSortKey(b); if(sa !== sb) return sb - sa; return a < b ? -1 : (a > b ? 1 : 0); });
  return uniq.map(function(t){ var n = cnt[t]; return n > 1 ? (t + "(" + n + ")") : t; }).join(" ");
}

/* Interactive click-to-move controller. Mounts into `wrap` (board) + `ctrl`
   (controls), calls onComplete(canonicalNotation) when a full legal play is
   composed and the panelist submits. */
function ClickBoard(pos, wrap, ctrl, onSubmit){
  var init = pos.board;
  var targets = {}; for(var t = 0; t < pos.legal.length; t++){ targets[pos.legal[t].s] = pos.legal[t].n; }
  var hasLegal = pos.legal.length > 0;
  var live, remaining, used, selected, highlights;

  function initLive(){ return { points: init.points.slice(), barX: init.bar.x, barO: init.bar.o, offX: init.off.x, offO: init.off.o }; }
  function rebuild(){
    live = initLive(); remaining = expandDice(init.dice); selected = null;
    for(var i = 0; i < used.length; i++){ live = applyHop(live, used[i]); var idx = remaining.indexOf(used[i].die); if(idx >= 0) remaining.splice(idx, 1); }
  }
  function reset(){ used = []; rebuild(); }
  function currentSig(){ return sigOf(live); }
  function isComplete(){ return hasLegal ? !!targets[currentSig()] : false; }
  /* Canonical string to display / submit for a completed play: prefer the rollout
     move's spelling when the composed play resolves to one (e.g. a bear-in double
     "14/6" rather than the raw hop chain "14/12/10/8/6") — scores identically. */
  function playText(){
    var raw = targets[currentSig()];
    var res = scoreChecker(pos, raw);
    return (res && res.matched) ? res.matched : raw;
  }

  function computeHighlights(){
    highlights = {};
    if(selected === null) return;
    var seen = {};
    for(var i = 0; i < remaining.length; i++){ var d = remaining[i]; if(seen[d]) continue; seen[d] = 1;
      var hops = legalHops(live, d);
      for(var k = 0; k < hops.length; k++){ var h = hops[k]; if(h.frm !== selected) continue;
        var ns = applyHop(live, h); var rem = remaining.slice(); rem.splice(rem.indexOf(d), 1);
        if(reachTarget(ns, rem, targets)){ highlights[h.to === "off" ? "off" : String(h.to)] = h; } } }
  }
  /* a board point / bar is a valid *source* if selecting it exposes >=1 dest */
  function sourceHasMove(bi){
    var save = selected; selected = bi; computeHighlights(); var any = false;
    for(var k in highlights){ if(highlights.hasOwnProperty(k)){ any = true; break; } }
    selected = save; computeHighlights(); return any;
  }

  function playHop(h){ used.push(h); live = applyHop(live, h); var idx = remaining.indexOf(h.die); if(idx >= 0) remaining.splice(idx, 1); selected = null; }

  function onHit(bi){
    if(isComplete()) return;
    if(selected !== null){ var key = (bi === "off") ? "off" : String(bi);
      if(highlights[key]){ playHop(highlights[key]); paint(); return; } }
    if(live.barX > 0){ selected = (selected === "bar") ? null : (sourceHasMove("bar") ? "bar" : null); }
    else if(typeof bi === "number" && live.points[bi] > 0){ selected = (selected === bi) ? null : (sourceHasMove(bi) ? bi : null); }
    else { selected = null; }
    paint();
  }

  function composedText(){
    if(isComplete()) return playText();
    if(used.length === 0) return "—";
    return formatHopsJS(used);
  }

  function paint(){
    computeHighlights();
    var bd = { points: live.points, bar: {x: live.barX, o: live.barO}, off: {x: live.offX, o: live.offO},
      dice: init.dice, cube: init.cube, score: init.score };
    wrap.innerHTML = renderBoardSVG(bd, { interactive: true, selected: selected, highlights: highlights, remaining: remaining });
    var svg = wrap.querySelector("svg");
    svg.addEventListener("click", function(e){
      var node = e.target; while(node && node !== svg && !(node.getAttribute && node.getAttribute("data-bi") !== null && node.getAttribute("data-bi") !== undefined)) node = node.parentNode;
      if(!node || node === svg) return;
      var raw = node.getAttribute("data-bi"); if(raw === null) return;
      onHit(raw === "bar" || raw === "off" ? raw : parseInt(raw, 10));
    });
    renderControls();
  }

  function renderControls(){
    ctrl.innerHTML = "";
    if(!hasLegal){
      ctrl.appendChild(el("p", { class: "warn", text: "No legal move for this roll — you must pass." }));
      var pb = el("button", { class: "btn", text: "Submit: Cannot Move" });
      pb.onclick = function(){ onSubmit("Cannot Move"); };
      ctrl.appendChild(el("div", { class: "btn-row" }, [pb]));
      return;
    }
    var status = selected === null
      ? (used.length ? "Click another White checker to keep moving, or Undo." : "Click one of your White checkers to move it.")
      : "Now click a highlighted point (or the same checker to deselect).";
    ctrl.appendChild(el("div", { class: "cb-status small muted", text: status }));
    ctrl.appendChild(el("div", { class: "cb-move" }, [
      el("span", { class: "small muted", text: "Your play: " }),
      el("b", { class: "cb-notation" + (isComplete() ? " done" : ""), text: composedText() })
    ]));
    var undo = el("button", { class: "btn secondary", text: "Undo" });
    undo.disabled = used.length === 0;
    undo.onclick = function(){ used.pop(); rebuild(); paint(); };
    var rst = el("button", { class: "btn secondary", text: "Reset" });
    rst.disabled = used.length === 0;
    rst.onclick = function(){ reset(); paint(); };
    var sub = el("button", { class: "btn", text: "Submit play" });
    sub.disabled = !isComplete();
    sub.onclick = function(){ if(isComplete()) onSubmit(playText()); };
    ctrl.appendChild(el("div", { class: "btn-row" }, [sub, undo, rst]));
  }

  reset();
  paint();
  return { paint: paint };
}

/* ---- screens ---- */
function modeChooser(current){
  /* Two radio cards: practice (feedback after each answer) vs blind run. */
  function card(mode, title, desc){
    var id = "mode-" + mode;
    var radio = el("input", { type: "radio", name: "runmode", id: id, value: mode });
    if(current === mode) radio.checked = true;
    var lab = el("label", { class: "mode-card", for: id }, [
      radio, el("b", { text: title }), el("span", { class: "small muted", text: desc })
    ]);
    return lab;
  }
  return el("div", { class: "mode-opts" }, [
    card(MODES.practice, "Practice — feedback after each answer",
      "See the engine's best play and your equity loss right after every position. Good for learning."),
    card(MODES.blind, "Blind panel run — results only at the end",
      "No feedback until you finish all " + TOTAL + " positions. Use this for a clean benchmark run.")
  ]);
}
function selectedMode(){
  var checked = document.querySelector('input[name="runmode"]:checked');
  return (checked && checked.value === MODES.blind) ? MODES.blind : MODES.practice;
}

function screenIntro(){
  clear();
  var nameVal = STATE.name || "";
  var input = el("input", { type: "text", id: "pname", value: nameVal,
    placeholder: "e.g. panelist-3 or your initials" });
  var answered = Object.keys(STATE.answers).length;
  var resumeNote = answered > 0 ? el("p", { class: "muted small",
    text: "Resuming: " + answered + " of " + TOTAL + " already answered on this device (mode locked to “" +
      (STATE.mode === MODES.blind ? "blind" : "practice") + "”)." }) : null;
  var chooser = answered > 0 ? null : modeChooser(STATE.mode || MODES.practice);
  var btn = el("button", { class: "btn", text: answered > 0 ? "Resume" : "Start" });
  btn.onclick = function(){
    var v = document.getElementById("pname").value.trim();
    if(!v){ document.getElementById("pname").focus(); return; }
    STATE.name = v;
    if(answered === 0) STATE.mode = selectedMode();   /* lock mode once a run begins */
    if(!STATE.started) STATE.started = new Date().toISOString();
    STATE.v = STATE_VERSION;
    saveState(STATE);
    routeToNext();
  };
  var reset = null;
  if(answered > 0){
    reset = el("button", { class: "btn secondary", text: "Start over (clear answers)" });
    reset.onclick = function(){
      if(confirm("Clear all saved answers on this device and start fresh?")){
        clearState(); STATE = freshState(); STATE_STALE = false; screenIntro();
      }
    };
  }
  app.appendChild(el("div", { class: "panel" }, [
    el("h1", { text: "Backgammon Human Benchmark — Pilot" }),
    el("p", { text: "You will see " + TOTAL + " backgammon positions, one at a time. For each, " +
      "give the play you think is best. Some are checker plays (click or type the move); some are cube decisions (pick a button)." }),
    el("p", { html: "In every diagram <b>you play the White checkers</b> (your opponent is Black), and it is " +
      "always your roll. You move your White checkers toward your home board (points 6 to 1, lower right) and " +
      "bear off on the right. Points are numbered from your perspective (24 = back checkers, 1 = ace point)." }),
    el("p", { class: "muted", html: "You cannot go back to change an earlier answer. Your progress is saved on this device, " +
      "so an accidental tab close will not lose it." }),
    el("hr", { class: "divider" }),
    chooser ? el("label", { text: "Run mode" }) : null,
    chooser,
    el("label", { for: "pname", text: "Your name or identifier" }),
    input,
    resumeNote,
    el("div", { class: "btn-row" }, [btn, reset])
  ]));
  input.focus();
}

/* Stale/corrupt saved-state recovery screen (offered instead of crashing). */
function screenStale(){
  clear();
  var fresh = el("button", { class: "btn", text: "Start fresh" });
  fresh.onclick = function(){ clearState(); STATE = freshState(); STATE_STALE = false; screenIntro(); };
  app.appendChild(el("div", { class: "panel" }, [
    el("h1", { text: "Saved progress can’t be restored" }),
    el("p", { html: "This device has saved answers from an <b>older version</b> of the quiz. To avoid mixing " +
      "incompatible data, please start a fresh run. (Your earlier run is not lost — if you already downloaded its " +
      "results JSON, that file is still valid.)" }),
    el("div", { class: "btn-row" }, [fresh])
  ]));
}

function contextChips(pos){
  var chips = [];
  if(pos.play_mode === "match" && pos.score.length){
    var s = "to " + pos.score.length + (pos.score.crawford ? ", Crawford" : "") +
      " — you " + pos.score.x + ", opp " + pos.score.o;
    chips.push(el("span", { class: "chip", html: "<b>Match</b> " + esc(s) }));
  } else {
    chips.push(el("span", { class: "chip", html: "<b>Money game</b>" }));
  }
  var owner = pos.cube.owner === "x" ? "you (White)" : (pos.cube.owner === "o" ? "opponent (Black)" : "centered");
  chips.push(el("span", { class: "chip", html: "<b>Cube:</b> " + pos.cube.value + " (" + owner + ")" }));
  if(pos.decision_type === "checker" && pos.dice.length === 2){
    chips.push(el("span", { class: "chip", html: "<b>Dice:</b> " + pos.dice[0] + "-" + pos.dice[1] }));
  }
  chips.push(el("span", { class: "chip", html: "<b>Pips:</b> you " + pos.pip.x + " · opp " + pos.pip.o }));
  return el("div", { class: "meta" }, chips);
}

function screenPosition(idx){
  clear();
  var pos = DATA[idx];
  var pct = Math.round((idx / TOTAL) * 100);
  var head = el("div", {}, [
    el("div", { class: "progress" }, [
      el("span", { text: "Position " + (idx + 1) + " / " + TOTAL }),
      el("span", { text: STATE.name || "" })
    ]),
    el("div", { class: "bar" }, [el("span", { style: "width:" + pct + "%" })])
  ]);

  var boardWrap = el("div", { class: "board-wrap" });
  var title = pos.decision_type === "cube"
    ? el("p", { html: "<b>You are White, on roll.</b> It is your cube decision — what is your action?" })
    : el("p", { html: "<b>You are White, on roll.</b> You rolled <b>" +
        (pos.dice.length === 2 ? pos.dice[0] + "-" + pos.dice[1] : "?") + "</b>. Click your checkers to compose your play." });

  var panel = el("div", { class: "panel" }, [head, boardWrap, title, contextChips(pos)]);

  if(pos.decision_type === "checker"){
    /* --- primary input: click-to-move on the single board engine --- */
    var ctrl = el("div", { class: "cb-controls" });
    var controller = new ClickBoard(pos, boardWrap, ctrl, function(notation){
      recordAndAdvance(idx, scoreChecker(pos, notation));
    });
    panel.appendChild(ctrl);

    /* --- secondary input: freeform text, with live legal-move validation --- */
    panel.appendChild(el("hr", { class: "divider" }));
    var details = el("details", { class: "text-entry" });
    details.appendChild(el("summary", { text: "Prefer to type the move? (advanced)" }));
    var warned = false;
    var input = el("input", { type: "text", id: "movein",
      placeholder: "e.g. 24/18 13/11", autocomplete: "off", autocapitalize: "off", spellcheck: "false" });
    var hint = el("div", { class: "hint",
      text: "Standard notation from your side: 24/18 13/11. Use * for a hit (8/4*), (2) for a repeated move (24/23(2)), bar/22 to enter, 6/off to bear off." });
    var live = el("div", { class: "hint live-valid" });
    var warnBox = el("div", { class: "warn", style: "display:none" });
    var submit = el("button", { class: "btn", text: "Submit typed play" });
    var submitAnyway = el("button", { class: "btn secondary", text: "Submit anyway", style: "display:none" });

    function updateLive(){
      var val = input.value.trim();
      if(!val){ live.textContent = ""; live.className = "hint live-valid"; return; }
      if(checkerMatches(pos, val)){ live.textContent = "✓ legal move for this roll"; live.className = "hint live-valid ok"; }
      else if(canonicalizeMove(val) === null){ live.textContent = "✗ can’t parse that notation yet"; live.className = "hint live-valid bad"; }
      else { live.textContent = "✗ not a legal move for this roll"; live.className = "hint live-valid bad"; }
    }
    function doScore(force){
      var val = input.value.trim();
      if(!val){ input.focus(); return; }
      var matches = checkerMatches(pos, val);
      if(!matches && !force){
        warnBox.style.display = "block";
        warnBox.textContent = "“" + val + "” is not recognised as a legal/listed move — check your notation, or submit anyway (it will be scored as your worst option).";
        submitAnyway.style.display = "inline-block";
        warned = true;
        return;
      }
      recordAndAdvance(idx, scoreChecker(pos, val));
    }
    submit.onclick = function(){ doScore(false); };
    submitAnyway.onclick = function(){ doScore(true); };
    input.addEventListener("keydown", function(e){ if(e.key === "Enter"){ doScore(false); } });
    input.addEventListener("input", function(){
      if(warned){ warnBox.style.display = "none"; submitAnyway.style.display = "none"; }
      updateLive();
    });
    details.appendChild(el("label", { for: "movein", text: "Type your play" }));
    details.appendChild(input);
    details.appendChild(hint);
    details.appendChild(live);
    details.appendChild(warnBox);
    details.appendChild(el("div", { class: "btn-row" }, [submit, submitAnyway]));
    panel.appendChild(details);
    app.appendChild(panel);
  } else {
    boardWrap.innerHTML = renderBoardSVG(pos.board, {});
    var opts = el("div", { class: "cube-opts" });
    pos.options.forEach(function(label){
      var b = el("button", { class: "btn", text: label });
      b.onclick = function(){ recordAndAdvance(idx, scoreCube(pos, label)); };
      opts.appendChild(b);
    });
    panel.appendChild(el("label", { text: "Your cube decision" }));
    panel.appendChild(opts);
    app.appendChild(panel);
  }
  window.scrollTo(0, 0);
}

/* Guards against a double-click / double-submit recording an answer twice or
   double-advancing: once an answer for `idx` exists we never overwrite it, and a
   reentrant call while a submit is in flight is ignored. */
var ADVANCING = false;
function recordAndAdvance(idx, res){
  if(ADVANCING) return;
  var pos = DATA[idx];
  if(STATE.answers[pos.position_id]) return;   /* already answered — no re-record */
  ADVANCING = true;
  try {
    STATE.answers[pos.position_id] = {
      chosen: res.chosen, is_best: res.is_best, equity_loss: res.equity_loss,
      matched: res.matched, parse_failed: !!res.parse_failed
    };
    saveState(STATE);
    if(STATE.mode === MODES.blind){ routeToNext(); }
    else { screenFeedback(idx); }
  } finally {
    ADVANCING = false;
  }
}

function routeToNext(){
  var i = firstUnanswered();
  if(i >= TOTAL) screenResults();
  else screenPosition(i);
}

/* ---- per-answer feedback (practice mode) ---- */
function moveRank(pos, moveStr){
  if(!moveStr || !pos.moves) return null;
  var canon = canonicalizeMove(moveStr);
  for(var i = 0; i < pos.moves.length; i++){
    if(pos.moves[i].move === moveStr || pos.moves[i].canonical === canon) return pos.moves[i].rank;
  }
  return null;
}
function cubeRank(pos, label){
  /* 1-based rank of an action among the posed options, ascending by error. */
  var order = pos.options.slice().sort(function(a, b){
    return (pos.error_mp[a] || 0) - (pos.error_mp[b] || 0);
  });
  var k = order.indexOf(label);
  return k >= 0 ? (k + 1) : null;
}

function screenFeedback(idx){
  clear();
  var pos = DATA[idx];
  var a = STATE.answers[pos.position_id];
  var pct = Math.round(((idx + 1) / TOTAL) * 100);
  var head = el("div", {}, [
    el("div", { class: "progress" }, [
      el("span", { text: "Feedback — position " + (idx + 1) + " / " + TOTAL }),
      el("span", { text: STATE.name || "" })
    ]),
    el("div", { class: "bar" }, [el("span", { style: "width:" + pct + "%" })])
  ]);

  var isBest = !!a.is_best;
  var verdict = el("div", { class: "verdict " + (isBest ? "ok" : "bad") },
    [ isBest ? "✓ Best play" : "✗ Not the best play" ]);

  var lossMp = a.equity_loss * MP_PER_POINT;
  var best = bestMoveText(pos);
  var rows = [];
  rows.push(el("div", { class: "fb-row" }, [ el("span", { class: "fb-k", text: "Your answer" }),
    el("span", { class: "fb-v", text: a.chosen + (a.parse_failed ? "  (not recognised — scored as worst option)" : "") }) ]));
  rows.push(el("div", { class: "fb-row" }, [ el("span", { class: "fb-k", text: pos.decision_type === "cube" ? "Best action" : "Best play" }),
    el("span", { class: "fb-v", text: best }) ]));
  rows.push(el("div", { class: "fb-row" }, [ el("span", { class: "fb-k", text: "Your equity loss" }),
    el("span", { class: "fb-v", text: fmt(lossMp, 1) + " mpt" + (isBest ? "  (0 — best)" : "") }) ]));

  var rank = pos.decision_type === "cube" ? cubeRank(pos, a.chosen) : moveRank(pos, a.matched);
  var nlisted = pos.decision_type === "cube" ? pos.options.length : (pos.moves ? pos.moves.length : 0);
  if(rank){
    rows.push(el("div", { class: "fb-row" }, [ el("span", { class: "fb-k", text: "Rank of your choice" }),
      el("span", { class: "fb-v", text: "#" + rank + " of " + nlisted + (pos.decision_type === "cube" ? " actions" : " listed moves") }) ]));
  } else if(pos.decision_type === "checker") {
    rows.push(el("div", { class: "fb-row" }, [ el("span", { class: "fb-k", text: "Rank of your choice" }),
      el("span", { class: "fb-v", text: "not among the " + nlisted + " listed moves" }) ]));
  }

  var kids = [head, verdict, el("div", { class: "fb-grid" }, rows)];

  /* Checker: redraw the board with the engine's best play applied (if resolvable). */
  if(pos.decision_type === "checker" && pos.best_after){
    var bd = { points: pos.best_after.points, bar: pos.best_after.bar, off: pos.best_after.off,
      dice: [], cube: pos.board.cube, score: pos.board.score };
    var fbWrap = el("div", { class: "board-wrap" });
    fbWrap.innerHTML = renderBoardSVG(bd, { diceLabel: "▶ best play: " + best });
    kids.push(el("p", { class: "small muted", text: "The board below shows the position after the engine's best play." }));
    kids.push(fbWrap);
  }

  var next = el("button", { class: "btn", text: (idx + 1 >= TOTAL) ? "See results" : "Next position" });
  next.onclick = function(){ routeToNext(); };
  kids.push(el("div", { class: "btn-row" }, [next]));

  app.appendChild(el("div", { class: "panel" }, kids));
  window.scrollTo(0, 0);
}

/* ---- results ---- */
function computeAggregate(){
  var losses = [], best = [], parseFail = 0;
  var byTier = {}, byType = {};
  for(var i = 0; i < TOTAL; i++){
    var pos = DATA[i];
    var a = STATE.answers[pos.position_id];
    if(!a) continue;
    losses.push(a.equity_loss);
    best.push(a.is_best ? 1 : 0);
    if(a.parse_failed) parseFail++;
    (byTier[pos.tier] = byTier[pos.tier] || []).push(a);
    (byType[pos.decision_type] = byType[pos.decision_type] || []).push(a);
  }
  function rollup(rows){
    var l = rows.map(function(r){ return r.equity_loss; });
    var b = rows.map(function(r){ return r.is_best ? 1 : 0; });
    var mean = l.length ? l.reduce(function(x, y){ return x + y; }, 0) / l.length : 0;
    var acc = b.length ? b.reduce(function(x, y){ return x + y; }, 0) / b.length : 0;
    return { benchpr: BENCHPR_CONSTANT * mean, best_move_accuracy: acc, mean_equity_loss: mean, n: rows.length };
  }
  var mean = losses.length ? losses.reduce(function(x, y){ return x + y; }, 0) / losses.length : 0;
  var acc = best.length ? best.reduce(function(x, y){ return x + y; }, 0) / best.length : 0;
  var perTier = {}, perType = {};
  Object.keys(byTier).sort().forEach(function(k){ perTier[k] = rollup(byTier[k]); });
  Object.keys(byType).sort().forEach(function(k){ perType[k] = rollup(byType[k]); });
  return {
    benchpr: BENCHPR_CONSTANT * mean, benchpr_label: "BenchPR (PR-calibrated)",
    best_move_accuracy: acc, mean_equity_loss: mean, n: losses.length,
    parse_failures: parseFail, per_tier: perTier, per_decision_type: perType
  };
}

function buildResultsJSON(agg){
  var slug = (STATE.name || "anon").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  var ts = new Date().toISOString();
  var decisions = [];
  for(var i = 0; i < TOTAL; i++){
    var pos = DATA[i];
    var a = STATE.answers[pos.position_id];
    if(!a) continue;
    decisions.push({ position_id: pos.position_id, chosen: a.chosen,
      is_best: a.is_best, equity_loss: a.equity_loss });
  }
  var manifest = {
    dataset_hash: MANIFEST.dataset_hash, prompt_version: MANIFEST.prompt_version,
    ascii_render_version: MANIFEST.ascii_render_version,
    image_render_version: MANIFEST.image_render_version, timestamp: ts
  };
  return {
    kind: "human", run_id: "human-" + (slug || "anon") + "-" + ts.replace(/[:.]/g, "").slice(0, 15),
    model: "human-panel/" + (STATE.name || "anon"), track: "text",
    mode: (STATE.mode === MODES.blind) ? MODES.blind : MODES.practice,
    manifest: manifest, aggregate: agg, decisions: decisions
  };
}

function bestMoveText(pos){
  if(pos.decision_type === "cube") return pos.best_action || "—";
  return pos.best_move || "—";
}

function screenResults(){
  clear();
  var agg = computeAggregate();
  var payload = buildResultsJSON(agg);

  var stats = el("div", { class: "headline" }, [
    el("div", { class: "stat" }, [ el("div", { class: "n", text: fmt(agg.benchpr, 2) }), el("div", { class: "l", text: "BenchPR (lower is better)" }) ]),
    el("div", { class: "stat" }, [ el("div", { class: "n", text: fmt(agg.mean_equity_loss, 4) }), el("div", { class: "l", text: "Mean equity loss" }) ]),
    el("div", { class: "stat" }, [ el("div", { class: "n", text: Math.round(agg.best_move_accuracy * 100) + "%" }), el("div", { class: "l", text: "Best-move accuracy" }) ]),
    el("div", { class: "stat" }, [ el("div", { class: "n", text: String(agg.n) }), el("div", { class: "l", text: "Positions scored" }) ])
  ]);

  /* per-tier + per-type tables */
  function breakdownTable(caption, obj, keyLabel){
    var rows = [ el("tr", {}, [ el("th", { text: keyLabel }), el("th", { class: "num", text: "n" }),
      el("th", { class: "num", text: "BenchPR" }), el("th", { class: "num", text: "Best-move" }) ]) ];
    Object.keys(obj).forEach(function(k){
      var r = obj[k];
      rows.push(el("tr", {}, [ el("td", { text: k }), el("td", { class: "num", text: String(r.n) }),
        el("td", { class: "num", text: fmt(r.benchpr, 2) }),
        el("td", { class: "num", text: Math.round(r.best_move_accuracy * 100) + "%" }) ]));
    });
    return el("div", {}, [ el("h2", { text: caption }), el("table", {}, rows) ]);
  }

  /* per-position review */
  var revRows = [ el("tr", {}, [ el("th", { text: "#" }), el("th", { text: "Type" }), el("th", { text: "Tier" }),
    el("th", { text: "Your answer" }), el("th", { text: "Best" }), el("th", { class: "num", text: "Eq. loss" }) ]) ];
  for(var i = 0; i < TOTAL; i++){
    var pos = DATA[i];
    var a = STATE.answers[pos.position_id];
    if(!a) continue;
    revRows.push(el("tr", {}, [
      el("td", { text: String(i + 1) }),
      el("td", { text: pos.decision_type }),
      el("td", { text: pos.tier }),
      el("td", { class: a.is_best ? "tag-ok" : "", text: a.chosen + (a.is_best ? " ✓" : "") }),
      el("td", { text: bestMoveText(pos) }),
      el("td", { class: "num" + (a.equity_loss > 0 ? " tag-bad" : ""), text: fmt(a.equity_loss, 4) })
    ]));
  }

  var dlBtn = el("button", { class: "btn", text: "Download results JSON" });
  dlBtn.onclick = function(){
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var slug = (STATE.name || "anon").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    var a = el("a", { href: url, download: "human-panel_" + (slug || "anon") + "_text.json" });
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function(){ URL.revokeObjectURL(url); }, 1000);
  };

  app.appendChild(el("div", { class: "panel" }, [
    el("h1", { text: "Results" }),
    el("p", { class: "muted", html: "Panelist: <b>" + esc(STATE.name || "anon") + "</b>. Engine ground truth is revealed below. " +
      "BenchPR = 500 × mean equity loss (same axis as XG Performance Rating): PR 2–4 world-class, 5–8 strong expert, 10–15 intermediate." }),
    stats,
    breakdownTable("By tier", agg.per_tier, "Tier"),
    breakdownTable("By decision type", agg.per_decision_type, "Type"),
    el("h2", { text: "Per-position review" }),
    el("div", { class: "board-wrap" }, [ el("table", {}, revRows) ]),
    el("hr", { class: "divider" }),
    el("div", { class: "btn-row" }, [dlBtn]),
    el("p", { class: "hint", text: "Send the downloaded JSON back to whoever ran the panel. It lands in results/ as a human run." })
  ]));
  window.scrollTo(0, 0);
}

/* ---- boot + error boundary ---- */
function screenError(err){
  clear();
  app.appendChild(el("div", { class: "panel" }, [
    el("h1", { text: "Something went wrong" }),
    el("p", { text: "The quiz hit an unexpected error. Your saved answers on this device are not lost — " +
      "try reloading the page. If it keeps happening, use “Start over” from the intro to reset." }),
    el("pre", { class: "errbox", text: String((err && (err.stack || err.message)) || err || "unknown error") })
  ]));
}
function boot(){
  if(STATE_STALE){ screenStale(); return; }
  if(STATE.name && firstUnanswered() < TOTAL){ screenPosition(firstUnanswered()); }
  else if(STATE.name && Object.keys(STATE.answers).length >= TOTAL){ screenResults(); }
  else { screenIntro(); }
}
/* Last-resort handler so an exception in any click handler surfaces a readable
   box instead of leaving a dead/blank page. */
var _errShown = false;
window.onerror = function(msg, src, line, col, err){
  if(_errShown) return false;
  _errShown = true;
  try { screenError(err || msg); } catch(_){ try { app.textContent = "Fatal error: " + msg; } catch(__){} }
  return false;
};
try { boot(); }
catch(e){ _errShown = true; try { screenError(e); } catch(_){ app.textContent = "Fatal error: " + e; } }
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the human-benchmark pilot quiz HTML.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output HTML path")
    parser.add_argument("--timestamp", default=None,
                        help="fixed ISO manifest timestamp (default: now, UTC)")
    args = parser.parse_args(argv)

    records = load_positions()
    if len(records) != 50:
        print(f"WARNING: expected 50 pilot positions, found {len(records)}", file=sys.stderr)
    data = build_data(records)
    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(records, ts)
    html = render_html(data, manifest)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)

    n_checker = sum(1 for d in data if d["decision_type"] == "checker")
    n_cube = sum(1 for d in data if d["decision_type"] == "cube")
    print(f"Wrote {args.out}")
    print(f"  positions: {len(data)}  (checker {n_checker}, cube {n_cube})")
    print(f"  dataset_hash: {manifest['dataset_hash']}")
    print(f"  size: {len(html):,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
