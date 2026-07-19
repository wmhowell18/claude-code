#!/usr/bin/env python3
"""Build a single self-contained human-benchmark quiz HTML for the pilot set.

Reads the 50 pilot position records (``positions/pilot/bg-*.json``) and their GNU
BG rollouts (``rollouts/gnubg/bg-*.json``), re-renders each board SVG in the
mover frame (``render/svg.py``) and emits ONE fully self-contained HTML file at
``site/public/human-benchmark-pilot.html`` — every SVG embedded, all CSS/JS inline,
zero network requests — so it can be emailed to panelists and opened locally.

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
import re
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
import render.svg as svgrender  # noqa: E402

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


# Checker colors for the panel: the on-roll player ("x") plays WHITE, the
# opponent ("o") plays BLACK. render/svg.py already draws the mover in a light
# tone and the opponent dark; we push both to true white/black so the UI can use
# the "white"/"black" vocabulary the panel asked for. Patched at import time
# (module globals are read at render call time), scoped to this process only.
svgrender._C_X = "#ffffff"       # mover / you = white
svgrender._C_X_EDGE = "#8a8a8a"
svgrender._C_O = "#000000"       # opponent = black
svgrender._C_O_EDGE = "#5a5a5a"


def _svg_white_black(svg: str) -> str:
    """Relabel the X/O text baked into the SVG to White/Black (colors now match)."""
    svg = svg.replace("X pip", "White pip").replace("(X on roll)", "(White on roll)")
    svg = svg.replace("O pip", "Black pip")
    svg = re.sub(r"\bX (\d+)-(\d+) O\b", r"White \1-\2 Black", svg)  # match score
    svg = svg.replace("Cube: ", "Cube: ")
    svg = re.sub(r"\(x\)", "(White)", svg)
    svg = re.sub(r"\(o\)", "(Black)", svg)
    svg = svg.replace("(center)", "(centered)")
    return svg


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

        # Render/score in the on-roll player's frame (board_json authoritative;
        # only color-mirrored for the checker positions whose rollout demands it).
        mover = _display_board(rec, roll)
        svg = _svg_white_black(svgrender.render(mover).strip())
        pip_x, pip_o = pip_counts(mover)

        entry: dict = {
            "position_id": pid,
            "decision_type": rec["decision_type"],
            "tier": rec["tier"],
            "play_mode": rec["play_mode"],
            "score": {
                "x": int(mover.score.get("x", 0)),
                "o": int(mover.score.get("o", 0)),
                "length": int(mover.score.get("length", 0)),
                "crawford": bool(mover.score.get("crawford", False)),
            },
            "cube": {
                "value": int(mover.cube.get("value", 1)),
                "owner": mover.cube.get("owner", "center"),
            },
            "dice": [int(d) for d in mover.dice],
            "pip": {"x": pip_x, "o": pip_o},
            "svg": svg,
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
"""

_JS = r"""
"use strict";
/* ---- data ---- */
var DATA = JSON.parse(document.getElementById("bench-data").textContent);
var MANIFEST = JSON.parse(document.getElementById("bench-manifest").textContent);
var TOTAL = DATA.length;
var STORE_KEY = "bg-human-bench-pilot-v1";

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
function loadState(){
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; }
  catch(e){ return {}; }
}
function saveState(st){
  try { localStorage.setItem(STORE_KEY, JSON.stringify(st)); } catch(e){}
}
var STATE = loadState();
/* STATE = { name, started, answers: { position_id: {chosen,is_best,equity_loss,matched,parse_failed} } } */
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

/* ---- screens ---- */
function screenIntro(){
  clear();
  var nameVal = STATE.name || "";
  var input = el("input", { type: "text", id: "pname", value: nameVal,
    placeholder: "e.g. panelist-3 or your initials" });
  var answered = Object.keys(STATE.answers).length;
  var resumeNote = answered > 0 ? el("p", { class: "muted small",
    text: "Resuming: " + answered + " of " + TOTAL + " already answered on this device." }) : null;
  var btn = el("button", { class: "btn", text: answered > 0 ? "Resume" : "Start" });
  btn.onclick = function(){
    var v = document.getElementById("pname").value.trim();
    if(!v){ document.getElementById("pname").focus(); return; }
    STATE.name = v;
    if(!STATE.started) STATE.started = new Date().toISOString();
    saveState(STATE);
    routeToNext();
  };
  var reset = null;
  if(answered > 0){
    reset = el("button", { class: "btn secondary", text: "Start over (clear answers)" });
    reset.onclick = function(){
      if(confirm("Clear all saved answers on this device and start fresh?")){
        STATE = { answers: {} }; saveState(STATE); screenIntro();
      }
    };
  }
  app.appendChild(el("div", { class: "panel" }, [
    el("h1", { text: "Backgammon Human Benchmark — Pilot" }),
    el("p", { text: "You will see " + TOTAL + " backgammon positions, one at a time. For each, " +
      "give the play you think is best. Some are checker plays (type the move); some are cube decisions (pick a button)." }),
    el("p", { html: "In every diagram <b>you play the White checkers</b> (your opponent is Black), and it is " +
      "always your roll. You move your White checkers toward your home board (points 6 to 1, lower right) and " +
      "bear off on the right. Points are numbered from your perspective (24 = back checkers, 1 = ace point)." }),
    el("p", { class: "muted", html: "This is a <b>blind</b> test: no engine evaluation or “best move” is shown until you finish all " +
      TOTAL + " positions. You cannot go back to change an earlier answer. Your progress is saved on this device, " +
      "so an accidental tab close will not lose it." }),
    el("hr", { class: "divider" }),
    el("label", { for: "pname", text: "Your name or identifier" }),
    input,
    resumeNote,
    el("div", { class: "btn-row" }, [btn, reset])
  ]));
  input.focus();
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

  var boardWrap = el("div", { class: "board-wrap", html: pos.svg });
  var title = pos.decision_type === "cube"
    ? el("p", { html: "<b>You are White, on roll.</b> It is your cube decision — what is your action?" })
    : el("p", { html: "<b>You are White, on roll.</b> You rolled <b>" +
        (pos.dice.length === 2 ? pos.dice[0] + "-" + pos.dice[1] : "?") + "</b>. What is your play?" });

  var panel = el("div", { class: "panel" }, [head, boardWrap, title, contextChips(pos)]);

  if(pos.decision_type === "checker"){
    var warned = false;
    var input = el("input", { type: "text", id: "movein",
      placeholder: "e.g. 24/18 13/11", autocomplete: "off", autocapitalize: "off", spellcheck: "false" });
    var hint = el("div", { class: "hint",
      text: "Standard notation from your side: 24/18 13/11. Use * for a hit (8/4*), (2) for a repeated move (24/23(2)), bar/22 to enter, 6/off to bear off." });
    var warnBox = el("div", { class: "warn", style: "display:none" });
    var submit = el("button", { class: "btn", text: "Submit play" });
    var submitAnyway = el("button", { class: "btn secondary", text: "Submit anyway", style: "display:none" });

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
      var res = scoreChecker(pos, val);
      recordAndAdvance(idx, res);
    }
    submit.onclick = function(){ doScore(false); };
    submitAnyway.onclick = function(){ doScore(true); };
    input.addEventListener("keydown", function(e){ if(e.key === "Enter"){ doScore(false); } });
    input.addEventListener("input", function(){
      if(warned){ warnBox.style.display = "none"; submitAnyway.style.display = "none"; }
    });

    panel.appendChild(el("label", { for: "movein", text: "Your play" }));
    panel.appendChild(input);
    panel.appendChild(hint);
    panel.appendChild(warnBox);
    panel.appendChild(el("div", { class: "btn-row" }, [submit, submitAnyway]));
    app.appendChild(panel);
    input.focus();
  } else {
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

function recordAndAdvance(idx, res){
  var pos = DATA[idx];
  STATE.answers[pos.position_id] = {
    chosen: res.chosen, is_best: res.is_best, equity_loss: res.equity_loss,
    matched: res.matched, parse_failed: !!res.parse_failed
  };
  saveState(STATE);
  routeToNext();
}

function routeToNext(){
  var i = firstUnanswered();
  if(i >= TOTAL) screenResults();
  else screenPosition(i);
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

/* ---- boot ---- */
if(STATE.name && firstUnanswered() < TOTAL){ screenPosition(firstUnanswered()); }
else if(STATE.name && firstUnanswered() >= TOTAL && Object.keys(STATE.answers).length === TOTAL){ screenResults(); }
else { screenIntro(); }
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
