"""GNU Backgammon integration (PLAN.md §1.3, §2.1, §7).

GNU BG is our **authoritative** ground-truth and self-play engine throughout the
benchmark (XG is only an optional future cross-check). It exposes a *command
language* driven either over a
tty (``gnubg -t``) by piping newline-separated commands, or via its embedded
Python (``gnubg -p script.py``). We use the command language.

This module is deliberately split into three testable layers:

* **command builders** — pure functions returning ``list[str]`` of gnubg
  commands (self-play generation, position rollouts, exports). No I/O.
* **output parsers** — pure functions turning gnubg's exported *match* text and
  *rollout* text into structured Python (per-decision boards; rollout records
  conforming to ``schema/rollout.schema.json``). No I/O.
* **one** invocation shim, :func:`run_gnubg`, that actually spawns the
  subprocess. Everything else is pure so the pipeline is unit-testable with
  canned gnubg output and :func:`run_gnubg` is *never called by tests* (gnubg is
  not installed in CI).

The self-play parser and the rollout ``move`` records lean on
:mod:`bgcore.board` / :mod:`bgcore.moves` / :mod:`bgcore.notation` so that the
positions we extract are the exact mover-relative boards the rest of the
pipeline consumes.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from bgcore.board import Board, flip, validate

# ---------------------------------------------------------------------------
# Rollout settings (mirror of PLAN.md §1.3 table). The dataclass is the single
# source of truth for both the command builder and the ``rollout_meta`` blob.
# ---------------------------------------------------------------------------

SETTINGS_VERSION = "gnubg-rollout-1"


@dataclass(frozen=True)
class RolloutSettings:
    """Standardized rollout settings, defaulting to the PLAN §1.3 targets.

    ``trials`` 1296 is the money-safe standard; the deep tier uses 5184.
    ``chequer_ply``/``cube_ply`` are gnubg evaluation plies. ``variance_reduction``
    and ``quasi_random`` (antithetic / quasi-random dice) default ON.
    """

    trials: int = 1296
    chequer_ply: int = 2
    cube_ply: int = 2
    variance_reduction: bool = True
    quasi_random: bool = True
    truncation: int = 0  # 0 = no truncation (short games); >0 truncates races
    seed: int = 0
    met: str = "gnubg-default"
    settings_version: str = SETTINGS_VERSION

    def to_meta(self) -> dict[str, Any]:
        """Serialize to the ``rollout_meta`` blob (schema/rollout.schema.json)."""
        return {
            "trials": int(self.trials),
            "chequer_ply": f"{int(self.chequer_ply)}-ply",
            "cube_ply": f"{int(self.cube_ply)}-ply",
            "truncation": "none" if not self.truncation else f"{int(self.truncation)}-ply",
            "variance_reduction": bool(self.variance_reduction),
            "antithetic_dice": bool(self.quasi_random),
            "met": self.met,
            "seed": int(self.seed),
            "settings_version": self.settings_version,
            "engine": "gnubg",
        }


# ---------------------------------------------------------------------------
# Command builders (pure)
# ---------------------------------------------------------------------------

_ONOFF = {True: "on", False: "off"}


def engine_commands(plies: int = 2) -> list[str]:
    """Commands to make **both** players gnubg bots at ``plies``-ply.

    gnubg has no ``set player both`` verb, so each seat (0 and 1) is configured
    explicitly for chequer play and cube decisions.
    """
    cmds: list[str] = []
    for seat in (0, 1):
        cmds.append(f"set player {seat} gnubg")
        cmds.append(f"set player {seat} chequer evaluation plies {int(plies)}")
        cmds.append(f"set player {seat} cube evaluation plies {int(plies)}")
    return cmds


def selfplay_commands(
    games: int,
    *,
    seed: int = 0,
    plies: int = 2,
    match_length: int | None = None,
    export_path: str = "data/selfplay/session.mat",
    export_format: str = "mat",
) -> list[str]:
    """Build the command script that self-plays ``games`` and exports them.

    ``match_length=None`` gives money sessions: ``games`` is the number of games
    played (each explicit ``new game`` runs to completion via automatic roll/move,
    with automatic *game* OFF so a money session — which never ends on its own —
    does not loop forever). A positive ``match_length`` plays ``games`` full
    matches of that length: automatic *game* is ON so the match chains its own
    games and stops when a player reaches ``match_length``. A deterministic RNG
    (``mersenne`` + ``seed``) makes runs reproducible. Games are exported with
    :func:`export_commands`.
    """
    if games < 1:
        raise ValueError("games must be >= 1")
    cmds: list[str] = []
    cmds += engine_commands(plies)
    cmds.append("set rng mersenne")  # a real RNG so 'set seed' is honoured
    cmds.append(f"set seed {int(seed)}")
    # let the bots roll and move unattended
    cmds.append("set automatic roll on")
    cmds.append("set automatic move on")
    cmds.append("set automatic doubles 0")
    if match_length is None:
        # money: bound the session ourselves — auto game OFF, one game at a time.
        cmds.append("set automatic game off")
        cmds.append("new session")
        for _ in range(games):
            cmds.append("new game")
    else:
        if match_length < 1:
            raise ValueError("match_length must be >= 1")
        # match: auto game ON is safe (bounded by match length). A gnubg session
        # holds one match at a time and ``export match`` only writes the current
        # one, so a single call plays exactly one match (loop the call in Python,
        # with distinct seeds, to gather several matches across varied scores).
        cmds.append("set automatic game on")
        cmds.append(f"new match {int(match_length)}")
        cmds.append("new game")
    cmds += export_commands(export_path, fmt=export_format, scope="match")
    return cmds


def export_commands(path: str, *, fmt: str = "mat", scope: str = "match") -> list[str]:
    """Commands to export the current ``game``/``match`` to ``path``.

    ``fmt`` is a gnubg export format keyword (``mat`` = Jellyfish match, ``sgf``,
    ``text``). ``scope`` is ``match`` or ``game``.
    """
    if scope not in ("match", "game"):
        raise ValueError("scope must be 'match' or 'game'")
    if fmt == "sgf":
        return [f"save match {shlex.quote(path)}"]
    return [f"export {scope} {fmt} {shlex.quote(path)}"]


def set_position_commands(xgid: str | None = None, position_id: str | None = None) -> list[str]:
    """Commands to load a position by XGID or by GNU BG Position ID."""
    if xgid:
        return [f"set xgid {xgid}"]
    if position_id:
        return [f"set board {position_id}"]
    raise ValueError("provide xgid or position_id")


def rollout_commands(
    *,
    xgid: str | None = None,
    position_id: str | None = None,
    settings: RolloutSettings | None = None,
    cube: bool = False,
) -> list[str]:
    """Build the command script that rolls out one loaded position.

    Mirrors the PLAN §1.3 settings table via :class:`RolloutSettings`. When
    ``cube`` is true a cube-decision rollout is requested (``rollout =cube``);
    otherwise a chequer-play rollout over the legal moves. Deterministic given
    ``settings.seed``.
    """
    s = settings or RolloutSettings()
    cmds: list[str] = []
    cmds += set_position_commands(xgid=xgid, position_id=position_id)
    cmds.append(f"set rollout trials {int(s.trials)}")
    cmds.append(f"set rollout chequerplay plies {int(s.chequer_ply)}")
    cmds.append(f"set rollout cubedecision plies {int(s.cube_ply)}")
    cmds.append(f"set rollout varredn {_ONOFF[bool(s.variance_reduction)]}")
    cmds.append(f"set rollout quasirandom {_ONOFF[bool(s.quasi_random)]}")
    if s.truncation and s.truncation > 0:
        cmds.append("set rollout truncation enable on")
        cmds.append(f"set rollout truncation plies {int(s.truncation)}")
    else:
        cmds.append("set rollout truncation enable off")
    cmds.append(f"set rollout seed {int(s.seed)}")
    cmds.append("rollout =cube" if cube else "rollout")
    return cmds


def script(commands: list[str]) -> str:
    """Join a command list into the newline-terminated text gnubg reads on stdin."""
    return "".join(c.rstrip("\n") + "\n" for c in commands)


# ---------------------------------------------------------------------------
# Invocation shim — the ONLY function that touches a subprocess. Tests never
# call this (gnubg is not installed). Everything above/below is pure.
# ---------------------------------------------------------------------------


def _export_path_of(commands: list[str]) -> str | None:
    """Return the file path an ``export``/``save match`` command writes to, if any.

    gnubg cannot write a .mat export to stdout, so self-play commands export to a
    file; :func:`run_gnubg` reads that file back. Rollout commands have no export
    and fall through to stdout.
    """
    for c in reversed(commands):
        s = c.strip()
        if s.startswith("export ") or s.startswith("save match "):
            toks = shlex.split(s)
            if toks:
                return toks[-1]
    return None


def run_gnubg(
    commands: list[str],
    *,
    binary: str = "gnubg",
    timeout: float = 3600.0,
) -> str:
    """Run gnubg headless, feeding ``commands`` on stdin.

    Returns the exported *match text* when the command list contains an
    ``export``/``save match`` (self-play): gnubg writes the .mat to a file rather
    than stdout, so we read that file back. Otherwise (e.g. a rollout) the gnubg
    stdout is returned. Isolated here so the rest of the module stays pure and
    testable. Not exercised in CI (no gnubg binary). Callers in the pipeline
    inject this (or a canned stand-in) as a plain ``Callable[[list[str]], str]``.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415 - kept local so importing this module is I/O-free

    proc = subprocess.run(  # noqa: S603
        [binary, "-t", "-q"],
        input=script(commands),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    path = _export_path_of(commands)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return proc.stdout


# ---------------------------------------------------------------------------
# Self-play match parser (.mat text -> per-decision boards)
# ---------------------------------------------------------------------------


@dataclass
class RawEvent:
    """One half-move parsed from a .mat cell, in reading order."""

    kind: str  # "move" | "double" | "take" | "drop" | "win"
    seat: int  # 0 = left column / player 1, 1 = right column / player 2
    dice: list[int] = field(default_factory=list)
    move: str = ""
    cube_value: int = 0


@dataclass
class Decision:
    """A candidate decision extracted from self-play (board + light context)."""

    board: Board
    decision_type: str  # "checker" | "cube"
    game_index: int
    move_number: int
    seat: int
    play_mode: str  # "money" | "match"
    source: str = "gnubg-selfplay"


_MATCH_LEN_RE = re.compile(r"^\s*(\d+)\s+point match", re.IGNORECASE)
_GAME_RE = re.compile(r"^\s*Game\s+(\d+)", re.IGNORECASE)
# a numbered ply line: "  3) <left cell>   <right cell>"
_PLY_RE = re.compile(r"^(\s*\d+\)\s?)(.*)$")
_CELL_MOVE_RE = re.compile(r"^([1-6]{2}):\s*(.+?)\s*$")
# a cell always begins with the dice ("53:") or a cube/result keyword. The two
# move columns sit at fixed offsets (left ~col 5, right ~col 33), so we split on
# the right column rather than on a run of spaces — a wide 4-checker move can
# leave only a single space before the right cell, which a space-run split would
# wrongly merge into one cell.
_CELL_START_RE = re.compile(
    r"[1-6]{2}:|Doubles?|Takes?|Drops?|Passe?s?|Cannot|Wins?", re.IGNORECASE
)
_RIGHT_COL_MIN = 20
# a dance / forced no-move cell carries only the dice: "64:"
_CELL_DANCE_RE = re.compile(r"^([1-6]{2}):\s*$")
_CELL_DOUBLE_RE = re.compile(r"Doubles?\s*=>\s*(\d+)", re.IGNORECASE)
_REP_RE = re.compile(r"\((\d+)\)$")


def _gnubg_hops(move: str) -> list[tuple[Any, Any]]:
    """Expand a gnubg .mat move cell into its ordered single-die hops.

    gnubg writes each checker's play in its own point numbers (24 = back, 1 =
    ace), entering from the bar as point ``25`` and bearing off to point ``0``,
    with a checker's multi-die play as space-separated segments that share the
    intermediate point (``24/18 18/16``). Rather than reconcile that spelling
    with :mod:`bgcore.notation`'s merged slash-chains, we flatten it to the exact
    sequence of (from, to) single-die hops gnubg applied, in ``bgcore`` **board
    indices** (``"bar"`` / ``"off"`` at the ends). Applying these hops in order
    reproduces gnubg's resulting position faithfully (checkers are fungible, so
    the one-checker-chain vs. two-checker ambiguity is irrelevant to the layout).
    Repeated tokens (``13/9(3)``) are expanded to that many identical hop chains.
    """
    def _pt(sym: str) -> Any:
        core = sym.rstrip("*")
        if core in ("0",):
            return "off"
        if core in ("25",):
            return "bar"
        return 25 - int(core)  # notation point -> board index

    hops: list[tuple[Any, Any]] = []
    for raw in move.split():
        rep = 1
        mrep = _REP_RE.search(raw)
        if mrep:
            rep = int(mrep.group(1))
            raw = raw[: mrep.start()]
        segs = raw.split("/")
        for _ in range(rep):
            for a, b in zip(segs, segs[1:]):
                hops.append((_pt(a), _pt(b)))
    return hops


def _apply_gnubg_move(board: Board, move: str) -> Board:
    """Apply a gnubg .mat move to ``board`` (mover-relative) via its raw hops.

    Reproduces gnubg's own game record faithfully instead of round-tripping
    through legal-move matching: each single-die hop decrements its source
    (bar/point), sends any lone opponent checker on the destination to the
    opponent bar (a hit), and increments the destination (or bears off). The
    returned board keeps the mover's perspective with ``dice`` cleared; use
    :func:`bgcore.board.flip` to pass the turn.
    """
    nb = board.copy()
    for frm, to in _gnubg_hops(move):
        if frm == "bar":
            nb.bar["x"] -= 1
        else:
            nb.points[frm] -= 1
        if to == "off":
            nb.off["x"] += 1
        else:
            if nb.points[to] == -1:  # hit a lone opponent checker
                nb.points[to] = 0
                nb.bar["o"] += 1
            nb.points[to] += 1
    nb.dice = []
    nb.refresh_pip()
    return nb


def _split_cells(line: str, body_start: int) -> list[str]:
    """Split a full ply line into its (up to two) column cells by column offset.

    ``body_start`` is the index just past the ``"NN) "`` prefix. The left cell
    runs from there to the start of the right cell; the right cell is the first
    cell-start token (dice or a cube/result keyword) at or beyond
    ``_RIGHT_COL_MIN``. Cells are returned stripped; empty cells are dropped so
    cube/dance events (which occupy a single column) parse cleanly.
    """
    right_col: int | None = None
    for mm in _CELL_START_RE.finditer(line):
        if mm.start() >= max(_RIGHT_COL_MIN, body_start):
            right_col = mm.start()
            break
    if right_col is None:
        cells = [line[body_start:]]
    else:
        cells = [line[body_start:right_col], line[right_col:]]
    return [c.strip() for c in cells if c.strip()]


def _parse_cell(cell: str, seat: int) -> RawEvent | None:
    m = _CELL_DOUBLE_RE.search(cell)
    if m:
        return RawEvent(kind="double", seat=seat, cube_value=int(m.group(1)))
    low = cell.lower()
    if low.startswith("take"):
        return RawEvent(kind="take", seat=seat)
    if low.startswith("drop") or low.startswith("pass"):
        return RawEvent(kind="drop", seat=seat)
    if "wins" in low:
        return RawEvent(kind="win", seat=seat)
    m = _CELL_MOVE_RE.match(cell)
    if m:
        dice = [int(m.group(1)[0]), int(m.group(1)[1])]
        move = m.group(2).strip()
        return RawEvent(kind="move", seat=seat, dice=dice, move=move)
    m = _CELL_DANCE_RE.match(cell)
    if m:  # rolled but no legal play (a dance): turn passes, no decision
        dice = [int(m.group(1)[0]), int(m.group(1)[1])]
        return RawEvent(kind="nomove", seat=seat, dice=dice)
    return None


def parse_match_events(text: str) -> list[tuple[int, list[RawEvent]]]:
    """Parse .mat text into ``(game_index, [RawEvent, ...])`` in reading order.

    Pure tokenizer: no board reconstruction. ``match_length`` (see
    :func:`match_length_of`) tells callers whether it was money or match play.
    """
    games: list[tuple[int, list[RawEvent]]] = []
    cur_game: int | None = None
    events: list[RawEvent] = []
    for line in text.splitlines():
        gm = _GAME_RE.match(line)
        if gm:
            if cur_game is not None:
                games.append((cur_game, events))
            cur_game = int(gm.group(1))
            events = []
            continue
        pm = _PLY_RE.match(line)
        if pm and cur_game is not None:
            body_start = len(pm.group(1))
            for idx, cell in enumerate(_split_cells(line, body_start)):
                ev = _parse_cell(cell, seat=idx)
                if ev is not None:
                    events.append(ev)
    if cur_game is not None:
        games.append((cur_game, events))
    return games


def match_length_of(text: str) -> int:
    """Return the match length declared in .mat header (0 == money session)."""
    for line in text.splitlines():
        m = _MATCH_LEN_RE.match(line)
        if m:
            return int(m.group(1))
    return 0


def parse_match(text: str, *, validate_boards: bool = True) -> list[Decision]:
    """Replay .mat self-play text into a stream of candidate :class:`Decision`.

    Reconstruction uses :func:`bgcore.moves.apply_move` and
    :func:`bgcore.board.flip`: the board is always kept mover-relative for the
    player about to act. Each checker ply yields a ``checker`` decision (board
    with dice set) *before* the move is applied; each double yields a ``cube``
    decision (board with the cube still centred) before the cube turns.

    Raises :class:`bgcore.moves.IllegalMove` if a canned move is not legal in the
    reconstructed position (a good signal the .mat or the engine disagree).
    """
    match_len = match_length_of(text)
    play_mode = "match" if match_len > 0 else "money"
    decisions: list[Decision] = []
    for game_index, events in parse_match_events(text):
        board = Board.starting_position()
        board.score = {"x": 0, "o": 0, "length": match_len, "crawford": False}
        board.decision_type = "cube"
        move_number = 0
        i = 0
        while i < len(events):
            ev = events[i]
            if ev.kind == "move":
                move_number += 1
                board.dice = list(ev.dice)
                board.decision_type = "checker"
                board.refresh_pip()
                cand = board.copy()
                if validate_boards:
                    validate(cand)
                decisions.append(
                    Decision(
                        board=cand,
                        decision_type="checker",
                        game_index=game_index,
                        move_number=move_number,
                        seat=ev.seat,
                        play_mode=play_mode,
                    )
                )
                board = _apply_gnubg_move(board, ev.move)
                board = flip(board)
                board.decision_type = "cube"
                i += 1
            elif ev.kind == "nomove":
                # a dance (forced no play): the turn passes with no decision to
                # record, but we must still flip so alternation stays in sync.
                board = flip(board)
                board.decision_type = "cube"
                i += 1
            elif ev.kind == "double":
                move_number += 1
                board.dice = []
                board.decision_type = "cube"
                board.refresh_pip()
                cand = board.copy()
                if validate_boards:
                    validate(cand)
                decisions.append(
                    Decision(
                        board=cand,
                        decision_type="cube",
                        game_index=game_index,
                        move_number=move_number,
                        seat=ev.seat,
                        play_mode=play_mode,
                    )
                )
                # resolve the response without flipping (doubler acts again on take)
                resp = events[i + 1] if i + 1 < len(events) else None
                if resp is not None and resp.kind == "take":
                    board.cube = {"value": int(ev.cube_value), "owner": "o"}
                    i += 2
                else:  # drop / end of game
                    i += 2 if resp is not None else 1
                    break
            else:  # take without a preceding double, win, or stray token
                i += 1
    return decisions


# ---------------------------------------------------------------------------
# Rollout output parsers (gnubg rollout text -> schema/rollout.schema.json)
# ---------------------------------------------------------------------------

# checker rollout move line, e.g.
#   "  1. Rollout       24/18 13/11              Eq.:  +0.1523 ( -0.0122)"
_RO_MOVE_RE = re.compile(
    r"^\s*(?P<rank>\d+)\.\s+(?:Rollout|Cubeful|Cubeless|\d+-ply)\s+"
    r"(?P<move>.+?)\s+Eq\.?\s*:\s*(?P<eq>[+-]?\d+\.\d+)"
    r"(?:\s*\(\s*(?P<diff>[+-]?\d+\.\d+)\s*\))?\s*$"
)
# a following stat line carrying the rollout std error in brackets: "[ 0.0042 ]"
_RO_STDERR_RE = re.compile(r"\[\s*(?P<se>\d+\.\d+)\s*\]")

_TRIALS_RE = re.compile(r"(\d+)\s+games", re.IGNORECASE)


def _to_mp(equity_delta: float) -> float:
    """Millipoints from an equity delta (1.0 equity == 1000 millipoints)."""
    return round(equity_delta * 1000.0, 1)


def parse_checker_rollout(
    text: str,
    *,
    settings: RolloutSettings | None = None,
    position_id: str = "",
    xgid: str = "",
    phase: str = "",
) -> dict[str, Any]:
    """Parse gnubg checker-play rollout text into a rollout record.

    Returns a dict conforming to ``schema/rollout.schema.json`` with
    ``decision_type == "checker"``: every parsed move gets ``equity``,
    ``error_mp`` (millipoints vs. best), ``std_err`` and ``rank``. Also fills
    ``best_move`` / ``best_equity`` / ``second_best_move`` / ``equity_gap``.
    """
    s = settings or RolloutSettings()
    lines = text.splitlines()
    moves: list[dict[str, Any]] = []
    for li, line in enumerate(lines):
        m = _RO_MOVE_RE.match(line)
        if not m:
            continue
        eq = float(m.group("eq"))
        std_err = None
        # std error usually sits on the next 1-2 lines
        for look in lines[li + 1 : li + 3]:
            se = _RO_STDERR_RE.search(look)
            if se:
                std_err = float(se.group("se"))
                break
        moves.append(
            {
                "rank": int(m.group("rank")),
                "move": m.group("move").strip(),
                "equity": eq,
                "std_err": std_err,
            }
        )
    if not moves:
        raise ValueError("no rollout move lines found in gnubg output")

    moves.sort(key=lambda d: d["rank"])
    best_equity = max(mv["equity"] for mv in moves)
    for mv in moves:
        mv["error_mp"] = _to_mp(best_equity - mv["equity"])
        if mv["std_err"] is None:
            mv.pop("std_err")

    ordered = sorted(moves, key=lambda d: (-d["equity"], d["rank"]))
    best_move = ordered[0]["move"]
    second_best = ordered[1]["move"] if len(ordered) > 1 else None
    gap = round(best_equity - ordered[1]["equity"], 6) if len(ordered) > 1 else 0.0

    meta = s.to_meta()
    tm = _TRIALS_RE.search(text)
    if tm:
        meta["trials"] = int(tm.group(1))

    record: dict[str, Any] = {
        "position_id": position_id,
        "engine": "gnubg",
        "decision_type": "checker",
        "rollout_meta": meta,
        "checker": {"moves": moves},
        "best_move": best_move,
        "best_equity": best_equity,
        "equity_gap": gap,
    }
    if xgid:
        record["xgid"] = xgid
    if second_best is not None:
        record["second_best_move"] = second_best
    if phase:
        record["phase"] = phase
    return record


_CUBE_ND_RE = re.compile(r"No\s+double\b.*?([+-]?\d+\.\d+)", re.IGNORECASE)
_CUBE_DT_RE = re.compile(r"Double,?\s*take\b.*?([+-]?\d+\.\d+)", re.IGNORECASE)
_CUBE_DP_RE = re.compile(r"Double,?\s*pass\b.*?([+-]?\d+\.\d+)", re.IGNORECASE)
_CUBE_BEST_RE = re.compile(r"Best\s+cube\s+action\s*:\s*(.+?)\s*$", re.IGNORECASE)

_CUBE_ACTIONS = {
    "no double": "No double",
    "double, take": "Double, Take",
    "double take": "Double, Take",
    "double, pass": "Double, Pass",
    "double pass": "Double, Pass",
    "too good": "Too good",
    "too good, pass": "Too good",
}


def _canon_cube_action(raw: str) -> str:
    low = re.sub(r"\s+", " ", raw.strip().lower()).rstrip(".")
    for key, val in _CUBE_ACTIONS.items():
        if low.startswith(key):
            return val
    return raw.strip()


def parse_cube_rollout(
    text: str,
    *,
    settings: RolloutSettings | None = None,
    position_id: str = "",
    xgid: str = "",
    phase: str = "",
) -> dict[str, Any]:
    """Parse gnubg cube-decision rollout text into a rollout record.

    Returns a dict conforming to ``schema/rollout.schema.json`` with
    ``decision_type == "cube"``: the three canonical equities (no double /
    double-take / double-pass), the derived ``best_action`` and an ``error_mp``
    map giving the millipoint cost of each wrong action.
    """
    s = settings or RolloutSettings()
    nd = _CUBE_ND_RE.search(text)
    dt = _CUBE_DT_RE.search(text)
    dp = _CUBE_DP_RE.search(text)
    if not (nd and dt and dp):
        raise ValueError("cube rollout output missing one of no-double/take/pass equities")
    nd_eq = float(nd.group(1))
    dt_eq = float(dt.group(1))
    dp_eq = float(dp.group(1))

    # Doubler's realizable equity if doubling: opponent picks the action worse
    # for the doubler (min of take/pass). Best overall = max(no-double, double).
    double_eq = min(dt_eq, dp_eq)
    best_eq = max(nd_eq, double_eq)

    bm = _CUBE_BEST_RE.search(text)
    if bm:
        best_action = _canon_cube_action(bm.group(1))
    else:
        best_action = "No double" if nd_eq >= double_eq else (
            "Double, Pass" if dp_eq < dt_eq else "Double, Take"
        )

    # Millipoint error of each named action, all >= 0. "No double" is the
    # doubler's error (cost of not doubling); "Double, Take"/"Double, Pass" are
    # the receiver's response errors (cost of the wrong take/pass), measured
    # against the response that minimizes the doubler's equity.
    best_response = min(dt_eq, dp_eq)
    error_mp = {
        "No double": _to_mp(max(0.0, best_eq - nd_eq)),
        "Double, Take": _to_mp(max(0.0, dt_eq - best_response)),
        "Double, Pass": _to_mp(max(0.0, dp_eq - best_response)),
    }

    meta = s.to_meta()
    tm = _TRIALS_RE.search(text)
    if tm:
        meta["trials"] = int(tm.group(1))

    record: dict[str, Any] = {
        "position_id": position_id,
        "engine": "gnubg",
        "decision_type": "cube",
        "rollout_meta": meta,
        "cube": {
            "no_double_equity": nd_eq,
            "double_take_equity": dt_eq,
            "double_pass_equity": dp_eq,
            "best_action": best_action,
            "error_mp": error_mp,
        },
        "best_equity": best_eq,
        "phase": phase or "cube-action",
    }
    if xgid:
        record["xgid"] = xgid
    return record


def parse_rollout(text: str, *, cube: bool = False, **kw: Any) -> dict[str, Any]:
    """Dispatch to :func:`parse_cube_rollout` or :func:`parse_checker_rollout`."""
    return parse_cube_rollout(text, **kw) if cube else parse_checker_rollout(text, **kw)
