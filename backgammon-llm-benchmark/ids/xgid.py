"""XGID <-> board_json (PLAN.md §1.1).

Parses/serializes the canonical XGID (board + cube + dice + score + turn),
converts to and from :class:`bgcore.board.Board`, and produces the normalized
(color/symmetry) dedup key. Round-trip XGID<->board must be lossless (Phase 0
acceptance, PLAN.md §7).

XGID field layout
=================
``XGID=<pos>:<cube>:<cubeowner>:<turn>:<dice>:<scoreB>:<scoreT>:<crawJac>:<matchlen>:<maxcube>``

* **pos** — 26 chars, indices 0..25. ``-`` = empty; ``A..O`` = 1..15 checkers of
  the *bottom* player ("player 1"); ``a..o`` = 1..15 of the *top* player
  ("player 2"). Index 0 = top player's bar, index 25 = bottom player's bar.
  Indices 1..24 are board points. The bottom player moves 24->1 (home 1..6) and
  the top player is its mirror. This is the opposite orientation to
  :mod:`bgcore.board` (whose mover advances low->high), so index mapping is
  ``board_index = 25 - xgid_index`` for the bottom-on-roll case.
* **cube** — cube value as a base-2 exponent (0 -> value 1, 1 -> 2, ...).
* **cubeowner** — ``0`` center, ``1`` bottom player owns, ``-1`` top player owns.
* **turn** — player on roll: ``1`` bottom, ``-1`` top, ``0`` = unset (treated as 1).
* **dice** — two digits (e.g. ``52``); ``00`` = no roll (cube decision).
* **scoreB / scoreT** — match scores of the bottom / top player.
* **crawJac** — for match play (matchlen>0): ``1`` iff Crawford game. For money
  (matchlen==0): the Jacoby flag (not represented in ``board_json``; see below).
* **matchlen** — match length; ``0`` = money game.
* **maxcube** — max cube as an exponent (XG default ``10`` = 1024). Not stored in
  ``board_json``.

Mapping to the mover-relative :class:`~bgcore.board.Board`: ``board.turn`` records
the mover's physical seat (``"x"`` = bottom, ``"o"`` = top) so the XGID turn/score/
owner fields reconstruct exactly. ``off`` is derived (XGID omits it). The only
fields not represented in ``board_json`` are **maxcube** and the **money Jacoby**
bit; both default (10 / 0) on encode, so ``board -> XGID -> board`` is lossless and
``XGID -> board -> XGID`` is lossless for the standard ``maxcube=10`` money/match
strings the benchmark emits.
"""

from __future__ import annotations

from bgcore.board import Board, canonical_key


def _enc(n: int) -> str:
    if n == 0:
        return "-"
    if n > 0:
        return chr(64 + n)  # 'A'..'O'
    return chr(96 - n)  # 'a'..'o'


def _dec(c: str) -> int:
    if c == "-":
        return 0
    o = ord(c)
    if 65 <= o <= 90:
        return o - 64
    if 97 <= o <= 122:
        return -(o - 96)
    raise ValueError(f"bad XGID position char {c!r}")


def board_to_xgid(board: Board, *, max_cube: int = 10, jacoby: int = 0) -> str:
    """Encode a :class:`~bgcore.board.Board` to a canonical XGID string."""
    sc = [0] * 26  # signed XGID counts, positive = bottom (X), negative = top (O)
    if board.turn == "x":  # mover is the bottom seat
        for i in range(1, 25):
            sc[i] = board.points[25 - i]
        sc[25] = board.bar["x"]
        sc[0] = -board.bar["o"]
        turn_field = 1
        score_b, score_t = board.score["x"], board.score["o"]
    else:  # mover is the top seat
        for i in range(1, 25):
            sc[i] = -board.points[i]
        sc[0] = -board.bar["x"]
        sc[25] = board.bar["o"]
        turn_field = -1
        score_b, score_t = board.score["o"], board.score["x"]

    pos = "".join(_enc(sc[i]) for i in range(26))

    value = int(board.cube["value"])
    cube_exp = value.bit_length() - 1
    owner = board.cube["owner"]
    if owner == "center":
        owner_field = 0
    else:
        owner_field = 1 if (owner == "x") == (board.turn == "x") else -1

    dice = board.dice
    dice_field = f"{dice[0]}{dice[1]}" if len(dice) == 2 else "00"

    length = int(board.score.get("length", 0))
    if length > 0:
        craw_jac = 1 if board.score.get("crawford") else 0
    else:
        craw_jac = int(jacoby)

    fields = [
        pos,
        str(cube_exp),
        str(owner_field),
        str(turn_field),
        dice_field,
        str(int(score_b)),
        str(int(score_t)),
        str(craw_jac),
        str(length),
        str(int(max_cube)),
    ]
    return "XGID=" + ":".join(fields)


def parse_xgid(s: str) -> dict:
    """Decode an XGID into a structured dict.

    Keys: ``board`` (:class:`~bgcore.board.Board`), ``max_cube`` (int) and
    ``jacoby`` (int, meaningful only for money games).
    """
    text = s.strip()
    if text.upper().startswith("XGID="):
        text = text[5:]
    parts = text.split(":")
    if len(parts) < 5:
        raise ValueError(f"XGID has too few fields: {s!r}")
    # pad missing trailing fields with defaults
    defaults = ["", "0", "0", "1", "00", "0", "0", "0", "0", "10"]
    for i in range(len(parts), 10):
        parts.append(defaults[i])

    pos = parts[0]
    if len(pos) != 26:
        raise ValueError(f"XGID position must be 26 chars, got {len(pos)}")
    sc = [_dec(c) for c in pos]

    cube_exp = int(parts[1])
    owner_field = int(parts[2])
    turn_field = int(parts[3])
    dice_field = parts[4]
    score_b = int(parts[5])
    score_t = int(parts[6])
    craw_jac = int(parts[7]) if parts[7] not in ("", "-") else 0
    length = int(parts[8]) if parts[8] not in ("", "-") else 0
    max_cube = int(parts[9]) if parts[9] not in ("", "-") else 10

    turn = "o" if turn_field == -1 else "x"

    points = [0] * 26
    bar = {"x": 0, "o": 0}
    if turn == "x":  # mover = bottom
        for i in range(1, 25):
            points[25 - i] = sc[i]
        bar["x"] = sc[25]
        bar["o"] = -sc[0]
        score_x, score_o = score_b, score_t
    else:  # mover = top
        for i in range(1, 25):
            points[i] = -sc[i]
        bar["x"] = -sc[0]
        bar["o"] = sc[25]
        score_x, score_o = score_t, score_b

    x_on = sum(v for v in points[1:25] if v > 0) + bar["x"]
    o_on = -sum(v for v in points[1:25] if v < 0) + bar["o"]
    off = {"x": 15 - x_on, "o": 15 - o_on}

    if owner_field == 0:
        owner = "center"
    else:
        # owner_field == +1 -> bottom seat owns; map back through turn
        owner = "x" if (owner_field == 1) == (turn == "x") else "o"

    if dice_field in ("00", "0", ""):
        dice: list[int] = []
    else:
        dice = [int(dice_field[0]), int(dice_field[1])]

    board = Board(
        points=points,
        bar=bar,
        off=off,
        turn=turn,
        dice=dice,
        cube={"value": 2 ** cube_exp, "owner": owner},
        score={
            "x": score_x,
            "o": score_o,
            "length": length,
            "crawford": bool(craw_jac) if length > 0 else False,
        },
        decision_type="checker" if dice else "cube",
    )
    board.refresh_pip()
    return {"board": board, "max_cube": max_cube, "jacoby": craw_jac if length == 0 else 0}


def xgid_to_board(s: str) -> Board:
    """Decode an XGID string directly to a :class:`~bgcore.board.Board`."""
    return parse_xgid(s)["board"]


def normalized_key(s: str) -> str:
    """Color/symmetry-normalized dedup key for an XGID (PLAN.md §2.3)."""
    return canonical_key(xgid_to_board(s))
