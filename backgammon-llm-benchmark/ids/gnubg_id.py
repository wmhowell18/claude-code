"""GNU BG Position ID + Match ID <-> board_json (PLAN.md §1.1).

Parses/serializes the secondary GNU BG ID pair for interop with GNU BG (our
interim rollout engine). Enables round-trip with :class:`bgcore.board.Board` and
cross-checks against the XGID.

Position ID
===========
14-char base64 of an 80-bit (10-byte) key. The key packs, LSB-first within each
byte, two 25-slot blocks (player on roll first, then opponent). Each slot ``j`` is
a *unary* run: ``count`` one-bits followed by a terminating zero-bit. Slots
0..23 are the player's points (slot ``j`` = own point ``j+1``; slot 0 = ace
point); slot 24 is the bar. Total one-bits 30, zero-bits 50 = 80 bits.

Board mapping (mover-relative frame of :mod:`bgcore.board`): a mover checker on
board index ``p`` is the mover's own point ``25 - p`` -> gnubg slot ``24 - p``; an
opponent checker on board index ``p`` is the opponent's own point ``p`` -> slot
``p - 1``. Bars go in slot 24. Off is derived (the ID omits it). The base64 step
uses the gnubg alphabet (standard ``A-Za-z0-9+/``) with padding stripped; encoding
the trailing 10th byte yields the last two characters.

The Position ID encodes **checkers only** — it is orientation-relative to the
player on roll and carries no cube/dice/score. Those live in the Match ID.

Match ID
========
12-char base64 of a 9-byte key packing (LSB-first, in order): cube value (log2, 4
bits), cube owner (2 bits: 0=on-roll owns, 1=opponent owns, 3=centered), player
on roll (1), Crawford (1), game state (3), turn (1), double-offered (1),
resignation (2), die 0 (3), die 1 (3), match length (15), on-roll score (15),
opponent score (15) = 66 bits, zero-padded to 72.

Fields Match ID carries that ``board_json`` does **not** model — game state,
double-offered, and resignation — are written as defaults (playing / no / none)
and ignored on read. Conversely ``board_json`` has ``off`` counts and a cube
*value* beyond match reach that the IDs don't carry (off is recomputed; cube value
round-trips via its log2). The checker component round-trips losslessly.
"""

from __future__ import annotations

import base64

from bgcore.board import Board


_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
# gnubg uses the standard base64 alphabet, so Python's base64 matches it.


# -- bit helpers (LSB-first packing, matching gnubg) ----------------------


def _pack(bits: list[int], nbytes: int) -> bytes:
    buf = bytearray(nbytes)
    for i, b in enumerate(bits):
        if b:
            buf[i // 8] |= 1 << (i % 8)
    return bytes(buf)


def _unpack(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for k in range(8):
            bits.append((byte >> k) & 1)
    return bits


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    pad = (-len(s)) % 4
    return base64.b64decode(s + "=" * pad)


# -- Position ID ----------------------------------------------------------


def position_id(board: Board) -> str:
    """Encode the checker layout to a 14-char GNU BG Position ID."""
    mover = [0] * 25
    opp = [0] * 25
    for p in range(1, 25):
        v = board.points[p]
        if v > 0:
            mover[24 - p] += v
        elif v < 0:
            opp[p - 1] += -v
    mover[24] = int(board.bar["x"])
    opp[24] = int(board.bar["o"])

    bits: list[int] = []
    for counts in (mover, opp):
        for j in range(25):
            bits.extend([1] * counts[j])
            bits.append(0)
    return _b64(_pack(bits, 10))


def position_id_to_board(pid: str) -> Board:
    """Decode a Position ID to a :class:`~bgcore.board.Board` (checkers only).

    Cube/dice/score are left at defaults; combine with :func:`parse_match_id`
    (see :func:`ids_to_board`) for a full position.
    """
    bits = _unpack(_unb64(pid))
    blocks: list[list[int]] = []
    idx = 0
    for _player in range(2):
        counts = []
        for _j in range(25):
            c = 0
            while bits[idx] == 1:
                c += 1
                idx += 1
            idx += 1  # terminating zero
            counts.append(c)
        blocks.append(counts)
    mover, opp = blocks

    points = [0] * 26
    for j in range(24):
        if mover[j]:
            points[24 - j] += mover[j]
        if opp[j]:
            points[j + 1] -= opp[j]
    bar = {"x": mover[24], "o": opp[24]}
    x_on = sum(v for v in points[1:25] if v > 0) + bar["x"]
    o_on = -sum(v for v in points[1:25] if v < 0) + bar["o"]
    off = {"x": 15 - x_on, "o": 15 - o_on}
    board = Board(points=points, bar=bar, off=off, turn="x", dice=[], decision_type="cube")
    board.refresh_pip()
    return board


# -- Match ID -------------------------------------------------------------


def _add(bits: list[int], value: int, nbits: int) -> None:
    for k in range(nbits):
        bits.append((value >> k) & 1)


def _read(bits: list[int], pos: int, nbits: int) -> tuple[int, int]:
    v = 0
    for k in range(nbits):
        v |= bits[pos + k] << k
    return v, pos + nbits


def match_id(board: Board) -> str:
    """Encode cube/dice/score context to a 12-char GNU BG Match ID."""
    value = int(board.cube["value"])
    cube_log2 = value.bit_length() - 1
    owner = board.cube["owner"]
    owner_field = 3 if owner == "center" else (0 if owner == "x" else 1)
    crawford = 1 if board.score.get("crawford") else 0
    dice = board.dice
    d0, d1 = (dice[0], dice[1]) if len(dice) == 2 else (0, 0)
    length = int(board.score.get("length", 0))
    score0 = int(board.score["x"])
    score1 = int(board.score["o"])

    bits: list[int] = []
    _add(bits, cube_log2, 4)
    _add(bits, owner_field, 2)
    _add(bits, 0, 1)  # player on roll = 0 (mover)
    _add(bits, crawford, 1)
    _add(bits, 1, 3)  # game state: playing
    _add(bits, 0, 1)  # turn = mover
    _add(bits, 0, 1)  # double offered
    _add(bits, 0, 2)  # resignation
    _add(bits, d0, 3)
    _add(bits, d1, 3)
    _add(bits, length, 15)
    _add(bits, score0, 15)
    _add(bits, score1, 15)
    return _b64(_pack(bits, 9))


def parse_match_id(mid: str) -> dict:
    """Decode a Match ID into a dict of the fields ``board_json`` models."""
    bits = _unpack(_unb64(mid))
    pos = 0
    cube_log2, pos = _read(bits, pos, 4)
    owner_field, pos = _read(bits, pos, 2)
    _on_roll, pos = _read(bits, pos, 1)
    crawford, pos = _read(bits, pos, 1)
    _gamestate, pos = _read(bits, pos, 3)
    _turn, pos = _read(bits, pos, 1)
    _doubled, pos = _read(bits, pos, 1)
    _resigned, pos = _read(bits, pos, 2)
    d0, pos = _read(bits, pos, 3)
    d1, pos = _read(bits, pos, 3)
    length, pos = _read(bits, pos, 15)
    score0, pos = _read(bits, pos, 15)
    score1, pos = _read(bits, pos, 15)

    owner = "center" if owner_field == 3 else ("x" if owner_field == 0 else "o")
    dice = [d0, d1] if (d0 and d1) else []
    return {
        "cube": {"value": 2 ** cube_log2, "owner": owner},
        "dice": dice,
        "score": {
            "x": score0,
            "o": score1,
            "length": length,
            "crawford": bool(crawford) if length > 0 else False,
        },
    }


# -- combined -------------------------------------------------------------


def board_to_ids(board: Board) -> dict:
    """Return ``{"position_id", "match_id"}`` for a board."""
    return {"position_id": position_id(board), "match_id": match_id(board)}


def ids_to_board(position: str, match: str) -> Board:
    """Reconstruct a full :class:`~bgcore.board.Board` from both GNU BG IDs."""
    board = position_id_to_board(position)
    meta = parse_match_id(match)
    board.cube = meta["cube"]
    board.dice = meta["dice"]
    board.score = meta["score"]
    board.decision_type = "checker" if board.dice else "cube"
    board.refresh_pip()
    return board
