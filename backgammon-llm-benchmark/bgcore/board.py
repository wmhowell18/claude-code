"""Canonical board model (PLAN.md §1.1).

This module pins down the exact semantics of ``board_json`` (see
``schema/position.schema.json``) and provides validation, pip counting,
perspective flip, canonical normalization (for dedup), and JSON round-tripping.

The 26-item ``points`` array — semantics
========================================

``points`` is a list of 26 integers, indices ``0..25``. The board is always
recorded **from the mover's perspective** (the player *to move*):

* ``points[p] > 0`` — ``p`` holds that many **mover** ("x") checkers.
* ``points[p] < 0`` — ``p`` holds ``abs(points[p])`` **opponent** ("o") checkers.
* ``points[0]`` and ``points[25]`` are reserved sentinels and are always ``0``;
  checkers on the bar / borne off live in the separate ``bar`` and ``off`` fields.

Direction of travel (mover-relative): **the mover advances from low indices to
high indices** and bears checkers off *beyond* index 24.

* Mover home board = indices ``19..24`` (index 24 is the mover's ace point,
  1 pip from off; index 19 is 6 pips from off).
* Mover bar re-entry lands on index ``d`` for a die ``d`` (indices ``1..6``).
* Opponent is the mirror image: it advances high→low, bears off beyond index 1,
  its home board is indices ``1..6``, and its bar re-entry lands on index
  ``25 - d``.

Pip counts::

    pip_x = 25 * bar.x + sum(points[p] * (25 - p) for points[p] > 0)
    pip_o = 25 * bar.o + sum(-points[p] * p       for points[p] < 0)

Labels ``x`` / ``o``
--------------------
``x`` denotes the **mover** and ``o`` the **opponent** everywhere they appear as a
count (``bar.x``, ``off.x``, ``score.x``, ``pip.x``, ``cube.owner``). Because the
layout is already mover-relative, the ``turn`` field carries a different, narrow
job: it records the mover's **physical seat** as tracked by external IDs — ``"x"``
when the mover is the ID's "player 1 / bottom" seat, ``"o"`` when it is the
"player 2 / top" seat. This is what lets XGID / GNU BG IDs round-trip losslessly
(see :mod:`ids.xgid`). Perspective :func:`flip` toggles it. A freshly constructed
mover-relative record uses ``turn = "x"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any


POINTS_LEN = 26
HOME_START = 19  # mover home board = indices 19..24
HOME_END = 24
CHECKERS_PER_SIDE = 15


class BoardError(ValueError):
    """Raised when a board fails structural / rules validation."""


@dataclass
class Board:
    """A backgammon position from the mover's perspective (see module docstring).

    Attributes mirror ``board_json`` exactly. ``points`` is the 26-item array;
    ``bar``/``off`` map side label -> count; ``dice`` is the roll (empty for cube
    decisions); ``cube`` is ``{"value": int, "owner": "center"|"x"|"o"}``;
    ``score`` is ``{"x", "o", "length", "crawford"}``; ``pip`` is optional/derived;
    ``turn`` is the mover's physical seat; ``decision_type`` is ``"checker"`` or
    ``"cube"``.
    """

    points: list[int] = field(default_factory=lambda: [0] * POINTS_LEN)
    bar: dict[str, int] = field(default_factory=lambda: {"x": 0, "o": 0})
    off: dict[str, int] = field(default_factory=lambda: {"x": 0, "o": 0})
    turn: str = "x"
    dice: list[int] = field(default_factory=list)
    cube: dict[str, Any] = field(default_factory=lambda: {"value": 1, "owner": "center"})
    score: dict[str, Any] = field(
        default_factory=lambda: {"x": 0, "o": 0, "length": 0, "crawford": False}
    )
    pip: dict[str, int] | None = None
    decision_type: str = "checker"

    # -- construction -----------------------------------------------------

    @classmethod
    def starting_position(cls, dice: list[int] | None = None) -> "Board":
        """The standard opening layout (money game), mover perspective."""
        pts = [0] * POINTS_LEN
        # mover (x, positive)
        pts[1] = 2
        pts[12] = 5
        pts[17] = 3
        pts[19] = 5
        # opponent (o, negative) — mirror
        pts[24] = -2
        pts[13] = -5
        pts[8] = -3
        pts[6] = -5
        b = cls(points=pts, dice=list(dice or []))
        b.decision_type = "checker" if b.dice else "cube"
        b.refresh_pip()
        return b

    # -- JSON -------------------------------------------------------------

    @classmethod
    def from_json(cls, data: dict[str, Any] | str) -> "Board":
        """Build a :class:`Board` from a ``board_json`` dict (or its JSON text)."""
        if isinstance(data, str):
            data = json.loads(data)
        points = list(data["points"])
        if len(points) != POINTS_LEN:
            raise BoardError(f"points must have {POINTS_LEN} items, got {len(points)}")
        bar = {"x": int(data.get("bar", {}).get("x", 0)), "o": int(data.get("bar", {}).get("o", 0))}
        off = {"x": int(data.get("off", {}).get("x", 0)), "o": int(data.get("off", {}).get("o", 0))}
        cube_in = data.get("cube", {}) or {}
        cube = {"value": int(cube_in.get("value", 1)), "owner": cube_in.get("owner", "center")}
        score_in = data.get("score", {}) or {}
        score = {
            "x": int(score_in.get("x", 0)),
            "o": int(score_in.get("o", 0)),
            "length": int(score_in.get("length", 0)),
            "crawford": bool(score_in.get("crawford", False)),
        }
        pip = None
        if data.get("pip") is not None:
            pip = {"x": int(data["pip"]["x"]), "o": int(data["pip"]["o"])}
        board = cls(
            points=[int(v) for v in points],
            bar=bar,
            off=off,
            turn=data.get("turn", "x"),
            dice=[int(d) for d in data.get("dice", [])],
            cube=cube,
            score=score,
            pip=pip,
            decision_type=data.get("decision_type", "checker" if data.get("dice") else "cube"),
        )
        return board

    def to_json(self, *, include_pip: bool = True) -> dict[str, Any]:
        """Serialize to a ``board_json`` dict matching the schema exactly."""
        out: dict[str, Any] = {
            "points": list(self.points),
            "bar": {"x": int(self.bar["x"]), "o": int(self.bar["o"])},
            "off": {"x": int(self.off["x"]), "o": int(self.off["o"])},
            "turn": self.turn,
            "dice": list(self.dice),
            "cube": {"value": int(self.cube["value"]), "owner": self.cube["owner"]},
            "score": {
                "x": int(self.score["x"]),
                "o": int(self.score["o"]),
                "length": int(self.score.get("length", 0)),
                "crawford": bool(self.score.get("crawford", False)),
            },
            "decision_type": self.decision_type,
        }
        if include_pip:
            px, po = pip_counts(self)
            out["pip"] = {"x": px, "o": po}
        return out

    def to_json_str(self, **kw: Any) -> str:
        return json.dumps(self.to_json(**kw), sort_keys=True, separators=(",", ":"))

    # -- convenience ------------------------------------------------------

    def copy(self) -> "Board":
        return replace(
            self,
            points=list(self.points),
            bar=dict(self.bar),
            off=dict(self.off),
            dice=list(self.dice),
            cube=dict(self.cube),
            score=dict(self.score),
            pip=dict(self.pip) if self.pip else None,
        )

    def refresh_pip(self) -> "Board":
        px, po = pip_counts(self)
        self.pip = {"x": px, "o": po}
        return self

    def all_home(self) -> bool:
        """True if every mover checker is in the home board (bearoff eligible)."""
        if self.bar["x"] > 0:
            return False
        for p in range(1, HOME_START):
            if self.points[p] > 0:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return self.to_json() == other.to_json()


# -- module-level helpers -------------------------------------------------


def pip_counts(board: Board) -> tuple[int, int]:
    """Return ``(pip_x, pip_o)`` — mover and opponent pip counts."""
    px = 25 * int(board.bar["x"])
    po = 25 * int(board.bar["o"])
    for p in range(1, 25):
        v = board.points[p]
        if v > 0:
            px += v * (25 - p)
        elif v < 0:
            po += (-v) * p
    return px, po


def validate(board: Board, *, strict: bool = True) -> list[str]:
    """Validate structure and (optionally strict) checker conservation.

    Returns a list of human-readable error strings. With ``strict=True`` a
    non-empty list raises :class:`BoardError`; with ``strict=False`` the list is
    returned so callers can inspect partial boards.
    """
    errs: list[str] = []
    pts = board.points
    if len(pts) != POINTS_LEN:
        errs.append(f"points length {len(pts)} != {POINTS_LEN}")
    else:
        if pts[0] != 0 or pts[25] != 0:
            errs.append("points[0] and points[25] must be 0 (reserved sentinels)")
        for p in range(1, 25):
            if abs(pts[p]) > CHECKERS_PER_SIDE:
                errs.append(f"points[{p}]={pts[p]} exceeds {CHECKERS_PER_SIDE}")
    for side in ("x", "o"):
        if board.bar[side] < 0 or board.off[side] < 0:
            errs.append(f"negative bar/off for {side}")
    x_on = sum(v for v in pts[1:25] if v > 0)
    o_on = -sum(v for v in pts[1:25] if v < 0)
    x_total = x_on + int(board.bar["x"]) + int(board.off["x"])
    o_total = o_on + int(board.bar["o"]) + int(board.off["o"])
    if x_total != CHECKERS_PER_SIDE:
        errs.append(f"mover (x) checker count {x_total} != {CHECKERS_PER_SIDE}")
    if o_total != CHECKERS_PER_SIDE:
        errs.append(f"opponent (o) checker count {o_total} != {CHECKERS_PER_SIDE}")
    if board.turn not in ("x", "o"):
        errs.append(f"turn {board.turn!r} not in x/o")
    if board.decision_type not in ("checker", "cube"):
        errs.append(f"decision_type {board.decision_type!r} invalid")
    for d in board.dice:
        if not (1 <= int(d) <= 6):
            errs.append(f"die {d} out of range 1..6")
    if len(board.dice) not in (0, 2):
        errs.append(f"dice must have 0 or 2 entries, got {len(board.dice)}")
    cv = int(board.cube["value"])
    if cv < 1 or (cv & (cv - 1)) != 0:
        errs.append(f"cube value {cv} must be a positive power of two")
    if board.cube["owner"] not in ("center", "x", "o"):
        errs.append(f"cube owner {board.cube['owner']!r} invalid")
    if board.score.get("length", 0) < 0:
        errs.append("negative match length")
    if strict and errs:
        raise BoardError("; ".join(errs))
    return errs


def flip(board: Board) -> Board:
    """Return the board viewed from the *opponent's* perspective.

    Mirrors the checker layout (``points[p] -> -points[25 - p]``), swaps the
    ``x``/``o`` roles for bar/off/score/pip/cube-owner, and toggles ``turn``.
    Dice are preserved verbatim (they belong to whoever the caller then treats as
    on roll). This is the color/perspective flip used for symmetry checks.
    """
    new_pts = [0] * POINTS_LEN
    for p in range(1, 25):
        new_pts[p] = -board.points[25 - p]
    owner = board.cube["owner"]
    new_owner = {"center": "center", "x": "o", "o": "x"}[owner]
    b = Board(
        points=new_pts,
        bar={"x": int(board.bar["o"]), "o": int(board.bar["x"])},
        off={"x": int(board.off["o"]), "o": int(board.off["x"])},
        turn="o" if board.turn == "x" else "x",
        dice=list(board.dice),
        cube={"value": int(board.cube["value"]), "owner": new_owner},
        score={
            "x": int(board.score["o"]),
            "o": int(board.score["x"]),
            "length": int(board.score.get("length", 0)),
            "crawford": bool(board.score.get("crawford", False)),
        },
        decision_type=board.decision_type,
    )
    b.refresh_pip()
    return b


def normalize(board: Board) -> Board:
    """Return a canonical form used for dedup (PLAN.md §2.3).

    Because the layout is already mover-relative, normalization only removes
    incidental representational freedom: dice are sorted descending, ``turn`` is
    forced to ``"x"`` (physical seat is irrelevant for dedup), the pip cache is
    recomputed, and for money games (``length == 0``) the score/crawford context
    is zeroed. The checker layout, bar/off, cube, and (for match play) the score
    are preserved because they change the *decision*.
    """
    b = board.copy()
    b.turn = "x"
    b.dice = sorted((int(d) for d in b.dice), reverse=True)
    length = int(b.score.get("length", 0))
    if length == 0:
        b.score = {"x": 0, "o": 0, "length": 0, "crawford": False}
    b.refresh_pip()
    return b


def canonical_key(board: Board) -> str:
    """A stable, hashable dedup key derived from :func:`normalize`."""
    return normalize(board).to_json_str(include_pip=False)
