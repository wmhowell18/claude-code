"""Deterministic ASCII board render (PLAN.md §1.1).

Renders a :class:`bgcore.board.Board` to a fixed, versioned monospace template
(point stacks with counts, bar in the middle, header with pips/cube/dice/score).
Output is deterministic given the board; the template is versioned by
``ASCII_RENDER_VERSION``.

Orientation is the mover's own numbering (mover = ``X`` moves 24 -> 1, home board
points 1-6; opponent = ``O``). Points 13-24 are the top row, 12-1 the bottom row,
with the bar gutter between the two halves. Every board row is 43 columns wide.
"""

from __future__ import annotations

from bgcore.board import Board, pip_counts

ASCII_RENDER_VERSION = "ascii-1"

_MAX_STACK = 5
_TOP = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
_BOT = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def _point(board: Board, n: int) -> tuple[str, int]:
    v = board.points[25 - n]
    if v > 0:
        return "X", v
    if v < 0:
        return "O", -v
    return " ", 0


def _cell(char: str, count: int, level: int) -> str:
    """A 3-char cell for stack ``count`` at display ``level`` (0 = nearest border)."""
    if count > level:
        if count > _MAX_STACK and level == _MAX_STACK - 1:
            return f"{count:>2} "
        return f" {char} "
    return "   "


def _row(board: Board, pts: list[int], level: int, gutter: str) -> str:
    left = "".join(_cell(*_point(board, n), level) for n in pts[:6])
    right = "".join(_cell(*_point(board, n), level) for n in pts[6:])
    return f"|{left}|{gutter}|{right}|"


def _border(pts: list[int]) -> str:
    left = "".join(f"{n:>2}-" for n in pts[:6])
    right = "".join(f"{n:>2}-" for n in pts[6:])
    return f"+{left}+BAR+{right}+"


def render(board: Board) -> str:
    """Render ``board`` to a deterministic fixed-width ASCII string."""
    px, po = pip_counts(board)

    if board.score.get("length", 0):
        crawf = " (Crawford)" if board.score.get("crawford") else ""
        ctx = (
            f"Match to {board.score['length']}{crawf}"
            f"  X:{board.score['x']} O:{board.score['o']}"
        )
    else:
        ctx = "Money game"
    cube_owner = {"center": "center", "x": "X", "o": "O"}[board.cube["owner"]]

    lines: list[str] = [
        f"[{ASCII_RENDER_VERSION}] {ctx}",
        f"Cube: {board.cube['value']} ({cube_owner})",
        (
            f"On roll: X   Dice: {board.dice[0]}-{board.dice[1]}"
            if board.dice
            else "On roll: X   [cube decision]"
        ),
        f"{_border(_TOP)}  O off:{int(board.off['o']):>2} pip:{po:>3}",
    ]

    # top half stacks grow downward toward the bar
    for level in range(_MAX_STACK - 1, -1, -1):
        lines.append(_row(board, _TOP, level, "   "))
    lines.append(f"|{' ' * 18}|BAR|{' ' * 18}|  bar X:{int(board.bar['x'])} O:{int(board.bar['o'])}")
    # bottom half stacks grow upward from the bottom border
    for level in range(_MAX_STACK):
        lines.append(_row(board, _BOT, level, "   "))

    lines.append(f"{_border(_BOT)}  X off:{int(board.off['x']):>2} pip:{px:>3}  <== on roll")
    return "\n".join(lines)
