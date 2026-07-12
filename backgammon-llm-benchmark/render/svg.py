"""Board -> SVG (PLAN.md §1.2).

Authors the SVG programmatically in one consistent house style, showing all
decision-relevant state (24 points + checkers with counts, bar/off trays, dice,
cube, score/match length, pip counts, point numbering, player-on-roll). SVG
source is committed so PNGs regenerate losslessly at any DPI.

Output is deterministic (no timestamps, no randomness) and versioned by
``IMAGE_RENDER_VERSION``. Coordinates use a fixed 1000x700 viewBox. Orientation
matches the ASCII render: mover = ``X`` (light checkers), opponent = ``O`` (dark),
points 13-24 on top and 12-1 on the bottom, bear-off tray on the right.
"""

from __future__ import annotations

from bgcore.board import Board, pip_counts

IMAGE_RENDER_VERSION = "svg-1"

# geometry
_W, _H = 1000, 700
_X0 = 30          # left edge of play field
_COLW = 60
_BARW = 40
_TRAYW = 60
_YTOP = 90        # top border of points
_YBOT = 610       # bottom border of points
_PTH = 210        # triangle height
_CR = 24          # checker radius

# theme
_C_BG = "#14110f"
_C_BOARD = "#3a2c22"
_C_PT_A = "#c9a06a"
_C_PT_B = "#7a5a3c"
_C_BAR = "#241a12"
_C_X = "#efe6d5"      # mover checkers (light)
_C_X_EDGE = "#b9ac93"
_C_O = "#141414"      # opponent checkers (dark)
_C_O_EDGE = "#000000"
_C_TEXT = "#f2ead9"
_C_DIE = "#f4efe6"
_C_PIP = "#1a1a1a"

_TOP = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
_BOT = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def _f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _bar_left() -> float:
    return _X0 + 6 * _COLW


def _col_x(idx: int) -> float:
    """Left x of visual column ``idx`` 0..11 (0..5 left half, 6..11 right half)."""
    if idx < 6:
        return _X0 + idx * _COLW
    return _bar_left() + _BARW + (idx - 6) * _COLW


def _point_column(n: int) -> tuple[int, bool]:
    """Return (visual column 0..11, is_top) for notation point ``n``."""
    if n in _TOP:
        return _TOP.index(n), True
    return _BOT.index(n), False


def _triangle(col: int, top: bool, light: bool) -> str:
    x = _col_x(col)
    cx = x + _COLW / 2
    fill = _C_PT_A if light else _C_PT_B
    if top:
        pts = f"{_f(x)},{_YTOP} {_f(x + _COLW)},{_YTOP} {_f(cx)},{_f(_YTOP + _PTH)}"
    else:
        pts = f"{_f(x)},{_YBOT} {_f(x + _COLW)},{_YBOT} {_f(cx)},{_f(_YBOT - _PTH)}"
    return f'<polygon points="{pts}" fill="{fill}" stroke="#241a12" stroke-width="1"/>'


def _checker(cx: float, cy: float, side: str) -> str:
    fill = _C_X if side == "X" else _C_O
    edge = _C_X_EDGE if side == "X" else _C_O_EDGE
    return (
        f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_CR}" fill="{fill}" '
        f'stroke="{edge}" stroke-width="2"/>'
    )


def _stack(col: int, top: bool, side: str, count: int) -> list[str]:
    out: list[str] = []
    cx = _col_x(col) + _COLW / 2
    shown = min(count, 5)
    for i in range(shown):
        if top:
            cy = _YTOP + _CR + i * (2 * _CR + 1)
        else:
            cy = _YBOT - _CR - i * (2 * _CR + 1)
        out.append(_checker(cx, cy, side))
    if count > 5:
        if top:
            cy = _YTOP + _CR + 4 * (2 * _CR + 1)
        else:
            cy = _YBOT - _CR - 4 * (2 * _CR + 1)
        label_fill = _C_PIP if side == "X" else _C_TEXT
        out.append(
            f'<text x="{_f(cx)}" y="{_f(cy + 5)}" text-anchor="middle" '
            f'font-family="monospace" font-size="20" fill="{label_fill}">{count}</text>'
        )
    return out


_PIP_LAYOUT = {
    1: [(1, 1)],
    2: [(0, 0), (2, 2)],
    3: [(0, 0), (1, 1), (2, 2)],
    4: [(0, 0), (2, 0), (0, 2), (2, 2)],
    5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
    6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)],
}


def _die(x: float, y: float, value: int, size: float = 54) -> list[str]:
    out = [
        f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(size)}" height="{_f(size)}" rx="8" '
        f'fill="{_C_DIE}" stroke="#000" stroke-width="1.5"/>'
    ]
    step = size / 3
    r = size / 12
    for gx, gy in _PIP_LAYOUT.get(value, []):
        cx = x + step * (gx + 0.5)
        cy = y + step * (gy + 0.5)
        out.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="{_C_PIP}"/>')
    return out


def _text(x: float, y: float, s: str, size: int = 20, anchor: str = "start", weight: str = "normal") -> str:
    esc = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{anchor}" font-family="monospace" '
        f'font-size="{size}" font-weight="{weight}" fill="{_C_TEXT}">{esc}</text>'
    )


def render(board: Board) -> str:
    """Render ``board`` to a self-contained, deterministic SVG string."""
    px, po = pip_counts(board)
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'width="{_W}" height="{_H}" data-render-version="{IMAGE_RENDER_VERSION}">'
    )
    parts.append(f'<rect x="0" y="0" width="{_W}" height="{_H}" fill="{_C_BG}"/>')
    # board frame + bar + tray
    play_right = _bar_left() + _BARW + 6 * _COLW
    parts.append(
        f'<rect x="{_X0}" y="{_YTOP}" width="{_f(play_right - _X0)}" '
        f'height="{_f(_YBOT - _YTOP)}" fill="{_C_BOARD}" stroke="#000" stroke-width="2"/>'
    )
    parts.append(
        f'<rect x="{_f(_bar_left())}" y="{_YTOP}" width="{_BARW}" '
        f'height="{_f(_YBOT - _YTOP)}" fill="{_C_BAR}"/>'
    )
    parts.append(
        f'<rect x="{_f(play_right + 10)}" y="{_YTOP}" width="{_TRAYW}" '
        f'height="{_f(_YBOT - _YTOP)}" fill="{_C_BAR}" stroke="#000" stroke-width="2"/>'
    )

    # points
    for n in _TOP + _BOT:
        col, top = _point_column(n)
        light = (col + (0 if top else 1)) % 2 == 0
        parts.append(_triangle(col, top, light))

    # point numbers
    for n in _TOP:
        col, _ = _point_column(n)
        parts.append(_text(_col_x(col) + _COLW / 2, _YTOP - 8, str(n), size=16, anchor="middle"))
    for n in _BOT:
        col, _ = _point_column(n)
        parts.append(_text(_col_x(col) + _COLW / 2, _YBOT + 22, str(n), size=16, anchor="middle"))

    # checkers
    for n in _TOP + _BOT:
        v = board.points[25 - n]
        if v == 0:
            continue
        col, top = _point_column(n)
        parts.extend(_stack(col, top, "X" if v > 0 else "O", abs(v)))

    # bar checkers
    barcx = _bar_left() + _BARW / 2
    for i in range(int(board.bar["o"])):
        parts.append(_checker(barcx, _YTOP + 40 + i * (2 * _CR + 1), "O"))
    for i in range(int(board.bar["x"])):
        parts.append(_checker(barcx, _YBOT - 40 - i * (2 * _CR + 1), "X"))

    # borne off (stubs in the tray)
    trayx = play_right + 10 + _TRAYW / 2
    for i in range(int(board.off["o"])):
        parts.append(_checker(trayx, _YTOP + 20 + i * 14, "O"))
    for i in range(int(board.off["x"])):
        parts.append(_checker(trayx, _YBOT - 20 - i * 14, "X"))

    # cube
    cube_val = int(board.cube["value"])
    owner = board.cube["owner"]
    cube_y = {"center": (_YTOP + _YBOT) / 2 - 27, "o": _YTOP + 6, "x": _YBOT - 60}[owner]
    parts.append(
        f'<rect x="{_f(_X0 - 6)}" y="{_f(cube_y)}" width="54" height="54" rx="8" '
        f'fill="#e8e2d2" stroke="#000" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{_f(_X0 - 6 + 27)}" y="{_f(cube_y + 36)}" text-anchor="middle" '
        f'font-family="monospace" font-size="26" fill="#1a1a1a">{cube_val}</text>'
    )

    # dice or cube-decision marker
    if board.dice:
        dx = _bar_left() + _BARW + 3 * _COLW
        parts.extend(_die(dx, (_YTOP + _YBOT) / 2 - 27, board.dice[0]))
        parts.extend(_die(dx + 70, (_YTOP + _YBOT) / 2 - 27, board.dice[1]))
    else:
        parts.append(
            _text(_bar_left() + _BARW + 3 * _COLW, (_YTOP + _YBOT) / 2, "[cube decision]", size=22)
        )

    # header text
    if board.score.get("length", 0):
        crawf = " Crawford" if board.score.get("crawford") else ""
        ctx = f"Match to {board.score['length']}{crawf}  X {board.score['x']}-{board.score['o']} O"
    else:
        ctx = "Money game"
    parts.append(_text(_X0, 40, ctx, size=24, weight="bold"))
    parts.append(_text(_X0, 66, f"Cube: {cube_val} ({owner})", size=18))
    parts.append(_text(play_right, 40, f"O pip {po}", size=20, anchor="end"))
    parts.append(_text(play_right, 66, f"X pip {px}  (X on roll)", size=20, anchor="end"))
    # on-roll marker
    parts.append(
        f'<polygon points="{_f(_X0 - 20)},{_f(_YBOT - 20)} {_f(_X0 - 20)},{_f(_YBOT)} '
        f'{_f(_X0 - 4)},{_f(_YBOT - 10)}" fill="{_C_X}"/>'
    )

    parts.append("</svg>")
    return "".join(parts)
