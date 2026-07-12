"""Tests: ASCII / SVG / raster render determinism + content (PLAN.md §1.1-1.2)."""

import io

import pytest

from bgcore.board import Board
from render import ascii as ascii_render
from render import svg as svg_render


def _match_board():
    b = Board.starting_position([6, 4])
    b.cube = {"value": 2, "owner": "x"}
    b.score = {"x": 2, "o": 5, "length": 7, "crawford": True}
    b.refresh_pip()
    return b


# -- ASCII ----------------------------------------------------------------


def test_ascii_deterministic():
    b = Board.starting_position([3, 1])
    assert ascii_render.render(b) == ascii_render.render(b)


def test_ascii_fixed_width_board_rows():
    out = ascii_render.render(Board.starting_position([3, 1]))
    board_rows = [ln for ln in out.splitlines() if ln.startswith(("|", "+"))]
    assert len(board_rows) == 13  # 2 borders + 11 checker/bar rows
    for ln in board_rows:
        frame = ln[:43]  # the fixed frame, before any trailing annotations
        assert len(frame) == 43
        if ln.startswith("+"):
            assert frame[0] == "+" and frame[42] == "+"
        else:
            assert frame[0] == "|" and frame[42] == "|"


def test_ascii_contains_required_state():
    out = ascii_render.render(_match_board())
    assert ascii_render.ASCII_RENDER_VERSION in out
    assert "Match to 7" in out and "Crawford" in out
    assert "Cube: 2" in out
    assert "Dice: 6-4" in out
    assert "pip:167" in out
    assert "on roll" in out
    assert "X" in out and "O" in out


def test_ascii_money_vs_cube_decision():
    b = Board.starting_position()  # no dice
    out = ascii_render.render(b)
    assert "Money game" in out
    assert "[cube decision]" in out


# -- SVG ------------------------------------------------------------------


def test_svg_deterministic():
    b = _match_board()
    assert svg_render.render(b) == svg_render.render(b)


def test_svg_is_wellformed_and_versioned():
    out = svg_render.render(Board.starting_position([5, 3]))
    assert out.startswith("<svg") and out.rstrip().endswith("</svg>")
    assert svg_render.IMAGE_RENDER_VERSION in out


def test_svg_contains_required_elements():
    out = svg_render.render(_match_board())
    assert "Match to 7" in out and "Crawford" in out
    assert "Cube: 2" in out
    assert "pip 167" in out
    assert "on roll" in out.lower()
    # point numbers 1..24 all present as <text> labels
    for n in (1, 6, 12, 13, 19, 24):
        assert f">{n}<" in out
    # dice drawn as rects with pips (no [cube decision] marker when a roll is set)
    assert "[cube decision]" not in out


def test_svg_cube_decision_marker():
    b = Board.starting_position()
    assert "[cube decision]" in svg_render.render(b)


# -- raster ---------------------------------------------------------------


def test_raster_produces_valid_png():
    from render import raster

    png = raster.board_to_png(Board.starting_position([6, 4]), width=640)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    assert im.format == "PNG"
    assert im.size[0] == 640
