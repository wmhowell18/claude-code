"""Tests: move parsing / end-position equivalence + cube notation (PLAN.md §4.3)."""

import pytest

from bgcore.board import Board
from bgcore import moves, notation


# -- move notation round-trips -------------------------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        ("24/18 13/11", "24/18 13/11"),
        ("13/11 24/18", "24/18 13/11"),   # order-independent -> canonical
        ("24/18*", "24/18*"),
        ("6/off", "6/off"),
        ("bar/22", "bar/22"),
        ("13/11(2)", "13/11(2)"),
        ("bar/24* 13/11", "bar/24* 13/11"),
        ("Cannot Move", "Cannot Move"),
        ("cannot move", "Cannot Move"),
    ],
)
def test_move_format_roundtrip(s, expected):
    once = notation.format_move(notation.parse_move(s))
    assert once == expected
    # idempotent / stable
    assert notation.format_move(notation.parse_move(once)) == expected


def test_parse_bad_move_raises():
    with pytest.raises(notation.NotationError):
        notation.parse_move("banana")
    with pytest.raises(notation.NotationError):
        notation.parse_move("99/1")


# -- equivalence by resulting position -----------------------------------


def test_order_independent_equivalence():
    b = Board.starting_position([6, 2])
    assert moves.moves_equivalent(b, "24/18 13/11", "13/11 24/18")


def test_combined_and_split_die_equivalence():
    # one checker moved 13->7 (a 4 then a 2) reaches the same place however spelled
    b = Board.starting_position([4, 2])
    assert moves.moves_equivalent(b, "13/9/7", "13/7")
    assert moves.moves_equivalent(b, "13/11/7", "13/7")


def test_non_equivalent_moves():
    b = Board.starting_position([3, 1])
    assert not moves.moves_equivalent(b, "8/5 6/5", "24/23 24/21")


# -- cube notation --------------------------------------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        ("double", "double"),
        ("DOUBLE", "double"),
        ("redouble", "double"),
        ("no double", "no double"),
        ("No Double", "no double"),
        ("roll", "no double"),
        ("take", "take"),
        ("pass", "pass"),
        ("drop", "pass"),
        ("beaver", "beaver"),
    ],
)
def test_cube_parse(s, expected):
    assert notation.parse_cube(s) == expected
    assert notation.format_cube(notation.parse_cube(s)) == expected


def test_cube_answer_two_part():
    assert notation.parse_cube_answer("Double, Take") == ("double", "take")
    assert notation.parse_cube_answer("double / pass") == ("double", "pass")
    assert notation.parse_cube_answer("No double") == ("no double", None)


def test_cube_parse_bad():
    with pytest.raises(notation.NotationError):
        notation.parse_cube("banana")
