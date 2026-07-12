"""Tests: bgcore.board model — pip counts, flip, normalize, JSON (PLAN.md §1.1)."""

import pytest

from bgcore.board import Board, BoardError, canonical_key, flip, normalize, pip_counts, validate


def test_starting_position_pips():
    b = Board.starting_position([3, 1])
    assert pip_counts(b) == (167, 167)
    validate(b)  # does not raise


def test_json_roundtrip():
    b = Board.starting_position([6, 4])
    b.cube = {"value": 2, "owner": "x"}
    b.score = {"x": 3, "o": 1, "length": 7, "crawford": False}
    data = b.to_json()
    b2 = Board.from_json(data)
    assert b == b2
    assert b2.to_json() == data
    # pip is computed into the JSON
    assert data["pip"] == {"x": 167, "o": 167}


def test_from_json_string():
    b = Board.starting_position()
    s = b.to_json_str()
    assert Board.from_json(s) == b


def test_flip_is_involution():
    b = Board.starting_position([5, 2])
    b.cube = {"value": 4, "owner": "x"}
    b.score = {"x": 2, "o": 5, "length": 11, "crawford": True}
    ff = flip(flip(b))
    assert ff.points == b.points
    assert ff.bar == b.bar and ff.off == b.off
    assert ff.cube == b.cube and ff.score == b.score
    assert ff.turn == b.turn


def test_flip_swaps_sides():
    b = Board(points=[0] * 26)
    b.points[1] = 3      # mover blot-stack
    b.points[24] = -2    # opponent
    b.bar = {"x": 1, "o": 0}
    b.cube = {"value": 2, "owner": "x"}
    fb = flip(b)
    # mover's 3 on index 1 become opponent's 3 on index 24
    assert fb.points[24] == -3
    assert fb.points[1] == 2
    assert fb.bar == {"x": 0, "o": 1}
    assert fb.cube["owner"] == "o"
    assert fb.turn == "o"


def test_normalize_dedup_key_ignores_seat_and_dice_order():
    a = Board.starting_position([1, 3])
    b = Board.starting_position([3, 1])
    b.turn = "o"
    assert canonical_key(a) == canonical_key(b)


def test_normalize_money_zeroes_score():
    b = Board.starting_position()
    b.score = {"x": 4, "o": 2, "length": 0, "crawford": False}
    n = normalize(b)
    assert n.score == {"x": 0, "o": 0, "length": 0, "crawford": False}


def test_validate_detects_bad_counts():
    b = Board.starting_position()
    b.points[1] = 9  # too many mover checkers overall
    errs = validate(b, strict=False)
    assert any("checker count" in e for e in errs)
    with pytest.raises(BoardError):
        validate(b, strict=True)


def test_validate_reserved_sentinels():
    b = Board.starting_position()
    b.points[0] = 1
    errs = validate(b, strict=False)
    assert any("reserved" in e for e in errs)


def test_all_home():
    b = Board(points=[0] * 26)
    b.points[19] = 5
    b.points[24] = 10
    assert b.all_home()
    b.points[10] = 1
    assert not b.all_home()
