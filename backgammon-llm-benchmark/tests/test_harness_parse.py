"""Tests: answer extraction on messy model outputs (PLAN.md §4.3)."""

import pytest

from bgcore.board import Board
from harness import parse


def _checker():
    return Board.starting_position([3, 1])


def _cube():
    return Board.starting_position()


def test_verbose_reasoning_then_move_line():
    b = _checker()
    text = (
        "First I consider splitting the back checkers.\n"
        "Actually building the 5-point is standard here.\n"
        "MOVE: 8/5 6/5\n"
    )
    oc = parse.parse_answer(b, text)
    assert oc.status == "parsed"
    assert oc.move == "8/5 6/5"
    assert not oc.used_fallback


def test_final_answer_prefix_accepted():
    b = _checker()
    oc = parse.parse_answer(b, "blah\nFINAL ANSWER: 8/5 6/5")
    assert oc.status == "parsed"
    assert oc.move == "8/5 6/5"


def test_wrong_case_and_extra_markup():
    b = _checker()
    oc = parse.parse_answer(b, "move:  `8/5 6/5` ")
    assert oc.status == "parsed"
    assert oc.move == "8/5 6/5"


def test_last_move_line_wins():
    b = _checker()
    text = "MOVE: 24/21\nOn reflection:\nMOVE: 8/5 6/5"
    oc = parse.parse_answer(b, text)
    assert oc.move == "8/5 6/5"


def test_fallback_to_move_like_line():
    b = _checker()
    oc = parse.parse_answer(b, "I would play 8/5 6/5 and be happy.")
    assert oc.status == "parsed"
    assert oc.used_fallback
    assert oc.move == "8/5 6/5"


def test_illegal_move_flagged():
    b = _checker()
    oc = parse.parse_answer(b, "MOVE: 8/2 6/1")  # not legal for a 3-1
    assert oc.status == "illegal"
    assert oc.move is None


def test_garbage_is_unparseable():
    b = _checker()
    oc = parse.parse_answer(b, "I have no idea, sorry.")
    assert oc.status == "unparseable"


def test_cannot_move_recognized():
    b = _checker()
    oc = parse.parse_answer(b, "MOVE: Cannot Move")
    # starting position has legal moves, so 'cannot move' is not legal here
    assert oc.status == "illegal"


def test_rollout_move_matched():
    b = _checker()
    oc = parse.parse_answer(b, "MOVE: 6/5 8/5", rollout_moves=["8/5 6/5", "24/21"])
    assert oc.status == "parsed"
    assert oc.rollout_move == "8/5 6/5"  # equivalent regardless of token order


def test_cube_drop_synonym():
    b = _cube()
    oc = parse.parse_answer(b, "This is a clear pass.\nACTION: drop")
    assert oc.status == "parsed"
    assert oc.cube_action == ("pass", None)


def test_cube_double_take_two_part():
    b = _cube()
    oc = parse.parse_answer(b, "ACTION: Double, Take")
    assert oc.status == "parsed"
    assert oc.cube_action == ("double", "take")
    assert oc.move == "double, take"


def test_cube_no_double():
    b = _cube()
    oc = parse.parse_answer(b, "ACTION: no double")
    assert oc.cube_action == ("no double", None)


def test_cube_fallback_line():
    b = _cube()
    oc = parse.parse_answer(b, "I think you should double here.")
    assert oc.status == "parsed"
    assert oc.used_fallback
    assert oc.cube_action[0] == "double"


def test_cube_garbage_unparseable():
    b = _cube()
    oc = parse.parse_answer(b, "hmmmm not sure")
    assert oc.status == "unparseable"
