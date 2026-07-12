"""Tests: XGID / GNU BG ID round-trip losslessness (PLAN.md §7)."""

import pytest

from bgcore.board import Board
from ids import gnubg_id, xgid


def _race_board():
    b = Board(points=[0] * 26, dice=[6, 5])
    b.points[20] = 4
    b.points[22] = 3
    b.points[24] = 3
    b.points[5] = -3
    b.points[3] = -4
    b.points[1] = -3
    b.off = {"x": 5, "o": 5}
    b.refresh_pip()
    return b


def _bearoff_board():
    b = Board(points=[0] * 26, dice=[3, 1])
    b.points[19] = 2
    b.points[20] = 2
    b.points[24] = 3
    b.points[6] = -2
    b.points[4] = -2
    b.points[1] = -4
    b.off = {"x": 8, "o": 7}
    b.refresh_pip()
    return b


def _bar_board():
    b = Board(points=[0] * 26, dice=[4, 2])
    b.points[1] = 2
    b.points[12] = 4
    b.points[17] = 3
    b.points[19] = 4
    b.points[24] = -2
    b.points[13] = -5
    b.points[8] = -3
    b.points[6] = -4
    b.bar = {"x": 2, "o": 1}
    b.refresh_pip()
    return b


def _match_cube_board():
    b = Board.starting_position()          # cube decision (no dice)
    b.decision_type = "cube"
    b.cube = {"value": 2, "owner": "o"}
    b.score = {"x": 3, "o": 5, "length": 7, "crawford": False}
    return b


ALL_BOARDS = {
    "start": Board.starting_position([3, 1]),
    "race": _race_board(),
    "bearoff": _bearoff_board(),
    "bar": _bar_board(),
    "cube": _match_cube_board(),
}


# -- XGID -----------------------------------------------------------------


def test_xgid_start_string():
    b = Board.starting_position()
    assert xgid.board_to_xgid(b) == "XGID=-b----E-C---eE---c-e----B-:0:0:1:00:0:0:0:0:10"


@pytest.mark.parametrize("name", list(ALL_BOARDS))
def test_xgid_board_roundtrip(name):
    b = ALL_BOARDS[name]
    s = xgid.board_to_xgid(b)
    b2 = xgid.xgid_to_board(s)
    assert b2.points == b.points
    assert b2.bar == b.bar and b2.off == b.off
    assert b2.dice == b.dice
    assert b2.cube == b.cube
    assert b2.score == b.score
    assert b2.turn == b.turn
    # string round-trip too
    assert xgid.board_to_xgid(b2) == s


def test_xgid_turn_o_roundtrip():
    b = ALL_BOARDS["race"].copy()
    b.turn = "o"
    s = xgid.board_to_xgid(b)
    assert ":-1:" in s  # turn field encodes the top seat
    assert xgid.board_to_xgid(xgid.xgid_to_board(s)) == s


def test_xgid_parse_ignores_prefix_case_and_pads():
    s = "xgid=-b----E-C---eE---c-e----B-:0:0:1:31:0:0:0:0:10"
    b = xgid.xgid_to_board(s)
    assert b.dice == [3, 1]


# -- GNU BG ---------------------------------------------------------------


def test_gnubg_start_position_id():
    assert gnubg_id.position_id(Board.starting_position()) == "4HPwATDgc/ABMA"


@pytest.mark.parametrize("name", list(ALL_BOARDS))
def test_gnubg_checker_roundtrip(name):
    b = ALL_BOARDS[name]
    pid = gnubg_id.position_id(b)
    assert len(pid) == 14
    b2 = gnubg_id.position_id_to_board(pid)
    assert b2.points == b.points
    assert b2.bar == b.bar
    assert b2.off == b.off


@pytest.mark.parametrize("name", list(ALL_BOARDS))
def test_gnubg_full_roundtrip(name):
    b = ALL_BOARDS[name]
    ids = gnubg_id.board_to_ids(b)
    assert len(ids["match_id"]) == 12
    b2 = gnubg_id.ids_to_board(ids["position_id"], ids["match_id"])
    assert b2.points == b.points
    assert b2.bar == b.bar and b2.off == b.off
    assert b2.dice == b.dice
    assert b2.cube == b.cube
    assert b2.score == b.score


def test_xgid_and_gnubg_agree_on_layout():
    for b in ALL_BOARDS.values():
        via_xgid = xgid.xgid_to_board(xgid.board_to_xgid(b))
        via_gnubg = gnubg_id.position_id_to_board(gnubg_id.position_id(b))
        assert via_xgid.points == via_gnubg.points
