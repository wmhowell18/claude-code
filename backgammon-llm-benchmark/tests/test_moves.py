"""Tests: bgcore.moves — legal move generation and rules (PLAN.md §4.3)."""

from bgcore.board import Board
from bgcore import moves


def _board(points_map, *, dice, bar=None, off=None):
    b = Board(points=[0] * 26, dice=list(dice))
    for idx, v in points_map.items():
        b.points[idx] = v
    if bar:
        b.bar = {"x": bar.get("x", 0), "o": bar.get("o", 0)}
    if off:
        b.off = {"x": off.get("x", 0), "o": off.get("o", 0)}
    return b


def test_starting_31_makes_five_point():
    b = Board.starting_position([3, 1])
    legal = moves.legal_moves(b)
    assert "8/5 6/5" in legal            # the money-play best
    assert moves.is_legal(b, "8/5 6/5")


def test_no_legal_move_returns_empty():
    # mover on the bar, every entry point (1..6) blocked by 2+ opponent checkers
    b = _board({i: -2 for i in range(1, 7)}, dice=[3, 4], bar={"x": 1})
    assert moves.legal_moves(b) == []
    assert moves.generate_moves(b) == []
    assert moves.is_legal(b, "Cannot Move")


def test_bar_entry_has_priority():
    # one entry point open (4), the other (3) blocked
    b = _board({3: -2, 4: 0, 7: 0}, dice=[3, 4], bar={"x": 1})
    for mv in moves.legal_moves(b):
        assert mv.startswith("bar/")


def test_larger_die_rule():
    # single checker; either die playable alone but not both -> must use the 6
    b = _board({13: 1, 24: -2}, dice=[6, 5])
    assert moves.legal_moves(b) == ["12/6"]


def test_must_play_both_when_possible():
    # a play exists that uses both dice, so single-die plays are excluded
    b = _board({13: 1, 24: -2}, dice=[6, 2])
    legal = moves.legal_moves(b)
    assert len(legal) == 1
    assert moves.moves_equivalent(b, legal[0], "12/4")  # both dice, one checker 13->21


def test_bearoff_overflow_off_highest_point():
    # one checker on the 5-point (board index 20); all home; 6 must bear it off
    b = _board({20: 1}, dice=[6, 4], off={"x": 14})
    legal = moves.legal_moves(b)
    # 4 moves 5->1, then 6 bears the checker off -> chain 5/1/off
    assert legal == ["5/1/off"]


def test_bearoff_exact_and_within_board():
    # checkers on the 6-point (idx19) and 1-point (idx24); die 1 bears off the ace
    b = _board({19: 1, 24: 1}, dice=[1, 3], off={"x": 13})
    legal = moves.legal_moves(b)
    # die 1 bears off 24 (1/off); die 3 moves 19->22 (6/3). both dice used.
    # tokens are ordered by descending start point, so "6/3" precedes "1/off".
    assert "6/3 1/off" in legal


def test_doubles_play_four_checkers():
    b = Board.starting_position([2, 2])
    assert moves.is_legal(b, "24/22(2) 13/11(2)")
    # every legal double move consumes all four dice
    for mv in moves.generate_moves(b):
        total = sum(len(ch["pts"]) - 1 for ch in moves._chains_from_hops(mv.hops))
        assert total == 4


def test_hit_generates_star():
    # opponent blot on the mover's landing point
    b = _board({1: 2, 7: -1}, dice=[6, 3])
    legal = moves.legal_moves(b)
    assert any("*" in mv for mv in legal)


def test_apply_move_updates_layout_and_hits():
    b = _board({1: 2, 7: -1}, dice=[6, 3])
    # both checkers leave point 24 (board idx 1); the 6 hits the blot on 18 (idx 7)
    nb = moves.apply_move(b, "24/18* 24/21")
    assert nb.points[7] == 1        # mover now on idx 7 (point 18)
    assert nb.points[4] == 1        # mover now on idx 4 (point 21)
    assert nb.points[1] == 0        # point 24 vacated
    assert nb.bar["o"] == 1         # opponent blot sent to bar
    assert nb.dice == []
