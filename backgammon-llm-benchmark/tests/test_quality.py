"""Tests for the quiz-eligibility quality gate (generate/quality.py).

Two layers: synthetic unit tests pinning each rule (forced / unscoreable / trivial
/ cube), and an integration test over the REAL pilot data asserting the gate gates
out exactly the eight degenerate positions the audit found (and nothing else).
"""

from __future__ import annotations

import glob
import json
import os

from bgcore.board import Board
from generate import quality

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_DIR = os.path.join(REPO_ROOT, "positions", "pilot")
ROLL_DIR = os.path.join(REPO_ROOT, "rollouts", "gnubg")


# -- synthetic unit tests: one per rule ------------------------------------


def _checker_record(board: Board) -> dict:
    return {"position_id": "syn", "decision_type": "checker", "board_json": board.to_json()}


def _checker_rollout(moves: list[tuple[str, float]]) -> dict:
    return {"decision_type": "checker",
            "checker": {"moves": [{"move": m, "error_mp": e} for m, e in moves]}}


def test_forced_single_legal_move_is_ineligible():
    # one checker, single die playable -> exactly one legal move
    b = Board(points=[0] * 26, dice=[6, 5])
    b.points[13] = 1
    b.points[24] = -2  # block the 5 so only the 6 plays (12/6): one legal move
    rec = _checker_record(b)
    roll = _checker_rollout([("12/6", 0.0)])
    ok, reason = quality.quiz_eligible(rec, roll)
    assert ok is False and reason.startswith("forced"), reason


def test_unscoreable_single_scored_move_is_ineligible():
    # a genuine 2-legal-move position, but the rollout only scored one move
    b = Board.starting_position([3, 1])
    rec = _checker_record(b)
    roll = _checker_rollout([("8/5 6/5", 0.0)])
    ok, reason = quality.quiz_eligible(rec, roll)
    assert ok is False and reason.startswith("unscoreable"), reason


def test_trivial_zero_spread_is_ineligible():
    b = Board.starting_position([3, 1])
    rec = _checker_record(b)
    roll = _checker_rollout([("8/5 6/5", 0.0), ("24/21 6/5", 0.0), ("13/10 6/5", 0.0)])
    ok, reason = quality.quiz_eligible(rec, roll)
    assert ok is False and reason.startswith("trivial"), reason


def test_meaningful_spread_is_eligible():
    b = Board.starting_position([3, 1])
    rec = _checker_record(b)
    roll = _checker_rollout([("8/5 6/5", 0.0), ("24/21 13/10", 55.0)])
    ok, reason = quality.quiz_eligible(rec, roll)
    assert ok is True and reason is None


def test_spread_just_below_threshold_is_ineligible():
    b = Board.starting_position([3, 1])
    rec = _checker_record(b)
    roll = _checker_rollout([("8/5 6/5", 0.0), ("24/21 13/10", quality.MIN_CHECKER_SPREAD_MP - 0.1)])
    ok, reason = quality.quiz_eligible(rec, roll)
    assert ok is False and reason.startswith("trivial")


def test_cube_requires_all_three_actions():
    rec = {"position_id": "c", "decision_type": "cube", "board_json": Board.starting_position([]).to_json()}
    two = {"decision_type": "cube", "cube": {"error_mp": {"No double": 0.0, "Double, Take": 40.0}}}
    ok, reason = quality.quiz_eligible(rec, two)
    assert ok is False and reason.startswith("cube")
    three = {"decision_type": "cube", "cube": {"error_mp": {"No double": 0.0, "Double, Take": 40.0, "Double, Pass": 900.0}}}
    ok2, reason2 = quality.quiz_eligible(rec, three)
    assert ok2 is True and reason2 is None


# -- integration: the real pilot set ---------------------------------------

# Derived by audit (see build output). 1-based display order over sorted files.
EXPECTED_FORCED = {
    "bg-2dff31791e5c29a8",  # #11
    "bg-4474e236bf67b9b0",  # #19
    "bg-6211351d8ca6977d",  # #21
    "bg-6aa50863df140c18",  # #25
    "bg-c853084b55fcc7b5",  # #40
}
EXPECTED_TRIVIAL = {
    "bg-0d3cf1acba7e656d",  # #5
    "bg-77105a01ad87534f",  # #26
    "bg-c57b3f4707c48e9b",  # #37
}
EXPECTED_EXCLUDED = EXPECTED_FORCED | EXPECTED_TRIVIAL


def _pilot_pairs():
    pairs = []
    for path in sorted(glob.glob(os.path.join(POS_DIR, "bg-*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        roll = json.load(open(os.path.join(ROLL_DIR, rec["position_id"] + ".json"), encoding="utf-8"))
        pairs.append((rec, roll))
    return pairs


def test_pilot_gate_excludes_exactly_the_eight_bad_positions():
    pairs = _pilot_pairs()
    assert len(pairs) == 50
    eligible, excluded = quality.filter_eligible(pairs)
    excl_ids = {pid for pid, _ in excluded}
    assert excl_ids == EXPECTED_EXCLUDED, excl_ids
    assert len(eligible) == 42
    checker = sum(1 for r in eligible if r["decision_type"] == "checker")
    cube = sum(1 for r in eligible if r["decision_type"] == "cube")
    assert (checker, cube) == (27, 15)


def test_pilot_exclusion_reasons_match_categories():
    pairs = _pilot_pairs()
    _, excluded = quality.filter_eligible(pairs)
    forced = {pid for pid, reason in excluded if reason.startswith("forced")}
    trivial = {pid for pid, reason in excluded if reason.startswith("trivial")}
    assert forced == EXPECTED_FORCED, forced
    assert trivial == EXPECTED_TRIVIAL, trivial


def test_all_cube_positions_are_eligible():
    for rec, roll in _pilot_pairs():
        if rec["decision_type"] != "cube":
            continue
        ok, reason = quality.quiz_eligible(rec, roll)
        assert ok is True, (rec["position_id"], reason)
