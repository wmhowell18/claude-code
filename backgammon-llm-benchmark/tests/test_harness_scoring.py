"""Tests: equity-loss lookup, worst-legal penalty, BenchPR math (PLAN.md §4.4)."""

import pytest

from bgcore.board import Board
from harness import parse, scoring
from harness.parse import ParseOutcome


def _checker():
    return Board.starting_position([3, 1])


def _cube():
    return Board.starting_position()


CHECKER_ROLLOUT = {
    "decision_type": "checker",
    "checker": {
        "moves": [
            {"move": "8/5 6/5", "equity": 0.10, "error_mp": 0.0, "std_err": 0.002, "rank": 1},
            {"move": "24/21 6/5", "equity": 0.06, "error_mp": 40.0, "std_err": 0.002, "rank": 2},
            {"move": "24/23 8/5", "equity": -0.20, "error_mp": 300.0, "std_err": 0.003, "rank": 3},
        ]
    },
    "best_move": "8/5 6/5",
}


def test_best_move_scores_zero():
    b = _checker()
    oc = parse.parse_answer(b, "MOVE: 8/5 6/5")
    sc = scoring.score_decision(CHECKER_ROLLOUT, b, oc)
    assert sc.equity_loss == 0.0
    assert sc.is_best
    assert not sc.parse_failed


def test_order_independent_lookup():
    b = _checker()
    oc = parse.parse_answer(b, "MOVE: 6/5 8/5")  # same position, reversed order
    sc = scoring.score_decision(CHECKER_ROLLOUT, b, oc)
    assert sc.is_best and sc.equity_loss == 0.0


def test_suboptimal_move_equity_loss():
    b = _checker()
    oc = parse.parse_answer(b, "MOVE: 24/21 6/5")  # a legal, second-best play
    sc = scoring.score_decision(CHECKER_ROLLOUT, b, oc)
    assert sc.equity_loss_mp == 40.0
    assert sc.equity_loss == pytest.approx(0.04)
    assert not sc.is_best


def test_within_noise_scores_zero():
    b = _checker()
    # a move with small error but within its std-error band counts as best
    rollout = {
        "decision_type": "checker",
        "checker": {"moves": [
            {"move": "8/5 6/5", "equity": 0.1, "error_mp": 0.0, "std_err": 0.005, "rank": 1},
            {"move": "24/21 6/5", "equity": 0.098, "error_mp": 2.0, "std_err": 0.005, "rank": 2},
        ]},
    }
    oc = parse.parse_answer(b, "MOVE: 24/21 6/5")
    sc = scoring.score_decision(rollout, b, oc)
    assert sc.is_best  # 2 mpt = 0.002 pts <= 0.005 std-err
    assert sc.equity_loss == 0.0


def test_illegal_scored_worst_legal():
    b = _checker()
    oc = parse.parse_answer(b, "MOVE: 8/2 6/1")  # illegal for 3-1
    sc = scoring.score_decision(CHECKER_ROLLOUT, b, oc)
    assert sc.parse_failed
    assert sc.equity_loss_mp == 300.0  # worst move's error


def test_unparseable_scored_worst_legal():
    b = _checker()
    oc = parse.parse_answer(b, "no clue")
    sc = scoring.score_decision(CHECKER_ROLLOUT, b, oc)
    assert sc.parse_failed
    assert sc.equity_loss_mp == 300.0


def test_worst_error_mp_helper():
    assert scoring.worst_error_mp(CHECKER_ROLLOUT) == 300.0


# -- cube -----------------------------------------------------------------

CUBE_ROLLOUT = {
    "decision_type": "cube",
    "cube": {
        "no_double_equity": 0.30,
        "double_take_equity": 0.28,
        "double_pass_equity": 1.0,
        "best_action": "No double",
        "error_mp": {"No double": 0.0, "Double, Take": 60.0, "Double, Pass": 700.0},
    },
}


def test_cube_best_action_zero():
    b = _cube()
    oc = parse.parse_answer(b, "ACTION: no double")
    sc = scoring.score_decision(CUBE_ROLLOUT, b, oc)
    assert sc.is_best and sc.equity_loss == 0.0


def test_cube_wrong_action_loss():
    b = _cube()
    oc = parse.parse_answer(b, "ACTION: double, take")
    sc = scoring.score_decision(CUBE_ROLLOUT, b, oc)
    assert sc.equity_loss_mp == 60.0
    assert not sc.is_best


def test_cube_unparseable_worst():
    b = _cube()
    oc = parse.parse_answer(b, "unsure")
    sc = scoring.score_decision(CUBE_ROLLOUT, b, oc)
    assert sc.parse_failed
    assert sc.equity_loss_mp == 700.0


def test_cube_bare_double_correct_when_best_is_double():
    b = _cube()
    rollout = {
        "decision_type": "cube",
        "cube": {"best_action": "Double, Take",
                 "error_mp": {"Double, Take": 0.0, "No double": 90.0}},
    }
    oc = ParseOutcome("parsed", "cube", answer_text="double",
                      move="double", cube_action=("double", None))
    sc = scoring.score_cube(rollout, oc)
    assert sc.is_best and sc.equity_loss == 0.0


# -- aggregation / BenchPR ------------------------------------------------


def test_benchpr_constant():
    # mean equity loss of 0.008 pts -> PR 4
    assert scoring.benchpr([0.008, 0.008]) == pytest.approx(4.0)


def test_aggregate_rollups_and_accuracy():
    rows = [
        {"equity_loss": 0.0, "is_best": True, "tier": "T1", "track": "text", "decision_type": "checker"},
        {"equity_loss": 0.020, "is_best": False, "tier": "T3", "track": "text", "decision_type": "checker"},
        {"equity_loss": 0.010, "is_best": False, "tier": "T3", "track": "text", "decision_type": "cube", "parse_failed": True},
    ]
    agg = scoring.aggregate(rows)
    assert agg["n"] == 3
    assert agg["best_move_accuracy"] == pytest.approx(1 / 3)
    assert agg["benchpr"] == pytest.approx(500 * (0.03 / 3))
    assert set(agg["per_tier"]) == {"T1", "T3"}
    assert agg["per_tier"]["T3"]["n"] == 2
    assert set(agg["per_decision_type"]) == {"checker", "cube"}
    assert agg["parse_failures"] == 1
    lo, hi = agg["benchpr_ci95"]
    assert lo <= agg["benchpr"] <= hi


def test_bootstrap_ci_deterministic():
    losses = [0.0, 0.01, 0.02, 0.005, 0.03]
    a = scoring.bootstrap_benchpr_ci(losses, seed=1)
    b = scoring.bootstrap_benchpr_ci(losses, seed=1)
    assert a == b
    assert a[0] <= a[1]


def test_empty_aggregate_safe():
    agg = scoring.aggregate([])
    assert agg["benchpr"] == 0.0
    assert agg["best_move_accuracy"] == 0.0
    assert agg["n"] == 0
