"""Tests: phase classifier + human-difficulty tiering (PLAN.md §3)."""

import pytest

from bgcore.board import Board, validate
from generate import tiering
from generate.tiering import HumanErrorModel, TaxonomyPrior, assign_tier, classify_phase, tier_for


# --------------------------------------------------------------------------
# hand-built board fixtures, one per phase (all validated to conserve checkers)
# --------------------------------------------------------------------------


def _b(points, **kw):
    b = Board(points=list(points), **kw)
    b.refresh_pip()
    validate(b)
    return b


def board_opening():
    return Board.starting_position([3, 1])


def board_race():
    pts = [0] * 26
    pts[15], pts[20], pts[24] = 3, 4, 3      # mover, all past the opponent
    pts[10], pts[5], pts[1] = -3, -4, -3
    return _b(pts, dice=[6, 5], off={"x": 5, "o": 5})


def board_bearoff():
    pts = [0] * 26
    pts[19], pts[20], pts[24] = 2, 2, 3
    pts[6], pts[4], pts[1] = -2, -2, -4
    return _b(pts, dice=[3, 1], off={"x": 8, "o": 7})


def board_backgame():
    pts = [0] * 26
    pts[1], pts[3] = 2, 2                      # two deep anchors in opp home
    pts[13], pts[8], pts[6] = 5, 3, 3
    pts[2], pts[4], pts[5], pts[10] = -3, -4, -4, -4   # opponent well advanced
    return _b(pts, dice=[5, 2])


def board_blitz():
    pts = [0] * 26
    pts[19], pts[20], pts[21], pts[17], pts[13], pts[24] = 2, 2, 2, 2, 4, 3
    pts[6], pts[8], pts[4], pts[1], pts[5] = -3, -3, -3, -1, -3
    return _b(pts, dice=[3, 3], bar={"x": 0, "o": 2})   # opponent has 2 on the bar


def board_priming():
    pts = [0] * 26
    for p in (15, 16, 17, 18, 19):            # a five-prime
        pts[p] = 2
    pts[13], pts[6] = 3, 2
    pts[24], pts[23], pts[3], pts[2], pts[1] = -3, -2, -4, -3, -3
    return _b(pts, dice=[4, 2])


def board_holding():
    pts = [0] * 26
    pts[7], pts[12], pts[15], pts[18], pts[19] = 2, 5, 3, 3, 2   # single anchor idx7
    pts[20], pts[6], pts[5], pts[4], pts[2] = -2, -3, -3, -3, -4
    return _b(pts, dice=[6, 3])


def board_cube():
    b = Board.starting_position()
    b.decision_type = "cube"
    return b


PHASE_CASES = [
    (board_opening, "opening-ish"),
    (board_race, "race"),
    (board_bearoff, "bearoff"),
    (board_backgame, "backgame"),
    (board_blitz, "blitz"),
    (board_priming, "priming"),
    (board_holding, "holding-game"),
    (board_cube, "cube-action"),
]


@pytest.mark.parametrize("factory,expected", PHASE_CASES)
def test_classify_phase(factory, expected):
    assert classify_phase(factory()) == expected


def test_all_phase_tags_are_in_taxonomy():
    assert set(tiering.PHASES) == set(tiering._PHASE_PRIOR)  # noqa: SLF001


# --------------------------------------------------------------------------
# threshold table (PLAN §3.2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "miss,eel,tier",
    [
        (0.005, 0.5, "T1"),   # experts essentially never err
        (0.05, 1.5, "T2"),    # 2-10%
        (0.20, 6.0, "T3"),    # 10-30%
        (0.40, 8.0, "T4"),    # >30%
        (0.05, 12.0, "T4"),   # EEL > 10 forces T4 regardless of miss rate
        (0.015, 1.5, "T2"),   # miss < 2% but EEL >= 1 -> not T1
    ],
)
def test_tier_for_thresholds(miss, eel, tier):
    assert tier_for(miss, eel) == tier


def test_taxonomy_prior_maps_phases_to_expected_tiers():
    prior = TaxonomyPrior()
    assert isinstance(prior, HumanErrorModel)
    assert assign_tier(board_race()).tier == "T1"
    assert assign_tier(board_backgame()).tier == "T4"
    assert assign_tier(board_blitz()).tier == "T3"
    r = assign_tier(board_opening())
    assert r.tier == "T2"
    assert r.difficulty_source == "taxonomy-prior"
    assert r.phase == "opening-ish"
    assert 0.0 <= r.expert_miss_rate <= 1.0


def test_cube_window_nudges_difficulty_up():
    prior = TaxonomyPrior()
    board = board_cube()
    base_miss, base_eel = prior.predict(board, None)
    tight = {
        "decision_type": "cube",
        "cube": {"no_double_equity": 0.50, "double_take_equity": 0.52},
    }
    miss, eel = prior.predict(board, tight)
    assert miss > base_miss
    assert eel > base_eel


def test_bar_contact_nudges_difficulty_up():
    prior = TaxonomyPrior()
    miss, eel = prior.predict(board_blitz(), None)   # opponent on the bar
    base_miss, base_eel = tiering._PHASE_PRIOR["blitz"]  # noqa: SLF001
    assert miss > base_miss and eel > base_eel


def test_custom_error_model_is_pluggable():
    class AlwaysHard:
        source = "unit-test"

        def predict(self, board, rollout=None):
            return 0.9, 25.0

    r = assign_tier(board_race(), model=AlwaysHard())
    assert r.tier == "T4"
    assert r.difficulty_source == "unit-test"
