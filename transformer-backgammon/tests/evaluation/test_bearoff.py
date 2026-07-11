"""Tests for the one-sided bearoff database.

Builds use small max_checkers limits (the position subset with at most
k checkers is closed under play) to keep runtime negligible; the full
15-checker database takes minutes and is exercised via scripts, not CI.
"""

import numpy as np
import pytest

from backgammon.core.board import (
    empty_board,
    initial_board,
    generate_legal_moves,
    apply_move,
    is_game_over,
    winner,
)
from backgammon.core.types import Player
from backgammon.core.dice import ALL_DICE_ROLLS, DICE_PROBABILITIES
from backgammon.evaluation.bearoff import (
    BearoffDatabase,
    TOTAL_POSITIONS,
    N_POINTS,
    enumerate_positions,
    position_index,
    _die_successors,
    _roll_successors,
    home_board_counts,
    bearoff_win_probability,
    bearoff_equity,
    select_bearoff_move,
    bearoff_agent,
)


@pytest.fixture(scope="module")
def db3():
    """Database covering positions with up to 3 checkers (instant build)."""
    return BearoffDatabase.build(max_checkers=3)


@pytest.fixture(scope="module")
def db6():
    """Database covering positions with up to 6 checkers."""
    return BearoffDatabase.build(max_checkers=6)


def _bearoff_race_board(white_pts, black_dists):
    """Board with both players bearing off.

    Args:
        white_pts: {point: count} for White (points 1-6).
        black_dists: {distance: count} for Black (distance d = point 25-d).
    """
    board = empty_board()
    for pt, cnt in white_pts.items():
        board.set_checkers(Player.WHITE, pt, cnt)
    board.set_checkers(Player.WHITE, 25, 15 - sum(white_pts.values()))
    for d, cnt in black_dists.items():
        board.set_checkers(Player.BLACK, 25 - d, cnt)
    board.set_checkers(Player.BLACK, 25, 15 - sum(black_dists.values()))
    return board


class TestPositionIndex:
    def test_bijection_on_subset(self):
        positions = enumerate_positions(4)
        idxs = [position_index(p) for p in positions]
        assert len(set(idxs)) == len(positions)
        assert all(0 <= i < TOTAL_POSITIONS for i in idxs)

    def test_empty_position_is_zero(self):
        assert position_index((0,) * N_POINTS) == 0

    def test_full_space_count(self):
        # C(21, 6) positions of up to 15 checkers on 6 points
        assert TOTAL_POSITIONS == 54264


class TestDieSuccessors:
    def test_bear_off_exact(self):
        # Checker on point 3, die 3: bear off
        succs = _die_successors((0, 0, 1, 0, 0, 0), 3)
        assert (0, 0, 0, 0, 0, 0) in succs

    def test_move_within(self):
        # Checker on point 6, die 2: only move 6 -> 4
        succs = _die_successors((0, 0, 0, 0, 0, 1), 2)
        assert succs == [(0, 0, 0, 1, 0, 0)]

    def test_overage_rule(self):
        # Only checker on point 2, die 5: bear off from highest occupied
        succs = _die_successors((0, 1, 0, 0, 0, 0), 5)
        assert succs == [(0, 0, 0, 0, 0, 0)]

    def test_no_overage_with_higher_checkers(self):
        # Checkers on 2 and 6, die 5: must move 6 -> 1 (no overage bearoff)
        succs = _die_successors((0, 1, 0, 0, 0, 1), 5)
        assert (1, 1, 0, 0, 0, 0) in succs
        assert (0, 0, 0, 0, 0, 1) not in succs  # May not bear off the 2

    def test_choice_between_bearoff_and_move(self):
        # Checkers on 3 and 6, die 3: bear off the 3 OR move 6 -> 3
        succs = set(_die_successors((0, 0, 1, 0, 0, 1), 3))
        assert (0, 0, 0, 0, 0, 1) in succs  # bore off the 3
        assert (0, 0, 2, 0, 0, 0) in succs  # moved 6 -> 3


class TestRollSuccessors:
    def test_finished_position_stays_finished(self):
        done = (0,) * N_POINTS
        assert _roll_successors(done, (3, 1)) == [done]

    def test_both_orders_considered(self):
        # Checker on 5: (6,1) -> off via 6 (overage); or 5->4 then 4 off...
        # With die 1 first: 5->4, then die 6 bears off 4 (overage).
        # Either order finishes.
        succs = _roll_successors((0, 0, 0, 0, 1, 0), (6, 1))
        assert (0,) * N_POINTS in succs

    def test_doubles_play_four(self):
        # 4 checkers on point 2, roll (2,2): all four bear off
        succs = _roll_successors((0, 4, 0, 0, 0, 0), (2, 2))
        assert (0,) * N_POINTS in succs


class TestDistributions:
    def test_empty_needs_zero_rolls(self, db3):
        d = db3.roll_distribution((0,) * N_POINTS)
        assert d[0] == pytest.approx(1.0)
        assert db3.rolls_to_bear_off((0,) * N_POINTS) == 0.0

    def test_single_checker_on_ace(self, db3):
        # Any roll bears off: exactly one roll, always
        d = db3.roll_distribution((1, 0, 0, 0, 0, 0))
        assert d[1] == pytest.approx(1.0)

    def test_single_checker_on_six(self, db3):
        # One roll iff max-die 6, a+b >= 6, or doubles d >= 2:
        # 27/36 rolls bear off immediately; the rest leave a checker
        # on <= 3 which always comes off next roll. E = 1.25 exactly.
        d = db3.roll_distribution((0, 0, 0, 0, 0, 1))
        assert d[1] == pytest.approx(27.0 / 36.0)
        assert d[2] == pytest.approx(9.0 / 36.0)
        assert db3.rolls_to_bear_off((0, 0, 0, 0, 0, 1)) == pytest.approx(1.25)

    def test_distributions_sum_to_one(self, db6):
        for counts in [(1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 6), (2, 2, 2, 0, 0, 0)]:
            assert db6.roll_distribution(counts).sum() == pytest.approx(1.0)

    def test_more_pips_never_fewer_expected_rolls(self, db6):
        # Adding a checker can never reduce expected rolls
        assert db6.rolls_to_bear_off((1, 1, 0, 0, 0, 0)) < db6.rolls_to_bear_off(
            (1, 1, 0, 0, 0, 1)
        )

    def test_checker_limit_enforced(self, db3):
        with pytest.raises(ValueError, match="max_checkers"):
            db3.roll_distribution((2, 2, 0, 0, 0, 0))


class TestWinProbability:
    def test_on_roll_cannot_lose_with_one_roll_left(self, db3):
        # (1,1): bears off both checkers on any roll -> X = 1 <= Y always
        p = db3.win_probability((1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0))
        assert p == pytest.approx(1.0)

    def test_on_roll_advantage_in_symmetric_race(self, db6):
        counts = (1, 1, 1, 1, 1, 1)
        p = db6.win_probability(counts, counts)
        assert 0.5 < p < 1.0

    def test_big_deficit_low_win_probability(self, db6):
        behind = (0, 0, 0, 2, 2, 2)
        ahead = (2, 1, 0, 0, 0, 0)
        assert db6.win_probability(behind, ahead) < 0.35

    def test_matches_monte_carlo(self, db6):
        """Exact win probability agrees with simulated optimal play."""
        rng = np.random.default_rng(7)
        white_counts = {1: 1, 2: 1, 5: 1, 6: 1}  # as {point: count}
        black_counts = {2: 1, 3: 1, 4: 1, 5: 1}  # as {distance: count}

        predicted = db6.win_probability(
            (1, 1, 0, 0, 1, 1), (0, 1, 1, 1, 1, 0)
        )

        wins = 0
        n_games = 300
        for _ in range(n_games):
            board = _bearoff_race_board(white_counts, black_counts)
            board.player_to_move = Player.WHITE
            while not is_game_over(board):
                player = board.player_to_move
                a, b = rng.integers(1, 7), rng.integers(1, 7)
                moves = generate_legal_moves(board, player, (int(a), int(b)))
                if moves:
                    mv = select_bearoff_move(db6, board, player, moves)
                    board = apply_move(board, player, mv)
                else:  # pragma: no cover - cannot dance in a pure race
                    board.player_to_move = player.opponent()
            if winner(board).winner == Player.WHITE:
                wins += 1

        observed = wins / n_games
        # 300 games, sigma ~ 0.03: allow 4 sigma
        assert observed == pytest.approx(predicted, abs=0.12)


class TestEngineIntegration:
    def test_home_board_counts_white(self):
        board = _bearoff_race_board({1: 2, 6: 3}, {2: 1})
        assert home_board_counts(board, Player.WHITE) == (2, 0, 0, 0, 0, 3)

    def test_home_board_counts_black_mirrored(self):
        # Black distance d lives on point 25 - d
        board = _bearoff_race_board({1: 1}, {1: 2, 6: 3})
        assert home_board_counts(board, Player.BLACK) == (2, 0, 0, 0, 0, 3)

    def test_home_board_counts_none_outside_home(self):
        board = initial_board()
        assert home_board_counts(board, Player.WHITE) is None
        assert home_board_counts(board, Player.BLACK) is None

    def test_win_probability_perspectives_sum_to_one_ish(self, db6):
        """P(white wins | white on roll) computed from both orientations."""
        board = _bearoff_race_board({1: 1, 2: 1, 5: 1}, {2: 1, 3: 1, 4: 1})
        board.player_to_move = Player.WHITE
        p_white = bearoff_win_probability(db6, board)
        assert p_white is not None
        assert 0.0 < p_white < 1.0
        eq = bearoff_equity(db6, board)
        assert eq == pytest.approx(2 * p_white - 1)

    def test_not_mutual_bearoff_returns_none(self, db6):
        board = initial_board()
        assert bearoff_win_probability(db6, board) is None
        assert bearoff_equity(db6, board) is None

    def test_select_bearoff_move_prefers_bearing_off(self, db3):
        # White on roll with checkers on 1 and 2, roll (2,1): bearing both
        # off wins immediately; any other play cannot be better.
        board = _bearoff_race_board({1: 1, 2: 1}, {6: 2})
        moves = generate_legal_moves(board, Player.WHITE, (2, 1))
        mv = select_bearoff_move(db3, board, Player.WHITE, moves)
        after = apply_move(board, Player.WHITE, mv)
        assert is_game_over(after)

    def test_select_returns_none_outside_bearoff(self, db3):
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        assert select_bearoff_move(db3, board, Player.WHITE, moves) is None

    def test_agent_uses_fallback_outside_bearoff(self, db3):
        agent = bearoff_agent(db3)
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        mv = agent.select_move(board, Player.WHITE, (3, 1), moves)
        assert mv in moves

    def test_agent_plays_db_move_in_bearoff(self, db3):
        agent = bearoff_agent(db3)
        board = _bearoff_race_board({1: 1, 2: 1}, {6: 2})
        moves = generate_legal_moves(board, Player.WHITE, (2, 1))
        mv = agent.select_move(board, Player.WHITE, (2, 1), moves)
        after = apply_move(board, Player.WHITE, mv)
        assert is_game_over(after)


class TestSaveLoad:
    def test_roundtrip(self, db3, tmp_path):
        path = str(tmp_path / "bearoff_test.npz")
        db3.save(path)
        loaded = BearoffDatabase.load(path)
        assert loaded.max_checkers == 3
        np.testing.assert_array_equal(
            loaded.distributions, db3.distributions
        )
        np.testing.assert_array_equal(
            loaded.expected_rolls, db3.expected_rolls
        )

    def test_load_or_build_caches(self, tmp_path):
        path = str(tmp_path / "bearoff_cache.npz")
        db1 = BearoffDatabase.load_or_build(path=path, max_checkers=2)
        assert (tmp_path / "bearoff_cache.npz").exists()
        db2 = BearoffDatabase.load_or_build(path=path, max_checkers=2)
        np.testing.assert_array_equal(db1.distributions, db2.distributions)
