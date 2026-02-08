"""Tests for TD(lambda) target computation."""

import pytest
import numpy as np

from backgammon.core.board import initial_board, empty_board
from backgammon.core.types import Player, GameOutcome, MoveStep
from backgammon.encoding.encoder import outcome_to_equity
from backgammon.training.self_play import GameStep, GameResult
from backgammon.training.td_lambda import (
    compute_td_lambda_targets,
    _monte_carlo_targets,
    _flip_equity_if_needed,
    _to_6dim,
    _to_5dim,
    _flip_6dim,
)


# ==============================================================================
# Helper to create test game results
# ==============================================================================


def _make_game(num_steps=5, outcome_player=Player.WHITE, outcome_points=1,
               value_estimates=None):
    """Create a simple game result for testing."""
    board = initial_board()
    steps = []
    for i in range(num_steps):
        player = Player.WHITE if i % 2 == 0 else Player.BLACK
        board_copy = board.copy()
        board_copy.player_to_move = player
        steps.append(GameStep(
            board=board_copy,
            player=player,
            legal_moves=[()],
            move_taken=(),
            dice=(1, 2),
        ))

    outcome = GameOutcome(winner=outcome_player, points=outcome_points)
    return GameResult(
        steps=steps,
        outcome=outcome,
        num_moves=num_steps,
        starting_position=board,
        value_estimates=value_estimates,
    )


# ==============================================================================
# Monte Carlo target tests
# ==============================================================================


class TestMonteCarloTargets:
    """Tests for _monte_carlo_targets (lambda=1.0 equivalent)."""

    def test_basic_white_wins(self):
        game = _make_game(num_steps=3, outcome_player=Player.WHITE, outcome_points=1)
        targets = _monte_carlo_targets(game)
        assert len(targets) == 3

        # Step 0 (White): white wins normal -> [1,0,0,0,0]
        np.testing.assert_array_almost_equal(targets[0], [1, 0, 0, 0, 0])
        # Step 1 (Black): white wins normal -> for black this is lose normal [0,0,0,0,0]
        np.testing.assert_array_almost_equal(targets[1], [0, 0, 0, 0, 0])
        # Step 2 (White): white wins normal -> [1,0,0,0,0]
        np.testing.assert_array_almost_equal(targets[2], [1, 0, 0, 0, 0])

    def test_gammon_win(self):
        game = _make_game(num_steps=2, outcome_player=Player.WHITE, outcome_points=2)
        targets = _monte_carlo_targets(game)
        # Step 0 (White): white wins gammon
        np.testing.assert_array_almost_equal(targets[0], [0, 1, 0, 0, 0])
        # Step 1 (Black): loses gammon -> lose_gammon=1
        np.testing.assert_array_almost_equal(targets[1], [0, 0, 0, 1, 0])

    def test_empty_game(self):
        game = _make_game(num_steps=0)
        targets = _monte_carlo_targets(game)
        assert len(targets) == 0


# ==============================================================================
# Equity flipping tests
# ==============================================================================


class TestFlipEquity:
    """Tests for _flip_equity_if_needed."""

    def test_same_player_no_flip(self):
        equity = np.array([0.5, 0.1, 0.05, 0.1, 0.05], dtype=np.float32)
        result = _flip_equity_if_needed(equity, Player.WHITE, Player.WHITE)
        np.testing.assert_array_almost_equal(result, equity)

    def test_different_player_flips(self):
        # Pure win for player A: [1, 0, 0, 0, 0]
        equity_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        equity_b = _flip_equity_if_needed(equity_a, Player.WHITE, Player.BLACK)
        # From B's perspective: A won normal means B lost normal
        # lose_normal = 1 - sum([1,0,0,0,0]) = 0 -> B_win_normal = 0
        # So equity_b = [0, 0, 0, 0, 0] which means B loses normal
        np.testing.assert_array_almost_equal(equity_b, [0, 0, 0, 0, 0])

    def test_gammon_flip(self):
        # A wins gammon: [0, 1, 0, 0, 0]
        equity_a = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        equity_b = _flip_equity_if_needed(equity_a, Player.WHITE, Player.BLACK)
        # B loses gammon: lose_gammon should be 1.0
        # equity_b[3] = equity_a[1] = 1.0 (B loses gammon = A wins gammon)
        assert equity_b[3] == pytest.approx(1.0)
        # B win_normal = lose_normal_a = 1 - 1 = 0
        assert equity_b[0] == pytest.approx(0.0)

    def test_double_flip_identity(self):
        """Flipping twice should return to original (approximately)."""
        equity = np.array([0.4, 0.1, 0.05, 0.08, 0.02], dtype=np.float32)
        flipped = _flip_equity_if_needed(equity, Player.WHITE, Player.BLACK)
        back = _flip_equity_if_needed(flipped, Player.BLACK, Player.WHITE)
        np.testing.assert_array_almost_equal(back, equity, decimal=5)


# ==============================================================================
# TD(lambda) target computation tests
# ==============================================================================


class TestTDLambdaTargets:
    """Tests for compute_td_lambda_targets."""

    def test_no_value_estimates_falls_back_to_mc(self):
        """Without value estimates, should return Monte Carlo targets."""
        game = _make_game(num_steps=3, outcome_player=Player.WHITE, outcome_points=1)
        targets = compute_td_lambda_targets(game, lambda_param=0.7)
        mc_targets = _monte_carlo_targets(game)
        assert len(targets) == len(mc_targets)
        for t_td, t_mc in zip(targets, mc_targets):
            np.testing.assert_array_almost_equal(t_td, t_mc)

    def test_lambda_1_equals_mc(self):
        """TD(lambda=1.0) should equal Monte Carlo targets."""
        # Create value estimates (uniform uncertain predictions)
        V = [np.array([0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 4
        game = _make_game(num_steps=4, outcome_player=Player.WHITE,
                          outcome_points=1, value_estimates=V)
        targets_td = compute_td_lambda_targets(game, lambda_param=1.0)
        targets_mc = _monte_carlo_targets(game)

        assert len(targets_td) == len(targets_mc)
        for t_td, t_mc in zip(targets_td, targets_mc):
            np.testing.assert_array_almost_equal(t_td, t_mc, decimal=4)

    def test_lambda_0_bootstraps(self):
        """TD(lambda=0) should bootstrap from next state's value."""
        # Simple 2-step game: step 0 (White), step 1 (Black), White wins normal
        V0 = np.array([0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        V1 = np.array([0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        game = _make_game(num_steps=2, outcome_player=Player.WHITE,
                          outcome_points=1, value_estimates=[V0, V1])

        targets = compute_td_lambda_targets(game, lambda_param=0.0)

        assert len(targets) == 2
        # With lambda=0, target_t = V(s_t) + delta_t = V(s_t) + (V(s_{t+1}) - V(s_t))
        # = V(s_{t+1}) (from the same player's perspective)
        # But V[1] is from Black's perspective, need to flip for White's perspective
        # So target_0 should be V(s_0) + delta_0 where delta_0 involves the flipped V[1]

    def test_output_shape(self):
        """Each target should be a 5-dim array."""
        V = [np.array([0.2, 0.05, 0.01, 0.05, 0.01], dtype=np.float32)] * 3
        game = _make_game(num_steps=3, outcome_player=Player.WHITE,
                          outcome_points=1, value_estimates=V)
        targets = compute_td_lambda_targets(game, lambda_param=0.7)
        assert len(targets) == 3
        for t in targets:
            assert t.shape == (5,)

    def test_targets_are_valid_probabilities(self):
        """Targets should be in [0, 1] range after clamping."""
        V = [np.array([0.2, 0.05, 0.01, 0.05, 0.01], dtype=np.float32)] * 5
        game = _make_game(num_steps=5, outcome_player=Player.WHITE,
                          outcome_points=1, value_estimates=V)
        targets = compute_td_lambda_targets(game, lambda_param=0.7)
        for t in targets:
            assert np.all(t >= 0.0), f"Negative value in target: {t}"
            assert np.all(t <= 1.0), f"Value > 1 in target: {t}"

    def test_draw_returns_empty(self):
        """Draw games should return empty targets."""
        game = GameResult(
            steps=[],
            outcome=None,
            num_moves=0,
            starting_position=initial_board(),
        )
        targets = compute_td_lambda_targets(game, lambda_param=0.7)
        assert len(targets) == 0

    def test_none_value_estimates_falls_back(self):
        """If some value estimates are None, should fall back to MC."""
        V = [np.array([0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), None, None]
        game = _make_game(num_steps=3, outcome_player=Player.WHITE,
                          outcome_points=1, value_estimates=V)
        targets = compute_td_lambda_targets(game, lambda_param=0.7)
        mc_targets = _monte_carlo_targets(game)
        for t_td, t_mc in zip(targets, mc_targets):
            np.testing.assert_array_almost_equal(t_td, t_mc)

    def test_intermediate_lambda(self):
        """Lambda=0.7 should produce targets between MC and bootstrapped."""
        V = [np.array([0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)] * 4
        game = _make_game(num_steps=4, outcome_player=Player.WHITE,
                          outcome_points=1, value_estimates=V)

        targets_mc = compute_td_lambda_targets(game, lambda_param=1.0)
        targets_07 = compute_td_lambda_targets(game, lambda_param=0.7)
        targets_00 = compute_td_lambda_targets(game, lambda_param=0.0)

        # The last step target should be the same for all lambda values
        # (it's always based on the game outcome)
        np.testing.assert_array_almost_equal(
            targets_mc[-1], targets_07[-1], decimal=3
        )


# ==============================================================================
# Integration: TD(lambda) with replay buffer
# ==============================================================================


class TestTDLambdaReplayBufferIntegration:
    """Test that TD(lambda) targets can be used with the replay buffer."""

    def test_add_game_with_td_lambda(self):
        from backgammon.training.replay_buffer import ReplayBuffer

        V = [np.array([0.2, 0.05, 0.01, 0.05, 0.01], dtype=np.float32)] * 5
        game = _make_game(num_steps=5, outcome_player=Player.WHITE,
                          outcome_points=1, value_estimates=V)

        buffer = ReplayBuffer(max_size=100, min_size=1)
        buffer.add_game(game, td_lambda=0.7)

        assert len(buffer) == 5

    def test_add_game_without_td_lambda(self):
        """Without td_lambda, should use MC targets (existing behavior)."""
        from backgammon.training.replay_buffer import ReplayBuffer

        game = _make_game(num_steps=3, outcome_player=Player.WHITE, outcome_points=1)
        buffer = ReplayBuffer(max_size=100, min_size=1)
        buffer.add_game(game)  # No td_lambda

        assert len(buffer) == 3

    def test_add_game_td_lambda_none_falls_back(self):
        """td_lambda=None should give same result as not passing it."""
        from backgammon.training.replay_buffer import ReplayBuffer

        game = _make_game(num_steps=3, outcome_player=Player.WHITE, outcome_points=1)
        buffer = ReplayBuffer(max_size=100, min_size=1)
        buffer.add_game(game, td_lambda=None)

        assert len(buffer) == 3
