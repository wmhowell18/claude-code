"""Tests for self-play helper functions that had zero coverage:
- _equity_to_value_np: 5-dim equity → scalar value conversion
- _terminal_value_for_player: terminal board → value from player perspective

Also tests EMA parameter utilities from train.py.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from backgammon.training.self_play import (
    _equity_to_value_np,
    _terminal_value_for_player,
)
from backgammon.training.train import _init_ema_params, _update_ema_params
from backgammon.core.board import initial_board, Board
from backgammon.core.types import Player


class TestEquityToValueNp:
    """Test _equity_to_value_np conversion."""

    def test_pure_win_normal(self):
        """100% win_normal should give value +1."""
        equity = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        value = _equity_to_value_np(equity)
        np.testing.assert_allclose(value, 1.0)

    def test_pure_lose_normal(self):
        """100% lose_normal (all zeros) should give value -1."""
        equity = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        value = _equity_to_value_np(equity)
        # lose_normal = 1 - 0 = 1, so lose_value = 1.0
        np.testing.assert_allclose(value, -1.0)

    def test_pure_win_gammon(self):
        """100% win_gammon should give value +2."""
        equity = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        value = _equity_to_value_np(equity)
        np.testing.assert_allclose(value, 2.0)

    def test_pure_win_backgammon(self):
        """100% win_bg should give value +3."""
        equity = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        value = _equity_to_value_np(equity)
        np.testing.assert_allclose(value, 3.0)

    def test_pure_lose_gammon(self):
        """100% lose_gammon should give value -2."""
        equity = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        value = _equity_to_value_np(equity)
        np.testing.assert_allclose(value, -2.0)

    def test_pure_lose_backgammon(self):
        """100% lose_bg should give value -3."""
        equity = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        value = _equity_to_value_np(equity)
        np.testing.assert_allclose(value, -3.0)

    def test_even_game(self):
        """50% win / 50% lose normal should give value ~0."""
        equity = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        # lose_normal = 0.5, so value = 0.5 - 0.5 = 0.0
        value = _equity_to_value_np(equity)
        np.testing.assert_allclose(value, 0.0, atol=1e-6)

    def test_batch_input(self):
        """Should handle batch of equities."""
        equity = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        values = _equity_to_value_np(equity)
        assert values.shape == (2,)
        np.testing.assert_allclose(values[0], 1.0)
        np.testing.assert_allclose(values[1], -1.0)

    def test_value_range(self):
        """Values should be in [-3, +3]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            equity = rng.dirichlet(np.ones(6))  # Random valid 6-dim equity
            value = _equity_to_value_np(equity)
            assert -3.0 <= value <= 3.0, f"Value {value} out of range for equity {equity}"


class TestTerminalValueForPlayer:
    """Test _terminal_value_for_player."""

    def test_non_terminal_returns_zero(self):
        """Non-terminal board should return 0."""
        board = initial_board()
        value = _terminal_value_for_player(board, Player.WHITE)
        assert value == 0.0

    def test_white_win_from_white_perspective(self):
        """White winning should be positive for White."""
        board = initial_board()
        # Set all white checkers to borne off
        new_white = np.zeros(26, dtype=np.int32)
        new_white[25] = 15  # All borne off
        board = Board(
            white_checkers=new_white,
            black_checkers=board.black_checkers.copy(),
            player_to_move=board.player_to_move,
        )
        value = _terminal_value_for_player(board, Player.WHITE)
        assert value > 0

    def test_white_win_from_black_perspective(self):
        """White winning should be negative for Black."""
        board = initial_board()
        new_white = np.zeros(26, dtype=np.int32)
        new_white[25] = 15
        board = Board(
            white_checkers=new_white,
            black_checkers=board.black_checkers.copy(),
            player_to_move=board.player_to_move,
        )
        value = _terminal_value_for_player(board, Player.BLACK)
        assert value < 0


class TestEMAParams:
    """Test EMA parameter utilities."""

    def test_init_ema_params_copies(self):
        """EMA init should create a copy, not a reference."""
        params = {'layer': {'kernel': jnp.array([1.0, 2.0, 3.0])}}
        ema = _init_ema_params(params)

        # Values should be equal
        np.testing.assert_array_equal(ema['layer']['kernel'], params['layer']['kernel'])

        # But should be different objects (copy, not alias)
        assert ema['layer']['kernel'] is not params['layer']['kernel']

    def test_update_ema_moves_toward_params(self):
        """EMA update should move ema_params toward current params."""
        ema_params = {'w': jnp.array([0.0, 0.0])}
        params = {'w': jnp.array([1.0, 1.0])}

        updated = _update_ema_params(ema_params, params, decay=0.9)

        # ema = 0.9 * 0.0 + 0.1 * 1.0 = 0.1
        np.testing.assert_allclose(updated['w'], [0.1, 0.1], atol=1e-6)

    def test_update_ema_decay_one_stays_put(self):
        """With decay=1.0, EMA should not change."""
        ema_params = {'w': jnp.array([5.0])}
        params = {'w': jnp.array([100.0])}

        updated = _update_ema_params(ema_params, params, decay=1.0)
        np.testing.assert_allclose(updated['w'], [5.0])

    def test_update_ema_decay_zero_copies_params(self):
        """With decay=0.0, EMA should equal current params."""
        ema_params = {'w': jnp.array([5.0])}
        params = {'w': jnp.array([100.0])}

        updated = _update_ema_params(ema_params, params, decay=0.0)
        np.testing.assert_allclose(updated['w'], [100.0])

    def test_ema_convergence(self):
        """Repeated updates should converge to the target."""
        ema_params = {'w': jnp.array([0.0])}
        params = {'w': jnp.array([1.0])}

        for _ in range(100):
            ema_params = _update_ema_params(ema_params, params, decay=0.99)

        np.testing.assert_allclose(ema_params['w'], [1.0], atol=0.1)
