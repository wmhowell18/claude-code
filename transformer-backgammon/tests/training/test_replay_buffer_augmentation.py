"""Tests for replay buffer features that had zero test coverage:
- Color-flip augmentation (_flip_equity_target, add_game with augmentation)
- Weighted sampling (position weighting in sample_batch)
- Pre-encoding correctness (_encode_board_fast_single)
"""

import pytest
import numpy as np
import jax.numpy as jnp

from backgammon.training.replay_buffer import (
    ReplayBuffer,
    _flip_equity_target,
    _encode_board_fast_single,
)
from backgammon.training.self_play import GameResult, GameStep
from backgammon.core.board import initial_board
from backgammon.core.types import Player, GameOutcome


def _make_game(num_steps=10, winner=Player.WHITE, points=1):
    """Create a minimal game for buffer tests."""
    board = initial_board()
    steps = [
        GameStep(
            board=board,
            player=Player.WHITE if i % 2 == 0 else Player.BLACK,
            legal_moves=[(), ((6, 5),)],
            move_taken=((6, 5),),
            dice=(3, 4),
        )
        for i in range(num_steps)
    ]
    return GameResult(
        steps=steps,
        outcome=GameOutcome(winner=winner, points=points),
        num_moves=num_steps,
        starting_position=board,
    )


class TestFlipEquityTarget:
    """Test _flip_equity_target correctness."""

    def test_flip_preserves_probability_sum(self):
        """Flipped target should still sum to 1.0 (valid 6-dim distribution)."""
        target = np.array([0.5, 0.1, 0.02, 0.27, 0.08, 0.03], dtype=np.float32)
        flipped = _flip_equity_target(target)

        np.testing.assert_allclose(target.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(flipped.sum(), 1.0, atol=1e-6)

    def test_flip_swaps_win_lose(self):
        """Win/lose categories should swap: [w_n,w_g,w_bg,l_n,l_g,l_bg] -> [l_n,l_g,l_bg,w_n,w_g,w_bg]."""
        target = np.array([0.5, 0.2, 0.05, 0.12, 0.1, 0.03], dtype=np.float32)
        flipped = _flip_equity_target(target)

        # new win_normal = old lose_normal
        assert flipped[0] == target[3]
        # new win_gammon = old lose_gammon
        assert flipped[1] == target[4]
        # new win_bg = old lose_bg
        assert flipped[2] == target[5]
        # new lose_normal = old win_normal
        assert flipped[3] == target[0]
        # new lose_gammon = old win_gammon
        assert flipped[4] == target[1]
        # new lose_bg = old win_bg
        assert flipped[5] == target[2]

    def test_flip_win_normal_becomes_lose_normal(self):
        """New win_normal should equal old lose_normal and vice versa."""
        target = np.array([0.4, 0.1, 0.02, 0.37, 0.08, 0.03], dtype=np.float32)
        flipped = _flip_equity_target(target)

        np.testing.assert_allclose(flipped[0], target[3], atol=1e-6)  # new win_n = old lose_n
        np.testing.assert_allclose(flipped[3], target[0], atol=1e-6)  # new lose_n = old win_n

    def test_double_flip_is_identity(self):
        """Flipping twice should return the original."""
        target = np.array([0.5, 0.15, 0.05, 0.17, 0.1, 0.03], dtype=np.float32)
        double_flipped = _flip_equity_target(_flip_equity_target(target))
        np.testing.assert_allclose(double_flipped, target, atol=1e-6)

    def test_flip_pure_win(self):
        """Target with 100% win_normal should flip to 100% lose_normal."""
        target = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        flipped = _flip_equity_target(target)

        # Old lose_normal = 0.0, so new win_normal = 0.0
        assert flipped[0] == 0.0
        # All probability moves to lose_normal
        assert flipped[3] == 1.0
        np.testing.assert_allclose(flipped.sum(), 1.0, atol=1e-6)


class TestEncodeBoardFastSingle:
    """Test the fast board encoding used in replay buffer."""

    def test_output_shape(self):
        """Should produce (26, 10) array."""
        board = initial_board()
        encoded = _encode_board_fast_single(board)
        assert encoded.shape == (26, 10)
        assert encoded.dtype == np.float32

    def test_checker_normalization(self):
        """Raw checker features should be normalized by 15."""
        board = initial_board()
        encoded = _encode_board_fast_single(board)

        # Checker values should be in [0, 1] (divided by 15)
        assert encoded[:, 0].min() >= 0.0
        assert encoded[:, 0].max() <= 1.0
        assert encoded[:, 1].min() >= 0.0
        assert encoded[:, 1].max() <= 1.0

    def test_global_features_broadcast(self):
        """Global features (cols 2-9) should be identical across all 26 positions."""
        board = initial_board()
        encoded = _encode_board_fast_single(board)

        global_feats = encoded[:, 2:10]
        # All 26 rows should have the same global features
        for row in range(1, 26):
            np.testing.assert_array_equal(global_feats[0], global_feats[row])


class TestColorFlipAugmentation:
    """Test replay buffer with color-flip augmentation enabled."""

    def test_augmentation_doubles_data(self):
        """With color-flip on, adding a game should add 2x the steps."""
        buffer_plain = ReplayBuffer(max_size=1000, min_size=1, use_color_flip_augmentation=False)
        buffer_flip = ReplayBuffer(max_size=1000, min_size=1, use_color_flip_augmentation=True)

        game = _make_game(num_steps=5)
        buffer_plain.add_game(game)
        buffer_flip.add_game(game)

        assert len(buffer_flip) == 2 * len(buffer_plain)

    def test_augmented_buffer_samples_correctly(self):
        """Buffer with augmented data should produce valid batches."""
        buffer = ReplayBuffer(
            max_size=1000, min_size=1,
            use_color_flip_augmentation=True,
        )

        game = _make_game(num_steps=10)
        buffer.add_game(game)

        batch = buffer.sample_batch(4)
        assert batch['board_encoding'].shape == (4, 26, 10)
        assert batch['equity_target'].shape == (4, 6)


class TestWeightedSampling:
    """Test position-weighted sampling in replay buffer."""

    def test_weighted_sampling_runs(self):
        """Weighted sampling should work without errors."""
        buffer = ReplayBuffer(
            max_size=1000, min_size=1,
            use_position_weighting=True,
        )

        # Add enough data
        for _ in range(3):
            game = _make_game(num_steps=10)
            buffer.add_game(game)

        batch = buffer.sample_batch(8)
        assert batch['board_encoding'].shape[0] == 8

    def test_weighted_vs_uniform_produces_batches(self):
        """Both weighted and uniform buffers should produce valid batches."""
        for weighted in [True, False]:
            buffer = ReplayBuffer(
                max_size=1000, min_size=1,
                use_position_weighting=weighted,
            )
            for _ in range(3):
                buffer.add_game(_make_game(num_steps=10))

            batch = buffer.sample_batch(4)
            assert jnp.all(jnp.isfinite(batch['board_encoding']))
            assert jnp.all(jnp.isfinite(batch['equity_target']))
