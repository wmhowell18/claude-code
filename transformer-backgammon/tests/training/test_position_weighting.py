"""Tests for position weighting and validation split features."""

import pytest
import numpy as np

from backgammon.training.replay_buffer import (
    ReplayBuffer,
    _compute_position_weight,
)
from backgammon.training.self_play import GameStep, GameResult
from backgammon.core.board import initial_board, empty_board
from backgammon.core.types import Player, GameOutcome


def _make_dummy_step(player=Player.WHITE):
    """Create a minimal game step for testing."""
    board = initial_board()
    return GameStep(
        board=board,
        player=player,
        dice=(3, 1),
        legal_moves=[()],
        move_taken=(),
    )


def _make_dummy_game(num_steps=10, outcome=None):
    """Create a dummy game result."""
    if outcome is None:
        outcome = GameOutcome(winner=Player.WHITE, points=1)
    steps = [_make_dummy_step() for _ in range(num_steps)]
    return GameResult(
        steps=steps,
        outcome=outcome,
        num_moves=num_steps,
        starting_position=initial_board(),
    )


# ==============================================================================
# Position weight computation tests
# ==============================================================================


class TestComputePositionWeight:
    def test_base_weight_is_one(self):
        """Equal game progress and certain outcome should be near 1.0."""
        # Pure win target: [1, 0, 0, 0, 0]
        target = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        weight = _compute_position_weight(target, 0, 10)
        assert weight >= 1.0

    def test_late_game_higher_weight(self):
        """Later positions in a game should have higher weight."""
        target = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        early_weight = _compute_position_weight(target, 0, 20)
        late_weight = _compute_position_weight(target, 19, 20)
        assert late_weight > early_weight

    def test_uncertain_position_higher_weight(self):
        """Positions near 50/50 should have higher weight than certain ones."""
        certain = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # win_prob=1.0
        uncertain = np.array([0.5, 0.0, 0.0, 0.0, 0.0])  # win_prob=0.5

        w_certain = _compute_position_weight(certain, 5, 10)
        w_uncertain = _compute_position_weight(uncertain, 5, 10)
        assert w_uncertain > w_certain

    def test_weight_always_positive(self):
        """Weight should always be >= 1.0."""
        for win_prob in [0.0, 0.2, 0.5, 0.8, 1.0]:
            target = np.array([win_prob, 0.0, 0.0, 0.0, 0.0])
            for step_idx in range(10):
                w = _compute_position_weight(target, step_idx, 10)
                assert w >= 1.0

    def test_single_step_game(self):
        """Single-step game should not crash."""
        target = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        w = _compute_position_weight(target, 0, 1)
        assert w >= 1.0

    def test_gammon_targets(self):
        """Gammon targets should work correctly."""
        # Gammon win: [0, 1, 0, 0, 0] → win_prob = 1.0
        gammon = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        w = _compute_position_weight(gammon, 5, 10)
        assert w >= 1.0

    def test_max_weight_bounded(self):
        """Weight shouldn't be unreasonably large."""
        # Most uncertain position at end of game
        target = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        w = _compute_position_weight(target, 99, 100)
        assert w <= 3.0  # game_progress(0.5) + uncertainty(0.5) = 2.0


# ==============================================================================
# Weighted replay buffer tests
# ==============================================================================


class TestWeightedReplayBuffer:
    def test_weighted_buffer_stores_weights(self):
        """Buffer with weighting should store weights."""
        buf = ReplayBuffer(
            max_size=1000, min_size=1,
            use_position_weighting=True,
        )
        game = _make_dummy_game(num_steps=5)
        buf.add_game(game)

        assert len(buf._weights) == len(buf._steps)
        assert all(w >= 1.0 for w in buf._weights)

    def test_unweighted_buffer_uniform_weights(self):
        """Buffer without weighting should have all weights = 1.0."""
        buf = ReplayBuffer(
            max_size=1000, min_size=1,
            use_position_weighting=False,
        )
        game = _make_dummy_game(num_steps=5)
        buf.add_game(game)

        assert all(w == 1.0 for w in buf._weights)

    def test_weighted_sampling_runs(self):
        """Weighted sampling should produce valid batches."""
        buf = ReplayBuffer(
            max_size=1000, min_size=1,
            use_position_weighting=True,
        )
        game = _make_dummy_game(num_steps=20)
        buf.add_game(game)

        batch = buf.sample_batch(5)
        assert batch['board_encoding'].shape[0] == 5
        assert batch['equity_target'].shape[0] == 5

    def test_clear_resets_weights(self):
        """Clearing buffer should also clear weights."""
        buf = ReplayBuffer(
            max_size=1000, min_size=1,
            use_position_weighting=True,
        )
        game = _make_dummy_game(num_steps=5)
        buf.add_game(game)
        assert len(buf._weights) > 0

        buf.clear()
        assert len(buf._weights) == 0

    def test_eviction_updates_weights(self):
        """When buffer is full, evicted entries should have updated weights."""
        buf = ReplayBuffer(
            max_size=5, min_size=1,
            use_position_weighting=True,
        )
        # Fill buffer
        game1 = _make_dummy_game(num_steps=5)
        buf.add_game(game1)
        assert len(buf._weights) == 5

        # Add more (triggers eviction)
        game2 = _make_dummy_game(num_steps=3)
        buf.add_game(game2)
        assert len(buf._weights) == 5  # Still max_size


# ==============================================================================
# Validation split tests
# ==============================================================================


class TestValidationSplit:
    def test_training_config_has_validation_fields(self):
        """TrainingConfig should have validation settings."""
        from backgammon.training.train import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, 'validation_fraction')
        assert hasattr(config, 'early_stopping_patience')
        assert hasattr(config, 'use_early_stopping')
        assert 0 < config.validation_fraction < 1

    def test_training_config_has_position_weighting(self):
        """TrainingConfig should have position weighting flag."""
        from backgammon.training.train import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, 'use_position_weighting')
        assert config.use_position_weighting is True
