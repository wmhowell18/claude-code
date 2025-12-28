"""Tests for experience replay buffer."""

import pytest
import numpy as np
import jax.numpy as jnp

from backgammon.training.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
)
from backgammon.training.self_play import GameResult, GameStep
from backgammon.core.board import initial_board
from backgammon.core.types import Player


def create_dummy_game(num_steps: int = 10, outcome: str = 'white_wins') -> GameResult:
    """Create a dummy game for testing."""
    board = initial_board()
    steps = []

    for i in range(num_steps):
        step = GameStep(
            board=board,
            player=Player.WHITE if i % 2 == 0 else Player.BLACK,
            legal_moves=[(), ((6, 5),), ((8, 7),)],
            move_taken=((6, 5),),
            dice=(3, 4),
        )
        steps.append(step)

    return GameResult(
        steps=steps,
        outcome=outcome,
        num_moves=num_steps,
        starting_position=board,
    )


class TestReplayBuffer:
    """Test basic replay buffer functionality."""

    def test_buffer_creation(self):
        """Test buffer can be created with default params."""
        buffer = ReplayBuffer(max_size=1000, min_size=100)
        assert len(buffer) == 0
        assert not buffer.is_ready()

    def test_add_single_game(self):
        """Test adding a single game to buffer."""
        buffer = ReplayBuffer(max_size=1000, min_size=10)
        game = create_dummy_game(num_steps=15)

        buffer.add_game(game)

        # Should have 15 steps in buffer
        assert len(buffer) == 15

    def test_add_multiple_games(self):
        """Test adding multiple games."""
        buffer = ReplayBuffer(max_size=1000, min_size=10)

        for _ in range(5):
            game = create_dummy_game(num_steps=10)
            buffer.add_game(game)

        # Should have 50 steps total
        assert len(buffer) == 50
        assert buffer.is_ready()

    def test_buffer_ready_threshold(self):
        """Test buffer becomes ready after min_size."""
        buffer = ReplayBuffer(max_size=1000, min_size=25)

        assert not buffer.is_ready()

        # Add 2 games (20 steps)
        for _ in range(2):
            buffer.add_game(create_dummy_game(num_steps=10))

        assert len(buffer) == 20
        assert not buffer.is_ready()  # Still below min_size

        # Add 1 more game (30 steps total)
        buffer.add_game(create_dummy_game(num_steps=10))
        assert buffer.is_ready()

    def test_fifo_eviction(self):
        """Test FIFO eviction when buffer is full."""
        buffer = ReplayBuffer(max_size=25, min_size=10, eviction_policy='fifo')

        # Fill buffer
        for _ in range(3):
            buffer.add_game(create_dummy_game(num_steps=10))

        # Should be at max capacity (30 > 25, so truncated to 25)
        assert len(buffer) == 25

        # Add more games - should stay at max
        buffer.add_game(create_dummy_game(num_steps=10))
        assert len(buffer) == 25

    def test_random_eviction(self):
        """Test random eviction policy."""
        buffer = ReplayBuffer(max_size=25, min_size=10, eviction_policy='random')

        # Fill buffer
        for _ in range(3):
            buffer.add_game(create_dummy_game(num_steps=10))

        assert len(buffer) == 25

        # Add more - should maintain size
        buffer.add_game(create_dummy_game(num_steps=5))
        assert len(buffer) == 25

    def test_sample_batch_not_ready(self):
        """Test sampling fails if buffer not ready."""
        buffer = ReplayBuffer(max_size=1000, min_size=50)
        buffer.add_game(create_dummy_game(num_steps=10))

        with pytest.raises(ValueError, match="Buffer not ready"):
            buffer.sample_batch(32)

    def test_sample_batch_basic(self):
        """Test basic batch sampling."""
        buffer = ReplayBuffer(max_size=1000, min_size=10)

        # Add games
        for _ in range(5):
            buffer.add_game(create_dummy_game(num_steps=10))

        # Sample batch
        batch = buffer.sample_batch(32)

        # Check batch structure
        assert 'board_encoding' in batch
        assert 'target_policy' in batch
        assert 'value_target' in batch
        assert 'action_mask' in batch

        # Check shapes (raw_encoding_config has feature_dim=2)
        # Board encoding is (batch, 26 positions, 2 features)
        # Action space size is 1024 (from action_encoder.ACTION_SPACE_SIZE)
        assert batch['board_encoding'].shape == (32, 26, 2)
        assert batch['target_policy'].shape == (32, 1024)
        assert batch['value_target'].shape == (32,)
        assert batch['action_mask'].shape == (32, 1024)

        # Check types
        assert batch['board_encoding'].dtype == jnp.float32
        assert batch['target_policy'].dtype == jnp.float32
        assert batch['value_target'].dtype == jnp.float32
        assert batch['action_mask'].dtype == jnp.bool_

    def test_sample_batch_respects_size(self):
        """Test sampling respects buffer size."""
        buffer = ReplayBuffer(max_size=1000, min_size=10)

        # Add only 20 steps
        for _ in range(2):
            buffer.add_game(create_dummy_game(num_steps=10))

        # Request 32 but should only get 20
        batch = buffer.sample_batch(32)
        assert batch['board_encoding'].shape[0] == 20

    def test_value_targets_from_outcome(self):
        """Test value targets are correctly computed from game outcome."""
        buffer = ReplayBuffer(max_size=1000, min_size=5)

        # Add white win
        white_win = create_dummy_game(num_steps=6, outcome='white_wins')
        buffer.add_game(white_win)

        # Sample and check values
        batch = buffer.sample_batch(6)
        values = batch['value_target']

        # White moves (steps 0, 2, 4) should have +1
        # Black moves (steps 1, 3, 5) should have -1
        # We can't know exact order after sampling, but all should be ±1
        assert all(v in [-1.0, 1.0] for v in values)

    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = ReplayBuffer(max_size=1000, min_size=10)

        for _ in range(3):
            buffer.add_game(create_dummy_game(num_steps=10))

        assert len(buffer) == 30

        buffer.clear()
        assert len(buffer) == 0
        assert not buffer.is_ready()

    def test_statistics(self):
        """Test buffer statistics."""
        buffer = ReplayBuffer(max_size=100, min_size=10)

        # Empty buffer
        stats = buffer.get_statistics()
        assert stats['size'] == 0
        assert stats['utilization'] == 0.0

        # Add games
        for _ in range(3):
            buffer.add_game(create_dummy_game(num_steps=10))

        stats = buffer.get_statistics()
        assert stats['size'] == 30
        assert stats['utilization'] == 0.3  # 30/100
        assert 'avg_value' in stats
        assert 'std_value' in stats
        assert 'min_value' in stats
        assert 'max_value' in stats


class TestPrioritizedReplayBuffer:
    """Test prioritized replay buffer."""

    def test_prioritized_creation(self):
        """Test prioritized buffer creation."""
        buffer = PrioritizedReplayBuffer(
            max_size=1000,
            min_size=100,
            alpha=0.6,
            beta=0.4,
        )
        assert len(buffer) == 0
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4

    def test_prioritized_add_game(self):
        """Test adding games to prioritized buffer."""
        buffer = PrioritizedReplayBuffer(max_size=1000, min_size=10)

        game = create_dummy_game(num_steps=15)
        buffer.add_game(game)

        assert len(buffer) == 15
        assert len(buffer._priorities) == 15

    def test_prioritized_sample_returns_weights(self):
        """Test prioritized sampling returns weights and indices."""
        buffer = PrioritizedReplayBuffer(max_size=1000, min_size=10)

        # Add games
        for _ in range(5):
            buffer.add_game(create_dummy_game(num_steps=10))

        # Sample
        batch, indices, weights = buffer.sample_batch(32)

        # Check returns
        assert 'board_encoding' in batch
        assert len(indices) == 32
        assert len(weights) == 32

        # Weights should be normalized to max of 1.0
        assert weights.max() == pytest.approx(1.0)
        assert all(w > 0 for w in weights)

    def test_update_priorities(self):
        """Test updating priorities."""
        buffer = PrioritizedReplayBuffer(max_size=1000, min_size=10)

        # Add games
        for _ in range(3):
            buffer.add_game(create_dummy_game(num_steps=10))

        # Sample
        batch, indices, weights = buffer.sample_batch(20)

        # Update priorities with new values
        new_priorities = np.random.rand(20) * 10
        buffer.update_priorities(indices, new_priorities)

        # Priorities should be updated
        for idx, priority in zip(indices, new_priorities):
            # Should be priority + epsilon
            assert buffer._priorities[idx] == pytest.approx(priority + 1e-6)

    def test_prioritized_clear(self):
        """Test clearing prioritized buffer."""
        buffer = PrioritizedReplayBuffer(max_size=1000, min_size=10)

        buffer.add_game(create_dummy_game(num_steps=20))
        assert len(buffer) == 20
        assert len(buffer._priorities) == 20

        buffer.clear()
        assert len(buffer) == 0
        assert len(buffer._priorities) == 0


class TestReplayBufferIntegration:
    """Integration tests for replay buffer."""

    def test_multiple_game_outcomes(self):
        """Test buffer handles different game outcomes."""
        buffer = ReplayBuffer(max_size=1000, min_size=10)

        # Add mix of white and black wins
        buffer.add_game(create_dummy_game(num_steps=10, outcome='white_wins'))
        buffer.add_game(create_dummy_game(num_steps=10, outcome='black_wins'))
        buffer.add_game(create_dummy_game(num_steps=10, outcome='white_wins'))

        assert len(buffer) == 30

        # Sample and verify we get variety
        batch = buffer.sample_batch(30)
        values = batch['value_target']

        # Should have both +1 and -1 values
        assert any(v == 1.0 for v in values)
        assert any(v == -1.0 for v in values)

    def test_replay_buffer_with_training_loop(self):
        """Test buffer works in a simple training loop simulation."""
        buffer = ReplayBuffer(max_size=500, min_size=50)

        # Simulate training loop
        for epoch in range(5):
            # Generate games
            for _ in range(3):
                game = create_dummy_game(num_steps=10)
                buffer.add_game(game)

            # Once ready, sample batches
            if buffer.is_ready():
                for _ in range(2):
                    batch = buffer.sample_batch(32)
                    assert batch['board_encoding'].shape[0] == 32

        # Should have accumulated games
        assert len(buffer) > 50
        assert buffer.is_ready()
