"""End-to-end integration tests for the complete training pipeline.

Tests that all components work together: encoder, network, self-play,
replay buffer, training loop, and checkpointing.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import shutil
from pathlib import Path

from backgammon.training.train import (
    TrainingConfig,
    TrainingPhase,
    create_train_state,
    train,
)
from backgammon.training.replay_buffer import ReplayBuffer
from backgammon.training.self_play import generate_training_batch
from backgammon.core.board import get_early_training_variants
from backgammon.evaluation.agents import pip_count_agent, random_agent
from backgammon.evaluation.network_agent import create_neural_agent


class TestTrainingPhase:
    """Test training phase management."""

    def test_phase_progression(self):
        """Test that phases progress correctly."""
        config = TrainingConfig(
            warmstart_games=10,
            early_phase_games=20,
            mid_phase_games=30,
            late_phase_games=40,
        )

        phase_manager = TrainingPhase(config)

        # Initially warmstart
        assert phase_manager.total_games_played == 0
        phase, _ = phase_manager.get_current_phase()
        assert phase == "warmstart"
        assert phase_manager.should_use_pip_count_warmstart()

        # After 10 games -> early
        phase_manager.total_games_played = 10
        phase, _ = phase_manager.get_current_phase()
        assert phase == "early"
        assert not phase_manager.should_use_pip_count_warmstart()

        # After 30 games -> mid
        phase_manager.total_games_played = 30
        phase, _ = phase_manager.get_current_phase()
        assert phase == "mid"

        # After 60 games -> late
        phase_manager.total_games_played = 60
        phase, _ = phase_manager.get_current_phase()
        assert phase == "late"


class TestCreateTrainState:
    """Test training state creation."""

    def test_create_state(self):
        """Test creating training state."""
        config = TrainingConfig()
        rng = jax.random.PRNGKey(42)

        state = create_train_state(config, rng)

        # Check state structure
        assert state is not None
        assert hasattr(state, 'params')
        assert hasattr(state, 'apply_fn')
        assert hasattr(state, 'tx')
        assert hasattr(state, 'opt_state')

    def test_state_inference(self):
        """Test that state can perform inference."""
        config = TrainingConfig(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            train_policy=True,  # Enable policy for this test
        )
        rng = jax.random.PRNGKey(42)

        state = create_train_state(config, rng)

        # Create dummy input (26 positions x 2 features)
        dummy_input = jnp.zeros((1, 26, 2))

        # Run forward pass
        output = state.apply_fn(
            {'params': state.params},
            dummy_input,
            training=False,
        )

        # Check outputs (network returns tuple of (equity, policy_logits, attention_weights))
        equity, policy_logits, _ = output
        assert equity.shape == (1, 5)  # Equity prediction (5 outcomes)
        assert policy_logits.shape == (1, 1024)  # Action space size


class TestSelfPlayIntegration:
    """Test self-play with neural network agents."""

    def test_neural_agent_self_play(self):
        """Test that neural agent can play games."""
        # Create small network
        config = TrainingConfig(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            train_policy=True,  # Enable policy for neural agent
        )
        rng_key = jax.random.PRNGKey(42)
        state = create_train_state(config, rng_key)

        # Create neural agent
        neural_agent = create_neural_agent(
            state=state,
            temperature=0.3,
            name="TestAgent",
        )

        # Generate games
        rng = np.random.default_rng(42)
        games = generate_training_batch(
            num_games=3,
            get_variant_fn=get_early_training_variants,
            white_agent=neural_agent,
            black_agent=neural_agent,
            rng=rng,
        )

        # Verify games were played
        assert len(games) == 3
        for game in games:
            assert game.num_moves > 0
            assert len(game.steps) > 0


class TestReplayBufferIntegration:
    """Test replay buffer with neural network training."""

    def test_buffer_with_neural_games(self):
        """Test replay buffer with games from neural agents."""
        # Create network and agent
        config = TrainingConfig(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            train_policy=True,  # Enable policy for neural agent
        )
        rng_key = jax.random.PRNGKey(42)
        state = create_train_state(config, rng_key)

        neural_agent = create_neural_agent(state=state, temperature=0.5)

        # Generate games
        rng = np.random.default_rng(42)
        games = generate_training_batch(
            num_games=5,
            get_variant_fn=get_early_training_variants,
            white_agent=neural_agent,
            black_agent=pip_count_agent(),
            rng=rng,
        )

        # Add to replay buffer
        buffer = ReplayBuffer(max_size=1000, min_size=10)
        for game in games:
            buffer.add_game(game)

        # Verify buffer filled
        assert len(buffer) > 0
        assert buffer.is_ready()

        # Sample batch
        batch = buffer.sample_batch(16)

        # Verify batch structure
        assert batch['board_encoding'].shape[0] == 16
        assert batch['target_policy'].shape == (16, 1024)
        assert batch['equity_target'].shape == (16, 5)
        assert batch['action_mask'].shape == (16, 1024)


class TestEndToEndTraining:
    """Test complete training pipeline."""

    def test_minimal_training_run(self):
        """Test a minimal training run with all components."""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            # Create minimal config for fast test
            config = TrainingConfig(
                # Very small for testing
                warmstart_games=5,
                early_phase_games=5,
                mid_phase_games=5,
                late_phase_games=5,

                # Small batches
                games_per_batch=2,

                # Frequent checkpointing
                checkpoint_every_n_batches=5,
                log_every_n_batches=2,

                # Small network
                embed_dim=64,
                num_heads=4,
                num_layers=2,
                ff_dim=256,

                # Enable policy for neural self-play
                train_policy=True,

                # Replay buffer
                replay_buffer_size=100,
                replay_buffer_min_size=10,
                training_batch_size=8,
                train_steps_per_game_batch=2,

                # Paths
                checkpoint_dir=str(checkpoint_dir),
                log_dir=str(log_dir),

                # Seed for reproducibility
                seed=42,
            )

            # Run training
            train(config)

            # Verify training completed
            assert checkpoint_dir.exists()
            assert log_dir.exists()

            # Verify logs were created
            log_file = log_dir / "training_log.jsonl"
            assert log_file.exists()

            # Read and verify log entries
            with open(log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0

                # Verify log format
                import json
                first_entry = json.loads(lines[0])
                assert 'phase' in first_entry
                assert 'batch_num' in first_entry
                assert 'total_games' in first_entry
                assert 'loss' in first_entry

        finally:
            # Clean up
            shutil.rmtree(temp_dir)

    def test_warmstart_to_neural_transition(self):
        """Test transition from warmstart (pip count) to neural self-play."""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            # Config with clear warmstart/neural transition
            config = TrainingConfig(
                warmstart_games=4,  # 2 batches of warmstart
                early_phase_games=4,  # 2 batches of neural
                mid_phase_games=0,
                late_phase_games=0,

                games_per_batch=2,
                checkpoint_every_n_batches=10,
                log_every_n_batches=1,

                embed_dim=64,
                num_heads=4,
                num_layers=2,
                ff_dim=256,

                # Enable policy for neural self-play
                train_policy=True,

                replay_buffer_size=100,
                replay_buffer_min_size=5,
                training_batch_size=4,
                train_steps_per_game_batch=1,

                checkpoint_dir=str(checkpoint_dir),
                log_dir=str(log_dir),

                seed=42,
            )

            # Run training
            train(config)

            # Read logs
            log_file = log_dir / "training_log.jsonl"
            with open(log_file, 'r') as f:
                import json
                entries = [json.loads(line) for line in f]

            # Verify we have entries from both phases
            phases = set(entry['phase'] for entry in entries)
            assert 'warmstart' in phases
            assert 'early' in phases

            # Verify total games played
            final_entry = entries[-1]
            assert final_entry['total_games'] >= 8

        finally:
            shutil.rmtree(temp_dir)


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_checkpoint_saving(self):
        """Test that checkpoints are saved correctly."""
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            config = TrainingConfig(
                warmstart_games=6,
                early_phase_games=0,
                mid_phase_games=0,
                late_phase_games=0,

                games_per_batch=2,
                checkpoint_every_n_batches=2,  # Save every 2 batches
                log_every_n_batches=1,

                embed_dim=64,
                num_heads=4,
                num_layers=2,
                ff_dim=256,

                checkpoint_dir=str(checkpoint_dir),
                log_dir=str(log_dir),

                seed=42,
            )

            # Run training
            train(config)

            # Verify checkpoints exist
            assert checkpoint_dir.exists()
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_*"))
            assert len(checkpoint_files) > 0

        finally:
            shutil.rmtree(temp_dir)


class TestConfigurationVariations:
    """Test different configuration variations."""

    def test_value_only_mode(self):
        """Test value-only training mode (no policy head)."""
        config = TrainingConfig(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            train_policy=False,  # Value-only mode
        )
        rng = jax.random.PRNGKey(42)
        state = create_train_state(config, rng)

        # Verify state was created
        assert state is not None

        # Test forward pass
        dummy_input = jnp.zeros((1, 26, 2))
        output = state.apply_fn(
            {'params': state.params},
            dummy_input,
            training=False,
        )

        # Check outputs - policy should be None in value-only mode
        equity, policy, _ = output
        assert equity.shape == (1, 5)  # Equity prediction
        assert policy is None  # No policy in value-only mode

    def test_different_network_sizes(self):
        """Test with different network configurations."""
        for embed_dim in [32, 64]:
            config = TrainingConfig(
                warmstart_games=2,
                early_phase_games=0,
                mid_phase_games=0,
                late_phase_games=0,

                games_per_batch=1,

                embed_dim=embed_dim,
                ff_dim=embed_dim * 4,
                num_heads=2 if embed_dim == 32 else 4,
                num_layers=1,
                train_policy=True,  # Enable policy for this test

                checkpoint_every_n_batches=100,
            )

            rng = jax.random.PRNGKey(42)
            state = create_train_state(config, rng)

            # Verify state was created
            assert state is not None

            # Test forward pass
            dummy_input = jnp.zeros((1, 26, 2))
            output = state.apply_fn(
                {'params': state.params},
                dummy_input,
                training=False,
            )

            equity, policy, _ = output
            assert equity.shape == (1, 5)  # Equity prediction
            assert policy.shape == (1, 1024)  # Policy logits
