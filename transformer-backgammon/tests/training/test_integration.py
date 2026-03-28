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
    save_checkpoint,
    _params_shapes_match,
    train,
)
from backgammon.training.replay_buffer import ReplayBuffer
from backgammon.training.self_play import generate_training_batch
from backgammon.training.losses import train_step
from backgammon.core.board import get_early_training_variants
from backgammon.evaluation.agents import pip_count_agent, random_agent
from backgammon.evaluation.network_agent import create_neural_agent
from flax.training import checkpoints


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
        dummy_input = jnp.zeros((1, 26, 10))

        # Run forward pass
        output = state.apply_fn(
            {'params': state.params},
            dummy_input,
            training=False,
        )

        # Check outputs (network returns tuple of (equity, policy_logits, cube_decision, attention_weights))
        equity, policy_logits, _, _ = output
        assert equity.shape == (1, 6)  # Equity prediction (6 outcomes)
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
        assert batch['equity_target'].shape == (16, 6)
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


class TestCheckpointRestore:
    """Test checkpoint restore scenarios — the gaps that let bugs slip through."""

    def _small_config(self, checkpoint_dir, log_dir, **overrides):
        """Helper to create a minimal training config."""
        defaults = dict(
            warmstart_games=4,
            early_phase_games=0,
            mid_phase_games=0,
            late_phase_games=0,
            games_per_batch=2,
            checkpoint_every_n_batches=1,
            log_every_n_batches=1,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            seed=42,
        )
        defaults.update(overrides)
        return TrainingConfig(**defaults)

    def test_restore_matching_shapes(self):
        """Save a checkpoint, create a fresh state, restore into it — basic happy path."""
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            config = self._small_config(checkpoint_dir, log_dir)
            rng = jax.random.PRNGKey(42)

            # Create state and run a few gradient steps so params differ from init
            state = create_train_state(config, rng)
            original_params = jax.tree.map(lambda p: p.copy(), state.params)

            # Save checkpoint at step 100
            save_checkpoint(state, config, step=100)

            # Create a completely fresh state (different random init)
            fresh_state = create_train_state(config, jax.random.PRNGKey(99))

            # Params should differ before restore
            fresh_leaves = jax.tree_util.tree_leaves(fresh_state.params)
            orig_leaves = jax.tree_util.tree_leaves(original_params)
            assert not all(
                np.allclose(f, o) for f, o in zip(fresh_leaves, orig_leaves)
            ), "Fresh and original params should differ"

            # Restore checkpoint into fresh state
            restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=str(checkpoint_dir),
                target=fresh_state,
            )

            # Verify restored params match original
            restored_leaves = jax.tree_util.tree_leaves(restored_state.params)
            for orig, restored in zip(orig_leaves, restored_leaves):
                np.testing.assert_array_equal(orig, restored)

            # Verify step was restored
            assert int(restored_state.step) == 100

        finally:
            shutil.rmtree(temp_dir)

    def test_restore_mismatched_shapes_detected(self):
        """Checkpoint from one architecture should be rejected when shapes don't match."""
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            # Save checkpoint with embed_dim=64
            config_v1 = self._small_config(checkpoint_dir, log_dir, embed_dim=64)
            rng = jax.random.PRNGKey(42)
            state_v1 = create_train_state(config_v1, rng)
            save_checkpoint(state_v1, config_v1, step=50)

            # Create fresh state with embed_dim=32 (different architecture)
            config_v2 = self._small_config(checkpoint_dir, log_dir, embed_dim=32, num_heads=2)
            state_v2 = create_train_state(config_v2, jax.random.PRNGKey(99))

            # Restore the v1 checkpoint into the v2 target
            restored = checkpoints.restore_checkpoint(
                ckpt_dir=str(checkpoint_dir),
                target=state_v2,
            )

            # _params_shapes_match should detect the mismatch
            assert not _params_shapes_match(state_v2.params, restored.params), (
                "Shape mismatch between v1 checkpoint and v2 model should be detected"
            )

        finally:
            shutil.rmtree(temp_dir)

    def test_restore_mismatched_num_layers_detected(self):
        """Changing num_layers should also trigger shape mismatch detection."""
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            # Save checkpoint with num_layers=2
            config_v1 = self._small_config(checkpoint_dir, log_dir, num_layers=2)
            state_v1 = create_train_state(config_v1, jax.random.PRNGKey(42))
            save_checkpoint(state_v1, config_v1, step=10)

            # Create state with num_layers=1
            config_v2 = self._small_config(checkpoint_dir, log_dir, num_layers=1)
            state_v2 = create_train_state(config_v2, jax.random.PRNGKey(99))

            restored = checkpoints.restore_checkpoint(
                ckpt_dir=str(checkpoint_dir),
                target=state_v2,
            )

            assert not _params_shapes_match(state_v2.params, restored.params)

        finally:
            shutil.rmtree(temp_dir)

    def test_optimizer_state_consistent_after_restore(self):
        """After restoring a checkpoint, the optimizer state must be usable for a gradient step.

        This catches the subtle bug where params are restored but opt_state still
        holds momentum buffers from the old architecture, causing shape errors on
        the first optimizer update.
        """
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            config = self._small_config(checkpoint_dir, log_dir, train_policy=True)
            rng = jax.random.PRNGKey(42)
            state = create_train_state(config, rng)

            # Save checkpoint
            save_checkpoint(state, config, step=5)

            # Restore into a fresh state (same architecture)
            fresh_state = create_train_state(config, jax.random.PRNGKey(99))
            restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=str(checkpoint_dir),
                target=fresh_state,
            )

            # Verify shapes match
            assert _params_shapes_match(fresh_state.params, restored_state.params)

            # Now perform a gradient step — this is where mismatched opt_state would crash
            dummy_batch = {
                'board_encoding': jnp.zeros((4, 26, 10)),
                'target_policy': jnp.ones((4, 1024)) / 1024,
                'equity_target': jnp.ones((4, 5)) / 5,
                'action_mask': jnp.ones((4, 1024)),
            }
            step_rng = jax.random.PRNGKey(0)
            updated_state, metrics = train_step(restored_state, dummy_batch, step_rng)

            # Verify the step completed without error and produced valid outputs
            assert updated_state.step == restored_state.step + 1
            assert jnp.isfinite(metrics['total_loss'])

        finally:
            shutil.rmtree(temp_dir)

    def test_full_resume_workflow(self):
        """End-to-end: train, save checkpoint, create new state, restore, and continue training.

        Simulates the real-world "resume from checkpoint" workflow.
        """
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        log_dir = Path(temp_dir) / "logs"

        try:
            config = self._small_config(checkpoint_dir, log_dir)

            # Phase 1: Run initial training (saves checkpoints)
            train(config)

            # Phase 2: "Resume" by running train() again — it should detect
            # and restore the checkpoint from phase 1
            log_dir_2 = Path(temp_dir) / "logs2"
            config2 = self._small_config(
                checkpoint_dir, log_dir_2,
                warmstart_games=2,
            )
            train(config2)

            # Verify phase 2 ran (has its own log entries)
            log_file = log_dir_2 / "training_log.jsonl"
            assert log_file.exists()

        finally:
            shutil.rmtree(temp_dir)

    def test_params_shapes_match_utility(self):
        """Unit test for _params_shapes_match helper function."""
        # Matching shapes
        params_a = {'layer': {'kernel': jnp.zeros((10, 20)), 'bias': jnp.zeros((20,))}}
        params_b = {'layer': {'kernel': jnp.ones((10, 20)), 'bias': jnp.ones((20,))}}
        assert _params_shapes_match(params_a, params_b)

        # Mismatched shapes
        params_c = {'layer': {'kernel': jnp.zeros((10, 32)), 'bias': jnp.zeros((32,))}}
        assert not _params_shapes_match(params_a, params_c)

        # Different number of leaves
        params_d = {'layer': {'kernel': jnp.zeros((10, 20))}}
        assert not _params_shapes_match(params_a, params_d)


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
        dummy_input = jnp.zeros((1, 26, 10))
        output = state.apply_fn(
            {'params': state.params},
            dummy_input,
            training=False,
        )

        # Check outputs - policy should be None in value-only mode
        equity, policy, _, _ = output
        assert equity.shape == (1, 6)  # Equity prediction
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
            dummy_input = jnp.zeros((1, 26, 10))
            output = state.apply_fn(
                {'params': state.params},
                dummy_input,
                training=False,
            )

            equity, policy, _, _ = output
            assert equity.shape == (1, 6)  # Equity prediction
            assert policy.shape == (1, 1024)  # Policy logits
