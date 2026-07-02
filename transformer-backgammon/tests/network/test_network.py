"""Tests for transformer network module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from backgammon.core.types import TransformerConfig, EncodedBoard
from backgammon.encoding import encode_board, raw_encoding_config
from backgammon.core.board import initial_board
from backgammon.network.network import (
    # Model classes
    BackgammonTransformer,

    # Config presets
    small_transformer_config,
    medium_transformer_config,
    large_transformer_config,

    # Utilities
    init_network,
    count_parameters,
    parameter_stats,

    # Forward pass
    forward,
    forward_batch,
)

# The authoritative loss functions and train_step live in training/losses.py
# (the duplicates that used to live in network.py were dead code, removed
# Mar 2026 — TODO item 106).
import optax
from flax.training import train_state as flax_train_state
from backgammon.training.losses import (
    train_step as losses_train_step,
    compute_metrics,
)


def _make_train_state(model, variables, learning_rate=1e-4):
    """Build a minimal TrainState around an initialized model."""
    return flax_train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate),
    )


def _make_batch(features, targets):
    """Build a value-only training batch dict for losses.train_step."""
    return {
        'board_encoding': features,
        'equity_target': targets,
        'target_policy': jnp.zeros((1,), dtype=jnp.float32),
        'action_mask': jnp.ones((1,), dtype=jnp.bool_),
    }


class TestTransformerConfig:
    """Tests for transformer configuration."""

    def test_small_config(self):
        """Test small configuration preset."""
        config = small_transformer_config()
        assert config.num_layers == 2
        assert config.embed_dim == 64
        assert config.num_heads == 4
        assert config.ff_dim == 256

    def test_medium_config(self):
        """Test medium configuration preset."""
        config = medium_transformer_config()
        assert config.num_layers == 4
        assert config.embed_dim == 128
        assert config.num_heads == 8
        assert config.ff_dim == 512

    def test_large_config(self):
        """Test large configuration preset."""
        config = large_transformer_config()
        assert config.num_layers == 8
        assert config.embed_dim == 256
        assert config.num_heads == 16
        assert config.ff_dim == 1024

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = TransformerConfig(
            num_layers=2,
            embed_dim=64,
            num_heads=4,
        )
        # Should not raise

        # Invalid: embed_dim not divisible by num_heads
        with pytest.raises(AssertionError):
            TransformerConfig(
                num_layers=2,
                embed_dim=65,  # Not divisible by 4
                num_heads=4,
            )


class TestNetworkInitialization:
    """Tests for network initialization."""

    def test_init_network_small(self):
        """Test initializing small network."""
        config = small_transformer_config()
        rng_key = jax.random.PRNGKey(0)

        model, params = init_network(config, rng_key)

        assert model is not None
        assert params is not None

    def test_init_network_medium(self):
        """Test initializing medium network."""
        config = medium_transformer_config()
        rng_key = jax.random.PRNGKey(0)

        model, params = init_network(config, rng_key)

        assert model is not None
        assert params is not None

    def test_parameter_counting(self):
        """Test parameter counting."""
        config = small_transformer_config()
        rng_key = jax.random.PRNGKey(0)

        model, params = init_network(config, rng_key)
        num_params = count_parameters(params)

        # Small network should have reasonable number of parameters
        assert num_params > 10000  # At least 10K
        assert num_params < 10000000  # Less than 10M

    def test_parameter_stats(self):
        """Test parameter statistics."""
        config = small_transformer_config()
        rng_key = jax.random.PRNGKey(0)

        model, params = init_network(config, rng_key)
        mean, std, max_abs = parameter_stats(params)

        # Parameters should be initialized with reasonable values
        assert abs(mean) < 1.0  # Mean close to 0
        assert std > 0.0  # Some variance
        assert std < 1.0  # Not too large
        assert max_abs < 5.0  # No extreme values


class TestForwardPass:
    """Tests for forward pass."""

    @pytest.fixture
    def setup_network(self):
        """Set up network and test data."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        rng_key = jax.random.PRNGKey(42)

        model, params = init_network(config, rng_key)

        # Create test encoded board
        encoding_config = raw_encoding_config()
        board = initial_board()
        encoded = encode_board(encoding_config, board)

        return model, params, encoded, rng_key

    def test_forward_single(self, setup_network):
        """Test forward pass with single board."""
        model, params, encoded, rng_key = setup_network

        equity, policy, cube_dec, attn_weights = forward(model, params, encoded, training=False)

        # Check output shapes
        assert equity.shape == (1, 6)  # Batch=1, 6 equity components
        assert policy is None  # Policy head not enabled

        # Equity should be valid probabilities
        assert jnp.all(equity >= 0.0)
        assert jnp.all(equity <= 1.0)

    def test_forward_batch(self, setup_network):
        """Test forward pass with batch."""
        model, params, _, rng_key = setup_network

        # Create batch of boards
        encoding_config = raw_encoding_config()
        boards = [initial_board() for _ in range(4)]
        from backgammon.encoding.encoder import encode_boards
        encoded_batch = encode_boards(encoding_config, boards)

        equity, policy, cube_dec, attn_weights = forward_batch(
            model, params, encoded_batch, training=False
        )

        # Check output shapes
        assert equity.shape == (4, 6)  # Batch=4, 6 equity components

    def test_forward_training_mode(self, setup_network):
        """Test forward pass in training mode with dropout."""
        model, params, encoded, rng_key = setup_network

        rng_key1, rng_key2 = jax.random.split(rng_key)

        equity1, _, _, _ = forward(model, params, encoded, training=True, rng_key=rng_key1)
        equity2, _, _, _ = forward(model, params, encoded, training=True, rng_key=rng_key2)

        # Different dropout keys should give slightly different results
        # (though for small network difference might be small)
        assert equity1.shape == equity2.shape == (1, 6)

    def test_attention_weights(self):
        """Test returning attention weights."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        config.return_attention_weights = True
        rng_key = jax.random.PRNGKey(0)

        model, params = init_network(config, rng_key)

        encoding_config = raw_encoding_config()
        board = initial_board()
        encoded = encode_board(encoding_config, board)

        equity, policy, cube_dec, attn_weights = forward(model, params, encoded, training=False)

        # Should return attention weights
        assert attn_weights is not None
        # Shape: [batch, num_layers, num_heads, seq_len, seq_len]
        assert attn_weights.shape == (1, config.num_layers, config.num_heads, 26, 26)


class TestTraining:
    """Tests for training via the authoritative losses.train_step."""

    @pytest.fixture
    def setup_training(self):
        """Set up training components."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        rng_key = jax.random.PRNGKey(42)

        model, variables = init_network(config, rng_key)
        state = _make_train_state(model, variables, learning_rate=1e-4)

        # Create training data
        encoding_config = raw_encoding_config()
        boards = [initial_board() for _ in range(8)]
        from backgammon.encoding.encoder import encode_boards
        encoded_batch = encode_boards(encoding_config, boards)
        features = jnp.array(encoded_batch.position_features, dtype=jnp.float32)

        # Create dummy targets
        targets = jnp.ones((8, 6), dtype=jnp.float32) / 6.0  # Uniform distribution

        return state, features, targets, rng_key

    def test_create_train_state(self, setup_training):
        """Test creating training state."""
        state, _, _, _ = setup_training

        assert state is not None
        assert state.params is not None
        assert state.opt_state is not None

    def test_train_step(self, setup_training):
        """Test single training step."""
        state, features, targets, rng_key = setup_training

        # Perform training step
        new_state, metrics = losses_train_step(
            state, _make_batch(features, targets), rng_key
        )

        # State should be updated
        assert new_state.step == state.step + 1

        # Loss should be finite
        loss_value = metrics['total_loss']
        assert jnp.isfinite(loss_value)
        assert loss_value > 0.0

        # Parameters should change
        old_params_flat = jax.tree_util.tree_leaves(state.params)
        new_params_flat = jax.tree_util.tree_leaves(new_state.params)

        # At least some parameters should have changed
        changed = any(
            not jnp.allclose(old, new, atol=1e-6)
            for old, new in zip(old_params_flat, new_params_flat)
        )
        assert changed, "Parameters should change after training step"

    def test_eval_metrics(self, setup_training):
        """Test evaluation metrics computation."""
        state, features, targets, _ = setup_training

        metrics = compute_metrics(state, _make_batch(features, targets))

        # Loss should be finite
        assert np.isfinite(metrics['loss'])
        assert metrics['loss'] > 0.0
        assert 'equity_accuracy' in metrics

    def test_multiple_train_steps(self, setup_training):
        """Test multiple training steps keep the loss finite."""
        state, features, targets, rng_key = setup_training
        batch = _make_batch(features, targets)

        initial_loss = compute_metrics(state, batch)['loss']

        # Train for several steps
        for i in range(10):
            rng_key, step_key = jax.random.split(rng_key)
            state, _ = losses_train_step(state, batch, step_key)

        final_loss = compute_metrics(state, batch)['loss']

        # Loss should decrease (or at least not increase significantly)
        # Note: With random initialization and few steps, might not always decrease
        # So we just check it's still finite and reasonable
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)


class TestModelComponents:
    """Tests for individual model components."""

    def test_transformer_output_shapes(self):
        """Test that transformer produces correct output shapes."""
        config = small_transformer_config()
        config.input_feature_dim = 2

        model = BackgammonTransformer(config=config)
        rng_key = jax.random.PRNGKey(0)

        # Create dummy input
        dummy_input = jnp.ones((2, 26, 2), dtype=jnp.float32)  # batch=2

        # Initialize and run
        params = model.init(rng_key, dummy_input, training=False)
        equity, policy, cube_dec, attn = model.apply(params, dummy_input, training=False)

        # Check shapes
        assert equity.shape == (2, 6)
        assert policy is None  # Not using policy head
        assert cube_dec is None  # Not using cube head

    def test_transformer_with_policy_head(self):
        """Test transformer with policy head enabled."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        config.use_policy_head = True

        model = BackgammonTransformer(config=config)
        rng_key = jax.random.PRNGKey(0)

        dummy_input = jnp.ones((1, 26, 2), dtype=jnp.float32)

        params = model.init(rng_key, dummy_input, training=False)
        equity, policy, cube_dec, attn = model.apply(params, dummy_input, training=False)

        # Both heads should produce output
        assert equity.shape == (1, 6)
        assert policy is not None
        assert policy.shape == (1, config.num_actions)
        assert cube_dec is None  # Not using cube head


class TestBFloat16:
    """Tests for bfloat16 mixed precision."""

    def test_bfloat16_forward_pass(self):
        """Test forward pass with bfloat16 compute dtype."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        config.dtype = jnp.bfloat16
        rng_key = jax.random.PRNGKey(42)

        model, params = init_network(config, rng_key)

        # Input in float32 (model casts internally)
        dummy_input = jnp.ones((4, 26, 2), dtype=jnp.float32)
        equity, policy, cube_dec, attn = model.apply(params, dummy_input, training=False)

        # Output should be float32 (cast at softmax/output heads)
        assert equity.dtype == jnp.float32
        assert equity.shape == (4, 6)
        assert jnp.all(jnp.isfinite(equity))

        # Equity should be valid probabilities
        assert jnp.all(equity >= 0.0)
        assert jnp.all(equity <= 1.0)

    def test_bfloat16_with_policy_head(self):
        """Test bfloat16 forward pass with policy head enabled."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        config.dtype = jnp.bfloat16
        config.use_policy_head = True
        rng_key = jax.random.PRNGKey(42)

        model, params = init_network(config, rng_key)

        dummy_input = jnp.ones((2, 26, 2), dtype=jnp.float32)
        equity, policy, cube_dec, attn = model.apply(params, dummy_input, training=False)

        # Both outputs should be float32
        assert equity.dtype == jnp.float32
        assert policy.dtype == jnp.float32
        assert equity.shape == (2, 6)
        assert policy.shape == (2, config.num_actions)

    def test_bfloat16_params_are_float32(self):
        """Verify parameters are stored in float32 even with bfloat16 compute."""
        config = small_transformer_config()
        config.dtype = jnp.bfloat16
        rng_key = jax.random.PRNGKey(0)

        model, params = init_network(config, rng_key)

        # All parameters should be float32
        for leaf in jax.tree_util.tree_leaves(params):
            assert leaf.dtype == jnp.float32, f"Parameter dtype {leaf.dtype} should be float32"

    def test_bfloat16_training_step(self):
        """Test that training works end-to-end with bfloat16."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        config.dtype = jnp.bfloat16
        rng_key = jax.random.PRNGKey(42)

        model, variables = init_network(config, rng_key)
        state = _make_train_state(model, variables, learning_rate=1e-4)

        features = jnp.ones((8, 26, 2), dtype=jnp.float32)
        targets = jnp.ones((8, 6), dtype=jnp.float32) / 6.0

        new_state, metrics = losses_train_step(
            state, _make_batch(features, targets), rng_key
        )
        assert jnp.isfinite(metrics['total_loss'])
        assert new_state.step == state.step + 1

    def test_float32_backward_compat(self):
        """Test that dtype=None (default) still works as float32."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        # dtype defaults to None
        assert config.dtype is None

        rng_key = jax.random.PRNGKey(42)
        model, params = init_network(config, rng_key)

        dummy_input = jnp.ones((2, 26, 2), dtype=jnp.float32)
        equity, _, _, _ = model.apply(params, dummy_input, training=False)
        assert equity.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(equity))


class TestV6eConfig:
    """Tests for v6e TPU configuration."""

    def test_v6e_quick_config(self):
        """Test v6e quick training config is valid."""
        from backgammon.training.train import v6e_quick_training_config

        config = v6e_quick_training_config()

        # Verify it's a small model
        assert config.embed_dim == 64
        assert config.num_layers == 2
        assert config.compute_dtype == 'bfloat16'

        # Verify total games is reasonable for a quick run
        total = (config.warmstart_games + config.early_phase_games +
                 config.mid_phase_games + config.late_phase_games)
        assert total <= 5000  # Should be a quick run

    def test_v6e_config_creates_valid_state(self):
        """Test that v6e config produces a working training state."""
        from backgammon.training.train import v6e_quick_training_config, create_train_state

        config = v6e_quick_training_config()
        rng = jax.random.PRNGKey(42)
        state = create_train_state(config, rng)

        assert state is not None
        assert state.params is not None

        # Verify all params are float32 (even though compute dtype is bfloat16)
        for leaf in jax.tree_util.tree_leaves(state.params):
            assert leaf.dtype == jnp.float32
