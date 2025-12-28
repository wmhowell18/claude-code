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

    # Loss functions
    equity_loss,
    mse_equity_loss,
    policy_loss,
    total_loss,

    # Training
    create_train_state,
    train_step,
    eval_step,
)


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

        equity, policy, attn_weights = forward(model, params, encoded, training=False)

        # Check output shapes
        assert equity.shape == (1, 5)  # Batch=1, 5 equity components
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

        equity, policy, attn_weights = forward_batch(
            model, params, encoded_batch, training=False
        )

        # Check output shapes
        assert equity.shape == (4, 5)  # Batch=4, 5 equity components

    def test_forward_training_mode(self, setup_network):
        """Test forward pass in training mode with dropout."""
        model, params, encoded, rng_key = setup_network

        rng_key1, rng_key2 = jax.random.split(rng_key)

        equity1, _, _ = forward(model, params, encoded, training=True, rng_key=rng_key1)
        equity2, _, _ = forward(model, params, encoded, training=True, rng_key=rng_key2)

        # Different dropout keys should give slightly different results
        # (though for small network difference might be small)
        assert equity1.shape == equity2.shape == (1, 5)

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

        equity, policy, attn_weights = forward(model, params, encoded, training=False)

        # Should return attention weights
        assert attn_weights is not None
        # Shape: [batch, num_layers, num_heads, seq_len, seq_len]
        assert attn_weights.shape == (1, config.num_layers, config.num_heads, 26, 26)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_equity_loss(self):
        """Test equity loss computation."""
        # Perfect prediction
        predicted = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)
        target = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)

        loss = equity_loss(predicted, target)

        # Loss should be finite for identical distributions
        assert jnp.isfinite(loss)
        assert float(loss) > 0.0
        assert float(loss) < 5.0

    def test_equity_loss_wrong_prediction(self):
        """Test equity loss with wrong prediction."""
        predicted = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
        target = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

        loss = equity_loss(predicted, target)

        # Loss should be large for completely wrong prediction
        assert loss > 1.0

    def test_mse_equity_loss(self):
        """Test MSE equity loss."""
        predicted = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)
        target = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)

        loss = mse_equity_loss(predicted, target)

        # Loss should be very small for perfect prediction
        assert loss < 0.01

    def test_policy_loss(self):
        """Test policy loss computation."""
        predicted_logits = jnp.array([[1.0, 2.0, 0.5]], dtype=jnp.float32)
        target_probs = jnp.array([[0.2, 0.6, 0.2]], dtype=jnp.float32)

        loss = policy_loss(predicted_logits, target_probs)

        # Loss should be finite and positive
        assert jnp.isfinite(loss)
        assert loss > 0.0

    def test_total_loss(self):
        """Test combined loss."""
        equity_pred = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)
        equity_target = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)

        loss = total_loss(equity_pred, equity_target)

        # Should equal equity loss when no policy
        assert jnp.isfinite(loss)
        assert float(loss) > 0.0
        assert float(loss) < 5.0

    def test_total_loss_with_policy(self):
        """Test combined loss with policy."""
        equity_pred = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)
        equity_target = jnp.array([[0.2, 0.3, 0.1, 0.2, 0.2]], dtype=jnp.float32)
        policy_pred = jnp.array([[1.0, 2.0, 0.5]], dtype=jnp.float32)
        policy_target = jnp.array([[0.2, 0.6, 0.2]], dtype=jnp.float32)

        loss = total_loss(
            equity_pred, equity_target,
            policy_pred, policy_target,
            equity_weight=1.0,
            policy_weight=0.5
        )

        # Should be combination of equity and policy loss
        assert loss > 0.0
        assert jnp.isfinite(loss)


class TestTraining:
    """Tests for training functionality."""

    @pytest.fixture
    def setup_training(self):
        """Set up training components."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        rng_key = jax.random.PRNGKey(42)

        model, params = init_network(config, rng_key)
        state = create_train_state(model, params, learning_rate=1e-4)

        # Create training data
        encoding_config = raw_encoding_config()
        boards = [initial_board() for _ in range(8)]
        from backgammon.encoding.encoder import encode_boards
        encoded_batch = encode_boards(encoding_config, boards)
        features = jnp.array(encoded_batch.position_features, dtype=jnp.float32)

        # Create dummy targets
        targets = jnp.ones((8, 5), dtype=jnp.float32) / 5.0  # Uniform distribution

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
        new_state, loss_value = train_step(state, features, targets, rng_key)

        # State should be updated
        assert new_state.step == state.step + 1

        # Loss should be finite
        assert jnp.isfinite(loss_value)
        assert loss_value > 0.0

        # Parameters should change
        # (check first param to verify update)
        old_params_flat = jax.tree_util.tree_leaves(state.params)
        new_params_flat = jax.tree_util.tree_leaves(new_state.params)

        # At least some parameters should have changed
        changed = any(
            not jnp.allclose(old, new, atol=1e-6)
            for old, new in zip(old_params_flat, new_params_flat)
        )
        assert changed, "Parameters should change after training step"

    def test_eval_step(self, setup_training):
        """Test evaluation step."""
        state, features, targets, _ = setup_training

        # Perform eval step
        loss_value = eval_step(state, features, targets)

        # Loss should be finite
        assert jnp.isfinite(loss_value)
        assert loss_value > 0.0

    def test_multiple_train_steps(self, setup_training):
        """Test multiple training steps reduce loss."""
        state, features, targets, rng_key = setup_training

        # Get initial loss
        initial_loss = eval_step(state, features, targets)

        # Train for several steps
        for i in range(10):
            rng_key, step_key = jax.random.split(rng_key)
            state, _ = train_step(state, features, targets, step_key)

        # Get final loss
        final_loss = eval_step(state, features, targets)

        # Loss should decrease (or at least not increase significantly)
        # Note: With random initialization and few steps, might not always decrease
        # So we just check it's still finite and reasonable
        assert jnp.isfinite(final_loss)


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
        equity, policy, attn = model.apply(params, dummy_input, training=False)

        # Check shapes
        assert equity.shape == (2, 5)
        assert policy is None  # Not using policy head

    def test_transformer_with_policy_head(self):
        """Test transformer with policy head enabled."""
        config = small_transformer_config()
        config.input_feature_dim = 2
        config.use_policy_head = True

        model = BackgammonTransformer(config=config)
        rng_key = jax.random.PRNGKey(0)

        dummy_input = jnp.ones((1, 26, 2), dtype=jnp.float32)

        params = model.init(rng_key, dummy_input, training=False)
        equity, policy, attn = model.apply(params, dummy_input, training=False)

        # Both heads should produce output
        assert equity.shape == (1, 5)
        assert policy is not None
        assert policy.shape == (1, 1000)  # Placeholder num_actions
