"""Transformer neural network architecture using JAX/Flax.

This module implements a transformer-based neural network for backgammon position evaluation.
The architecture consists of:
- Input projection from board features to embedding dimension
- Positional embeddings for the 26 board positions
- Stack of transformer blocks with multi-head self-attention
- Value head for equity prediction
- Optional policy head for move selection

The implementation uses JAX for GPU acceleration and Flax for neural network layers.
"""

from typing import Any, Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax

from backgammon.core.types import (
    EncodedBoard,
    Equity,
    TransformerConfig,
)


# ==============================================================================
# CONFIGURATION PRESETS
# ==============================================================================


def small_transformer_config() -> TransformerConfig:
    """Small transformer for testing and rapid iteration.

    Returns:
        TransformerConfig with ~500K parameters
    """
    return TransformerConfig(
        num_layers=2,
        embed_dim=64,
        num_heads=4,
        ff_dim=256,
        dropout_rate=0.1,
        input_feature_dim=2,
    )


def medium_transformer_config() -> TransformerConfig:
    """Medium transformer for baseline training.

    Returns:
        TransformerConfig with ~2M parameters
    """
    return TransformerConfig(
        num_layers=4,
        embed_dim=128,
        num_heads=8,
        ff_dim=512,
        dropout_rate=0.1,
        input_feature_dim=2,
    )


def large_transformer_config() -> TransformerConfig:
    """Large transformer for serious training runs.

    Returns:
        TransformerConfig with ~10M parameters
    """
    return TransformerConfig(
        num_layers=8,
        embed_dim=256,
        num_heads=16,
        ff_dim=1024,
        dropout_rate=0.1,
        input_feature_dim=2,
    )


# ==============================================================================
# TRANSFORMER COMPONENTS
# ==============================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer.

    Attributes:
        num_heads: Number of attention heads
        embed_dim: Embedding dimension
        dropout_rate: Dropout probability
        return_attention_weights: Whether to return attention weights
        dtype: Compute dtype (None for float32, jnp.bfloat16 for TPU)
    """
    num_heads: int
    embed_dim: int
    dropout_rate: float = 0.1
    return_attention_weights: bool = False
    dtype: Any = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]
            training: Whether in training mode (for dropout)

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, seq_len, embed_dim]
            - attention_weights: [batch, num_heads, seq_len, seq_len] or None
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.num_heads

        # Linear projections for Q, K, V
        qkv = nn.Dense(
            3 * self.embed_dim, dtype=self.dtype, param_dtype=jnp.float32, name='qkv',
        )(x)
        qkv = jnp.reshape(
            qkv,
            (batch_size, seq_len, 3, self.num_heads, head_dim)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale

        # Softmax
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Apply dropout
        if training:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights,
                deterministic=not training
            )

        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape and project
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [batch, seq_len, heads, head_dim]
        attn_output = jnp.reshape(attn_output, (batch_size, seq_len, self.embed_dim))
        output = nn.Dense(
            self.embed_dim, dtype=self.dtype, param_dtype=jnp.float32, name='out',
        )(attn_output)

        if self.return_attention_weights:
            return output, attn_weights
        else:
            return output, None


class FeedForward(nn.Module):
    """Feed-forward network used in transformer blocks.

    Attributes:
        embed_dim: Embedding dimension
        ff_dim: Hidden dimension
        dropout_rate: Dropout probability
        dtype: Compute dtype (None for float32, jnp.bfloat16 for TPU)
    """
    embed_dim: int
    ff_dim: int
    dropout_rate: float = 0.1
    dtype: Any = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply feed-forward network.

        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]
            training: Whether in training mode (for dropout)

        Returns:
            Output tensor of shape [batch, seq_len, embed_dim]
        """
        x = nn.Dense(self.ff_dim, dtype=self.dtype, param_dtype=jnp.float32, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.embed_dim, dtype=self.dtype, param_dtype=jnp.float32, name='fc2')(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward.

    Attributes:
        config: Transformer configuration
        return_attention_weights: Whether to return attention weights
    """
    config: TransformerConfig
    return_attention_weights: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Apply transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]
            training: Whether in training mode

        Returns:
            Tuple of (output, attention_weights)
        """
        dtype = self.config.dtype

        # Multi-head self-attention with residual connection
        attn_output, attn_weights = MultiHeadAttention(
            num_heads=self.config.num_heads,
            embed_dim=self.config.embed_dim,
            dropout_rate=self.config.dropout_rate,
            return_attention_weights=self.return_attention_weights,
            dtype=dtype,
            name='attention'
        )(x, training=training)

        x = x + attn_output
        # LayerNorm in float32 for numerical stability (bfloat16 variance can underflow)
        x = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon, dtype=jnp.float32, name='ln1',
        )(x)
        if dtype is not None:
            x = x.astype(dtype)

        # Feed-forward with residual connection
        ff_output = FeedForward(
            embed_dim=self.config.embed_dim,
            ff_dim=self.config.ff_dim,
            dropout_rate=self.config.dropout_rate,
            dtype=dtype,
            name='ff'
        )(x, training=training)

        x = x + ff_output
        x = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon, dtype=jnp.float32, name='ln2',
        )(x)
        if dtype is not None:
            x = x.astype(dtype)

        return x, attn_weights


# ==============================================================================
# OUTPUT HEADS
# ==============================================================================


class ValueHead(nn.Module):
    """Value head for equity prediction.

    Converts transformer output to equity probabilities.
    Uses global pooling followed by MLP to produce 5 equity values.

    Attributes:
        config: Transformer configuration
        pool_type: Pooling strategy ("mean", "max", "cls")
    """
    config: TransformerConfig
    pool_type: str = "mean"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict equity from transformer output.

        Args:
            x: Transformer output of shape [batch, seq_len, embed_dim]

        Returns:
            Equity probabilities of shape [batch, 5] (always float32)
            (win_normal, win_gammon, win_backgammon, lose_gammon, lose_backgammon)
        """
        dtype = self.config.dtype
        batch_size, seq_len, embed_dim = x.shape

        # Global pooling
        if self.pool_type == "mean":
            pooled = jnp.mean(x, axis=1)  # [batch, embed_dim]
        elif self.pool_type == "max":
            pooled = jnp.max(x, axis=1)  # [batch, embed_dim]
        elif self.pool_type == "cls":
            pooled = x[:, 0, :]  # Use first token [batch, embed_dim]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # MLP head
        x = nn.Dense(
            self.config.ff_dim // 2, dtype=dtype, param_dtype=jnp.float32, name='fc1',
        )(pooled)
        x = nn.gelu(x)
        x = nn.Dense(5, dtype=dtype, param_dtype=jnp.float32, name='equity')(x)

        # Cast to float32 before softmax for numerical stability
        x = x.astype(jnp.float32)
        x = nn.softmax(x, axis=-1)

        return x


class PolicyHead(nn.Module):
    """Policy head for move prediction (optional).

    Converts transformer output to move probabilities.

    Attributes:
        config: Transformer configuration
        num_actions: Number of possible actions
    """
    config: TransformerConfig
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict move probabilities from transformer output.

        Args:
            x: Transformer output of shape [batch, seq_len, embed_dim]

        Returns:
            Policy logits of shape [batch, num_actions] (always float32)
        """
        dtype = self.config.dtype
        # Global mean pooling
        pooled = jnp.mean(x, axis=1)  # [batch, embed_dim]

        # MLP head
        x = nn.Dense(
            self.config.ff_dim // 2, dtype=dtype, param_dtype=jnp.float32, name='fc1',
        )(pooled)
        x = nn.gelu(x)
        x = nn.Dense(
            self.num_actions, dtype=dtype, param_dtype=jnp.float32, name='policy',
        )(x)

        # Cast logits to float32 for stable cross-entropy
        return x.astype(jnp.float32)


class CubeHead(nn.Module):
    """Cube decision head for double/no-double/take/pass predictions.

    Converts transformer output to cube action probabilities.
    Output has 4 dimensions: [no_double, double, take, pass].

    Attributes:
        config: Transformer configuration
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict cube action probabilities from transformer output.

        Args:
            x: Transformer output of shape [batch, seq_len, embed_dim]

        Returns:
            Cube decision logits of shape [batch, 4] (always float32)
            (no_double, double, take, pass)
        """
        dtype = self.config.dtype
        # Global mean pooling
        pooled = jnp.mean(x, axis=1)  # [batch, embed_dim]

        # MLP head
        x = nn.Dense(
            self.config.ff_dim // 4, dtype=dtype, param_dtype=jnp.float32, name='fc1',
        )(pooled)
        x = nn.gelu(x)
        x = nn.Dense(4, dtype=dtype, param_dtype=jnp.float32, name='cube_decision')(x)

        # Cast logits to float32 for stable downstream computation
        return x.astype(jnp.float32)


# ==============================================================================
# MAIN TRANSFORMER MODEL
# ==============================================================================


class BackgammonTransformer(nn.Module):
    """Complete transformer model for backgammon position evaluation.

    Architecture:
    1. Input projection: board features -> embedding dimension
    2. Positional embeddings (learned or fixed)
    3. Stack of transformer blocks
    4. Value head for equity prediction
    5. Optional policy head for move selection

    Attributes:
        config: Transformer configuration
    """
    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Forward pass through the transformer.

        Args:
            x: Input features of shape [batch, seq_len=26, feature_dim]
            training: Whether in training mode (for dropout)

        Returns:
            Tuple of (equity, policy, cube_decision, attention_weights)
            - equity: [batch, 5] - equity predictions
            - policy: [batch, num_actions] or None
            - cube_decision: [batch, 4] or None (no_double, double, take, pass)
            - attention_weights: [batch, num_layers, num_heads, seq_len, seq_len] or None
        """
        batch_size, seq_len, feature_dim = x.shape
        assert seq_len == 26, f"Expected seq_len=26, got {seq_len}"
        dtype = self.config.dtype

        # Cast input to compute dtype
        if dtype is not None:
            x = x.astype(dtype)

        # Input projection
        x = nn.Dense(
            self.config.embed_dim, dtype=dtype, param_dtype=jnp.float32, name='input_proj',
        )(x)

        # Add positional embeddings
        if self.config.use_learned_positional_encoding:
            pos_embed = self.param(
                'pos_embed',
                nn.initializers.normal(stddev=0.02),
                (1, seq_len, self.config.embed_dim)
            )
            if dtype is not None:
                pos_embed = pos_embed.astype(dtype)
            x = x + pos_embed
        else:
            # Fixed sinusoidal positional encoding
            pos_embed = self._get_sinusoidal_encoding(seq_len, self.config.embed_dim)
            if dtype is not None:
                pos_embed = pos_embed.astype(dtype)
            x = x + pos_embed

        # Apply transformer blocks
        all_attention_weights = [] if self.config.return_attention_weights else None

        for i in range(self.config.num_layers):
            x, attn_weights = TransformerBlock(
                config=self.config,
                return_attention_weights=self.config.return_attention_weights,
                name=f'block_{i}'
            )(x, training=training)

            if attn_weights is not None:
                all_attention_weights.append(attn_weights)

        # Value head (equity prediction)
        equity = ValueHead(
            config=self.config,
            pool_type="mean",
            name='value_head'
        )(x)

        # Policy head (optional)
        policy = None
        if self.config.use_policy_head:
            policy = PolicyHead(
                config=self.config,
                num_actions=self.config.num_actions,
                name='policy_head'
            )(x)

        # Cube decision head (optional)
        cube_decision = None
        if self.config.use_cube_head:
            cube_decision = CubeHead(
                config=self.config,
                name='cube_head'
            )(x)

        # Stack attention weights
        if all_attention_weights:
            attention_weights = jnp.stack(all_attention_weights, axis=1)
        else:
            attention_weights = None

        return equity, policy, cube_decision, attention_weights

    def _get_sinusoidal_encoding(
        self,
        seq_len: int,
        embed_dim: int
    ) -> jnp.ndarray:
        """Generate sinusoidal positional encodings.

        Args:
            seq_len: Sequence length
            embed_dim: Embedding dimension

        Returns:
            Positional encodings of shape [1, seq_len, embed_dim]
        """
        position = jnp.arange(seq_len)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, embed_dim, 2) * -(jnp.log(10000.0) / embed_dim)
        )

        pe = jnp.zeros((seq_len, embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return jnp.expand_dims(pe, axis=0)  # [1, seq_len, embed_dim]


# ==============================================================================
# NETWORK UTILITIES
# ==============================================================================


def init_network(config: TransformerConfig, rng_key: jax.random.PRNGKey):
    """Initialize network with random weights.

    Args:
        config: Transformer configuration
        rng_key: JAX random key

    Returns:
        Tuple of (model, params) where model is the Flax module and params are the weights
    """
    model = BackgammonTransformer(config=config)

    # Create dummy input to initialize parameters
    dummy_input = jnp.ones((1, 26, config.input_feature_dim), dtype=jnp.float32)

    # Initialize parameters
    params = model.init(rng_key, dummy_input, training=False)

    return model, params


def count_parameters(params) -> int:
    """Count total number of parameters.

    Args:
        params: Model parameters (Flax params dict)

    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def parameter_stats(params) -> Tuple[float, float, float]:
    """Get parameter statistics.

    Args:
        params: Model parameters

    Returns:
        Tuple of (mean, std, max_abs_value)
    """
    flat_params = jax.tree_util.tree_leaves(params)
    all_params = jnp.concatenate([x.ravel() for x in flat_params])

    mean = float(jnp.mean(all_params))
    std = float(jnp.std(all_params))
    max_abs = float(jnp.max(jnp.abs(all_params)))

    return mean, std, max_abs


# ==============================================================================
# FORWARD PASS
# ==============================================================================


def forward(
    model: BackgammonTransformer,
    params,
    encoded_board: EncodedBoard,
    training: bool = False,
    rng_key: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Perform forward pass through the network.

    Args:
        model: Flax model
        params: Model parameters
        encoded_board: Encoded board state
        training: Whether in training mode
        rng_key: Random key (for dropout in training mode)

    Returns:
        Tuple of (equity, policy, cube_decision, attention_weights)
    """
    # Extract features as JAX array
    features = jnp.array(encoded_board.position_features, dtype=jnp.float32)

    # Forward pass
    if training and rng_key is not None:
        equity, policy, cube_decision, attention_weights = model.apply(
            params,
            features,
            training=training,
            rngs={'dropout': rng_key}
        )
    else:
        equity, policy, cube_decision, attention_weights = model.apply(
            params,
            features,
            training=training
        )

    return equity, policy, cube_decision, attention_weights


def forward_batch(
    model: BackgammonTransformer,
    params,
    encoded_boards: EncodedBoard,
    training: bool = False,
    rng_key: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Batched forward pass (GPU-optimized).

    Args:
        model: Flax model
        params: Model parameters
        encoded_boards: Batch of encoded board states
        training: Whether in training mode
        rng_key: Random key (for dropout)

    Returns:
        Tuple of (equity_batch, policy_batch, cube_decision_batch, attention_weights_batch)
    """
    return forward(model, params, encoded_boards, training, rng_key)


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================


def equity_loss(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute equity prediction loss (cross-entropy).

    Args:
        predicted: Predicted equity of shape [batch, 5]
        target: Target equity of shape [batch, 5]

    Returns:
        Average cross-entropy loss (scalar JAX array)
    """
    # Cross-entropy loss
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    loss = -jnp.sum(target * jnp.log(predicted + epsilon), axis=-1)
    return jnp.mean(loss)


def mse_equity_loss(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute equity prediction loss (MSE).

    Args:
        predicted: Predicted equity of shape [batch, 5]
        target: Target equity of shape [batch, 5]

    Returns:
        Mean squared error loss (scalar JAX array)
    """
    loss = jnp.mean((predicted - target) ** 2, axis=-1)
    return jnp.mean(loss)


def policy_loss(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute policy prediction loss (cross-entropy).

    Args:
        predicted: Predicted policy logits of shape [batch, num_actions]
        target: Target policy distribution of shape [batch, num_actions]

    Returns:
        Average cross-entropy loss (scalar JAX array)
    """
    # Apply softmax to predicted logits
    predicted_probs = nn.softmax(predicted, axis=-1)

    # Cross-entropy
    epsilon = 1e-10
    loss = -jnp.sum(target * jnp.log(predicted_probs + epsilon), axis=-1)
    return jnp.mean(loss)


def total_loss(
    equity_pred: jnp.ndarray,
    equity_target: jnp.ndarray,
    policy_pred: Optional[jnp.ndarray] = None,
    policy_target: Optional[jnp.ndarray] = None,
    equity_weight: float = 1.0,
    policy_weight: float = 0.5
) -> jnp.ndarray:
    """Compute combined loss.

    Args:
        equity_pred: Predicted equity [batch, 5]
        equity_target: Target equity [batch, 5]
        policy_pred: Predicted policy [batch, num_actions] or None
        policy_target: Target policy [batch, num_actions] or None
        equity_weight: Weight for equity loss
        policy_weight: Weight for policy loss

    Returns:
        Combined weighted loss (scalar JAX array)
    """
    loss = equity_weight * equity_loss(equity_pred, equity_target)

    if policy_pred is not None and policy_target is not None:
        loss += policy_weight * policy_loss(policy_pred, policy_target)

    return loss


# ==============================================================================
# TRAINING STATE
# ==============================================================================


def create_train_state(
    model: BackgammonTransformer,
    params,
    learning_rate: float
) -> train_state.TrainState:
    """Create training state with Adam optimizer.

    Args:
        model: Flax model
        params: Initial parameters
        learning_rate: Learning rate for optimizer

    Returns:
        Flax training state
    """
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jax.jit
def train_step(
    state: train_state.TrainState,
    features: jnp.ndarray,
    equity_targets: jnp.ndarray,
    rng_key: jax.random.PRNGKey
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """Perform single training step with gradient descent.

    Args:
        state: Training state
        features: Input features [batch, 26, feature_dim]
        equity_targets: Target equities [batch, 5]
        rng_key: Random key for dropout

    Returns:
        Tuple of (updated_state, loss_value)
    """
    def loss_fn(params):
        equity_pred, _, _, _ = state.apply_fn(
            params,
            features,
            training=True,
            rngs={'dropout': rng_key}
        )
        return equity_loss(equity_pred, equity_targets)

    loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss_value


@jax.jit
def eval_step(
    state: train_state.TrainState,
    features: jnp.ndarray,
    equity_targets: jnp.ndarray
) -> jnp.ndarray:
    """Perform evaluation step (no gradient computation).

    Args:
        state: Training state
        features: Input features [batch, 26, feature_dim]
        equity_targets: Target equities [batch, 5]

    Returns:
        Loss value (scalar JAX array)
    """
    equity_pred, _, _, _ = state.apply_fn(
        state.params,
        features,
        training=False
    )
    return equity_loss(equity_pred, equity_targets)
