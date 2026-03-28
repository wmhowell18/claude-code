"""Stochastic MuZero network architecture for backgammon.

Implements the Stochastic MuZero framework (Antonoglou et al., 2022) adapted
for backgammon. The key insight: instead of separating neural network training
from search, learn a world model that predicts the next state, then use MCTS
with that model. The "stochastic" extension handles dice rolls as chance nodes.

Architecture:
  - Representation network h: observation -> hidden state
  - Dynamics network g: (hidden_state, action) -> (next_hidden_state, reward)
  - Prediction network f: hidden_state -> (policy, value)
  - Chance network c: hidden_state -> chance_outcome_distribution (dice probs)
  - Afterstate dynamics a: (hidden_state, chance_outcome) -> next_hidden_state

Reference: "Planning in Stochastic Environments with a Learned Model"
           (Antonoglou et al., 2022)
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

# 21 distinct dice outcomes: (1,1)..(6,6) with doubles counted once,
# non-doubles unordered. Matches standard backgammon dice probability.
NUM_DICE_OUTCOMES = 21

# Maximum action space size (from action encoder)
DEFAULT_NUM_ACTIONS = 1024


@dataclass
class MuZeroConfig:
    """Configuration for the Stochastic MuZero network.

    Attributes:
        hidden_dim: Hidden state dimension.
        num_actions: Size of the action space.
        num_chance_outcomes: Number of stochastic outcomes (21 for 2d6).
        representation_layers: Number of MLP layers in representation network.
        dynamics_layers: Number of MLP layers in dynamics network.
        prediction_layers: Number of MLP layers in prediction network.
        support_size: Size of categorical value support (value is represented
            as a distribution over [-support_size, support_size]).
        dropout_rate: Dropout probability.
        dtype: Compute dtype (None=float32, jnp.bfloat16 for TPU).
    """
    hidden_dim: int = 256
    num_actions: int = DEFAULT_NUM_ACTIONS
    num_chance_outcomes: int = NUM_DICE_OUTCOMES
    representation_layers: int = 2
    dynamics_layers: int = 2
    prediction_layers: int = 2
    support_size: int = 10
    dropout_rate: float = 0.1
    dtype: Any = None


class ResBlock(nn.Module):
    """Residual MLP block with RMSNorm."""
    dim: int
    dtype: Any = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        residual = x
        x = nn.RMSNorm(name='norm')(x)
        x = nn.Dense(self.dim, dtype=self.dtype, param_dtype=jnp.float32, name='fc1')(x)
        x = nn.silu(x)
        x = nn.Dense(self.dim, dtype=self.dtype, param_dtype=jnp.float32, name='fc2')(x)
        return residual + x


class RepresentationNetwork(nn.Module):
    """Encodes a raw observation (board features) into a hidden state.

    h: observation -> hidden_state
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(
        self, observation: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        """Encode observation to hidden state.

        Args:
            observation: Board features [batch, 26, feature_dim].

        Returns:
            Hidden state [batch, hidden_dim].
        """
        batch_size = observation.shape[0]
        # Flatten spatial board features
        x = observation.reshape(batch_size, -1)
        # Project to hidden dim
        x = nn.Dense(
            self.config.hidden_dim, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='proj',
        )(x)
        x = nn.silu(x)

        for i in range(self.config.representation_layers):
            x = ResBlock(self.config.hidden_dim, dtype=self.config.dtype, name=f'res_{i}')(
                x, training=training
            )

        # Normalize hidden state to unit scale (helps stability)
        x = nn.RMSNorm(name='out_norm')(x)
        return x


class DynamicsNetwork(nn.Module):
    """Predicts next hidden state and reward given action.

    g: (hidden_state, action) -> (next_hidden_state, reward)
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict next state and reward.

        Args:
            hidden_state: Current hidden state [batch, hidden_dim].
            action: One-hot action [batch, num_actions].

        Returns:
            Tuple of (next_hidden_state, reward_logits).
            - next_hidden_state: [batch, hidden_dim]
            - reward_logits: [batch, 2*support_size+1] categorical reward
        """
        # Project action to same dim as hidden state
        action_embed = nn.Dense(
            self.config.hidden_dim, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='action_proj',
        )(action)

        # Combine state and action
        x = hidden_state + action_embed

        for i in range(self.config.dynamics_layers):
            x = ResBlock(self.config.hidden_dim, dtype=self.config.dtype, name=f'res_{i}')(
                x, training=training
            )

        next_state = nn.RMSNorm(name='state_norm')(x)

        # Reward prediction (categorical)
        support_dim = 2 * self.config.support_size + 1
        reward_logits = nn.Dense(
            support_dim, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='reward',
        )(x).astype(jnp.float32)

        return next_state, reward_logits


class PredictionNetwork(nn.Module):
    """Predicts policy and value from hidden state.

    f: hidden_state -> (policy, value)
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(
        self, hidden_state: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict policy and value.

        Args:
            hidden_state: [batch, hidden_dim].

        Returns:
            Tuple of (policy_logits, value_logits).
            - policy_logits: [batch, num_actions]
            - value_logits: [batch, 2*support_size+1] categorical value
        """
        x = hidden_state
        for i in range(self.config.prediction_layers):
            x = ResBlock(self.config.hidden_dim, dtype=self.config.dtype, name=f'res_{i}')(
                x, training=training
            )

        # Policy head
        policy_logits = nn.Dense(
            self.config.num_actions, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='policy',
        )(x).astype(jnp.float32)

        # Value head (categorical distribution over support)
        support_dim = 2 * self.config.support_size + 1
        value_logits = nn.Dense(
            support_dim, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='value',
        )(x).astype(jnp.float32)

        return policy_logits, value_logits


class ChanceNetwork(nn.Module):
    """Predicts distribution over stochastic outcomes (dice rolls).

    c: hidden_state -> chance_outcome_distribution

    For backgammon, this predicts probabilities over 21 dice outcomes.
    In a perfect model this should learn the uniform distribution (each
    non-double has P=2/36, each double has P=1/36), but the model may
    learn position-dependent biases that improve planning.
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(
        self, hidden_state: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        """Predict chance outcome distribution.

        Args:
            hidden_state: [batch, hidden_dim].

        Returns:
            Chance logits [batch, num_chance_outcomes].
        """
        x = nn.Dense(
            self.config.hidden_dim // 2, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='fc1',
        )(hidden_state)
        x = nn.silu(x)
        x = nn.Dense(
            self.config.num_chance_outcomes, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='chance',
        )(x).astype(jnp.float32)
        return x


class AfterstateDynamics(nn.Module):
    """Transitions from afterstate to next state given chance outcome.

    a: (afterstate, chance_outcome) -> next_hidden_state

    In backgammon: after a player moves, dice are rolled (chance event),
    then the opponent faces a new state.
    """
    config: MuZeroConfig

    @nn.compact
    def __call__(
        self,
        afterstate: jnp.ndarray,
        chance_outcome: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """Apply chance outcome to afterstate.

        Args:
            afterstate: Hidden state after action [batch, hidden_dim].
            chance_outcome: One-hot chance outcome [batch, num_chance_outcomes].

        Returns:
            Next hidden state [batch, hidden_dim].
        """
        chance_embed = nn.Dense(
            self.config.hidden_dim, dtype=self.config.dtype,
            param_dtype=jnp.float32, name='chance_proj',
        )(chance_outcome)

        x = afterstate + chance_embed

        for i in range(self.config.dynamics_layers):
            x = ResBlock(self.config.hidden_dim, dtype=self.config.dtype, name=f'res_{i}')(
                x, training=training
            )

        return nn.RMSNorm(name='state_norm')(x)


class StochasticMuZeroNetwork(nn.Module):
    """Complete Stochastic MuZero network for backgammon.

    Combines all sub-networks into a single module for joint training.
    """
    config: MuZeroConfig

    def setup(self):
        self.representation = RepresentationNetwork(self.config, name='repr')
        self.dynamics = DynamicsNetwork(self.config, name='dynamics')
        self.prediction = PredictionNetwork(self.config, name='prediction')
        self.chance = ChanceNetwork(self.config, name='chance')
        self.afterstate = AfterstateDynamics(self.config, name='afterstate')

    def initial_inference(
        self, observation: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run initial inference from a raw observation.

        Args:
            observation: Board features [batch, 26, feature_dim].

        Returns:
            Tuple of (hidden_state, policy_logits, value_logits, chance_logits).
        """
        hidden_state = self.representation(observation, training=training)
        policy_logits, value_logits = self.prediction(hidden_state, training=training)
        chance_logits = self.chance(hidden_state, training=training)
        return hidden_state, policy_logits, value_logits, chance_logits

    def recurrent_inference(
        self,
        hidden_state: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run recurrent inference: apply action, get predictions.

        Args:
            hidden_state: Current hidden state [batch, hidden_dim].
            action: One-hot action [batch, num_actions].

        Returns:
            Tuple of (next_state, reward_logits, policy_logits, value_logits, chance_logits).
        """
        next_state, reward_logits = self.dynamics(hidden_state, action, training=training)
        policy_logits, value_logits = self.prediction(next_state, training=training)
        chance_logits = self.chance(next_state, training=training)
        return next_state, reward_logits, policy_logits, value_logits, chance_logits

    def chance_inference(
        self,
        afterstate: jnp.ndarray,
        chance_outcome: jnp.ndarray,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply a chance outcome (dice roll) to an afterstate.

        Args:
            afterstate: Hidden state after player action [batch, hidden_dim].
            chance_outcome: One-hot dice outcome [batch, num_chance_outcomes].

        Returns:
            Tuple of (next_state, policy_logits, value_logits, chance_logits).
        """
        next_state = self.afterstate(afterstate, chance_outcome, training=training)
        policy_logits, value_logits = self.prediction(next_state, training=training)
        chance_logits = self.chance(next_state, training=training)
        return next_state, policy_logits, value_logits, chance_logits

    def __call__(
        self, observation: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass (delegates to initial_inference for compatibility)."""
        return self.initial_inference(observation, training=training)


# ==============================================================================
# VALUE SUPPORT UTILITIES
# ==============================================================================


def scalar_to_support(scalar: jnp.ndarray, support_size: int) -> jnp.ndarray:
    """Convert scalar values to categorical support distribution.

    Maps a scalar value to a distribution over [-support_size, support_size]
    using a two-hot encoding (distributes mass between two adjacent bins).

    Args:
        scalar: Scalar values [batch].
        support_size: Half-width of the support.

    Returns:
        Support distribution [batch, 2*support_size+1].
    """
    # Clip to support range
    scalar = jnp.clip(scalar, -support_size, support_size)
    # Shift to [0, 2*support_size]
    shifted = scalar + support_size
    # Lower and upper bin indices
    lower = jnp.floor(shifted).astype(jnp.int32)
    upper = lower + 1
    # Fractional part determines weight distribution
    upper_weight = shifted - lower.astype(jnp.float32)
    lower_weight = 1.0 - upper_weight

    support_dim = 2 * support_size + 1
    batch_size = scalar.shape[0]

    # Clamp upper to valid range
    upper = jnp.minimum(upper, support_dim - 1)

    # Build two-hot distribution
    dist = jnp.zeros((batch_size, support_dim))
    dist = dist.at[jnp.arange(batch_size), lower].add(lower_weight)
    dist = dist.at[jnp.arange(batch_size), upper].add(upper_weight)
    return dist


def support_to_scalar(logits: jnp.ndarray, support_size: int) -> jnp.ndarray:
    """Convert categorical support logits to scalar values.

    Args:
        logits: Support logits [batch, 2*support_size+1].
        support_size: Half-width of the support.

    Returns:
        Scalar values [batch].
    """
    probs = jax.nn.softmax(logits, axis=-1)
    support = jnp.arange(-support_size, support_size + 1, dtype=jnp.float32)
    return jnp.sum(probs * support, axis=-1)


# ==============================================================================
# DICE OUTCOME ENCODING
# ==============================================================================

def get_dice_outcomes():
    """Get all 21 distinct dice outcomes for backgammon.

    Returns:
        List of (die1, die2) tuples where die1 <= die2.
    """
    outcomes = []
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            outcomes.append((d1, d2))
    return outcomes


def dice_to_index(die1: int, die2: int) -> int:
    """Convert a dice roll to its index in the 21-outcome space.

    Args:
        die1: First die (1-6).
        die2: Second die (1-6).

    Returns:
        Index in [0, 20].
    """
    d1, d2 = min(die1, die2), max(die1, die2)
    # Map (d1, d2) where d1 <= d2 to a flat index
    # (1,1)=0, (1,2)=1, ..., (1,6)=5, (2,2)=6, ..., (6,6)=20
    idx = 0
    for i in range(1, d1):
        idx += 7 - i
    idx += d2 - d1
    return idx


def get_true_dice_probs() -> jnp.ndarray:
    """Get the true probability distribution over 21 dice outcomes.

    Non-doubles have P=2/36, doubles have P=1/36.

    Returns:
        Probability vector of shape [21].
    """
    probs = []
    for d1, d2 in get_dice_outcomes():
        if d1 == d2:
            probs.append(1.0 / 36.0)
        else:
            probs.append(2.0 / 36.0)
    return jnp.array(probs, dtype=jnp.float32)
