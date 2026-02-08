"""Loss functions and training step for backgammon network.

Implements policy and value loss computation with JAX for GPU acceleration.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Dict
from flax.training import train_state

from backgammon.training.self_play import GameResult, GameStep


def compute_policy_loss(
    policy_logits: jnp.ndarray,
    target_policy: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute policy loss (cross-entropy).

    Args:
        policy_logits: Network policy logits (batch_size, num_actions)
        target_policy: Target policy distribution (batch_size, num_actions)
        mask: Mask for valid actions (batch_size, num_actions)

    Returns:
        Scalar loss
    """
    # Apply mask to logits (set invalid actions to -inf)
    masked_logits = jnp.where(mask, policy_logits, -1e9)

    # Log softmax for numerical stability
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)

    # Cross-entropy loss: -sum(target * log(pred))
    loss = -jnp.sum(target_policy * log_probs, axis=-1)

    # Average over batch
    return jnp.mean(loss)


def compute_equity_loss(
    equity_pred: jnp.ndarray,
    equity_target: jnp.ndarray,
) -> jnp.ndarray:
    """Compute equity loss (cross-entropy on 5-dim distribution).

    The network outputs a softmax over 5 equity outcomes. We use
    cross-entropy against the target equity distribution.

    Args:
        equity_pred: Predicted equity probabilities (batch_size, 5)
        equity_target: Target equity distribution (batch_size, 5)

    Returns:
        Scalar loss
    """
    # Cross-entropy: -sum(target * log(pred))
    epsilon = 1e-10
    loss = -jnp.sum(equity_target * jnp.log(equity_pred + epsilon), axis=-1)
    return jnp.mean(loss)


def compute_combined_loss(
    policy_logits: jnp.ndarray,
    equity_pred: jnp.ndarray,
    target_policy: jnp.ndarray,
    equity_target: jnp.ndarray,
    mask: jnp.ndarray,
    policy_weight: float = 1.0,
    equity_weight: float = 0.5,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute combined policy + equity loss.

    Args:
        policy_logits: Network policy logits (None for value-only mode)
        equity_pred: Network equity predictions (batch_size, 5)
        target_policy: Target policy (ignored in value-only mode)
        equity_target: Target equity distribution (batch_size, 5)
        mask: Action mask (ignored in value-only mode)
        policy_weight: Weight for policy loss
        equity_weight: Weight for equity loss

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    eq_loss = compute_equity_loss(equity_pred, equity_target)

    # Check if policy head is enabled
    if policy_logits is not None:
        # Combined policy + equity training
        pol_loss = compute_policy_loss(policy_logits, target_policy, mask)
        total_loss = policy_weight * pol_loss + equity_weight * eq_loss

        metrics = {
            'policy_loss': pol_loss,
            'equity_loss': eq_loss,
            'total_loss': total_loss,
        }
    else:
        # Equity-only training
        total_loss = equity_weight * eq_loss

        metrics = {
            'policy_loss': jnp.array(0.0),  # Placeholder
            'equity_loss': eq_loss,
            'total_loss': total_loss,
        }

    return total_loss, metrics


def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    """Single training step with gradient computation and parameter update.

    Args:
        state: Current training state
        batch: Training batch with keys:
            - 'board_encoding': (batch_size, 26, 2)
            - 'target_policy': (batch_size, num_actions)
            - 'value_target': (batch_size,)
            - 'action_mask': (batch_size, num_actions)
        rng: RNG key for dropout
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss

    Returns:
        Tuple of (updated_state, metrics)
    """

    def loss_fn(params):
        """Compute loss for gradient computation."""
        # Forward pass - network returns (equity, policy, attention_weights)
        # Pass RNG for dropout
        equity_pred, policy_logits, _ = state.apply_fn(
            {'params': params},
            batch['board_encoding'],
            training=True,
            rngs={'dropout': rng},
        )

        # Compute loss using equity distribution directly (5-dim cross-entropy)
        loss, metrics = compute_combined_loss(
            policy_logits,
            equity_pred,
            batch['target_policy'],
            batch['equity_target'],
            batch['action_mask'],
            policy_weight,
            value_weight,
        )

        return loss, metrics

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    # Add gradient norm to metrics
    grad_norm = optax.global_norm(grads)
    metrics['grad_norm'] = grad_norm

    return state, metrics


def prepare_training_batch(
    game_results: list[GameResult],
    max_batch_size: int = 256,
) -> Dict[str, jnp.ndarray]:
    """Prepare training batch from game results.

    Converts game trajectories into training examples with
    targets computed from game outcomes. Prefer using
    ReplayBuffer.sample_batch() instead for training.

    Args:
        game_results: List of completed games
        max_batch_size: Maximum batch size

    Returns:
        Training batch dictionary
    """
    from backgammon.encoding.encoder import encode_board, raw_encoding_config, outcome_to_equity
    from backgammon.encoding.action_encoder import encode_move_to_one_hot, create_action_mask

    encoding_config = raw_encoding_config()

    # Extract all (step, game) pairs so we can access the outcome
    all_items = []
    for game in game_results:
        if game.outcome is None:
            continue
        for step in game.steps:
            all_items.append((step, game.outcome))

    # Limit batch size
    if len(all_items) > max_batch_size:
        import random
        all_items = random.sample(all_items, max_batch_size)

    if not all_items:
        # Return minimal valid batch
        from backgammon.encoding.action_encoder import get_action_space_size
        action_size = get_action_space_size()
        return {
            'board_encoding': jnp.zeros((1, 26, encoding_config.feature_dim)),
            'target_policy': jnp.zeros((1, action_size)),
            'equity_target': jnp.zeros((1, 5)),
            'action_mask': jnp.ones((1, action_size)),
        }

    # Prepare batch data
    board_encodings = []
    target_policies = []
    equity_targets = []
    action_masks = []

    for step, outcome in all_items:
        # Encode board state
        encoded = encode_board(encoding_config, step.board)
        board_encodings.append(encoded.position_features[0])

        # Create target policy from move played
        target_policy = encode_move_to_one_hot(step.move_taken, step.legal_moves)
        target_policies.append(target_policy)

        # Compute equity target from game outcome
        equity = outcome_to_equity(outcome, step.player)
        equity_targets.append(equity.to_array())

        # Create action mask
        mask = create_action_mask(step.legal_moves)
        action_masks.append(mask)

    return {
        'board_encoding': jnp.array(board_encodings, dtype=jnp.float32),
        'target_policy': jnp.array(target_policies, dtype=jnp.float32),
        'equity_target': jnp.array(equity_targets, dtype=jnp.float32),
        'action_mask': jnp.array(action_masks, dtype=jnp.bool_),
    }


def compute_metrics(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
) -> Dict[str, float]:
    """Compute evaluation metrics on a batch.

    Args:
        state: Training state
        batch: Evaluation batch

    Returns:
        Dictionary of metrics
    """
    # Forward pass - network returns (equity, policy, attention_weights)
    equity_pred, policy_logits, _ = state.apply_fn(
        {'params': state.params},
        batch['board_encoding'],
        training=False,
    )

    # Compute loss using equity distribution directly
    loss, loss_metrics = compute_combined_loss(
        policy_logits,
        equity_pred,
        batch['target_policy'],
        batch['equity_target'],
        batch['action_mask'],
    )

    metrics = {
        'loss': float(loss),
        'policy_loss': float(loss_metrics['policy_loss']),
        'equity_loss': float(loss_metrics['equity_loss']),
    }

    # Add policy accuracy if policy head is enabled
    if policy_logits is not None:
        predictions = jnp.argmax(policy_logits, axis=-1)
        targets = jnp.argmax(batch['target_policy'], axis=-1)
        accuracy = jnp.mean(predictions == targets)
        metrics['accuracy'] = float(accuracy)
    else:
        metrics['accuracy'] = 0.0  # Placeholder for equity-only mode

    return metrics
