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


def compute_value_loss(
    value_pred: jnp.ndarray,
    value_target: jnp.ndarray,
) -> jnp.ndarray:
    """Compute value loss (MSE).

    Args:
        value_pred: Predicted values (batch_size,)
        value_target: Target values (batch_size,)

    Returns:
        Scalar loss
    """
    # Mean squared error
    return jnp.mean((value_pred - value_target) ** 2)


def compute_combined_loss(
    policy_logits: jnp.ndarray,
    value_pred: jnp.ndarray,
    target_policy: jnp.ndarray,
    value_target: jnp.ndarray,
    mask: jnp.ndarray,
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute combined policy + value loss.

    Args:
        policy_logits: Network policy logits
        value_pred: Network value predictions
        target_policy: Target policy
        value_target: Target values
        mask: Action mask
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    policy_loss = compute_policy_loss(policy_logits, target_policy, mask)
    value_loss = compute_value_loss(value_pred, value_target)

    total_loss = policy_weight * policy_loss + value_weight * value_loss

    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total_loss,
    }

    return total_loss, metrics


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    """Single training step with gradient computation and parameter update.

    Args:
        state: Current training state
        batch: Training batch with keys:
            - 'board_encoding': (batch_size, 26)
            - 'target_policy': (batch_size, num_actions)
            - 'value_target': (batch_size,)
            - 'action_mask': (batch_size, num_actions)
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss

    Returns:
        Tuple of (updated_state, metrics)
    """

    def loss_fn(params):
        """Compute loss for gradient computation."""
        # Forward pass
        logits = state.apply_fn(
            {'params': params},
            batch['board_encoding'],
            train=True,
        )

        # For now, assume logits are policy logits
        # In a full implementation, we'd have separate heads
        policy_logits = logits
        value_pred = jnp.zeros((logits.shape[0],))  # Placeholder

        # Compute loss
        loss, metrics = compute_combined_loss(
            policy_logits,
            value_pred,
            batch['target_policy'],
            batch['value_target'],
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
    targets computed from game outcomes.

    Args:
        game_results: List of completed games
        max_batch_size: Maximum batch size

    Returns:
        Training batch dictionary
    """
    # Extract all game steps
    all_steps = []
    for game in game_results:
        all_steps.extend(game.steps)

    # Limit batch size
    if len(all_steps) > max_batch_size:
        import random
        all_steps = random.sample(all_steps, max_batch_size)

    # Prepare batch data
    board_encodings = []
    target_policies = []
    value_targets = []
    action_masks = []

    for step in all_steps:
        # TODO: Encode board state
        # board_enc = encode_position(step.board, step.player)
        # board_encodings.append(board_enc)

        # TODO: Create target policy (e.g., from MCTS or move played)
        # target_policy = create_policy_target(step.move_taken, step.legal_moves)
        # target_policies.append(target_policy)

        # TODO: Compute value target from game outcome
        # value_target = compute_value_target(game.outcome, step.player)
        # value_targets.append(value_target)

        # TODO: Create action mask
        # mask = create_action_mask(step.legal_moves)
        # action_masks.append(mask)

        pass  # Placeholder

    # For now, return dummy batch
    batch_size = len(all_steps) if all_steps else 1
    return {
        'board_encoding': jnp.zeros((batch_size, 26)),
        'target_policy': jnp.zeros((batch_size, 256)),  # Placeholder action space
        'value_target': jnp.zeros((batch_size,)),
        'action_mask': jnp.ones((batch_size, 256)),
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
    # Forward pass
    logits = state.apply_fn(
        {'params': state.params},
        batch['board_encoding'],
        train=False,
    )

    policy_logits = logits
    value_pred = jnp.zeros((logits.shape[0],))

    # Compute accuracy (top-1)
    predictions = jnp.argmax(policy_logits, axis=-1)
    targets = jnp.argmax(batch['target_policy'], axis=-1)
    accuracy = jnp.mean(predictions == targets)

    # Compute loss
    loss, loss_metrics = compute_combined_loss(
        policy_logits,
        value_pred,
        batch['target_policy'],
        batch['value_target'],
        batch['action_mask'],
    )

    metrics = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'policy_loss': float(loss_metrics['policy_loss']),
        'value_loss': float(loss_metrics['value_loss']),
    }

    return metrics
