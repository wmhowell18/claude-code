"""Experience replay buffer for training data management.

Stores game trajectories and samples batches for training.
Breaks temporal correlation and improves sample efficiency.

Performance: Boards are encoded into a pre-allocated numpy array at
insertion time so that sample_batch() is a single vectorized gather +
JAX array conversion — no Python-level per-sample work and no retained
Board objects.

Note on color symmetry: the encoder canonicalizes every board to the
mover's perspective (mirrored point axis for Black), so White/Black
symmetry is built into the input representation itself. The old
"color-flip augmentation" is therefore unnecessary — and its previous
implementation was actively harmful: it stored the win/lose-swapped
equity target for a position that is strategically identical from the
mover's perspective, corrupting half the training labels.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import random

from backgammon.training.self_play import GameResult, GameStep
from backgammon.encoding.encoder import (
    encode_board_canonical,
    outcome_to_equity,
)

# Encoded board shape: 26 positions x (2 raw + 8 global) features
_ENCODED_SHAPE = (26, 10)
_EQUITY_DIM = 6


def _encode_board_fast_single(board) -> np.ndarray:
    """Canonical single-board encoding (mover perspective, feature_dim=10)."""
    return encode_board_canonical(board)


@dataclass
class ReplayBuffer:
    """Experience replay buffer for training data.

    Stores game trajectories and provides random sampling for training.
    This breaks temporal correlation and improves training stability.

    Args:
        max_size: Maximum number of game steps to store
        min_size: Minimum size before sampling is allowed
        eviction_policy: How to remove old data ('fifo' or 'random')
        use_position_weighting: Weight positions by importance for sampling
    """

    max_size: int = 100_000
    min_size: int = 1_000
    eviction_policy: str = 'fifo'

    # Position weighting
    use_position_weighting: bool = False

    # Internal storage: pre-allocated numpy arrays (grown lazily)
    _encoded: Optional[np.ndarray] = None       # (capacity, 26, 10)
    _targets: Optional[np.ndarray] = None       # (capacity, 6)
    _weights: Optional[np.ndarray] = None       # (capacity,)
    _size: int = 0
    _insertion_index: int = 0

    def __post_init__(self):
        """Allocate storage."""
        self._allocate()

    def _allocate(self):
        # Allocate in chunks to avoid reserving max_size upfront for small runs
        initial = min(self.max_size, 16_384)
        self._encoded = np.zeros((initial,) + _ENCODED_SHAPE, dtype=np.float32)
        self._targets = np.zeros((initial, _EQUITY_DIM), dtype=np.float32)
        self._weights = np.ones(initial, dtype=np.float64)
        self._size = 0
        self._insertion_index = 0

    def _grow(self, needed: int):
        """Grow storage arrays (doubling, capped at max_size)."""
        capacity = self._encoded.shape[0]
        if needed <= capacity:
            return
        new_capacity = min(self.max_size, max(needed, capacity * 2))
        for name in ('_encoded', '_targets', '_weights'):
            old = getattr(self, name)
            new = np.zeros((new_capacity,) + old.shape[1:], dtype=old.dtype)
            new[:capacity] = old
            if name == '_weights':
                new[capacity:] = 1.0
            setattr(self, name, new)

    def __len__(self) -> int:
        """Return number of steps in buffer."""
        return self._size

    def is_ready(self) -> bool:
        """Check if buffer has enough data for sampling."""
        return self._size >= self.min_size

    def add_game(
        self,
        game_result: GameResult,
        td_lambda: Optional[float] = None,
    ) -> None:
        """Add a complete game to the buffer.

        Args:
            game_result: Completed game with trajectory and outcome
            td_lambda: If set, use TD(lambda) targets instead of pure
                Monte Carlo targets. Requires value_estimates in game_result.
                Typical value: 0.7. If None, uses pure MC targets (lambda=1.0).
        """
        if game_result.outcome is None:
            return  # Skip draws (max moves reached)

        # Compute targets using TD(lambda) if available, else Monte Carlo
        if td_lambda is not None and game_result.value_estimates is not None:
            from backgammon.training.td_lambda import compute_td_lambda_targets
            targets = compute_td_lambda_targets(game_result, lambda_param=td_lambda)
        else:
            targets = None

        # Add each step with its equity target and weight
        num_steps = len(game_result.steps)
        for i, step in enumerate(game_result.steps):
            if targets is not None and i < len(targets):
                value_target = targets[i]
            else:
                # Fallback: pure Monte Carlo target
                equity = outcome_to_equity(game_result.outcome, step.player)
                value_target = equity.to_array()  # shape (6,)

            weight = _compute_position_weight(
                value_target, i, num_steps,
            ) if self.use_position_weighting else 1.0

            self._add_step(step, value_target, weight)

    def _add_step(self, step: GameStep, value_target, weight: float = 1.0) -> None:
        """Add single step to buffer with pre-encoded board features.

        Args:
            step: Game step (state, action, player)
            value_target: 6-dim equity target for this position
            weight: Sampling weight for this position
        """
        encoded = _encode_board_fast_single(step.board)
        value_arr = np.asarray(value_target, dtype=np.float32)

        if self._size < self.max_size:
            self._grow(self._size + 1)
            idx = self._size
            self._size += 1
        elif self.eviction_policy == 'fifo':
            # Overwrite oldest entry (circular buffer)
            idx = self._insertion_index % self.max_size
            self._insertion_index += 1
        elif self.eviction_policy == 'random':
            idx = random.randint(0, self.max_size - 1)
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

        self._encoded[idx] = encoded
        self._targets[idx] = value_arr
        self._weights[idx] = weight

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """Sample indices, weighted if position weighting is enabled."""
        n = self._size
        sample_size = min(batch_size, n)

        if self.use_position_weighting:
            weights = self._weights[:n]
            probs = weights / weights.sum()
            return np.random.choice(n, size=sample_size, replace=False, p=probs)
        return np.random.choice(n, size=sample_size, replace=False)

    def sample_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Sample random batch for training using pre-encoded features.

        Value-only training: 'target_policy' and 'action_mask' are returned
        as minimal (1,)-shaped placeholders — the loss ignores them when the
        policy head is disabled, and full-size dummies would waste ~10MB of
        host->device transfer per batch.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Training batch dictionary with:
                - 'board_encoding': (batch_size, 26, feature_dim)
                - 'equity_target': (batch_size, 6)
                - 'target_policy': (1,) placeholder
                - 'action_mask': (1,) placeholder

        Raises:
            ValueError: If buffer is not ready (too few samples)
        """
        if not self.is_ready():
            raise ValueError(
                f"Buffer not ready: {self._size} < {self.min_size}"
            )

        indices = self._sample_indices(batch_size)

        return {
            'board_encoding': jnp.asarray(self._encoded[indices]),
            'equity_target': jnp.asarray(self._targets[indices]),
            'target_policy': jnp.zeros((1,), dtype=jnp.float32),
            'action_mask': jnp.ones((1,), dtype=jnp.bool_),
        }

    def clear(self) -> None:
        """Clear all data from buffer."""
        self._allocate()

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        if self._size == 0:
            return {
                'size': 0,
                'utilization': 0.0,
                'avg_win_prob': 0.0,
            }

        # Win probability = sum of win components (indices 0,1,2)
        equities = self._targets[:self._size]
        win_probs = equities[:, 0] + equities[:, 1] + equities[:, 2]

        return {
            'size': self._size,
            'utilization': self._size / self.max_size,
            'avg_win_prob': float(np.mean(win_probs)),
            'std_win_prob': float(np.std(win_probs)),
        }


def _compute_position_weight(
    equity_target: np.ndarray,
    step_index: int,
    total_steps: int,
) -> float:
    """Compute sampling weight for a position.

    Positions that are more informative for training get higher weight:
    1. Endgame positions (later in the game) get upweighted because they
       have clearer targets and are critical for bearing off accuracy.
    2. Uncertain positions (equity near 50/50) get upweighted because
       they're harder to evaluate and more informative for learning.
    3. Extreme positions (clear win/loss) get downweighted because
       the network can learn them quickly.

    Args:
        equity_target: 6-dim equity target [wn, wg, wbg, ln, lg, lbg].
        step_index: Position in the game (0 = first move).
        total_steps: Total number of steps in the game.

    Returns:
        Sampling weight >= 1.0.
    """
    weight = 1.0

    # Game progress weighting: later positions are more valuable
    # (clearer signal, closer to outcome)
    if total_steps > 1:
        progress = step_index / (total_steps - 1)
        # Late game gets up to 1.5x weight
        weight += 0.5 * progress

    # Equity uncertainty weighting: positions near 50/50 are harder
    # Win probability = sum of win components
    win_prob = float(np.sum(np.asarray(equity_target)[:3]))
    # Uncertainty is highest at win_prob=0.5, lowest at 0 or 1
    # entropy-like: 4 * p * (1-p) peaks at 1.0 when p=0.5
    uncertainty = 4.0 * win_prob * (1.0 - win_prob)
    # Uncertain positions get up to 1.5x weight
    weight += 0.5 * uncertainty

    return weight


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized sampling.

    Samples important transitions more frequently based on TD-error
    or other priority metrics. This can improve learning efficiency.

    Args:
        max_size: Maximum buffer size
        min_size: Minimum size before sampling
        alpha: Priority exponent (0 = uniform, 1 = full priority)
        beta: Importance sampling exponent (0 = no correction, 1 = full)
    """

    def __init__(
        self,
        max_size: int = 100_000,
        min_size: int = 1_000,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(max_size=max_size, min_size=min_size)
        self.alpha = alpha
        self.beta = beta
        self._priorities = np.ones(self._encoded.shape[0], dtype=np.float64)

    def _grow(self, needed: int):
        capacity = self._encoded.shape[0]
        super()._grow(needed)
        new_capacity = self._encoded.shape[0]
        if new_capacity > capacity and hasattr(self, '_priorities'):
            new_p = np.ones(new_capacity, dtype=np.float64)
            new_p[:capacity] = self._priorities
            self._priorities = new_p

    def _add_step(self, step: GameStep, value_target, weight: float = 1.0) -> None:
        """Add step with max priority so new samples are seen quickly."""
        prev_size = self._size
        super()._add_step(step, value_target, weight)
        # Index that was just written
        if self._size > prev_size:
            idx = self._size - 1
        else:
            idx = (self._insertion_index - 1) % self.max_size
        max_priority = self._priorities[:max(self._size, 1)].max()
        self._priorities[idx] = max_priority if max_priority > 0 else 1.0

    def sample_batch(
        self,
        batch_size: int,
    ) -> Tuple[Dict[str, jnp.ndarray], np.ndarray, np.ndarray]:
        """Sample batch with priorities.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Tuple of (batch, indices, weights):
                - batch: Training batch dictionary
                - indices: Sampled indices (for priority updates)
                - weights: Importance sampling weights
        """
        if not self.is_ready():
            raise ValueError(
                f"Buffer not ready: {self._size} < {self.min_size}"
            )

        n = self._size
        priorities = self._priorities[:n]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        sample_size = min(batch_size, n)
        indices = np.random.choice(n, size=sample_size, replace=False, p=probs)

        # Compute importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max for stability

        batch = {
            'board_encoding': jnp.asarray(self._encoded[indices]),
            'equity_target': jnp.asarray(self._targets[indices]),
            'target_policy': jnp.zeros((1,), dtype=jnp.float32),
            'action_mask': jnp.ones((1,), dtype=jnp.bool_),
        }

        return batch, indices, weights

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of sampled transitions
            priorities: New priority values (e.g., TD-errors)
        """
        for idx, priority in zip(indices, priorities):
            # Add small epsilon to avoid zero priority
            self._priorities[idx] = float(priority) + 1e-6

    def clear(self) -> None:
        """Clear buffer and priorities."""
        super().clear()
        self._priorities = np.ones(self._encoded.shape[0], dtype=np.float64)
