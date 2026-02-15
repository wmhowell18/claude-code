"""Experience replay buffer for training data management.

Stores game trajectories and samples batches for training.
Breaks temporal correlation and improves sample efficiency.

Performance: Boards are pre-encoded at insertion time so that
sample_batch() only does numpy indexing + JAX array conversion,
eliminating ~13K Python function calls per batch.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random

from backgammon.training.self_play import GameResult, GameStep
from backgammon.encoding.encoder import encode_board, raw_encoding_config, outcome_to_equity
from backgammon.encoding.action_encoder import (
    encode_move_to_one_hot,
    create_action_mask,
    get_action_space_size,
)
from backgammon.core.types import Player


def _encode_board_fast_single(board) -> np.ndarray:
    """Fast single-board encoding for raw encoding (feature_dim=2).

    Directly slices numpy checker arrays instead of calling encode_board()
    which iterates 26 times per board in Python.

    Args:
        board: Board object.

    Returns:
        Array of shape (26, 2) with normalized checker counts.
    """
    features = np.empty((26, 2), dtype=np.float32)
    inv15 = np.float32(1.0 / 15.0)
    if board.player_to_move == Player.WHITE:
        features[:, 0] = board.white_checkers * inv15
        features[:, 1] = board.black_checkers * inv15
    else:
        features[:, 0] = board.black_checkers * inv15
        features[:, 1] = board.white_checkers * inv15
    return features


@dataclass
class ReplayBuffer:
    """Experience replay buffer for training data.

    Stores game trajectories and provides random sampling for training.
    This breaks temporal correlation and improves training stability.

    Args:
        max_size: Maximum number of game steps to store
        min_size: Minimum size before sampling is allowed
        eviction_policy: How to remove old data ('fifo' or 'random')
    """

    max_size: int = 100_000
    min_size: int = 1_000
    eviction_policy: str = 'fifo'

    # Position weighting
    use_position_weighting: bool = False

    # Internal storage
    _steps: List[Tuple[GameStep, float]] = field(default_factory=list)
    _weights: List[float] = field(default_factory=list)
    _insertion_index: int = 0
    _encoding_config: Optional[object] = None

    # Pre-encoded board features for fast sampling (eliminates re-encoding)
    _encoded_boards: List[np.ndarray] = field(default_factory=list)
    _equity_targets_list: List[np.ndarray] = field(default_factory=list)

    # Cached dummy arrays for value-only training (avoid recreating per sample)
    _dummy_policy: Optional[np.ndarray] = None
    _dummy_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize encoding config after creation."""
        if self._encoding_config is None:
            self._encoding_config = raw_encoding_config()
        # Pre-allocate dummy arrays for value-only training
        action_size = get_action_space_size()
        self._dummy_policy = np.zeros(action_size, dtype=np.float32)
        self._dummy_mask = np.ones(action_size, dtype=bool)

    def __len__(self) -> int:
        """Return number of steps in buffer."""
        return len(self._steps)

    def is_ready(self) -> bool:
        """Check if buffer has enough data for sampling."""
        return len(self._steps) >= self.min_size

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
                value_target = equity.to_array()  # shape (5,)

            weight = _compute_position_weight(
                value_target, i, num_steps,
            ) if self.use_position_weighting else 1.0

            self._add_step(step, value_target, weight)

    def _add_step(self, step: GameStep, value_target: float, weight: float = 1.0) -> None:
        """Add single step to buffer with pre-encoded board features.

        Pre-encodes the board at insertion time so sample_batch() only needs
        numpy indexing — no per-sample encode_board() calls.

        Args:
            step: Game step (state, action, player)
            value_target: Value target for this position
            weight: Sampling weight for this position
        """
        # Pre-encode the board (vectorized numpy, ~10x faster than encode_board)
        encoded = _encode_board_fast_single(step.board)
        value_arr = np.asarray(value_target, dtype=np.float32)

        if len(self._steps) < self.max_size:
            # Buffer not full yet, just append
            self._steps.append((step, value_target))
            self._weights.append(weight)
            self._encoded_boards.append(encoded)
            self._equity_targets_list.append(value_arr)
        else:
            # Buffer full, evict according to policy
            if self.eviction_policy == 'fifo':
                # Overwrite oldest entry (circular buffer)
                idx = self._insertion_index % self.max_size
                self._steps[idx] = (step, value_target)
                self._weights[idx] = weight
                self._encoded_boards[idx] = encoded
                self._equity_targets_list[idx] = value_arr
                self._insertion_index += 1
            elif self.eviction_policy == 'random':
                # Replace random entry
                idx = random.randint(0, self.max_size - 1)
                self._steps[idx] = (step, value_target)
                self._weights[idx] = weight
                self._encoded_boards[idx] = encoded
                self._equity_targets_list[idx] = value_arr
            else:
                raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

    def sample_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Sample random batch for training using pre-encoded features.

        Uses pre-encoded board features stored at insertion time, eliminating
        ~13K encode_board() Python calls per batch. For value-only training,
        also skips creating unused policy targets and action masks.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Training batch dictionary with:
                - 'board_encoding': (batch_size, 26, feature_dim)
                - 'target_policy': (batch_size, num_actions)
                - 'equity_target': (batch_size, 5)
                - 'action_mask': (batch_size, num_actions)

        Raises:
            ValueError: If buffer is not ready (too few samples)
        """
        if not self.is_ready():
            raise ValueError(
                f"Buffer not ready: {len(self._steps)} < {self.min_size}"
            )

        # Sample indices (weighted if position weighting is enabled)
        n = len(self._steps)
        sample_size = min(batch_size, n)

        if self.use_position_weighting and self._weights:
            # Weighted sampling: positions with higher weights are sampled more
            weights_arr = np.array(self._weights[:n])
            probs = weights_arr / weights_arr.sum()
            indices = np.random.choice(
                n, size=sample_size, replace=False, p=probs,
            )
        else:
            indices = np.array(random.sample(range(n), sample_size))

        # Fast path: use pre-encoded board features (no per-sample encoding)
        board_encodings = np.array([self._encoded_boards[i] for i in indices])
        equity_targets = np.array([self._equity_targets_list[i] for i in indices])

        # For value-only training, policy targets and action masks are unused
        # by the loss function. Use pre-allocated dummy arrays to avoid
        # creating 512 × 1024-element arrays per batch.
        action_size = get_action_space_size()
        dummy_policies = np.broadcast_to(
            self._dummy_policy, (sample_size, action_size)
        ).copy()  # copy() because broadcast_to returns read-only view
        dummy_masks = np.broadcast_to(
            self._dummy_mask, (sample_size, action_size)
        ).copy()

        # Convert to JAX arrays
        return {
            'board_encoding': jnp.array(board_encodings, dtype=jnp.float32),
            'target_policy': jnp.array(dummy_policies, dtype=jnp.float32),
            'equity_target': jnp.array(equity_targets, dtype=jnp.float32),
            'action_mask': jnp.array(dummy_masks, dtype=jnp.bool_),
        }

    def _create_policy_target(
        self,
        move_taken: tuple,
        legal_moves: List[tuple],
    ) -> np.ndarray:
        """Create target policy distribution.

        Creates a one-hot distribution on the move that was played.
        In a more sophisticated version, this would use MCTS visit counts
        or other policy improvement methods.

        Args:
            move_taken: The move that was actually played
            legal_moves: All legal moves in this position

        Returns:
            Target policy distribution (ACTION_SPACE_SIZE,)
        """
        # Use proper action encoder to create one-hot policy
        return encode_move_to_one_hot(move_taken, legal_moves)

    def _create_action_mask(
        self,
        legal_moves: List[tuple],
    ) -> np.ndarray:
        """Create mask of legal actions.

        Args:
            legal_moves: List of legal moves

        Returns:
            Boolean mask (ACTION_SPACE_SIZE,) - True for legal actions
        """
        # Use proper action encoder to create action mask
        return create_action_mask(legal_moves)

    def clear(self) -> None:
        """Clear all data from buffer."""
        self._steps.clear()
        self._weights.clear()
        self._encoded_boards.clear()
        self._equity_targets_list.clear()
        self._insertion_index = 0

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        if len(self._steps) == 0:
            return {
                'size': 0,
                'utilization': 0.0,
                'avg_win_prob': 0.0,
            }

        # Compute statistics from equity targets
        equities = np.array([v for _, v in self._steps])  # (N, 5)
        # Win probability = sum of win components (indices 0,1,2)
        win_probs = equities[:, 0] + equities[:, 1] + equities[:, 2]

        return {
            'size': len(self._steps),
            'utilization': len(self._steps) / self.max_size,
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
        equity_target: 5-dim equity target [wn, wg, wb, lg, lb].
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
    win_prob = float(np.sum(equity_target[:3]))
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
        self._priorities: List[float] = []

    def _add_step(self, step: GameStep, value_target: float, weight: float = 1.0) -> None:
        """Add step with default priority and pre-encoded features."""
        # New transitions get max priority
        max_priority = max(self._priorities) if self._priorities else 1.0

        # Pre-encode the board
        encoded = _encode_board_fast_single(step.board)
        value_arr = np.asarray(value_target, dtype=np.float32)

        if len(self._steps) < self.max_size:
            self._steps.append((step, value_target))
            self._priorities.append(max_priority)
            self._encoded_boards.append(encoded)
            self._equity_targets_list.append(value_arr)
        else:
            idx = self._insertion_index % self.max_size
            self._steps[idx] = (step, value_target)
            self._priorities[idx] = max_priority
            self._encoded_boards[idx] = encoded
            self._equity_targets_list[idx] = value_arr
            self._insertion_index += 1

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
                f"Buffer not ready: {len(self._steps)} < {self.min_size}"
            )

        # Compute sampling probabilities
        priorities = np.array(self._priorities[:len(self._steps)])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample indices
        sample_size = min(batch_size, len(self._steps))
        indices = np.random.choice(
            len(self._steps),
            size=sample_size,
            replace=False,
            p=probs,
        )

        # Compute importance sampling weights
        weights = (len(self._steps) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max for stability

        # Fast path: use pre-encoded board features
        board_encodings = np.array([self._encoded_boards[i] for i in indices])
        equity_targets = np.array([self._equity_targets_list[i] for i in indices])

        # Dummy policy/mask for value-only training
        action_size = get_action_space_size()
        dummy_policies = np.zeros((sample_size, action_size), dtype=np.float32)
        dummy_masks = np.ones((sample_size, action_size), dtype=bool)

        batch = {
            'board_encoding': jnp.array(board_encodings, dtype=jnp.float32),
            'target_policy': jnp.array(dummy_policies, dtype=jnp.float32),
            'equity_target': jnp.array(equity_targets, dtype=jnp.float32),
            'action_mask': jnp.array(dummy_masks, dtype=jnp.bool_),
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
        self._priorities.clear()
