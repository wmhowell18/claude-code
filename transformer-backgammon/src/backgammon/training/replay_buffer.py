"""Experience replay buffer for training data management.

Stores game trajectories and samples batches for training.
Breaks temporal correlation and improves sample efficiency.
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

    # Internal storage
    _steps: List[Tuple[GameStep, float]] = field(default_factory=list)
    _insertion_index: int = 0
    _encoding_config: Optional[object] = None

    def __post_init__(self):
        """Initialize encoding config after creation."""
        if self._encoding_config is None:
            self._encoding_config = raw_encoding_config()

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

        # Add each step with its equity target
        for i, step in enumerate(game_result.steps):
            if targets is not None and i < len(targets):
                value_target = targets[i]
            else:
                # Fallback: pure Monte Carlo target
                equity = outcome_to_equity(game_result.outcome, step.player)
                value_target = equity.to_array()  # shape (5,)

            self._add_step(step, value_target)

    def _add_step(self, step: GameStep, value_target: float) -> None:
        """Add single step to buffer.

        Args:
            step: Game step (state, action, player)
            value_target: Value target for this position
        """
        if len(self._steps) < self.max_size:
            # Buffer not full yet, just append
            self._steps.append((step, value_target))
        else:
            # Buffer full, evict according to policy
            if self.eviction_policy == 'fifo':
                # Overwrite oldest entry (circular buffer)
                idx = self._insertion_index % self.max_size
                self._steps[idx] = (step, value_target)
                self._insertion_index += 1
            elif self.eviction_policy == 'random':
                # Replace random entry
                idx = random.randint(0, self.max_size - 1)
                self._steps[idx] = (step, value_target)
            else:
                raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

    def sample_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Sample random batch for training.

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

        # Sample random indices
        sample_size = min(batch_size, len(self._steps))
        sampled_steps = random.sample(self._steps, sample_size)

        # Prepare batch arrays
        board_encodings = []
        target_policies = []
        equity_targets = []
        action_masks = []

        for step, equity_target in sampled_steps:
            # Encode board state
            encoded = encode_board(self._encoding_config, step.board)
            # Keep as (26, feature_dim) - don't flatten
            # Remove batch dimension: (1, 26, feature_dim) -> (26, feature_dim)
            board_enc = encoded.position_features[0]
            board_encodings.append(board_enc)

            # Create target policy from move taken
            target_policy = self._create_policy_target(
                step.move_taken,
                step.legal_moves,
            )
            target_policies.append(target_policy)

            # Equity target (already computed as 5-dim array)
            equity_targets.append(equity_target)

            # Create action mask from legal moves
            action_mask = self._create_action_mask(step.legal_moves)
            action_masks.append(action_mask)

        # Convert to JAX arrays
        return {
            'board_encoding': jnp.array(board_encodings, dtype=jnp.float32),
            'target_policy': jnp.array(target_policies, dtype=jnp.float32),
            'equity_target': jnp.array(equity_targets, dtype=jnp.float32),
            'action_mask': jnp.array(action_masks, dtype=jnp.bool_),
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

    def _add_step(self, step: GameStep, value_target: float) -> None:
        """Add step with default priority."""
        # New transitions get max priority
        max_priority = max(self._priorities) if self._priorities else 1.0

        if len(self._steps) < self.max_size:
            self._steps.append((step, value_target))
            self._priorities.append(max_priority)
        else:
            idx = self._insertion_index % self.max_size
            self._steps[idx] = (step, value_target)
            self._priorities[idx] = max_priority
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

        # Get samples
        sampled_steps = [self._steps[i] for i in indices]

        # Prepare batch (same as parent class)
        board_encodings = []
        target_policies = []
        equity_targets = []
        action_masks = []

        for step, equity_target in sampled_steps:
            encoded = encode_board(self._encoding_config, step.board)
            # Keep as (26, feature_dim) - don't flatten
            board_enc = encoded.position_features[0]
            board_encodings.append(board_enc)

            target_policy = self._create_policy_target(
                step.move_taken,
                step.legal_moves,
            )
            target_policies.append(target_policy)

            equity_targets.append(equity_target)

            action_mask = self._create_action_mask(step.legal_moves)
            action_masks.append(action_mask)

        batch = {
            'board_encoding': jnp.array(board_encodings, dtype=jnp.float32),
            'target_policy': jnp.array(target_policies, dtype=jnp.float32),
            'equity_target': jnp.array(equity_targets, dtype=jnp.float32),
            'action_mask': jnp.array(action_masks, dtype=jnp.bool_),
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
