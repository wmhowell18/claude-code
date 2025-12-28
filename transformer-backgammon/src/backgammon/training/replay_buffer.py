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
from backgammon.encoding.encoder import encode_board, raw_encoding_config
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

    def add_game(self, game_result: GameResult) -> None:
        """Add a complete game to the buffer.

        Args:
            game_result: Completed game with trajectory and outcome
        """
        # Compute value targets from game outcome
        # Winner gets +1, loser gets -1
        outcome_value = 1.0 if game_result.outcome == 'white_wins' else -1.0

        # Add each step with its value target
        for step in game_result.steps:
            # Value is from perspective of current player
            if step.player == Player.WHITE:
                value_target = outcome_value
            else:
                value_target = -outcome_value

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
                - 'board_encoding': (batch_size, 26)
                - 'target_policy': (batch_size, num_actions)
                - 'value_target': (batch_size,)
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
        value_targets = []
        action_masks = []

        for step, value_target in sampled_steps:
            # Encode board state
            encoded = encode_board(self._encoding_config, step.board)
            # Flatten the features (1, 26, feature_dim) -> (26 * feature_dim,)
            board_enc = encoded.position_features.reshape(-1)
            board_encodings.append(board_enc)

            # Create target policy from move taken
            # For now, use one-hot on the move that was played
            # TODO: In production, use MCTS policy or other improvement
            target_policy = self._create_policy_target(
                step.move_taken,
                step.legal_moves,
            )
            target_policies.append(target_policy)

            # Value target (already computed)
            value_targets.append(value_target)

            # Create action mask from legal moves
            action_mask = self._create_action_mask(step.legal_moves)
            action_masks.append(action_mask)

        # Convert to JAX arrays
        return {
            'board_encoding': jnp.array(board_encodings, dtype=jnp.float32),
            'target_policy': jnp.array(target_policies, dtype=jnp.float32),
            'value_target': jnp.array(value_targets, dtype=jnp.float32),
            'action_mask': jnp.array(action_masks, dtype=jnp.bool_),
        }

    def _create_policy_target(
        self,
        move_taken: tuple,
        legal_moves: List[tuple],
        num_actions: int = 256,
    ) -> np.ndarray:
        """Create target policy distribution.

        For now, creates a one-hot distribution on the move that was played.
        In a more sophisticated version, this would use MCTS visit counts
        or other policy improvement methods.

        Args:
            move_taken: The move that was actually played
            legal_moves: All legal moves in this position
            num_actions: Size of action space

        Returns:
            Target policy distribution (num_actions,)
        """
        # Create zero policy
        policy = np.zeros(num_actions, dtype=np.float32)

        # Find index of move taken
        # TODO: Proper move encoding
        # For now, use hash as placeholder
        move_idx = hash(move_taken) % num_actions
        policy[move_idx] = 1.0

        # Could also distribute probability among legal moves
        # for a softer target (helps exploration)

        return policy

    def _create_action_mask(
        self,
        legal_moves: List[tuple],
        num_actions: int = 256,
    ) -> np.ndarray:
        """Create mask of legal actions.

        Args:
            legal_moves: List of legal moves
            num_actions: Size of action space

        Returns:
            Boolean mask (num_actions,) - True for legal actions
        """
        mask = np.zeros(num_actions, dtype=bool)

        # Mark legal moves as valid
        # TODO: Proper move encoding
        for move in legal_moves:
            move_idx = hash(move) % num_actions
            mask[move_idx] = True

        return mask

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
                'avg_value': 0.0,
            }

        # Compute statistics
        values = [v for _, v in self._steps]

        return {
            'size': len(self._steps),
            'utilization': len(self._steps) / self.max_size,
            'avg_value': np.mean(values),
            'std_value': np.std(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
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
        value_targets = []
        action_masks = []

        for step, value_target in sampled_steps:
            encoded = encode_board(self._encoding_config, step.board)
            board_enc = encoded.position_features.reshape(-1)
            board_encodings.append(board_enc)

            target_policy = self._create_policy_target(
                step.move_taken,
                step.legal_moves,
            )
            target_policies.append(target_policy)

            value_targets.append(value_target)

            action_mask = self._create_action_mask(step.legal_moves)
            action_masks.append(action_mask)

        batch = {
            'board_encoding': jnp.array(board_encodings, dtype=jnp.float32),
            'target_policy': jnp.array(target_policies, dtype=jnp.float32),
            'value_target': jnp.array(value_targets, dtype=jnp.float32),
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
