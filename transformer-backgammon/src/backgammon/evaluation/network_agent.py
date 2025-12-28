"""Neural network agent for backgammon.

Wraps the transformer network to work with the Agent interface.
Handles board encoding, network inference, and move selection.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional
from flax.training import train_state

from backgammon.core.board import Board
from backgammon.core.types import Player, Move, Dice, LegalMoves
from backgammon.encoding.encoder import encode_board, raw_encoding_config
from backgammon.evaluation.agents import Agent


class NeuralNetworkAgent:
    """Agent that uses a neural network to select moves.

    The network outputs a policy distribution over all possible moves
    and a value estimate for the position.

    Args:
        state: Flax training state (model + params)
        temperature: Sampling temperature (0 = greedy, 1 = sample from policy)
        name: Agent name for logging
    """

    def __init__(
        self,
        state: train_state.TrainState,
        temperature: float = 0.0,
        name: str = "NeuralNet",
    ):
        self.state = state
        self.temperature = temperature
        self.name = name
        self.encoding_config = raw_encoding_config()

    def select_move(
        self,
        board: Board,
        player: Player,
        dice: Dice,
        legal_moves: LegalMoves,
    ) -> Move:
        """Select move using neural network policy.

        Args:
            board: Current board state
            player: Player to move
            dice: Dice roll
            legal_moves: List of legal moves

        Returns:
            Selected move (from legal_moves)
        """
        if not legal_moves:
            return ()

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Encode board state
        encoded = encode_board(self.encoding_config, board)
        encoded_board = encoded.position_features  # Shape: (1, 26, feature_dim)

        # Run network inference
        policy_logits, value = self._forward(encoded_board)

        # Compute probabilities for each legal move
        move_probs = self._score_legal_moves(policy_logits[0], legal_moves)

        # Select move based on temperature
        if self.temperature == 0.0:
            # Greedy: select best move
            best_idx = np.argmax(move_probs)
            return legal_moves[best_idx]
        else:
            # Sample from policy
            # Apply temperature
            adjusted_probs = move_probs ** (1.0 / self.temperature)
            adjusted_probs = adjusted_probs / adjusted_probs.sum()

            # Sample
            idx = np.random.choice(len(legal_moves), p=adjusted_probs)
            return legal_moves[idx]

    def _forward(self, encoded_board: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run forward pass through network.

        Args:
            encoded_board: Encoded board state (batch_size, 26)

        Returns:
            Tuple of (policy_logits, value)
                policy_logits: (batch_size, num_actions) - logits for each action
                value: (batch_size,) - value estimate
        """
        # Apply model (no gradient needed for inference)
        # Network returns (equity, policy, attention_weights)
        equity, policy_logits, _ = self.state.apply_fn(
            {'params': self.state.params},
            encoded_board,
            training=False,
        )

        # Extract value from equity (use expected value from equity distribution)
        # Equity is [batch_size, 5]: [win_normal, win_gammon, win_backgammon, lose_gammon, lose_backgammon]
        # Convert to single value estimate (simple weighted sum)
        # Wins are positive, losses are negative
        value = (
            equity[:, 0] * 1.0 +  # win normal
            equity[:, 1] * 2.0 +  # win gammon
            equity[:, 2] * 3.0 +  # win backgammon
            equity[:, 3] * (-2.0) +  # lose gammon
            equity[:, 4] * (-3.0)  # lose backgammon
        )

        return policy_logits, value

    def _score_legal_moves(
        self,
        policy_logits: jnp.ndarray,
        legal_moves: LegalMoves,
    ) -> np.ndarray:
        """Score each legal move using the policy network.

        Args:
            policy_logits: Raw logits from network (num_actions,)
            legal_moves: List of legal moves

        Returns:
            Probability distribution over legal_moves
        """
        # For each legal move, compute its score from the network
        # This requires encoding each move and looking up its logit

        move_scores = []
        for move in legal_moves:
            # Encode move to get its index in the action space
            # TODO: Implement proper move encoding
            # For now, use a simple hash as placeholder
            move_idx = hash(move) % len(policy_logits)
            move_scores.append(float(policy_logits[move_idx]))

        # Convert to probabilities (softmax over legal moves only)
        move_scores = np.array(move_scores)

        # Numerical stability: subtract max
        move_scores = move_scores - move_scores.max()
        exp_scores = np.exp(move_scores)
        probs = exp_scores / exp_scores.sum()

        return probs

    def get_value_estimate(self, board: Board, player: Player) -> float:
        """Get network's value estimate for a position.

        Args:
            board: Board state to evaluate
            player: Player to evaluate for

        Returns:
            Value estimate (higher is better for player)
        """
        # Encode board
        encoded = encode_board(self.encoding_config, board)
        encoded_board = encoded.position_features

        # Run forward pass
        _, value = self._forward(encoded_board)

        return float(value[0])


def create_neural_agent(
    state: train_state.TrainState,
    temperature: float = 0.0,
    name: str = "NeuralNet",
) -> Agent:
    """Create an Agent from a neural network.

    Args:
        state: Flax training state
        temperature: Sampling temperature
        name: Agent name

    Returns:
        Agent compatible with existing interface
    """
    network_agent = NeuralNetworkAgent(state, temperature, name)

    # Wrap in Agent interface
    return Agent(
        name=name,
        select_move_fn=network_agent.select_move,
    )


def create_mcts_agent(
    state: train_state.TrainState,
    num_simulations: int = 100,
    name: str = "MCTS",
) -> Agent:
    """Create an Agent using MCTS with neural network guidance.

    This is a more sophisticated agent that uses Monte Carlo Tree Search
    guided by the neural network for move selection.

    Args:
        state: Flax training state
        num_simulations: Number of MCTS simulations per move
        name: Agent name

    Returns:
        Agent using MCTS + neural network
    """
    # TODO: Implement MCTS
    # For now, just return a neural agent
    return create_neural_agent(state, temperature=0.0, name=name)
