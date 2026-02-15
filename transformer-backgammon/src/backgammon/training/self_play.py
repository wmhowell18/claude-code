"""Self-play game generation for training.

Generates training data by playing games between agents (warmstart pip count,
then neural network self-play). Supports recording network value estimates
for TD(lambda) target computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass, field

from backgammon.core.board import Board, apply_move, generate_legal_moves, is_game_over, winner
from backgammon.core.types import Player, Move, GameOutcome
from backgammon.core.dice import roll_dice
from backgammon.evaluation.agents import Agent, pip_count_agent, random_agent


class GameStep(NamedTuple):
    """Single step in a game (state, action, outcome).

    Attributes:
        board: Board state before this move.
        player: Player who moved.
        legal_moves: All legal moves available.
        move_taken: The move that was actually played.
        dice: Dice roll for this turn.
    """
    board: Board
    player: Player
    legal_moves: List[Move]
    move_taken: Move
    dice: Tuple[int, int]


@dataclass
class GameResult:
    """Result of a completed game.

    Attributes:
        steps: Complete game trajectory.
        outcome: Final game outcome (None for draws).
        num_moves: Total number of moves played.
        starting_position: Initial board state.
        value_estimates: Optional per-step value estimates from the network
            during self-play. Shape: list of 5-element arrays (equity dist)
            from the perspective of the player to move at each step.
            Used for TD(lambda) target computation.
    """
    steps: List[GameStep]
    outcome: GameOutcome
    num_moves: int
    starting_position: Board
    value_estimates: Optional[List[np.ndarray]] = None


def play_game(
    white_agent: Agent,
    black_agent: Agent,
    starting_position: Board,
    max_moves: int = 1000,
    rng: Optional[np.random.Generator] = None,
    record_value_estimates: bool = False,
) -> GameResult:
    """Play a single game between two agents.

    Args:
        white_agent: Agent playing white
        black_agent: Agent playing black
        starting_position: Initial board position
        max_moves: Maximum moves before declaring draw
        rng: Random number generator
        record_value_estimates: If True, capture network equity estimates
            at each step for TD(lambda) target computation. Only works
            when agents have a get_equity_estimate method (neural agents).

    Returns:
        GameResult with complete game trajectory
    """
    if rng is None:
        rng = np.random.default_rng()

    board = starting_position
    steps = []
    value_estimates = [] if record_value_estimates else None

    for move_num in range(max_moves):
        # Check if game is over
        if is_game_over(board):
            outcome = winner(board)
            return GameResult(
                steps=steps,
                outcome=outcome,
                num_moves=move_num,
                starting_position=starting_position,
                value_estimates=value_estimates,
            )

        # Current player
        current_player = board.player_to_move
        current_agent = white_agent if current_player == Player.WHITE else black_agent

        # Capture value estimate before move (for TD(lambda))
        if record_value_estimates:
            equity_est = _get_agent_equity(current_agent, board)
            value_estimates.append(equity_est)

        # Roll dice
        dice = roll_dice(rng)

        # Generate legal moves
        legal_moves = generate_legal_moves(board, current_player, dice)

        # Agent selects move
        move = current_agent.select_move(board, current_player, dice, legal_moves)

        # Record step
        steps.append(GameStep(
            board=board.copy(),
            player=current_player,
            legal_moves=legal_moves,
            move_taken=move,
            dice=dice,
        ))

        # Apply move
        board = apply_move(board, current_player, move)

    # Max moves reached - declare draw
    return GameResult(
        steps=steps,
        outcome=None,  # Draw
        num_moves=max_moves,
        starting_position=starting_position,
        value_estimates=value_estimates,
    )


def _get_agent_equity(agent: Agent, board: Board) -> Optional[np.ndarray]:
    """Try to get the equity estimate from an agent.

    Works with neural agents that have get_equity_estimate. Returns
    None for agents without this capability (pip count, random).

    Args:
        agent: Agent to query.
        board: Board state to evaluate.

    Returns:
        5-dim equity array or None if agent doesn't support it.
    """
    # The Agent wrapper stores the underlying function; check if the
    # agent's underlying object has get_equity_estimate.
    fn = agent.select_move_fn
    if hasattr(fn, '__self__') and hasattr(fn.__self__, 'get_equity_estimate'):
        return fn.__self__.get_equity_estimate(board)
    return None


def generate_training_batch(
    num_games: int,
    get_variant_fn,
    white_agent: Agent,
    black_agent: Agent,
    rng: Optional[np.random.Generator] = None,
    record_value_estimates: bool = False,
) -> List[GameResult]:
    """Generate a batch of training games.

    Args:
        num_games: Number of games to play
        get_variant_fn: Function returning position variants (e.g., get_early_training_variants)
        white_agent: Agent for white
        black_agent: Agent for black
        rng: Random number generator
        record_value_estimates: If True, record equity estimates for TD(lambda).

    Returns:
        List of completed game results
    """
    if rng is None:
        rng = np.random.default_rng()

    games = []
    variants = get_variant_fn()

    for _ in range(num_games):
        # Select random variant
        variant = variants[rng.integers(0, len(variants))]

        # Play game
        result = play_game(
            white_agent, black_agent, variant, rng=rng,
            record_value_estimates=record_value_estimates,
        )
        games.append(result)

    return games


def compute_game_statistics(games: List[GameResult]) -> dict:
    """Compute statistics from a batch of games.

    Args:
        games: List of game results

    Returns:
        Dictionary of statistics
    """
    total_games = len(games)
    white_wins = sum(1 for g in games if g.outcome and g.outcome.winner == Player.WHITE)
    black_wins = sum(1 for g in games if g.outcome and g.outcome.winner == Player.BLACK)
    draws = sum(1 for g in games if g.outcome is None)

    avg_moves = np.mean([g.num_moves for g in games])

    # Outcome types (normal, gammon, backgammon)
    gammons = sum(1 for g in games if g.outcome and g.outcome.points == 2)
    backgammons = sum(1 for g in games if g.outcome and g.outcome.points == 3)

    return {
        'total_games': total_games,
        'white_wins': white_wins,
        'black_wins': black_wins,
        'draws': draws,
        'white_win_rate': white_wins / total_games if total_games > 0 else 0.0,
        'avg_moves': avg_moves,
        'gammons': gammons,
        'backgammons': backgammons,
    }
