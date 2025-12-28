"""Self-play game generation for training.

Generates training data by playing games between agents (warmstart pip count,
then neural network self-play).
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass

from backgammon.core.board import Board, apply_move, generate_legal_moves, is_game_over, winner
from backgammon.core.types import Player, Move, GameOutcome
from backgammon.core.dice import roll_dice
from backgammon.evaluation.agents import Agent, pip_count_agent, random_agent


class GameStep(NamedTuple):
    """Single step in a game (state, action, outcome)."""
    board: Board
    player: Player
    legal_moves: List[Move]
    move_taken: Move
    dice: Tuple[int, int]


@dataclass
class GameResult:
    """Result of a completed game."""
    steps: List[GameStep]
    outcome: GameOutcome
    num_moves: int
    starting_position: Board


def play_game(
    white_agent: Agent,
    black_agent: Agent,
    starting_position: Board,
    max_moves: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> GameResult:
    """Play a single game between two agents.

    Args:
        white_agent: Agent playing white
        black_agent: Agent playing black
        starting_position: Initial board position
        max_moves: Maximum moves before declaring draw
        rng: Random number generator

    Returns:
        GameResult with complete game trajectory
    """
    if rng is None:
        rng = np.random.default_rng()

    board = starting_position
    steps = []

    for move_num in range(max_moves):
        # Check if game is over
        if is_game_over(board):
            outcome = winner(board)
            return GameResult(
                steps=steps,
                outcome=outcome,
                num_moves=move_num,
                starting_position=starting_position,
            )

        # Current player
        current_player = board.player_to_move
        current_agent = white_agent if current_player == Player.WHITE else black_agent

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
    )


def generate_training_batch(
    num_games: int,
    get_variant_fn,
    white_agent: Agent,
    black_agent: Agent,
    rng: Optional[np.random.Generator] = None,
) -> List[GameResult]:
    """Generate a batch of training games.

    Args:
        num_games: Number of games to play
        get_variant_fn: Function returning position variants (e.g., get_early_training_variants)
        white_agent: Agent for white
        black_agent: Agent for black
        rng: Random number generator

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
        result = play_game(white_agent, black_agent, variant, rng=rng)
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
