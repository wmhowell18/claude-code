"""Player agents for self-play and evaluation.

This module provides different types of agents that can play backgammon:
- Random agent: selects moves uniformly at random
- Pip count agent: uses simple heuristics (pip count + position penalties)
- Network agent: uses neural network for position evaluation

Agents are useful for:
- Self-play training data generation
- Evaluation and benchmarking
- Curriculum learning (start with simple opponents)
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np

from backgammon.core.types import Board, Player, Dice, Move, LegalMoves
from backgammon.core.board import (
    pip_count,
    apply_move,
    checkers_on_bar,
    checkers_borne_off,
    can_bear_off,
)


# ==============================================================================
# AGENT BASE CLASS
# ==============================================================================


@dataclass
class Agent:
    """Base agent class for playing backgammon.

    Attributes:
        name: Agent name for identification
        select_move_fn: Function that selects a move from legal moves
    """
    name: str
    select_move_fn: Callable[[Board, Player, Dice, LegalMoves], Move]

    def select_move(self, board: Board, player: Player, dice: Dice, legal_moves: LegalMoves) -> Move:
        """Select a move from legal moves.

        Args:
            board: Current board state
            player: Player to move
            dice: Dice roll
            legal_moves: List of legal moves

        Returns:
            Selected move
        """
        return self.select_move_fn(board, player, dice, legal_moves)


# ==============================================================================
# RANDOM AGENT
# ==============================================================================


def random_agent(seed: Optional[int] = None) -> Agent:
    """Create an agent that selects moves uniformly at random.

    Args:
        seed: Random seed (optional, for reproducibility)

    Returns:
        Random agent
    """
    rng = np.random.default_rng(seed)

    def select_random_move(board: Board, player: Player, dice: Dice, legal_moves: LegalMoves) -> Move:
        """Select a random legal move."""
        if not legal_moves:
            return ()  # No legal moves
        idx = rng.integers(0, len(legal_moves))
        return legal_moves[idx]

    return Agent(name="Random", select_move_fn=select_random_move)


# ==============================================================================
# PIP COUNT HEURISTIC AGENT
# ==============================================================================


@dataclass
class PipCountConfig:
    """Configuration for pip count heuristic agent.

    Attributes:
        blot_penalty: Penalty per exposed checker
        hit_bonus: Bonus for hitting opponent
        anchor_bonus: Bonus for anchors in opponent's home board
        prime_bonus: Bonus for consecutive made points
        escape_bonus: Bonus for escaping back checkers
        race_bonus: Bonus for being past contact (pure race)
        key_point_penalty: Penalty for breaking 6-point or 5-point (pre-bearoff)
        home_board_bonus: Bonus per made point in home board
        bearoff_bonus: Bonus per checker borne off
    """
    blot_penalty: float = 10.0
    hit_bonus: float = 15.0
    anchor_bonus: float = 5.0
    prime_bonus: float = 3.0
    escape_bonus: float = 8.0
    race_bonus: float = 20.0
    key_point_penalty: float = 25.0
    home_board_bonus: float = 4.0
    bearoff_bonus: float = 8.0


def pip_count_agent(config: Optional[PipCountConfig] = None) -> Agent:
    """Create an agent that uses pip count + heuristics to select moves.

    This agent evaluates each legal move by:
    1. Applying the move
    2. Computing pip count
    3. Adding bonuses/penalties for position features
    4. Selecting move with best score

    This is a simple but effective baseline agent, much better than random.

    Args:
        config: Heuristic configuration (uses defaults if None)

    Returns:
        Pip count heuristic agent
    """
    if config is None:
        config = PipCountConfig()

    def evaluate_position(board: Board, player: Player) -> float:
        """Evaluate a position from a player's perspective.

        Lower scores are better (we want to minimize pip count + penalties).

        Args:
            board: Board to evaluate
            player: Player whose perspective to evaluate from

        Returns:
            Position score (lower is better)
        """
        opponent = player.opponent()

        # Base score: pip count (lower is better)
        score = float(pip_count(board, player))

        # Count blots (exposed checkers)
        blots = 0
        for point in range(1, 25):
            if board.get_checkers(player, point) == 1:
                blots += 1

        # Penalty for blots
        score += blots * config.blot_penalty

        # Count opponent blots (hitting opportunities)
        opp_blots = 0
        for point in range(1, 25):
            if board.get_checkers(opponent, point) == 1:
                opp_blots += 1

        # Bonus for opponent blots (we might hit them)
        score -= opp_blots * (config.hit_bonus / 2)

        # Bonus for hitting opponent (if they're on bar)
        if checkers_on_bar(board, opponent) > 0:
            score -= config.hit_bonus

        # Bonus for anchors in opponent's home board
        if player == Player.WHITE:
            opp_home_start, opp_home_end = 19, 25
        else:
            opp_home_start, opp_home_end = 1, 7

        anchors = 0
        for point in range(opp_home_start, opp_home_end):
            if board.get_checkers(player, point) >= 2:
                anchors += 1

        score -= anchors * config.anchor_bonus

        # Bonus for prime (consecutive made points)
        max_prime = 0
        current_prime = 0
        for point in range(1, 25):
            if board.get_checkers(player, point) >= 2:
                current_prime += 1
                max_prime = max(max_prime, current_prime)
            else:
                current_prime = 0

        score -= max_prime * config.prime_bonus

        # Bonus for escaping back checkers
        if player == Player.WHITE:
            back_point = 24
            # Check if we have checkers on 24 or 23
            back_checkers = board.get_checkers(player, 24) + board.get_checkers(player, 23)
            if back_checkers == 0:
                # All back checkers escaped
                score -= config.escape_bonus
        else:
            # Black escapes from 1 or 2
            back_checkers = board.get_checkers(player, 1) + board.get_checkers(player, 2)
            if back_checkers == 0:
                score -= config.escape_bonus

        # Bonus for race (past contact)
        # Check if any of our checkers are behind opponent's checkers
        if player == Player.WHITE:
            our_furthest = 0
            their_furthest = 25
            for point in range(1, 25):
                if board.get_checkers(player, point) > 0:
                    our_furthest = max(our_furthest, point)
                if board.get_checkers(opponent, point) > 0:
                    their_furthest = min(their_furthest, point)

            if our_furthest < their_furthest:
                # We're past contact
                score -= config.race_bonus
        else:
            # Black perspective (moves 1→24)
            our_furthest = 25
            their_furthest = 0
            for point in range(1, 25):
                if board.get_checkers(player, point) > 0:
                    our_furthest = min(our_furthest, point)
                if board.get_checkers(opponent, point) > 0:
                    their_furthest = max(their_furthest, point)

            if our_furthest > their_furthest:
                # We're past contact
                score -= config.race_bonus

        # Penalty for breaking key home board points (6-point, 5-point)
        # These are the most valuable points — don't give them up before bearoff
        bearing_off = can_bear_off(board, player)
        if not bearing_off:
            if player == Player.WHITE:
                key_points = [6, 5]  # White's 6-point and 5-point
            else:
                key_points = [19, 20]  # Black's 6-point and 5-point

            for kp in key_points:
                if board.get_checkers(player, kp) < 2:
                    # Key point is broken — penalize
                    score += config.key_point_penalty

        # Bonus for home board strength (made points in home board)
        if player == Player.WHITE:
            home_range = range(1, 7)
        else:
            home_range = range(19, 25)

        made_points = sum(1 for p in home_range if board.get_checkers(player, p) >= 2)
        score -= made_points * config.home_board_bonus

        # Bonus for checkers already borne off (encourages finishing)
        off = checkers_borne_off(board, player)
        score -= off * config.bearoff_bonus

        return score

    def select_pip_count_move(board: Board, player: Player, dice: Dice, legal_moves: LegalMoves) -> Move:
        """Select move that minimizes pip count + penalties.

        Args:
            board: Current board
            player: Player to move
            dice: Dice roll
            legal_moves: Legal moves to choose from

        Returns:
            Best move according to pip count heuristic
        """
        if not legal_moves:
            return ()  # No legal moves

        best_move = legal_moves[0]
        best_score = float('inf')

        for move in legal_moves:
            # Apply move and evaluate
            new_board = apply_move(board.copy(), player, move)
            score = evaluate_position(new_board, player)

            if score < best_score:
                best_score = score
                best_move = move

        return best_move

    return Agent(name="PipCount", select_move_fn=select_pip_count_move)


# ==============================================================================
# GREEDY PIP COUNT (FASTER)
# ==============================================================================


def greedy_pip_count_agent() -> Agent:
    """Create a faster pip count agent that only considers pip count.

    This agent is much faster than the full pip count agent but less accurate.
    Just minimizes pip count without considering position features.

    Returns:
        Greedy pip count agent
    """
    def select_greedy_move(board: Board, player: Player, dice: Dice, legal_moves: LegalMoves) -> Move:
        """Select move that minimizes pip count only."""
        if not legal_moves:
            return ()

        best_move = legal_moves[0]
        best_pip_count = float('inf')

        for move in legal_moves:
            new_board = apply_move(board.copy(), player, move)
            pips = pip_count(new_board, player)

            if pips < best_pip_count:
                best_pip_count = pips
                best_move = move

        return best_move

    return Agent(name="GreedyPipCount", select_move_fn=select_greedy_move)


# ==============================================================================
# AGENT HELPERS
# ==============================================================================


def count_blots(board: Board, player: Player) -> int:
    """Count number of blots (exposed checkers) for a player.

    Args:
        board: Board state
        player: Player to count blots for

    Returns:
        Number of blots
    """
    blots = 0
    for point in range(1, 25):
        if board.get_checkers(player, point) == 1:
            blots += 1
    return blots


def has_anchor(board: Board, player: Player) -> bool:
    """Check if player has an anchor in opponent's home board.

    An anchor is 2+ checkers on a point in opponent's home.

    Args:
        board: Board state
        player: Player to check

    Returns:
        True if player has at least one anchor
    """
    if player == Player.WHITE:
        opp_home_start, opp_home_end = 19, 25
    else:
        opp_home_start, opp_home_end = 1, 7

    for point in range(opp_home_start, opp_home_end):
        if board.get_checkers(player, point) >= 2:
            return True
    return False


def is_past_contact(board: Board, player: Player) -> bool:
    """Check if player is past contact (pure race).

    Past contact means all of our checkers are ahead of all opponent checkers.

    Args:
        board: Board state
        player: Player to check

    Returns:
        True if past contact
    """
    opponent = player.opponent()

    if player == Player.WHITE:
        # White moves 24→1, so we want our furthest < their closest
        our_furthest = 0
        their_closest = 25

        for point in range(1, 25):
            if board.get_checkers(player, point) > 0:
                our_furthest = max(our_furthest, point)
            if board.get_checkers(opponent, point) > 0:
                their_closest = min(their_closest, point)

        return our_furthest < their_closest
    else:
        # Black moves 1→24, so we want our furthest > their closest
        our_furthest = 25
        their_closest = 0

        for point in range(1, 25):
            if board.get_checkers(player, point) > 0:
                our_furthest = min(our_furthest, point)
            if board.get_checkers(opponent, point) > 0:
                their_closest = max(their_closest, point)

        return our_furthest > their_closest
