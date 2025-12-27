"""Board representation and game rules.

This module implements the core backgammon game logic, including:
- Board initialization and manipulation
- Move generation
- Move application
- Game state queries

Board Layout:
    White moves from 24→1→off (home board: 1-6)
    Black moves from 1→24→off (home board: 19-24)

    Point numbering:
    13 14 15 16 17 18    19 20 21 22 23 24
    +------------------+------------------+
    |                  |                  |  Black home
    |                  |                  |
    |                  |                  |
    |                  |                  |
    |                  |                  |
    |                  |                  |  White home
    +------------------+------------------+
    12 11 10  9  8  7     6  5  4  3  2  1
"""

from typing import List, Tuple, Optional
import numpy as np
from backgammon.core.types import (
    Board,
    Player,
    Point,
    Move,
    MoveStep,
    Dice,
    LegalMoves,
    GameOutcome,
)
from backgammon.core.dice import dice_values


# ==============================================================================
# BOARD CONSTRUCTION
# ==============================================================================

def initial_board() -> Board:
    """Create the standard backgammon starting position.

    Standard setup:
    - White: 2 on 24, 5 on 13, 3 on 8, 5 on 6
    - Black: 2 on 1, 5 on 12, 3 on 17, 5 on 19

    Returns:
        Board in starting position with White to move
    """
    board = Board()

    # White checkers (moves 24→1)
    board.white_checkers[24] = 2   # Two on the 24-point
    board.white_checkers[13] = 5   # Five on the 13-point (mid-point)
    board.white_checkers[8] = 3    # Three on the 8-point
    board.white_checkers[6] = 5    # Five on the 6-point

    # Black checkers (moves 1→24)
    board.black_checkers[1] = 2    # Two on the 1-point
    board.black_checkers[12] = 5   # Five on the 12-point (mid-point)
    board.black_checkers[17] = 3   # Three on the 17-point
    board.black_checkers[19] = 5   # Five on the 19-point

    board.player_to_move = Player.WHITE
    return board


def empty_board() -> Board:
    """Create an empty board with no checkers.

    Returns:
        Empty board
    """
    return Board()


def copy_board(board: Board) -> Board:
    """Clone a board.

    Args:
        board: Board to copy

    Returns:
        Deep copy of the board
    """
    return board.copy()


def flip_board(board: Board) -> Board:
    """Flip board perspective (white ↔ black).

    This is useful for training - we can train a single network
    that always plays as "white" by flipping the board for black's moves.

    Args:
        board: Board to flip

    Returns:
        New board with colors swapped and points mirrored
    """
    flipped = Board()

    # Swap colors and mirror points
    # Point i for white becomes point (25-i) for black
    for i in range(26):
        mirror_i = 25 - i
        flipped.white_checkers[mirror_i] = board.black_checkers[i]
        flipped.black_checkers[mirror_i] = board.white_checkers[i]

    # Flip player
    flipped.player_to_move = board.player_to_move.opponent()

    return flipped


# ==============================================================================
# BOARD QUERIES
# ==============================================================================

def pip_count(board: Board, player: Player) -> int:
    """Calculate pip count for a player.

    Pip count = sum of (point_number × num_checkers) for all checkers.
    This measures how far a player is from bearing off all checkers.

    Args:
        board: Current board
        player: Which player

    Returns:
        Total pip count
    """
    checkers = board.white_checkers if player == Player.WHITE else board.black_checkers

    # For white, distance is point number (24 is far, 1 is close)
    # For black, distance is 25 - point number (1 is far, 24 is close)
    total = 0
    for point in range(26):
        count = checkers[point]
        if count > 0:
            if player == Player.WHITE:
                # White moves 24→1, so pip = point number
                # Bar (0) is treated as 25 pips
                if point == 0:
                    total += 25 * count
                else:
                    total += point * count
            else:
                # Black moves 1→24, so pip = 25 - point
                # Bar (0) is treated as 25 pips
                if point == 0:
                    total += 25 * count
                else:
                    total += (25 - point) * count

    return total


def checkers_on_bar(board: Board, player: Player) -> int:
    """Get number of checkers on the bar for a player.

    Args:
        board: Current board
        player: Which player

    Returns:
        Number of checkers on bar (point 0)
    """
    return board.get_checkers(player, 0)


def checkers_borne_off(board: Board, player: Player) -> int:
    """Get number of checkers borne off for a player.

    Args:
        board: Current board
        player: Which player

    Returns:
        Number of checkers borne off (point 25)
    """
    return board.get_checkers(player, 25)


def can_bear_off(board: Board, player: Player) -> bool:
    """Check if a player can bear off checkers.

    A player can bear off when all their checkers are in their home board.

    Args:
        board: Current board
        player: Which player

    Returns:
        True if player can bear off
    """
    checkers = board.white_checkers if player == Player.WHITE else board.black_checkers

    # Check if any checkers are outside home board or on bar
    if checkers[0] > 0:  # Checkers on bar
        return False

    if player == Player.WHITE:
        # White home board is 1-6
        for point in range(7, 25):  # Points 7-24
            if checkers[point] > 0:
                return False
    else:
        # Black home board is 19-24
        for point in range(1, 19):  # Points 1-18
            if checkers[point] > 0:
                return False

    return True


def is_game_over(board: Board) -> bool:
    """Check if the game is over.

    Game is over when one player has borne off all 15 checkers.

    Args:
        board: Current board

    Returns:
        True if game is over
    """
    return (
        checkers_borne_off(board, Player.WHITE) == 15
        or checkers_borne_off(board, Player.BLACK) == 15
    )


def winner(board: Board) -> Optional[GameOutcome]:
    """Determine the winner and outcome type.

    Args:
        board: Current board

    Returns:
        GameOutcome if game is over, None otherwise
    """
    white_off = checkers_borne_off(board, Player.WHITE)
    black_off = checkers_borne_off(board, Player.BLACK)

    if white_off == 15:
        # White won - determine points
        if black_off == 0:
            # Backgammon: opponent has borne off no checkers and has checkers in winner's home or on bar
            black_in_white_home = sum(board.black_checkers[1:7])
            black_on_bar = board.black_checkers[0]
            if black_in_white_home > 0 or black_on_bar > 0:
                return GameOutcome(winner=Player.WHITE, points=3)
            else:
                # Gammon: opponent has borne off no checkers
                return GameOutcome(winner=Player.WHITE, points=2)
        else:
            # Normal win
            return GameOutcome(winner=Player.WHITE, points=1)

    elif black_off == 15:
        # Black won - determine points
        if white_off == 0:
            # Backgammon
            white_in_black_home = sum(board.white_checkers[19:25])
            white_on_bar = board.white_checkers[0]
            if white_in_black_home > 0 or white_on_bar > 0:
                return GameOutcome(winner=Player.BLACK, points=3)
            else:
                # Gammon
                return GameOutcome(winner=Player.BLACK, points=2)
        else:
            # Normal win
            return GameOutcome(winner=Player.BLACK, points=1)

    return None


def is_valid_board(board: Board) -> Tuple[bool, str]:
    """Validate a board state.

    Args:
        board: Board to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check total checker counts
    white_total = sum(board.white_checkers)
    black_total = sum(board.black_checkers)

    if white_total != 15:
        return False, f"White has {white_total} checkers, should have 15"

    if black_total != 15:
        return False, f"Black has {black_total} checkers, should have 15"

    # Check for invalid checker counts (>15 on any point)
    for point in range(26):
        if board.white_checkers[point] > 15:
            return False, f"Point {point} has {board.white_checkers[point]} white checkers"
        if board.black_checkers[point] > 15:
            return False, f"Point {point} has {board.black_checkers[point]} black checkers"

    return True, ""


# ==============================================================================
# HELPER FUNCTIONS FOR MOVE GENERATION
# ==============================================================================

def _home_board_range(player: Player) -> range:
    """Get the range of points in a player's home board.

    Args:
        player: Which player

    Returns:
        Range of points (inclusive)
    """
    if player == Player.WHITE:
        return range(1, 7)  # 1-6
    else:
        return range(19, 25)  # 19-24


def _entry_point(player: Player, die: int) -> Point:
    """Get the entry point for a die value when entering from the bar.

    Args:
        player: Which player
        die: Die value (1-6)

    Returns:
        Point number
    """
    if player == Player.WHITE:
        return 25 - die  # White enters in 19-24 (opponent's home)
    else:
        return die  # Black enters in 1-6 (opponent's home)


def _can_land_on_point(board: Board, player: Player, point: Point) -> bool:
    """Check if a player can land on a point.

    You can land on a point if:
    - It's empty
    - You own it
    - Opponent has exactly 1 checker (blot - you can hit it)

    Args:
        board: Current board
        player: Which player is moving
        point: Point to land on

    Returns:
        True if player can land on point
    """
    opponent = player.opponent()
    opponent_checkers = board.get_checkers(opponent, point)

    # Can land if opponent has 0 or 1 checker
    return opponent_checkers <= 1


# ==============================================================================
# BOARD DISPLAY (for debugging)
# ==============================================================================

def board_to_string(board: Board) -> str:
    """Convert board to string representation.

    Args:
        board: Board to display

    Returns:
        ASCII art representation
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f"Player to move: {board.player_to_move}")
    lines.append(f"White pip count: {pip_count(board, Player.WHITE)}")
    lines.append(f"Black pip count: {pip_count(board, Player.BLACK)}")
    lines.append("")

    # Simple representation: show each point
    lines.append("Point | White | Black")
    lines.append("------+-------+------")

    for point in range(26):
        w = board.white_checkers[point]
        b = board.black_checkers[point]
        if point == 0:
            point_name = "BAR  "
        elif point == 25:
            point_name = "OFF  "
        else:
            point_name = f"{point:2d}   "

        lines.append(f"{point_name}|  {w:2d}   |  {b:2d}")

    lines.append("=" * 50)
    return "\n".join(lines)


def print_board(board: Board) -> None:
    """Print board to console.

    Args:
        board: Board to print
    """
    print(board_to_string(board))
