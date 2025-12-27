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


# ==============================================================================
# MOVE GENERATION
# ==============================================================================

def generate_legal_moves(board: Board, player: Player, dice: Dice) -> LegalMoves:
    """Generate all legal moves for a board, player, and dice roll.

    This is the core move generation algorithm. It handles:
    - Entering from the bar (required if checkers on bar)
    - Normal moves
    - Bearing off
    - Using all possible dice combinations

    Args:
        board: Current board state
        player: Player to move
        dice: Dice roll

    Returns:
        List of all legal moves
    """
    moves_list = dice_values(dice)
    all_moves = []

    # Generate all possible move sequences
    _generate_moves_recursive(
        board=board,
        player=player,
        dice_remaining=moves_list,
        current_move=[],
        all_moves=all_moves,
    )

    # If no moves found, player must pass
    if not all_moves:
        return [()]  # Empty move

    # Remove duplicate moves and filter for maximum dice usage
    unique_moves = _deduplicate_moves(all_moves)
    max_dice_used = _filter_best_moves(unique_moves)

    return max_dice_used


def _generate_moves_recursive(
    board: Board,
    player: Player,
    dice_remaining: List[int],
    current_move: List[MoveStep],
    all_moves: List[Move],
) -> None:
    """Recursively generate all possible move sequences.

    Args:
        board: Current board state
        player: Player to move
        dice_remaining: Dice values still available
        current_move: Move steps accumulated so far
        all_moves: List to append complete moves to
    """
    # Base case: no more dice
    if not dice_remaining:
        if current_move:  # Only add non-empty moves
            all_moves.append(tuple(current_move))
        return

    # Try to use the next die value
    die = dice_remaining[0]
    rest_dice = dice_remaining[1:]

    # Check if we must enter from bar first
    if checkers_on_bar(board, player) > 0:
        # Must enter from bar before any other move
        entry_moves = _generate_entry_moves(board, player, die)

        if entry_moves:
            for move_step in entry_moves:
                # Apply move temporarily
                new_board = board.copy()
                _apply_move_step(new_board, player, move_step)

                # Recurse with this move added
                _generate_moves_recursive(
                    board=new_board,
                    player=player,
                    dice_remaining=rest_dice,
                    current_move=current_move + [move_step],
                    all_moves=all_moves,
                )
        else:
            # Can't enter - try skipping this die
            _generate_moves_recursive(
                board=board,
                player=player,
                dice_remaining=rest_dice,
                current_move=current_move,
                all_moves=all_moves,
            )
        return

    # Normal moves (not entering from bar)
    possible_steps = _generate_single_die_moves(board, player, die)

    if possible_steps:
        for move_step in possible_steps:
            # Apply move temporarily
            new_board = board.copy()
            _apply_move_step(new_board, player, move_step)

            # Recurse with this move added
            _generate_moves_recursive(
                board=new_board,
                player=player,
                dice_remaining=rest_dice,
                current_move=current_move + [move_step],
                all_moves=all_moves,
            )
    else:
        # Can't use this die - try skipping it
        _generate_moves_recursive(
            board=board,
            player=player,
            dice_remaining=rest_dice,
            current_move=current_move,
            all_moves=all_moves,
        )

    # Also add the current move as-is (in case we can't use remaining dice)
    if current_move:
        all_moves.append(tuple(current_move))


def _generate_entry_moves(board: Board, player: Player, die: int) -> List[MoveStep]:
    """Generate moves entering from the bar.

    Args:
        board: Current board
        player: Player to move
        die: Die value to use

    Returns:
        List of possible entry moves (0 or 1 move)
    """
    entry_point = _entry_point(player, die)

    if _can_land_on_point(board, player, entry_point):
        opponent = player.opponent()
        hits = board.get_checkers(opponent, entry_point) == 1

        return [MoveStep(
            from_point=0,
            to_point=entry_point,
            die_used=die,
            hits_opponent=hits,
        )]

    return []


def _generate_single_die_moves(board: Board, player: Player, die: int) -> List[MoveStep]:
    """Generate all possible moves using a single die value.

    Args:
        board: Current board
        player: Player to move
        die: Die value to use

    Returns:
        List of possible move steps
    """
    moves = []
    checkers = board.white_checkers if player == Player.WHITE else board.black_checkers

    # Check each point where player has checkers
    for from_point in range(1, 25):
        if checkers[from_point] == 0:
            continue

        # Calculate destination
        if player == Player.WHITE:
            to_point = from_point - die  # White moves toward 0
        else:
            to_point = from_point + die  # Black moves toward 25

        # Check if this is a bearing off move
        if can_bear_off(board, player):
            if player == Player.WHITE:
                if to_point <= 0:
                    # Bearing off for white
                    if to_point == 0 or _can_bear_off_with_higher(board, player, from_point, die):
                        moves.append(MoveStep(
                            from_point=from_point,
                            to_point=25,  # Off
                            die_used=die,
                            hits_opponent=False,
                        ))
                    continue
            else:  # BLACK
                if to_point >= 25:
                    # Bearing off for black
                    if to_point == 25 or _can_bear_off_with_higher(board, player, from_point, die):
                        moves.append(MoveStep(
                            from_point=from_point,
                            to_point=25,  # Off
                            die_used=die,
                            hits_opponent=False,
                        ))
                    continue

        # Normal move (not bearing off)
        if 1 <= to_point <= 24:
            if _can_land_on_point(board, player, to_point):
                opponent = player.opponent()
                hits = board.get_checkers(opponent, to_point) == 1

                moves.append(MoveStep(
                    from_point=from_point,
                    to_point=to_point,
                    die_used=die,
                    hits_opponent=hits,
                ))

    return moves


def _can_bear_off_with_higher(board: Board, player: Player, from_point: int, die: int) -> bool:
    """Check if can bear off with a die higher than needed.

    When bearing off, if you roll higher than needed, you can bear off
    from the highest occupied point.

    Args:
        board: Current board
        player: Player
        from_point: Point trying to bear off from
        die: Die value

    Returns:
        True if this is the highest occupied point
    """
    checkers = board.white_checkers if player == Player.WHITE else board.black_checkers

    # Check if there are any checkers on higher points
    if player == Player.WHITE:
        # For white, higher point = larger number in home board
        for point in range(from_point + 1, 7):
            if checkers[point] > 0:
                return False
    else:
        # For black, higher point = smaller number in home board
        for point in range(19, from_point):
            if checkers[point] > 0:
                return False

    return True


def _deduplicate_moves(moves: List[Move]) -> List[Move]:
    """Remove duplicate moves.

    Args:
        moves: List of moves

    Returns:
        List with duplicates removed
    """
    # Convert to set of frozensets for deduplication
    unique = []
    seen = set()

    for move in moves:
        # Create a hashable representation
        key = tuple(sorted([
            (step.from_point, step.to_point, step.die_used)
            for step in move
        ]))

        if key not in seen:
            seen.add(key)
            unique.append(move)

    return unique


def _filter_best_moves(moves: List[Move]) -> List[Move]:
    """Filter moves to only those that use the maximum number of dice.

    In backgammon, you must use as many dice as possible.

    Args:
        moves: List of legal moves

    Returns:
        Moves that use maximum dice
    """
    if not moves:
        return moves

    max_dice = max(len(move) for move in moves)
    return [move for move in moves if len(move) == max_dice]


# ==============================================================================
# MOVE APPLICATION
# ==============================================================================

def apply_move(board: Board, player: Player, move: Move) -> Board:
    """Apply a move to a board, returning a new board.

    Args:
        board: Current board
        player: Player making the move
        move: Move to apply

    Returns:
        New board with move applied
    """
    new_board = board.copy()

    for step in move:
        _apply_move_step(new_board, player, step)

    # Switch player
    new_board.player_to_move = player.opponent()

    return new_board


def _apply_move_step(board: Board, player: Player, step: MoveStep) -> None:
    """Apply a single move step to a board (mutates board).

    Args:
        board: Board to modify
        player: Player making the move
        step: Move step to apply
    """
    opponent = player.opponent()

    # Remove checker from source
    count = board.get_checkers(player, step.from_point)
    assert count > 0, f"No checkers at point {step.from_point}"
    board.set_checkers(player, step.from_point, count - 1)

    # Handle hitting
    if step.hits_opponent:
        # Remove opponent's blot and put it on the bar
        board.set_checkers(opponent, step.to_point, 0)
        bar_count = board.get_checkers(opponent, 0)
        board.set_checkers(opponent, 0, bar_count + 1)

    # Add checker to destination
    dest_count = board.get_checkers(player, step.to_point)
    board.set_checkers(player, step.to_point, dest_count + 1)


def undo_move(board: Board, player: Player, move: Move) -> Board:
    """Undo a move (reverse it).

    This is useful for search algorithms.

    Args:
        board: Board after the move
        player: Player who made the move
        move: Move to undo

    Returns:
        Board before the move
    """
    # To undo, we need to reverse the move steps
    # This is complex because we need to track hits
    # For now, simpler to just store the board state before moving
    raise NotImplementedError("Use board.copy() before applying move instead")


def is_legal_move(board: Board, player: Player, dice: Dice, move: Move) -> bool:
    """Check if a specific move is legal.

    Args:
        board: Current board
        player: Player to move
        dice: Dice roll
        move: Move to check

    Returns:
        True if move is legal
    """
    legal_moves = generate_legal_moves(board, player, dice)
    return move in legal_moves
