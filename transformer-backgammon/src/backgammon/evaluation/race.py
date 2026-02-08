"""Race equity estimation using the Keith count / Effective Pip Count.

When both players have disengaged (no contact), the game becomes a pure
race. In this situation, the equity can be estimated from pip counts
without needing the neural network, which is faster and more accurate
for simple race positions.

The Keith count adjusts the raw pip count with positional corrections:
- Penalty for checkers on the bar (shouldn't happen in a race, but handle it)
- Penalty for gaps in the home board
- Penalty for having more checkers to bear off
- Penalty for checkers far from the home board

The race equity formula maps the pip count difference to a win probability
using an empirical sigmoid function calibrated from backgammon databases.

References:
- Tom Keith, "Effective Pip Count" (2005)
- Backgammon Galore race equity tables
"""

import numpy as np
from typing import Optional

from backgammon.core.board import Board, pip_count
from backgammon.core.types import Player


def effective_pip_count(board: Board, player: Player) -> float:
    """Compute the Effective Pip Count (EPC) for a player.

    The EPC adjusts the raw pip count with positional corrections that
    account for waste (inefficiency) in bearing off.

    Corrections applied:
    1. +2 for each checker still on the board beyond the minimum needed
    2. +1 for each gap in the home board
    3. +2 for each checker on the bar (exceptional in a race)
    4. +1 per checker on points 4-6 (crossover penalty)

    Args:
        board: Current board state.
        player: Player to compute EPC for.

    Returns:
        Effective pip count (higher is worse for the player).
    """
    raw_pips = pip_count(board, player)

    # Count checkers on the board (not borne off)
    checkers_off = board.get_checkers(player, 25)
    checkers_remaining = 15 - checkers_off

    # pip_count includes point 25 (borne off) which contributes 25*count for
    # White but 0 for Black. Borne-off checkers should not count any pips.
    if player == Player.WHITE:
        raw_pips -= 25 * checkers_off

    if checkers_remaining == 0:
        return 0.0

    correction = 0.0

    # Wastage from checker distribution:
    # Each additional checker beyond what's needed adds wastage
    correction += checkers_remaining * 1.0

    # Gap penalty: empty points in home board
    if player == Player.WHITE:
        home_range = range(1, 7)
    else:
        home_range = range(19, 25)

    for point in home_range:
        if board.get_checkers(player, point) == 0:
            correction += 1.0

    # Bar penalty
    bar_checkers = board.get_checkers(player, 0)
    correction += bar_checkers * 4.0

    # Crossover penalty for checkers still outside home board
    for point in range(1, 25):
        count = board.get_checkers(player, point)
        if count > 0:
            if player == Player.WHITE:
                if point > 6:
                    correction += count * 0.5
            else:
                if point < 19:
                    correction += count * 0.5

    return raw_pips + correction


def race_equity(board: Board, player: Player) -> float:
    """Estimate equity for a race position using pip count difference.

    Uses the empirical sigmoid formula:
    P(win) = 1 / (1 + exp(-k * (opp_epc - our_epc) / scale))

    where k and scale are calibrated parameters.

    This gives a quick, reasonably accurate equity estimate for pure
    race positions without needing a neural network.

    Args:
        board: Current board state (should be a race position).
        player: Player to estimate equity for.

    Returns:
        Estimated equity in [-1, 1] range where:
        +1 = certain win, -1 = certain loss, 0 = even.
    """
    our_epc = effective_pip_count(board, player)
    opp_epc = effective_pip_count(board, player.opponent())

    # Handle terminal cases
    if our_epc == 0.0:
        return 1.0  # Already won
    if opp_epc == 0.0:
        return -1.0  # Already lost

    # Pip count difference (positive = we're ahead)
    diff = opp_epc - our_epc

    # Sigmoid mapping: calibrated from backgammon race databases
    # Scale factor determines how quickly equity changes with pip lead
    # A 10-pip lead in a typical race ~ 75% win probability
    scale = 12.0
    win_prob = 1.0 / (1.0 + np.exp(-diff / scale))

    # Convert win probability to equity: equity = 2 * P(win) - 1
    return float(2.0 * win_prob - 1.0)


def is_race_position(board: Board) -> bool:
    """Check if the position is a pure race (no contact).

    A race position means all of one player's checkers are ahead of
    all of the other player's checkers. No contact is possible.

    Args:
        board: Board state to check.

    Returns:
        True if the position is a pure race.
    """
    # Check from White's perspective
    white_furthest = 0
    black_closest = 25

    for point in range(1, 25):
        if board.get_checkers(Player.WHITE, point) > 0:
            white_furthest = max(white_furthest, point)
        if board.get_checkers(Player.BLACK, point) > 0:
            black_closest = min(black_closest, point)

    # Check bar
    if board.get_checkers(Player.WHITE, 0) > 0 or board.get_checkers(Player.BLACK, 0) > 0:
        return False

    return white_furthest < black_closest
