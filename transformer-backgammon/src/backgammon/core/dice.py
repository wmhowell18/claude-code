"""Dice utilities for backgammon.

This module handles dice rolling, dice combinations, and related utilities.
"""

from typing import List, Tuple
import numpy as np
from backgammon.core.types import Dice


def all_dice_rolls() -> List[Dice]:
    """Generate all 21 unique dice outcomes.

    In backgammon, (2,3) and (3,2) are equivalent, so there are 21 unique rolls:
    - 6 doubles: (1,1), (2,2), (3,3), (4,4), (5,5), (6,6)
    - 15 non-doubles: (1,2), (1,3), ..., (5,6)

    Returns:
        List of all 21 unique dice combinations, sorted
    """
    rolls = []
    for die1 in range(1, 7):
        for die2 in range(die1, 7):  # die2 >= die1 to avoid duplicates
            rolls.append((die1, die2))
    return rolls


def is_doubles(dice: Dice) -> bool:
    """Check if dice roll is doubles.

    Args:
        dice: Dice roll tuple

    Returns:
        True if both dice show the same value
    """
    return dice[0] == dice[1]


def dice_values(dice: Dice) -> List[int]:
    """Get the dice values to use for moves.

    For doubles, you get 4 moves. For non-doubles, you get 2 moves.

    Args:
        dice: Dice roll tuple

    Returns:
        List of dice values (length 2 or 4)

    Examples:
        >>> dice_values((3, 5))
        [3, 5]
        >>> dice_values((4, 4))
        [4, 4, 4, 4]
    """
    if is_doubles(dice):
        return [dice[0]] * 4
    else:
        return [dice[0], dice[1]]


def roll_dice(rng_key: np.random.Generator) -> Dice:
    """Roll two dice.

    Args:
        rng_key: NumPy random generator

    Returns:
        Tuple of (die1, die2) where each is 1-6
    """
    die1 = int(rng_key.integers(1, 7))
    die2 = int(rng_key.integers(1, 7))
    return (die1, die2)


def canonicalize_dice(dice: Dice) -> Dice:
    """Canonicalize dice to standard form (smaller value first).

    Args:
        dice: Dice roll tuple

    Returns:
        Tuple with min value first, max value second

    Examples:
        >>> canonicalize_dice((5, 3))
        (3, 5)
        >>> canonicalize_dice((4, 4))
        (4, 4)
    """
    return tuple(sorted(dice))  # type: ignore


def dice_to_string(dice: Dice) -> str:
    """Convert dice to readable string.

    Args:
        dice: Dice roll tuple

    Returns:
        String representation

    Examples:
        >>> dice_to_string((3, 5))
        '3-5'
        >>> dice_to_string((4, 4))
        'Double 4s'
    """
    if is_doubles(dice):
        return f"Double {dice[0]}s"
    else:
        return f"{dice[0]}-{dice[1]}"


# Precompute all dice rolls for efficiency
ALL_DICE_ROLLS = all_dice_rolls()

# Probabilities for each dice outcome
# Doubles have probability 1/36, non-doubles have probability 2/36 = 1/18
DICE_PROBABILITIES = {
    dice: 1/36 if is_doubles(dice) else 1/18
    for dice in ALL_DICE_ROLLS
}
