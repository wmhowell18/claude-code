"""
Transformer Backgammon - GPU-optimized backgammon AI using transformer neural networks.
"""

__version__ = "0.1.0"

# Core exports
from backgammon.core.types import (
    Board,
    Move,
    MoveStep,
    Dice,
    Player,
    GameOutcome,
    Equity,
)

__all__ = [
    "Board",
    "Move",
    "MoveStep",
    "Dice",
    "Player",
    "GameOutcome",
    "Equity",
]
