"""Core backgammon domain layer (PLAN.md §1).

``bgcore`` is the canonical, dependency-light (stdlib-only) model that every other
module imports: the board representation (:mod:`bgcore.board`), full checker-play
rules (:mod:`bgcore.moves`), and XG-style move/cube notation
(:mod:`bgcore.notation`).

The single source of truth for a position's checker layout is the mover-perspective
26-item ``points`` array documented in :mod:`bgcore.board`.
"""

from bgcore.board import (
    Board,
    BoardError,
    canonical_key,
    flip,
    normalize,
    pip_counts,
    validate,
)

__all__ = [
    "Board",
    "BoardError",
    "canonical_key",
    "flip",
    "normalize",
    "pip_counts",
    "validate",
]
