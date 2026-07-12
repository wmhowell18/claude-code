"""Human-difficulty tiering (PLAN.md §3).

Tiers measure **how hard a decision is for top humans**, not how costly the
mistake is. The equity gap is deliberately *not* the tier axis (§3); it only
down-samples near-free decisions elsewhere in the pipeline.

Three pieces live here:

1. :func:`classify_phase` — a board-topology phase classifier producing one of
   the PLAN §3 phase tags (``race``, ``holding-game``, ``blitz``, ``priming``,
   ``backgame``, ``bearoff``, ``opening-ish``, ``cube-action``).
2. :class:`HumanErrorModel` — a pluggable protocol predicting
   ``expert_miss_rate`` and ``expected_expert_loss`` (EEL, millipoints).
   :class:`TaxonomyPrior` is the default implementation: the known-hard taxonomy
   prior of §3.1, seeded per phase and nudged by a few board features.
3. :func:`tier_for` / :func:`assign_tier` — the provisional §3.2 threshold table
   mapping ``(miss_rate, EEL)`` to ``T1``–``T4`` and recording
   ``difficulty_source``.

Everything is deterministic and stdlib-only; a real fitted human-error model
(§3.1 source 2) or panel calibration (source 3) can be dropped in later by
implementing the :class:`HumanErrorModel` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from bgcore.board import Board, flip, pip_counts

PHASES = (
    "race",
    "holding-game",
    "blitz",
    "priming",
    "backgame",
    "bearoff",
    "opening-ish",
    "cube-action",
)

TIERING_VERSION = "taxonomy-prior-1"


# ---------------------------------------------------------------------------
# Board feature helpers (mover-relative frame; see bgcore.board)
# ---------------------------------------------------------------------------


def _mover_min_index(board: Board) -> int:
    """Lowest board index holding a mover checker (0 if on the bar)."""
    if board.bar["x"] > 0:
        return 0
    for p in range(1, 25):
        if board.points[p] > 0:
            return p
    return 25


def _opp_max_index(board: Board) -> int:
    """Highest board index holding an opponent checker (25 if on the bar)."""
    if board.bar["o"] > 0:
        return 25
    for p in range(24, 0, -1):
        if board.points[p] < 0:
            return p
    return 0


def has_contact(board: Board) -> bool:
    """True if the two sides can still collide (not a pure race)."""
    return _mover_min_index(board) < _opp_max_index(board)


def _made_points(board: Board, lo: int, hi: int) -> list[int]:
    """Mover points (>=2 checkers) with board index in ``[lo, hi]``."""
    return [p for p in range(lo, hi + 1) if board.points[p] >= 2]


def _longest_prime(board: Board) -> int:
    """Longest run of consecutive mover-made points anywhere on the board."""
    best = run = 0
    for p in range(1, 25):
        if board.points[p] >= 2:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _opp_all_home(board: Board) -> bool:
    return flip(board).all_home()


def _near_start(board: Board) -> bool:
    """Heuristic 'still in the opening' test: high pips, nothing borne off."""
    px, po = pip_counts(board)
    return (
        px >= 152
        and po >= 152
        and board.off["x"] == 0
        and board.off["o"] == 0
        and board.bar["x"] == 0
        and board.bar["o"] == 0
    )


# ---------------------------------------------------------------------------
# Phase classifier
# ---------------------------------------------------------------------------


def classify_phase(board: Board) -> str:
    """Classify a board into one PLAN §3 phase tag from its topology.

    Priority-ordered so the most specific/decision-defining phase wins:
    cube-action → bearoff → race → opening-ish → backgame → blitz → priming →
    holding-game (fallback for generic contact positions).
    """
    if board.decision_type == "cube":
        return "cube-action"

    contact = has_contact(board)
    if not contact:
        # both bearing off, or a pure race toward the finish
        if board.all_home() and _opp_all_home(board):
            return "bearoff"
        return "race"

    # The special contact structures are checked before the generic
    # "opening-ish" bucket so a backgame / blitz / prime (which also have high
    # pip counts) isn't swallowed by the near-start heuristic.

    # backgame: two or more mover anchors deep in the opponent's home (1..6),
    # trailing in the race (holding on for a late shot).
    deep_anchors = _made_points(board, 1, 6)
    px, po = pip_counts(board)
    behind = px > po
    if len(deep_anchors) >= 2 and behind:
        return "backgame"

    # blitz: attacking with a made home board while the opponent is disrupted
    # (on the bar or with back blots in our home) — happens early/mid game.
    home_points_made = len(_made_points(board, 19, 24))
    opp_back_blots = any(board.points[p] == -1 for p in range(19, 25))
    if (board.bar["o"] > 0 or opp_back_blots) and home_points_made >= 2:
        return "blitz"

    # priming: a long wall (4+ consecutive made points) restraining the opponent
    if _longest_prime(board) >= 4:
        return "priming"

    # opening-ish: still near the start (high pips, nothing off/on-bar) with no
    # special structure yet.
    if _near_start(board):
        return "opening-ish"

    # holding-game default: contact position with (usually) a single anchor,
    # trailing in the race, waiting for a shot.
    return "holding-game"


# ---------------------------------------------------------------------------
# Human-error model protocol + taxonomy-prior default
# ---------------------------------------------------------------------------


@runtime_checkable
class HumanErrorModel(Protocol):
    """Predicts expert difficulty for a position (PLAN.md §3.1).

    ``predict`` returns ``(expert_miss_rate, expected_expert_loss)`` where the
    miss rate is a probability in ``[0, 1]`` that a reference PR-3 human picks a
    non-best move, and EEL is the predicted mean equity loss in millipoints.
    ``rollout`` (optional) is a rollout record (schema/rollout.schema.json) whose
    ``equity_gap`` and move spread can sharpen the estimate.
    """

    source: str

    def predict(self, board: Board, rollout: dict[str, Any] | None = ...) -> tuple[float, float]:
        ...


# Per-phase base priors: (miss_rate, EEL millipoints). Seeded from the §3.1
# known-hard taxonomy: backgame timing / prime-vs-prime / blitz windows /
# too-good decisions are error hot-spots; races and openings are near-free.
_PHASE_PRIOR: dict[str, tuple[float, float]] = {
    "race": (0.010, 0.5),
    "bearoff": (0.060, 2.0),
    "opening-ish": (0.030, 1.2),
    "holding-game": (0.140, 4.5),
    "priming": (0.200, 7.0),
    "blitz": (0.220, 7.5),
    "backgame": (0.360, 12.0),
    "cube-action": (0.180, 6.5),
}


@dataclass
class TaxonomyPrior:
    """Default :class:`HumanErrorModel`: the §3.1 known-hard taxonomy prior.

    Looks up a per-phase base ``(miss_rate, EEL)`` and applies small, bounded
    feature nudges (contact-with-checkers-on-the-bar, deep primes, tight cube
    windows). Deterministic; ``source`` is ``"taxonomy-prior"``.
    """

    source: str = "taxonomy-prior"

    def predict(
        self, board: Board, rollout: dict[str, Any] | None = None
    ) -> tuple[float, float]:
        phase = classify_phase(board)
        miss, eel = _PHASE_PRIOR.get(phase, (0.12, 4.0))

        # feature nudges -------------------------------------------------
        if board.bar["x"] > 0 or board.bar["o"] > 0:
            miss += 0.03
            eel += 1.0
        prime = _longest_prime(board)
        if prime >= 5:  # full/near-full prime: sharper timing decisions
            miss += 0.04
            eel += 1.5

        # cube windows: a tight double/take/pass margin is where experts bleed.
        if rollout is not None and rollout.get("decision_type") == "cube":
            cube = rollout.get("cube", {})
            dt = cube.get("double_take_equity")
            nd = cube.get("no_double_equity")
            if dt is not None and nd is not None and abs(dt - nd) < 0.05:
                miss += 0.06
                eel += 2.0

        miss = max(0.0, min(1.0, round(miss, 4)))
        eel = max(0.0, round(eel, 3))
        return miss, eel


# ---------------------------------------------------------------------------
# Threshold table (provisional, PLAN.md §3.2)
# ---------------------------------------------------------------------------


def tier_for(miss_rate: float, eel: float) -> str:
    """Map ``(miss_rate, EEL)`` to a tier per the provisional §3.2 table.

    * T4 — miss rate > 30% **or** EEL > 10 mpt (even world-class players err).
    * T1 — miss rate < 2% **and** EEL < 1 mpt (experts essentially never err).
    * T3 — miss rate 10–30%.
    * T2 — miss rate 2–10% (the remaining band).
    """
    if miss_rate > 0.30 or eel > 10.0:
        return "T4"
    if miss_rate < 0.02 and eel < 1.0:
        return "T1"
    if miss_rate >= 0.10:
        return "T3"
    return "T2"


@dataclass
class TierResult:
    tier: str
    expert_miss_rate: float
    expected_expert_loss: float
    phase: str
    difficulty_source: str

    def to_fields(self) -> dict[str, Any]:
        """Position-record fields for this tier assignment."""
        return {
            "tier": self.tier,
            "expert_miss_rate": self.expert_miss_rate,
            "expected_expert_loss": self.expected_expert_loss,
            "phase": self.phase,
            "difficulty_source": self.difficulty_source,
        }


def assign_tier(
    board: Board,
    *,
    model: HumanErrorModel | None = None,
    rollout: dict[str, Any] | None = None,
) -> TierResult:
    """Assign a tier to ``board`` using ``model`` (default :class:`TaxonomyPrior`).

    Records the phase and the model's ``source`` as ``difficulty_source`` so the
    provenance of every tier assignment is auditable (PLAN.md §3.1).
    """
    m = model or TaxonomyPrior()
    miss, eel = m.predict(board, rollout)
    phase = classify_phase(board)
    return TierResult(
        tier=tier_for(miss, eel),
        expert_miss_rate=miss,
        expected_expert_loss=eel,
        phase=phase,
        difficulty_source=getattr(m, "source", "taxonomy-prior"),
    )
