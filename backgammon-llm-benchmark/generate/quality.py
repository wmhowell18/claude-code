"""Quiz-eligibility quality gate (PLAN.md §3, §3.2).

A benchmark position only *measures* something if it poses a real decision. Three
kinds of degenerate position must be gated out of a quiz/benchmark set:

* **forced** — the mover has a single legal play, so there is nothing to choose;
* **unscoreable** — the rollout scored fewer than two moves, so any answer other
  than the one listed move can only ever tie or be penalised as "worst" with no
  gradient between real alternatives;
* **trivial** — every scored move has ~zero equity error (the classic end-of-race
  bear-off where "take two men off" is the whole story): all answers score 0 loss,
  so the position separates no one.

This module is the single source of truth for that gate. It is written to run at
**set-selection time** in the pipeline (after rollouts exist) and is reused by the
human-quiz builder so the quiz and the pipeline agree exactly. It depends only on
:mod:`bgcore`.

Threshold choice (``MIN_CHECKER_SPREAD_MP`` = 10 mpt = 0.01 equity points): the
sampler already down-samples near-free decisions below ``sample.NEAR_FREE_GAP``
(2 mpt); 10 mpt sits an order of magnitude above typical rollout noise (best-move
``std_err`` is ~1e-3 equity ≈ 1 mpt) while still admitting genuinely close — but
real — decisions. A trivial bear-off whose moves are all 0.0 mpt dies here; a
non-trivial bear-off with a real slotting/ordering choice keeps a >10 mpt gap and
survives, which is exactly the requested semantics (no bear-off special-casing).
"""

from __future__ import annotations

from typing import Any

from bgcore import moves as _moves
from bgcore.board import Board, flip

__all__ = [
    "MIN_CHECKER_SPREAD_MP",
    "MIN_LEGAL_MOVES",
    "MIN_SCORED_MOVES",
    "CUBE_ACTIONS_REQUIRED",
    "display_frame_board",
    "quiz_eligible",
    "filter_eligible",
]

MIN_CHECKER_SPREAD_MP = 10.0
MIN_LEGAL_MOVES = 2
MIN_SCORED_MOVES = 2
CUBE_ACTIONS_REQUIRED = 3


def _checker_moves(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    return list((rollout.get("checker") or {}).get("moves") or [])


def _scored(moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in moves if m.get("move")]


def display_frame_board(record: dict[str, Any], rollout: dict[str, Any]) -> Board:
    """Board in the frame the position is scored/shown in.

    Mirrors the human quiz's ``_display_board``: ``board_json`` is authoritative and
    already mover-relative, so cube positions are never flipped. A checker position
    is presented as ``flip(board_json)`` only when that is the orientation in which
    the rollout's move list is legal (a subset of the pilot rollouts were computed
    in the color-mirror frame). Ties go to ``board_json``.
    """
    b = Board.from_json(record["board_json"])
    if record.get("decision_type") == "cube":
        return b
    moves = _scored(_checker_moves(rollout))

    def n_legal(bd: Board) -> int:
        c = 0
        for m in moves:
            try:
                if _moves.is_legal(bd, m["move"]):
                    c += 1
            except Exception:  # noqa: BLE001
                pass
        return c

    fb = flip(b)
    return fb if n_legal(fb) > n_legal(b) else b


def quiz_eligible(record: dict[str, Any], rollout: dict[str, Any]) -> tuple[bool, str | None]:
    """Return ``(eligible, reason)`` — ``reason`` is ``None`` iff eligible.

    Applies, in order: (a) >=2 legal moves in the display frame, (b) >=2 scored
    rollout moves, (c) best-to-worst loss spread >= ``MIN_CHECKER_SPREAD_MP``. Cube
    positions require all three actions scored. ``reason`` names the first rule
    that fails, for logging.
    """
    dt = record.get("decision_type")
    if dt == "cube":
        errmap = (rollout.get("cube") or {}).get("error_mp") or {}
        if len(errmap) < CUBE_ACTIONS_REQUIRED:
            return False, ("cube: only %d action(s) scored (need %d)"
                           % (len(errmap), CUBE_ACTIONS_REQUIRED))
        return True, None

    board = display_frame_board(record, rollout)
    n_legal = len(_moves.generate_moves(board))
    if n_legal < MIN_LEGAL_MOVES:
        return False, "forced: only %d legal move%s in the display frame" % (
            n_legal, "" if n_legal == 1 else "s")

    scored = _scored(_checker_moves(rollout))
    if len(scored) < MIN_SCORED_MOVES:
        return False, "unscoreable: rollout lists only %d scored move%s" % (
            len(scored), "" if len(scored) == 1 else "s")

    spread = max((abs(float(m.get("error_mp", 0.0))) for m in scored), default=0.0)
    if spread < MIN_CHECKER_SPREAD_MP:
        return False, ("trivial: best-to-worst spread %.1f mpt < %.0f (no real decision)"
                       % (spread, MIN_CHECKER_SPREAD_MP))
    return True, None


def filter_eligible(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Partition ``(record, rollout)`` pairs into eligible records + exclusions.

    Returns ``(eligible_records, excluded)`` where ``excluded`` is a list of
    ``(position_id, reason)``. Order is preserved.
    """
    eligible: list[dict[str, Any]] = []
    excluded: list[tuple[str, str]] = []
    for record, rollout in pairs:
        ok, reason = quiz_eligible(record, rollout)
        if ok:
            eligible.append(record)
        else:
            excluded.append((record.get("position_id", "?"), reason or "ineligible"))
    return eligible, excluded
