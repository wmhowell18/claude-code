"""Answer extraction + move normalization (PLAN.md §4.3).

Pulls the model's committed answer out of free text: it looks for the strict
contract line emitted by :mod:`harness.prompts` — ``MOVE: <move>`` for checker
plays, ``ACTION: <...>`` for cube decisions — with tolerant fallbacks (also
accepts a ``FINAL ANSWER:`` prefix, or, failing that, the last plausible
move-looking / cube-looking line). The extracted answer is parsed via
:mod:`bgcore.notation`, checked for legality via :mod:`bgcore.moves`, and (for
checker plays) normalized to the canonical resulting-position move and matched
against the rollout's move list so order- and die-ordering-equivalent plays
compare equal.

The result is a structured :class:`ParseOutcome` with status ``parsed`` /
``illegal`` / ``unparseable``. Re-asking on ``unparseable`` up to N times is the
runner's policy, not this module's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bgcore import moves as _moves
from bgcore import notation as _notation
from bgcore.board import Board

__all__ = [
    "ParseStatus",
    "ParseOutcome",
    "extract_answer_text",
    "parse_checker",
    "parse_cube",
    "parse_answer",
]

ParseStatus = str  # "parsed" | "illegal" | "unparseable"

# Contract label lines, most-preferred first. Group 1 is the payload.
_MOVE_LINE_RE = re.compile(r"^\s*(?:final\s+answer|move)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE)
_ACTION_LINE_RE = re.compile(r"^\s*(?:final\s+answer|action)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE)

# A loose "looks like a move" matcher for the fallback path.
_MOVE_ISH_RE = re.compile(r"(?:bar|off|\d{1,2})(?:/(?:off|\d{1,2})\*?)+(?:\(\d+\))?", re.IGNORECASE)
_CANNOT_MOVE_RE = re.compile(r"\b(cannot move|can'?t move|no (?:legal )?(?:move|play)|dance)\b", re.IGNORECASE)
_CUBE_WORD_RE = re.compile(
    r"\b(no double|no-double|double|redouble|take|pass|drop|beaver|accept|decline)\b",
    re.IGNORECASE,
)


@dataclass
class ParseOutcome:
    """Structured result of parsing one model answer.

    Attributes
    ----------
    status:
        ``parsed`` (legal answer recovered), ``illegal`` (a move was parsed but is
        not legal here) or ``unparseable`` (no answer could be extracted/parsed).
    decision_type:
        ``checker`` or ``cube``.
    answer_text:
        The raw extracted answer string (post-label), or ``None``.
    move:
        Canonical resulting-position move notation (checker, when legal).
    rollout_move:
        The equivalent move string from the rollout's move list, when one was
        supplied and matched (lets scoring do a direct lookup).
    cube_action:
        ``(action, response)`` for cube decisions (e.g. ``("double", "take")``).
    detail:
        Human-readable note about how the answer was recovered or why it failed.
    used_fallback:
        True when the contract line was absent and a fallback heuristic was used.
    """

    status: ParseStatus
    decision_type: str
    answer_text: str | None = None
    move: str | None = None
    rollout_move: str | None = None
    cube_action: tuple[str, str | None] | None = None
    detail: str = ""
    used_fallback: bool = False
    raw: str = field(default="", repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "parsed"


# -- extraction -----------------------------------------------------------


def extract_answer_text(text: str, decision_type: str) -> tuple[str | None, bool]:
    """Return ``(answer_text, used_fallback)`` extracted from free text.

    Prefers the strict contract line (scanning from the bottom so a final answer
    wins over any earlier mention). Falls back to the last move-ish / cube-ish
    line when no contract line is present.
    """
    lines = text.splitlines()
    label_re = _ACTION_LINE_RE if decision_type == "cube" else _MOVE_LINE_RE
    for line in reversed(lines):
        m = label_re.match(line)
        if m:
            return m.group(1).strip().strip("`*_ "), False

    # Fallback: last plausible answer-looking line.
    if decision_type == "cube":
        for line in reversed(lines):
            if _CUBE_WORD_RE.search(line):
                return line.strip().strip("`*_ "), True
        return None, True

    for line in reversed(lines):
        if _CANNOT_MOVE_RE.search(line):
            return "Cannot Move", True
        m = _MOVE_ISH_RE.search(line)
        if m:
            # Grab all move tokens on the line, in order.
            toks = _MOVE_ISH_RE.findall(line)
            return " ".join(toks), True
    return None, True


# -- checker --------------------------------------------------------------


def _canonical_move(board: Board, chosen: str) -> str | None:
    """The canonical resulting-position notation for a legal ``chosen`` move."""
    for legal in _moves.legal_moves(board):
        if _moves.moves_equivalent(board, chosen, legal):
            return legal
    return None


def parse_checker(
    board: Board,
    text: str,
    *,
    rollout_moves: list[str] | None = None,
) -> ParseOutcome:
    """Extract + validate a checker play from ``text`` against ``board``."""
    answer, fallback = extract_answer_text(text, "checker")
    if not answer:
        return ParseOutcome(
            "unparseable", "checker", detail="no MOVE line and no move-like text",
            used_fallback=fallback, raw=text,
        )
    try:
        _notation.parse_move(answer)  # validate notation (raises on garbage)
    except _notation.NotationError as exc:
        return ParseOutcome(
            "unparseable", "checker", answer_text=answer,
            detail=f"notation error: {exc}", used_fallback=fallback, raw=text,
        )

    if not _moves.is_legal(board, answer):
        return ParseOutcome(
            "illegal", "checker", answer_text=answer,
            detail="parsed but not a legal play in this position",
            used_fallback=fallback, raw=text,
        )

    canonical = _canonical_move(board, answer)
    rollout_move = None
    if rollout_moves:
        for rm in rollout_moves:
            try:
                if _moves.moves_equivalent(board, answer, rm):
                    rollout_move = rm
                    break
            except Exception:  # noqa: BLE001 - a malformed rollout string shouldn't crash parse
                continue
    return ParseOutcome(
        "parsed", "checker", answer_text=answer, move=canonical,
        rollout_move=rollout_move,
        detail="matched rollout move" if rollout_move else "legal move",
        used_fallback=fallback, raw=text,
    )


# -- cube -----------------------------------------------------------------


def parse_cube(board: Board, text: str) -> ParseOutcome:
    """Extract + parse a cube decision from ``text``."""
    answer, fallback = extract_answer_text(text, "cube")
    if not answer:
        return ParseOutcome(
            "unparseable", "cube", detail="no ACTION line and no cube-like text",
            used_fallback=fallback, raw=text,
        )
    try:
        action, response = _notation.parse_cube_answer(answer)
    except _notation.NotationError as exc:
        return ParseOutcome(
            "unparseable", "cube", answer_text=answer,
            detail=f"cube parse error: {exc}", used_fallback=fallback, raw=text,
        )
    canonical = action if response is None else f"{action}, {response}"
    return ParseOutcome(
        "parsed", "cube", answer_text=answer, move=canonical,
        cube_action=(action, response), detail="parsed cube action",
        used_fallback=fallback, raw=text,
    )


def parse_answer(
    board: Board,
    text: str,
    *,
    decision_type: str | None = None,
    rollout_moves: list[str] | None = None,
) -> ParseOutcome:
    """Dispatch to :func:`parse_checker` / :func:`parse_cube` by decision type."""
    dt = decision_type or board.decision_type
    if dt == "cube":
        return parse_cube(board, text)
    return parse_checker(board, text, rollout_moves=rollout_moves)
