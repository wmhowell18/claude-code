"""Per-run cost tracking + budget accounting (PLAN.md §5.2-5.3).

Accumulates OpenRouter's authoritative per-request dollar cost into per-model and
per-run totals, and enforces the fixed-$ budget track: a :class:`BudgetGuard`
stops issuing requests once the budget would be exceeded and records the spend, so
positions still unanswered at exhaustion can be scored worst-case by the runner
(PLAN §5.3). The budget-spending *strategy* supported here is self-consistency:
draw ``k`` samples and take the majority vote by equivalent-move class.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from bgcore import moves as _moves
from bgcore.board import Board
from harness.parse import ParseOutcome

__all__ = ["CostTracker", "BudgetExceeded", "BudgetGuard", "majority_vote"]


class CostTracker:
    """Running totals of request cost, overall and per model."""

    def __init__(self) -> None:
        self._by_model: dict[str, float] = defaultdict(float)
        self._count_by_model: dict[str, int] = defaultdict(int)

    def add(self, model: str, cost_usd: float | None) -> float:
        c = float(cost_usd or 0.0)
        self._by_model[model] += c
        self._count_by_model[model] += 1
        return c

    @property
    def total(self) -> float:
        return sum(self._by_model.values())

    def by_model(self) -> dict[str, float]:
        return dict(self._by_model)

    def requests_by_model(self) -> dict[str, int]:
        return dict(self._count_by_model)

    def summary(self) -> dict[str, Any]:
        return {
            "total_usd": self.total,
            "by_model": self.by_model(),
            "requests_by_model": self.requests_by_model(),
        }


class BudgetExceeded(RuntimeError):
    """Raised by :meth:`BudgetGuard.spend` when a charge would exceed the budget."""


@dataclass
class BudgetGuard:
    """Fixed-dollar budget for the budget track (PLAN §5.3).

    ``budget_usd = None`` means untracked (always allowed). ``spent`` accumulates
    actual charges; :meth:`can_spend` gates *before* a request using an estimate,
    :meth:`spend` records the real cost after it.
    """

    budget_usd: float | None = None
    spent: float = 0.0
    stopped: bool = field(default=False)

    @property
    def tracked(self) -> bool:
        return self.budget_usd is not None

    @property
    def remaining(self) -> float:
        if self.budget_usd is None:
            return float("inf")
        return self.budget_usd - self.spent

    @property
    def exhausted(self) -> bool:
        return self.tracked and self.remaining <= 0

    def can_spend(self, estimate: float = 0.0) -> bool:
        """True if a request costing ~``estimate`` may be issued."""
        if not self.tracked:
            return True
        if self.stopped:
            return False
        return self.remaining - max(0.0, estimate) >= 0

    def spend(self, cost_usd: float | None, *, strict: bool = False) -> float:
        """Record a real charge. With ``strict`` raise if it exceeds the budget."""
        c = float(cost_usd or 0.0)
        if self.tracked and strict and c > self.remaining + 1e-12:
            raise BudgetExceeded(f"charge {c} exceeds remaining {self.remaining}")
        self.spent += c
        if self.exhausted:
            self.stopped = True
        return c


# -- self-consistency -----------------------------------------------------


def majority_vote(
    board: Board,
    outcomes: Sequence[ParseOutcome],
) -> tuple[ParseOutcome | None, list[tuple[ParseOutcome, int]]]:
    """Majority vote over samples by *equivalent-move class* (PLAN §4.6, §5.3).

    Checker answers are clustered by resulting position (``moves_equivalent``);
    cube answers by canonical action string. Only ``parsed`` outcomes vote.
    Returns ``(winner, tally)`` where ``tally`` is ``[(representative, votes), …]``
    ordered by descending votes (ties broken by first appearance). ``winner`` is
    ``None`` if no sample parsed.
    """
    parsed = [o for o in outcomes if o.status == "parsed"]
    if not parsed:
        return None, []

    clusters: list[dict[str, Any]] = []  # {rep, count, key}

    def same(a: ParseOutcome, b: ParseOutcome) -> bool:
        if a.decision_type != b.decision_type:
            return False
        if a.decision_type == "cube":
            return (a.cube_action or a.move) == (b.cube_action or b.move)
        try:
            return _moves.moves_equivalent(board, a.answer_text or "", b.answer_text or "")
        except Exception:  # noqa: BLE001
            return (a.move or a.answer_text) == (b.move or b.answer_text)

    for o in parsed:
        for c in clusters:
            if same(o, c["rep"]):
                c["count"] += 1
                break
        else:
            clusters.append({"rep": o, "count": 1})

    clusters.sort(key=lambda c: -c["count"])  # stable -> ties keep first appearance
    tally = [(c["rep"], c["count"]) for c in clusters]
    return tally[0][0], tally
