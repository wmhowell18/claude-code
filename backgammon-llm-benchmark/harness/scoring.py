"""Equity-loss scoring and BenchPR (PLAN.md §4.4, docs/SCORING.md).

Ground truth is a rollout record (``schema/rollout.schema.json``). Scoring is a
lookup: for a checker play we find the equivalent move in the rollout's move list
and read its ``error_mp`` (error vs. best, in **millipoints**); for a cube
decision we read the millipoint error of the chosen action. Moves that are
equivalent-best within rollout noise score **0**. Unparseable / illegal answers
are scored as the **worst legal move** (PLAN §4.6) so bad formatting is penalised
but cannot go beyond the real range of the position.

Units. Rollout errors are millipoints (1 mpt = 0.001 equity points). We report
per-decision ``equity_loss`` in **equity points** and

    BenchPR = 500 x mean(equity_loss_points)

which is computed identically to XG's Performance Rating, so models and humans
sit on the same axis (PLAN §4.4). Roll-ups are provided per tier / per track /
per decision type, plus a bootstrap confidence interval on BenchPR.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from bgcore import moves as _moves
from bgcore.board import Board
from harness.parse import ParseOutcome

__all__ = [
    "MP_PER_POINT",
    "BENCHPR_CONSTANT",
    "DecisionScore",
    "worst_error_mp",
    "score_decision",
    "score_checker",
    "score_cube",
    "benchpr",
    "mean_equity_loss",
    "aggregate",
    "bootstrap_benchpr_ci",
]

MP_PER_POINT = 1000.0
BENCHPR_CONSTANT = 500.0
_EPS = 1e-9


@dataclass
class DecisionScore:
    """Score for one decision.

    ``equity_loss`` is in equity points (>= 0); ``equity_loss_mp`` the same in
    millipoints. ``is_best`` marks a rollout-best (or within-noise) answer.
    ``parse_failed`` is set when the answer was illegal/unparseable and scored as
    worst-legal. ``matched`` is the rollout move/action the answer resolved to.
    """

    equity_loss: float
    equity_loss_mp: float
    is_best: bool
    parse_failed: bool
    chosen: str | None
    matched: str | None
    detail: str


# -- rollout accessors ----------------------------------------------------


def _checker_moves(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    return list((rollout.get("checker") or {}).get("moves") or [])


def _cube_errs(rollout: dict[str, Any]) -> dict[str, float]:
    return dict((rollout.get("cube") or {}).get("error_mp") or {})


def worst_error_mp(rollout: dict[str, Any], decision_type: str | None = None) -> float:
    """Millipoint error of the worst option — the penalty for a failed answer."""
    dt = decision_type or rollout.get("decision_type")
    if dt == "cube":
        vals = [abs(float(v)) for v in _cube_errs(rollout).values()]
        return max(vals) if vals else 0.0
    mvs = _checker_moves(rollout)
    if not mvs:
        return 0.0
    return max(abs(float(m.get("error_mp", 0.0))) for m in mvs)


def _best_std_err_points(rollout: dict[str, Any]) -> float:
    mvs = _checker_moves(rollout)
    if not mvs:
        return 0.0
    best = min(mvs, key=lambda m: abs(float(m.get("error_mp", 0.0))))
    return abs(float(best.get("std_err", 0.0) or 0.0))


def _mk(loss_mp: float, is_best: bool, parse_failed: bool, chosen: str | None,
        matched: str | None, detail: str) -> DecisionScore:
    loss_mp = max(0.0, float(loss_mp))
    return DecisionScore(
        equity_loss=loss_mp / MP_PER_POINT,
        equity_loss_mp=loss_mp,
        is_best=is_best,
        parse_failed=parse_failed,
        chosen=chosen,
        matched=matched,
        detail=detail,
    )


# -- checker scoring ------------------------------------------------------


def score_checker(
    rollout: dict[str, Any],
    board: Board,
    outcome: ParseOutcome,
    *,
    noise_points: float | None = None,
) -> DecisionScore:
    if outcome.status != "parsed":
        return _mk(
            worst_error_mp(rollout, "checker"), False, True,
            outcome.answer_text, None,
            f"{outcome.status} -> worst-legal penalty",
        )

    answer = outcome.answer_text or outcome.move or ""
    mvs = _checker_moves(rollout)
    matched = None
    for m in mvs:
        rm = m.get("move")
        if not rm:
            continue
        try:
            if _moves.moves_equivalent(board, answer, rm):
                matched = m
                break
        except Exception:  # noqa: BLE001
            continue

    if matched is None:
        # Legal but not among the stored (top-K) moves: conservative worst penalty.
        return _mk(
            worst_error_mp(rollout, "checker"), False, False,
            outcome.move or answer, None,
            "legal move absent from rollout list -> worst penalty",
        )

    err_mp = abs(float(matched.get("error_mp", 0.0)))
    tol_points = noise_points if noise_points is not None else _best_std_err_points(rollout)
    is_best = (err_mp / MP_PER_POINT) <= (tol_points + _EPS)
    loss_mp = 0.0 if is_best else err_mp
    return _mk(
        loss_mp, is_best, False, outcome.move or answer, matched.get("move"),
        "best (within noise)" if is_best else f"equity loss {err_mp:.1f} mpt",
    )


# -- cube scoring ---------------------------------------------------------


def _cube_label(action: str, response: str | None) -> str:
    if action == "no double":
        return "No double"
    if action == "double":
        if response == "take":
            return "Double, Take"
        if response == "pass":
            return "Double, Pass"
        return "Double"
    if action == "take":
        return "Take"
    if action == "pass":
        return "Pass"
    return action


def score_cube(rollout: dict[str, Any], outcome: ParseOutcome) -> DecisionScore:
    if outcome.status != "parsed" or not outcome.cube_action:
        return _mk(
            worst_error_mp(rollout, "cube"), False, True,
            outcome.answer_text, None,
            f"{outcome.status} -> worst-action penalty",
        )
    cube = rollout.get("cube") or {}
    best_action = str(cube.get("best_action", "")).strip().lower()
    errmap = {k.strip().lower(): abs(float(v)) for k, v in _cube_errs(rollout).items()}
    action, response = outcome.cube_action
    label = _cube_label(action, response)
    ll = label.lower()

    def result(loss_mp: float, is_best: bool, detail: str) -> DecisionScore:
        return _mk(loss_mp, is_best, False, label, best_action or None, detail)

    if ll == best_action:
        return result(0.0, True, "matched best action")
    if ll in errmap:
        v = errmap[ll]
        return result(v, v <= _EPS, f"cube error {v:.1f} mpt")
    # Doubling correctness heuristics when only the coarse action is known.
    if ll == "double" and best_action in ("double, take", "double, pass", "too good"):
        return result(0.0, True, "doubling correct (best is a double)")
    if ll == "take" and best_action == "double, take":
        return result(0.0, True, "take correct")
    if ll == "pass" and best_action == "double, pass":
        return result(0.0, True, "pass correct")
    worst = max(errmap.values(), default=0.0)
    return result(worst, False, "action not in rollout -> worst penalty")


def score_decision(
    rollout: dict[str, Any],
    board: Board,
    outcome: ParseOutcome,
    *,
    noise_points: float | None = None,
) -> DecisionScore:
    """Score one decision against its rollout record."""
    dt = outcome.decision_type or rollout.get("decision_type") or board.decision_type
    if dt == "cube":
        return score_cube(rollout, outcome)
    return score_checker(rollout, board, outcome, noise_points=noise_points)


# -- aggregation ----------------------------------------------------------


def mean_equity_loss(losses_points: Sequence[float]) -> float:
    return sum(losses_points) / len(losses_points) if losses_points else 0.0


def benchpr(losses_points: Sequence[float]) -> float:
    """``500 x mean(equity_loss)`` in equity points (lower is better)."""
    return BENCHPR_CONSTANT * mean_equity_loss(losses_points)


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [float(r["equity_loss"]) for r in rows]
    best = [bool(r["is_best"]) for r in rows]
    return {
        "benchpr": benchpr(losses),
        "best_move_accuracy": (sum(best) / len(best)) if best else 0.0,
        "mean_equity_loss": mean_equity_loss(losses),
        "n": len(rows),
    }


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-decision rows into headline + per-tier/track/type roll-ups.

    Each row needs ``equity_loss`` (points), ``is_best`` (bool) and optionally
    ``tier``, ``track``, ``decision_type``. Returns the aggregate block used by
    :mod:`harness.report`, including a bootstrap CI on BenchPR.
    """
    rows = list(rows)
    losses = [float(r["equity_loss"]) for r in rows]
    top = _rollup(rows)
    ci_low, ci_high = bootstrap_benchpr_ci(losses)

    def by(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            k = r.get(key)
            if k is None:
                continue
            groups.setdefault(str(k), []).append(r)
        return {k: _rollup(v) for k, v in sorted(groups.items())}

    return {
        "benchpr": top["benchpr"],
        "benchpr_ci95": [ci_low, ci_high],
        "best_move_accuracy": top["best_move_accuracy"],
        "mean_equity_loss": top["mean_equity_loss"],
        "n": top["n"],
        "parse_failures": sum(1 for r in rows if r.get("parse_failed")),
        "per_tier": by("tier"),
        "per_track": by("track"),
        "per_decision_type": by("decision_type"),
    }


def bootstrap_benchpr_ci(
    losses_points: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on BenchPR (deterministic given ``seed``)."""
    n = len(losses_points)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        v = benchpr(losses_points)
        return (v, v)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [losses_points[rng.randrange(n)] for _ in range(n)]
        stats.append(benchpr(sample))
    stats.sort()
    lo = stats[max(0, int(math.floor((alpha / 2) * n_boot)))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return (lo, hi)
