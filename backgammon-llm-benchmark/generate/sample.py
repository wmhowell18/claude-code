"""Stratified sampling of candidate decisions (PLAN.md §1.4, §3.3).

Selects a subset of candidate decisions toward the target strata:

* **decision type** — ~70% checker / 30% cube (PLAN §1.4).
* **play mode** — ~60% money / 40% match (PLAN §1.4).
* **phase diversity** — spread across the §3 phase tags rather than piling into
  whatever the bots played most.

Sampling is **deterministic given a seed**. Near-free decisions (rollout
``equity_gap`` < ~0.002) are down-sampled when gap info is available, since a
choice that barely matters measures nothing (PLAN §3, §3.2).

The sampler operates on any object exposing ``decision_type``, ``play_mode`` and
``phase`` attributes (e.g. :class:`generate.selfplay.Candidate`); an optional
``equity_gap`` attribute enables the near-free filter.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

NEAR_FREE_GAP = 0.002


@dataclass(frozen=True)
class SampleTargets:
    """Target stratum ratios (PLAN §1.4, §3.3)."""

    checker_ratio: float = 0.70
    money_ratio: float = 0.60
    phase_diversity: bool = True
    near_free_gap: float = NEAR_FREE_GAP

    def cube_ratio(self) -> float:
        return 1.0 - self.checker_ratio

    def match_ratio(self) -> float:
        return 1.0 - self.money_ratio


def _attr(c: Any, name: str, default: Any = None) -> Any:
    if isinstance(c, dict):
        return c.get(name, default)
    return getattr(c, name, default)


def _is_near_free(c: Any, gap_threshold: float) -> bool:
    gap = _attr(c, "equity_gap", None)
    if gap is None:
        return False
    # a large third-option blunder can keep a small-gap decision meaningful
    if _attr(c, "keep_near_free", False):
        return False
    return abs(float(gap)) < gap_threshold


def _largest_remainder(total: int, weights: dict[Any, float]) -> dict[Any, int]:
    """Apportion ``total`` items across keys by ``weights`` (largest-remainder)."""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    wsum = sum(weights.values()) or 1.0
    raw = {k: total * (w / wsum) for k, w in weights.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    used = sum(alloc.values())
    remainder = sorted(weights, key=lambda k: (raw[k] - alloc[k], k), reverse=True)
    for k in remainder[: total - used]:
        alloc[k] += 1
    return alloc


def stratified_sample(
    candidates: Sequence[Any],
    n: int,
    *,
    targets: SampleTargets | None = None,
    seed: int = 0,
) -> list[Any]:
    """Return up to ``n`` candidates stratified toward ``targets``, deterministically.

    Strategy: apportion ``n`` across the four ``decision_type × play_mode`` cells
    by the target ratios (largest-remainder), then within each cell fill by
    round-robin over phases (for diversity) drawing from a seed-shuffled order.
    Fully deterministic for a given ``(candidates order, n, targets, seed)``.
    """
    t = targets or SampleTargets()
    rng = random.Random(seed)

    pool = [c for c in candidates if not _is_near_free(c, t.near_free_gap)]
    if n >= len(pool):
        return list(pool)

    # cell weights: decision_type × play_mode
    cell_weights = {
        ("checker", "money"): t.checker_ratio * t.money_ratio,
        ("checker", "match"): t.checker_ratio * t.match_ratio(),
        ("cube", "money"): t.cube_ratio() * t.money_ratio,
        ("cube", "match"): t.cube_ratio() * t.match_ratio(),
    }

    # bucket candidates by cell, and within a cell by phase
    cells: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for c in pool:
        cell = (_attr(c, "decision_type", "checker"), _attr(c, "play_mode", "money"))
        if cell not in cell_weights:
            cell = ("checker", "money")
        cells[cell][_attr(c, "phase", "unknown")].append(c)

    # only weight cells that actually have candidates, then reapportion
    avail = {cell: w for cell, w in cell_weights.items() if cells.get(cell)}
    alloc = _largest_remainder(n, avail)

    selected: list[Any] = []
    for cell, want in alloc.items():
        phase_map = cells.get(cell, {})
        # deterministic shuffle within each phase bucket
        phases = sorted(phase_map)
        for ph in phases:
            rng.shuffle(phase_map[ph])
        picked = _round_robin(phase_map, phases, want)
        selected.extend(picked)

    # top up if rounding/availability left us short of n
    if len(selected) < n:
        chosen_ids = {id(c) for c in selected}
        leftovers = [c for c in pool if id(c) not in chosen_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    return selected


def _round_robin(phase_map: dict[str, list[Any]], phases: list[str], want: int) -> list[Any]:
    """Draw ``want`` items round-robin across phase buckets for diversity."""
    out: list[Any] = []
    idx = {ph: 0 for ph in phases}
    active = [ph for ph in phases if phase_map.get(ph)]
    while len(out) < want and active:
        still: list[str] = []
        for ph in active:
            if len(out) >= want:
                break
            bucket = phase_map[ph]
            if idx[ph] < len(bucket):
                out.append(bucket[idx[ph]])
                idx[ph] += 1
            if idx[ph] < len(bucket):
                still.append(ph)
        active = still
    return out


def stratum_counts(candidates: Sequence[Any]) -> dict[str, dict[str, int]]:
    """Summarize a candidate set by decision_type / play_mode / phase (QA helper)."""
    out: dict[str, dict[str, int]] = {
        "decision_type": defaultdict(int),
        "play_mode": defaultdict(int),
        "phase": defaultdict(int),
    }
    for c in candidates:
        out["decision_type"][_attr(c, "decision_type", "checker")] += 1
        out["play_mode"][_attr(c, "play_mode", "money")] += 1
        out["phase"][_attr(c, "phase", "unknown")] += 1
    return {k: dict(v) for k, v in out.items()}
