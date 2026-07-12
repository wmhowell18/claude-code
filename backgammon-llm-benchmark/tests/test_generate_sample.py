"""Tests: stratified, deterministic candidate sampling (PLAN.md §1.4, §3.3)."""

from dataclasses import dataclass

import pytest

from generate import sample
from generate.sample import SampleTargets, stratified_sample, stratum_counts


@dataclass
class Cand:
    decision_type: str
    play_mode: str
    phase: str
    equity_gap: float | None = None
    keep_near_free: bool = False
    tag: int = 0


def _pool(n_checker_money=40, n_checker_match=30, n_cube_money=15, n_cube_match=15):
    pool = []
    t = 0
    phases_checker = ["race", "holding-game", "priming", "blitz", "backgame", "bearoff"]
    for i in range(n_checker_money):
        pool.append(Cand("checker", "money", phases_checker[i % len(phases_checker)], tag=t)); t += 1
    for i in range(n_checker_match):
        pool.append(Cand("checker", "match", phases_checker[i % len(phases_checker)], tag=t)); t += 1
    for i in range(n_cube_money):
        pool.append(Cand("cube", "money", "cube-action", tag=t)); t += 1
    for i in range(n_cube_match):
        pool.append(Cand("cube", "match", "cube-action", tag=t)); t += 1
    return pool


def test_sample_is_deterministic_given_seed():
    pool = _pool()
    a = stratified_sample(pool, 30, seed=123)
    b = stratified_sample(pool, 30, seed=123)
    assert [c.tag for c in a] == [c.tag for c in b]


def test_sample_changes_with_seed():
    pool = _pool()
    a = stratified_sample(pool, 30, seed=1)
    b = stratified_sample(pool, 30, seed=2)
    assert [c.tag for c in a] != [c.tag for c in b]


def test_sample_respects_checker_cube_ratio():
    pool = _pool()
    out = stratified_sample(pool, 50, seed=7)
    counts = stratum_counts(out)
    checker = counts["decision_type"].get("checker", 0)
    cube = counts["decision_type"].get("cube", 0)
    # target 70/30 on 50 -> ~35/15; allow largest-remainder slack
    assert checker == pytest.approx(35, abs=3)
    assert cube == pytest.approx(15, abs=3)


def test_sample_respects_money_match_ratio():
    pool = _pool()
    out = stratified_sample(pool, 50, seed=9)
    counts = stratum_counts(out)
    money = counts["play_mode"].get("money", 0)
    match = counts["play_mode"].get("match", 0)
    assert money == pytest.approx(30, abs=4)
    assert match == pytest.approx(20, abs=4)


def test_sample_spreads_across_phases():
    pool = _pool()
    out = stratified_sample(pool, 30, seed=3)
    phases = stratum_counts(out)["phase"]
    # checker phases should be diverse, not all one bucket
    checker_phases = {p for p in phases if p != "cube-action"}
    assert len(checker_phases) >= 4


def test_near_free_decisions_downsampled():
    pool = [Cand("checker", "money", "race", equity_gap=0.0005, tag=i) for i in range(10)]
    pool += [Cand("checker", "money", "race", equity_gap=0.05, tag=100 + i) for i in range(10)]
    out = stratified_sample(pool, 20, seed=0)
    # the 10 near-free (gap < 0.002) ones are filtered out entirely
    assert all(c.equity_gap >= 0.002 for c in out)
    assert len(out) == 10


def test_near_free_kept_when_flagged():
    pool = [Cand("checker", "money", "race", equity_gap=0.0005, keep_near_free=True, tag=i)
            for i in range(5)]
    out = stratified_sample(pool, 5, seed=0)
    assert len(out) == 5


def test_sample_returns_all_when_target_exceeds_pool():
    pool = _pool(2, 2, 1, 1)
    out = stratified_sample(pool, 100, seed=0)
    assert len(out) == len(pool)


def test_targets_ratio_helpers():
    t = SampleTargets(checker_ratio=0.7, money_ratio=0.6)
    assert t.cube_ratio() == pytest.approx(0.3)
    assert t.match_ratio() == pytest.approx(0.4)
