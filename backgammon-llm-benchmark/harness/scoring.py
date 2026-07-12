"""Equity-loss scoring and BenchPR (PLAN.md §4.4).

Looks up the chosen decision's equity loss in the rollout data, aggregates
mean equity loss, best-move accuracy, and per-tier/per-track rollups, and
computes ``BenchPR = 500 x mean(equity_loss)`` on XG's money-equity scale.
The exact PR constant/cube handling is pinned and validated here (PLAN.md §9).
"""
