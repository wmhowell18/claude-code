"""Assign T1-T4 tiers (PLAN.md §3).

Tiers measure difficulty for top humans, not blunder cost. Estimates
expected_expert_loss / expert_miss_rate via the known-hard taxonomy prior, a
human-error model trained on analyzed public match corpora, and expert-panel
calibration. Equity gap is only a filter: down-samples near-free
(gap < ~0.002) decisions.
"""
