"""Assign T1-T4 tiers (PLAN.md §3).

Derives tiers from rollout data: primarily the best-vs-second equity gap
(blunder margin) combined with decision rarity and game phase. Down-samples
near-free (gap < ~0.002) positions.
"""
