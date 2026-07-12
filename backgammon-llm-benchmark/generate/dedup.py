"""Blocklist + intra-set dedup (PLAN.md §2.3).

Rejects candidates whose normalized XGID matches the versioned public-source
blocklist (``data/blocklist/``), and removes near-duplicates within the dataset
via color/symmetry normalization so tiers aren't inflated.
"""
