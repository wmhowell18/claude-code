"""Answer extraction + move normalization (PLAN.md §4.3).

Extracts the ``FINAL ANSWER`` line, parses slash-notation moves (bar//off,
combined dice) into the resulting board, and normalizes to end-positions so
order- and die-ordering-equivalent plays compare equal. Also parses cube actions.
"""
