"""Per-run cost tracking + budget accounting (PLAN.md §5.2-5.3).

Accumulates OpenRouter's authoritative per-request dollar cost, rolls it up
into results JSON, and enforces the fixed-$ budget track: stop when the budget
is exhausted; score unanswered positions as worst-case.
"""
