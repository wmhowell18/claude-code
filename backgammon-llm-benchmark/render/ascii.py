"""Deterministic ASCII board render (PLAN.md §1.1).

Renders ``board_json`` to a fixed, versioned monospace template (2-high point
stacks with counts, bar in the middle, header with pips/cube/dice/score). Output
is deterministic given the board; the template string carries ``ascii_render_version``.
"""
