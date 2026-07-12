"""Versioned prompt templates per track (PLAN.md §4.2).

Holds the fixed system prompt (notation rules + required FINAL ANSWER format)
and the per-track payload builders (text = ascii + board_json; image = png).
Each template carries a ``prompt_version`` so runs are reproducible.
"""
