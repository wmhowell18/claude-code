"""XGID <-> board_json (PLAN.md §1.1).

Parses/serializes the canonical XGID (board + cube + dice + score + turn),
converts to and from ``board_json``, and produces the normalized (color/symmetry)
dedup key. Round-trip XGID<->JSON must be lossless (Phase 0 acceptance, PLAN.md §7).
"""
