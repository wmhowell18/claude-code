"""Validate a single position record (PLAN.md §7 acceptance).

Checks a record against schema/position.schema.json and verifies the XGID<->JSON
round-trip is lossless and the ASCII/image renders regenerate deterministically.
"""
