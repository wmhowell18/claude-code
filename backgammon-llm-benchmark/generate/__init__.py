"""Dataset construction pipeline (PLAN.md §2-3).

Stages, each a stdlib-only module:

* :mod:`generate.gnubg` — GNU BG integration (command builders + output parsers;
  one isolated subprocess shim).
* :mod:`generate.selfplay` — self-play orchestration -> candidate decisions.
* :mod:`generate.dedup` — canonical/mirror dedup + public-source blocklist.
* :mod:`generate.sample` — deterministic stratified sampling toward §3.3 targets.
* :mod:`generate.tiering` — phase classifier + human-difficulty tiering (T1-T4).
* :mod:`generate.contamination` — canary, held-out hashing, redaction, findability.
* :mod:`generate.records` — assemble + validate full position records.

Flow: selfplay -> dedup -> sample -> tiering -> records (see
``scripts/build_dataset.py`` for the end-to-end wiring, with a canned ``--dry-run``).
"""
