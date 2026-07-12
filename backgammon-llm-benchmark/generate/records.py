"""Assemble and validate full position records (PLAN.md §1.1; schema/position.schema.json).

Turns a mover-relative :class:`bgcore.board.Board` plus tier/split/provenance
metadata into the single JSON record that is the source of truth for both tracks
(text + image). Fills the redundant representations — ``xgid``, ``gnubg_id``,
``board_json`` and ``ascii`` — from the core layer, records render versions and
paths, and structurally validates against ``schema/position.schema.json``.

Validation is dependency-free (a small structural checker) so it runs in CI
without ``jsonschema``; when ``jsonschema`` *is* installed, :func:`validate_record`
additionally validates against the committed schema.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from bgcore.board import Board, validate as _validate_board
from ids import gnubg_id as _gnubg
from ids import xgid as _xgid
from render import ascii as _ascii
from render import svg as _svg

_REQUIRED = ("position_id", "xgid", "board_json", "ascii", "tier", "decision_type")
_TIERS = {"T1", "T2", "T3", "T4"}
_DECISIONS = {"checker", "cube"}
_SPLITS = {"pilot", "dev", "heldout"}


def position_id_for(board: Board, *, prefix: str = "bg") -> str:
    """Stable internal id derived from the normalized XGID hash (PLAN.md §1.1)."""
    key = _xgid.normalized_key(_xgid.board_to_xgid(board))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def build_record(
    board: Board,
    *,
    tier: str,
    split: str,
    decision_type: str | None = None,
    play_mode: str | None = None,
    phase: str | None = None,
    expected_expert_loss: float | None = None,
    expert_miss_rate: float | None = None,
    difficulty_source: str | None = None,
    rollout_ref: str | None = None,
    canary: str | None = None,
    created: str | None = None,
    image_dir: str | None = None,
    include_image_paths: bool = True,
) -> dict[str, Any]:
    """Assemble a full position record from a board + tiering/provenance fields.

    Derives ``xgid``, ``gnubg_id`` (position + match), ``board_json`` and the
    deterministic ``ascii`` render, and records the ASCII/image render versions.
    Image *paths* are recorded (relative to ``image_dir``); actual PNG/SVG bytes
    are produced separately by :mod:`render` so this stays render-dependency-free.
    """
    _validate_board(board)
    xgid = _xgid.board_to_xgid(board)
    pid = position_id_for(board)
    dt = decision_type or board.decision_type

    record: dict[str, Any] = {
        "position_id": pid,
        "xgid": xgid,
        "gnubg_id": _gnubg.board_to_ids(board),
        "board_json": board.to_json(),
        "ascii": _ascii.render(board),
        "ascii_render_version": _ascii.ASCII_RENDER_VERSION,
        "decision_type": dt,
        "tier": tier,
        "split": split,
    }
    if play_mode is not None:
        record["play_mode"] = play_mode
    if phase is not None:
        record["phase"] = phase
    if expected_expert_loss is not None:
        record["expected_expert_loss"] = expected_expert_loss
    if expert_miss_rate is not None:
        record["expert_miss_rate"] = expert_miss_rate
    if difficulty_source is not None:
        record["difficulty_source"] = difficulty_source
    if rollout_ref is not None:
        record["rollout_ref"] = rollout_ref
    if canary is not None:
        record["canary"] = canary
    if created is not None:
        record["created"] = created
    if include_image_paths:
        base = image_dir.rstrip("/") + "/" if image_dir else ""
        record["image_svg"] = f"{base}{pid}.svg"
        record["image_png"] = f"{base}{pid}.png"
        record["image_render_version"] = _svg.IMAGE_RENDER_VERSION
    return record


def structural_errors(record: dict[str, Any]) -> list[str]:
    """Return human-readable structural problems with a record (no deps).

    Checks required fields, enum domains, ``board_json`` round-trip against the
    stored ``xgid``, and cross-field consistency. Empty list == structurally OK.
    """
    errs: list[str] = []
    for f in _REQUIRED:
        if f not in record or record[f] in (None, ""):
            errs.append(f"missing required field {f!r}")
    if errs:
        return errs

    if record["tier"] not in _TIERS:
        errs.append(f"tier {record['tier']!r} not in {sorted(_TIERS)}")
    if record["decision_type"] not in _DECISIONS:
        errs.append(f"decision_type {record['decision_type']!r} invalid")
    if "split" in record and record["split"] not in _SPLITS:
        errs.append(f"split {record['split']!r} invalid")

    # board_json must parse and conserve checkers
    try:
        board = Board.from_json(record["board_json"])
        _validate_board(board)
    except Exception as exc:  # noqa: BLE001 - surface as a structural error
        errs.append(f"board_json invalid: {exc}")
        return errs

    # xgid must round-trip to the same canonical position as board_json
    try:
        from bgcore.board import canonical_key

        if canonical_key(_xgid.xgid_to_board(record["xgid"])) != canonical_key(board):
            errs.append("xgid does not round-trip to board_json")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"xgid invalid: {exc}")

    if record["board_json"].get("decision_type") != record["decision_type"]:
        errs.append("decision_type disagrees with board_json.decision_type")

    mr = record.get("expert_miss_rate")
    if mr is not None and not (0.0 <= float(mr) <= 1.0):
        errs.append(f"expert_miss_rate {mr} out of [0,1]")
    return errs


def validate_record(record: dict[str, Any], *, strict: bool = True) -> list[str]:
    """Validate a record structurally (+ against the JSON Schema if available).

    Returns the list of errors; with ``strict=True`` a non-empty list raises
    :class:`ValueError`. Uses ``jsonschema`` against
    ``schema/position.schema.json`` when it is installed, always applying the
    dependency-free structural checks.
    """
    errs = structural_errors(record)
    errs.extend(_jsonschema_errors(record))
    if strict and errs:
        raise ValueError("; ".join(errs))
    return errs


def _jsonschema_errors(record: dict[str, Any]) -> list[str]:
    try:
        import jsonschema  # noqa: PLC0415
    except ImportError:
        return []
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "schema", "position.schema.json"
    )
    try:
        import json

        with open(schema_path, encoding="utf-8") as fh:
            schema = json.load(fh)
    except OSError:
        return []
    validator = jsonschema.Draft202012Validator(schema)
    return [f"schema: {e.message}" for e in validator.iter_errors(record)]
