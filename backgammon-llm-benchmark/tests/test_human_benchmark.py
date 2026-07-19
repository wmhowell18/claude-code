"""Tests for the human-benchmark quiz generator (scripts/build_human_benchmark.py).

The generator emits ONE self-contained HTML quiz driven by a single runtime JS
board engine: each position carries structured board data + its XGID, and each
checker position carries the enumerated legal-move set (``legal``) that the JS
click-to-move controller validates against. These tests pin the data contract and
cross-check the embedded legal set against the authoritative engine so a frame or
signature-format mistake can't slip through (the JS ``sigOf`` port must match the
Python ``_sig`` byte-for-byte).

Loaded by file path via importlib (``scripts`` is not an importable package).
"""

import importlib.util
import json
import os

import pytest

from bgcore import moves as bgmoves

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_PATH = os.path.join(REPO_ROOT, "scripts", "build_human_benchmark.py")


def _load_builder():
    spec = importlib.util.spec_from_file_location("bench_human_build", BUILD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hb = _load_builder()


@pytest.fixture(scope="module")
def records():
    return hb.load_positions()


@pytest.fixture(scope="module")
def data(records):
    return hb.build_data(records)


# -- data contract ----------------------------------------------------------


def test_pilot_has_fifty_positions(data):
    assert len(data) == 50
    assert sum(1 for d in data if d["decision_type"] == "checker") == 35
    assert sum(1 for d in data if d["decision_type"] == "cube") == 15


def test_every_entry_carries_structured_board_not_prebaked_svg(data):
    for entry in data:
        assert "svg" not in entry, "the pre-baked SVG string must be gone"
        board = entry["board"]
        assert len(board["points"]) == 26
        for key in ("bar", "off", "cube", "score", "pip", "dice"):
            assert key in board
        assert entry["xgid"].startswith("XGID=")


def test_checker_entries_carry_legal_set(data):
    for entry in data:
        if entry["decision_type"] != "checker":
            continue
        assert entry["legal"], "checker positions must embed a non-empty legal set"
        for mv in entry["legal"]:
            assert set(mv.keys()) == {"n", "s"}
            assert "|" in mv["s"]


def test_cube_entries_have_options_and_no_legal(data):
    for entry in data:
        if entry["decision_type"] != "cube":
            continue
        assert "legal" not in entry
        assert entry["options"], "cube positions keep their three-button options"


# -- the embedded legal set == the authoritative engine ---------------------


def _display_and_rollout(record):
    pid = record["position_id"]
    with open(os.path.join(hb.ROLL_DIR, pid + ".json"), encoding="utf-8") as fh:
        rollout = json.load(fh)
    return hb._display_board(record, rollout), rollout


def test_embedded_legal_sigs_match_generate_moves(records, data):
    by_id = {d["position_id"]: d for d in data}
    for record in records:
        if record["decision_type"] != "checker":
            continue
        mover, _ = _display_and_rollout(record)
        want = {
            hb._sig(lm.points, lm.bar["x"], lm.bar["o"], lm.off["x"])
            for lm in bgmoves.generate_moves(mover)
        }
        got = {mv["s"] for mv in by_id[record["position_id"]]["legal"]}
        assert got == want, record["position_id"]


def test_rollout_moves_legal_in_display_frame_are_in_embedded_set(records, data):
    """Every rollout move that IS legal in the display frame must resolve to an
    embedded legal signature (catches a frame/signature mismatch).

    A minority of pilot rollout moves are illegal in either frame (a known
    upstream data quirk — some gnubg move lists were computed in a numbering the
    engine can't reproduce); those are simply skipped here, exactly as the
    scorer's endpoint-map already tolerates them.
    """
    by_id = {d["position_id"]: d for d in data}
    for record in records:
        if record["decision_type"] != "checker":
            continue
        mover, rollout = _display_and_rollout(record)
        embedded = {mv["s"] for mv in by_id[record["position_id"]]["legal"]}
        for m in (rollout.get("checker") or {}).get("moves") or []:
            move = m.get("move")
            if not move:
                continue
            lm = bgmoves._match(mover, move)
            if lm in (None, "cannot-move"):
                continue  # illegal in this frame -> not expected in the set
            sig = hb._sig(lm.points, lm.bar["x"], lm.bar["o"], lm.off["x"])
            assert sig in embedded, (record["position_id"], move)
