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


def test_checker_entries_carry_best_after_board(records, data):
    """Every checker position embeds ``best_after`` — the board after the engine's
    best play — so the practice-mode feedback panel can redraw it. With the
    move-matching fix every pilot best move is legal, so none should be None."""
    by_id = {d["position_id"]: d for d in data}
    for record in records:
        if record["decision_type"] != "checker":
            continue
        entry = by_id[record["position_id"]]
        ba = entry["best_after"]
        assert ba is not None, ("best_after must resolve for every pilot position",
                                record["position_id"])
        assert len(ba["points"]) == 26
        assert set(ba["bar"].keys()) == {"x", "o"}
        assert set(ba["off"].keys()) == {"x", "o"}
        # It equals applying best_move to the display-frame mover.
        mover, _ = _display_and_rollout(record)
        after = bgmoves.apply_move(mover, entry["best_move"])
        assert ba["points"] == [int(v) for v in after.points]


def test_page_carries_practice_and_blind_modes():
    """The generated page must offer both run modes and the feedback machinery."""
    records = hb.load_positions()
    data = hb.build_data(records)
    manifest = hb.build_manifest(records, "2026-01-01T00:00:00Z")
    html = hb.render_html(data, manifest)
    for needle in ("screenFeedback", "STATE_VERSION", "modeChooser", "runmode",
                   "Blind panel run", "Practice — feedback after each answer", "screenStale"):
        assert needle in html, needle


def test_quality_gate_filters_pilot_to_42(records):
    """The quiz build gates out the 8 no-decision positions, landing at 42."""
    eligible, excluded = hb.filter_eligible(records)
    assert len(records) == 50
    assert len(eligible) == 42
    assert len(excluded) == 8
    checker = sum(1 for r in eligible if r["decision_type"] == "checker")
    cube = sum(1 for r in eligible if r["decision_type"] == "cube")
    assert (checker, cube) == (27, 15)
    # every exclusion carries a position_id + human-readable reason
    for e in excluded:
        assert e["position_id"] and e["reason"]
    data = hb.build_data(eligible)
    assert len(data) == 42


def test_manifest_records_effective_and_excluded(records):
    eligible, excluded = hb.filter_eligible(records)
    manifest = hb.build_manifest(records, "2026-01-01T00:00:00Z",
                                 excluded=excluded, effective=len(eligible))
    assert manifest["total_positions"] == 50
    assert manifest["effective_positions"] == 42
    assert len(manifest["excluded_positions"]) == 8
    # dataset_hash still covers the FULL pilot (comparability), unchanged by the gate
    assert manifest["dataset_hash"] == hb.build_manifest(records, "2026-01-01T00:00:00Z")["dataset_hash"]


def test_page_uses_one_click_automove_and_topmoves_feedback():
    """The page ships the one-click auto-move engine (dice order + swap + spent
    fading + rejection shake) and the top-3 feedback tables — and no longer uses
    the old select-then-pick-destination highlight UI."""
    records = hb.load_positions()
    html = hb.render_html(hb.build_data(records), hb.build_manifest(records, "2026-01-01T00:00:00Z"))
    for needle in ("pickDie", "trySwap", "diceSwappable", "usedCount", "bg-shake",
                   "fbTopMovesTable", "fbCubeTable"):
        assert needle in html, needle
    # the destination-highlight primary interaction is gone
    for gone in ("computeHighlights", "sourceHasMove", "destdot"):
        assert gone not in html, gone


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


def test_all_rollout_moves_resolve_in_display_frame(records):
    """Regression: after the move-matching fix, EVERY pilot rollout move resolves
    to a legal move in its display frame (previously 22 moves across 13 positions
    — single-checker spellings like ``10/3`` — mis-reported as illegal, so their
    best moves couldn't be composed/scored). None may remain unmatched."""
    unmatched = []
    for record in records:
        if record["decision_type"] != "checker":
            continue
        mover, rollout = _display_and_rollout(record)
        for m in (rollout.get("checker") or {}).get("moves") or []:
            move = m.get("move")
            if not move:
                continue
            if bgmoves._match(mover, move) in (None, "cannot-move"):
                unmatched.append((record["position_id"], move))
    assert unmatched == [], unmatched
