"""Tests: full position-record assembly + structural validation (PLAN.md §1.1)."""

import pytest

from bgcore.board import Board, validate
from generate import records, tiering
from ids import xgid as _xgid


def _board():
    pts = [0] * 26
    pts[7], pts[12], pts[15], pts[18], pts[19] = 2, 5, 3, 3, 2
    pts[20], pts[6], pts[5], pts[4], pts[2] = -2, -3, -3, -3, -4
    b = Board(points=pts, dice=[6, 3])
    b.refresh_pip()
    validate(b)
    return b


def _record(board):
    tr = tiering.assign_tier(board)
    return records.build_record(
        board, tier=tr.tier, split="dev", play_mode="money", phase=tr.phase,
        expected_expert_loss=tr.expected_expert_loss, expert_miss_rate=tr.expert_miss_rate,
        difficulty_source=tr.difficulty_source, created="2026-07-12T00:00:00Z",
    )


def test_build_record_has_all_representations():
    rec = _record(_board())
    for f in ("position_id", "xgid", "gnubg_id", "board_json", "ascii", "tier", "decision_type"):
        assert f in rec
    assert rec["gnubg_id"]["position_id"]
    assert rec["gnubg_id"]["match_id"]
    assert rec["ascii_render_version"] == "ascii-1"
    assert rec["image_svg"].endswith(".svg")
    assert rec["image_png"].endswith(".png")


def test_build_record_validates_structurally():
    rec = _record(_board())
    assert records.validate_record(rec) == []


def test_position_id_is_stable_and_normalized():
    b = _board()
    assert records.position_id_for(b) == records.position_id_for(b.copy())
    assert records.position_id_for(b).startswith("bg-")


def test_xgid_roundtrips_to_board_json():
    rec = _record(_board())
    from bgcore.board import canonical_key

    rt = _xgid.xgid_to_board(rec["xgid"])
    stored = Board.from_json(rec["board_json"])
    assert canonical_key(rt) == canonical_key(stored)


def test_validate_flags_missing_required_field():
    rec = _record(_board())
    del rec["tier"]
    errs = records.validate_record(rec, strict=False)
    assert any("tier" in e for e in errs)
    with pytest.raises(ValueError):
        records.validate_record(rec, strict=True)


def test_validate_flags_bad_tier_enum():
    rec = _record(_board())
    rec["tier"] = "T9"
    errs = records.validate_record(rec, strict=False)
    assert any("tier" in e for e in errs)


def test_validate_flags_xgid_board_mismatch():
    rec = _record(_board())
    rec["xgid"] = _xgid.board_to_xgid(Board.starting_position([3, 1]))
    errs = records.validate_record(rec, strict=False)
    assert any("round-trip" in e for e in errs)


def test_validate_flags_decision_type_disagreement():
    rec = _record(_board())
    rec["decision_type"] = "cube"  # board_json says checker
    errs = records.validate_record(rec, strict=False)
    assert any("decision_type" in e for e in errs)


def test_validate_flags_out_of_range_miss_rate():
    rec = _record(_board())
    rec["expert_miss_rate"] = 1.5
    errs = records.validate_record(rec, strict=False)
    assert any("expert_miss_rate" in e for e in errs)


def test_cube_record_decision_type():
    b = Board.starting_position()
    b.decision_type = "cube"
    rec = records.build_record(b, tier="T3", split="dev", phase="cube-action")
    assert rec["decision_type"] == "cube"
    assert records.validate_record(rec) == []
