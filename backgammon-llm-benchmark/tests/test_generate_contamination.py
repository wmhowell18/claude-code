"""Tests: canary / hashing / redaction / findability (PLAN.md §2.4-2.5)."""

import pytest

from bgcore.board import Board, flip, validate
from generate import contamination, records, tiering
from ids import xgid as _xgid


def _board():
    pts = [0] * 26
    pts[7], pts[12], pts[15], pts[18], pts[19] = 2, 5, 3, 3, 2
    pts[20], pts[6], pts[5], pts[4], pts[2] = -2, -3, -3, -3, -4
    b = Board(points=pts, dice=[6, 3])
    b.refresh_pip()
    validate(b)
    return b


def _record(board, split="heldout"):
    tr = tiering.assign_tier(board)
    return records.build_record(
        board, tier=tr.tier, split=split, play_mode="money", phase=tr.phase,
        expected_expert_loss=tr.expected_expert_loss, expert_miss_rate=tr.expert_miss_rate,
        difficulty_source=tr.difficulty_source, created="2026-07-12T00:00:00Z",
    )


def test_read_canary_from_manifest():
    token = contamination.read_canary("CANARY.md")
    assert token.startswith("BENCH-CANARY-")
    assert len(token) == len("BENCH-CANARY-") + 36


def test_read_canary_missing_token(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("no token here")
    with pytest.raises(ValueError):
        contamination.read_canary(str(p))


def test_inject_canary_does_not_mutate_input():
    rec = {"a": 1}
    out = contamination.inject_canary(rec, "BENCH-CANARY-xyz")
    assert out["canary"] == "BENCH-CANARY-xyz"
    assert "canary" not in rec


def test_hash_is_sha256_hex_and_stable():
    b = _board()
    h1 = contamination.hash_xgid(b)
    h2 = contamination.hash_xgid(_xgid.board_to_xgid(b))
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # valid hex


def test_hash_invariant_under_color_mirror():
    b = _board()
    assert contamination.hash_xgid(b) == contamination.hash_xgid(flip(b))


def test_hash_differs_for_different_positions():
    assert contamination.hash_xgid(_board()) != contamination.hash_xgid(
        Board.starting_position([3, 1])
    )


def test_hash_record_uses_xgid():
    rec = _record(_board())
    assert contamination.hash_record(rec) == contamination.hash_xgid(rec["xgid"])


def test_heldout_hashes_publishes_only_hashes():
    recs = [_record(_board()), _record(Board.starting_position([3, 1]))]
    payload = contamination.heldout_hashes(
        recs, canary="BENCH-CANARY-z", created="2026-07-12", dataset_version="v1"
    )
    assert payload["count"] == 2
    assert len(payload["hashes"]) == 2
    assert payload["hashes"] == sorted(payload["hashes"])
    # none of the actual position content leaks into the published payload
    blob = str(payload)
    assert "board_json" not in blob
    assert "XGID" not in blob


def test_redact_strips_answer_fields():
    rec = _record(_board())
    red = contamination.redact_record(rec, canary="BENCH-CANARY-z")
    for f in ("board_json", "ascii", "xgid", "gnubg_id", "rollout_ref", "image_svg", "image_png"):
        assert f not in red
    assert red["hash"] == contamination.hash_record(rec)
    assert red["tier"] == rec["tier"]
    assert red["redacted"] is True
    assert red["canary"] == "BENCH-CANARY-z"


def test_findability_report_lists_id_queries_no_web_calls():
    b = _board()
    report = contamination.findability_report(b).emit()
    assert report["web_calls_made"] == 0
    assert report["xgid"] == _xgid.board_to_xgid(b)
    # exact-match quoted XGID and the gnubg ids appear among the queries
    assert any(q == f'"{report["xgid"]}"' for q in report["queries"])
    assert report["gnubg_position_id"] in " ".join(report["queries"])
    assert report["gnubg_match_id"] in " ".join(report["queries"])


def test_findability_report_accepts_xgid_string():
    b = _board()
    xgid = _xgid.board_to_xgid(b)
    report = contamination.findability_report(xgid).emit()
    assert report["xgid"] == xgid
