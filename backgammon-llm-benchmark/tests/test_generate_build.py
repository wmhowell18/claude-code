"""Tests: end-to-end build_dataset dry-run + verify_position script (PLAN.md §7)."""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    path = os.path.join(_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_dataset = _load_script("build_dataset")
verify_position = _load_script("verify_position")


# --------------------------------------------------------------------------
# build_dataset --dry-run
# --------------------------------------------------------------------------


def test_build_dry_run_end_to_end():
    from generate import selfplay

    config = selfplay.RunConfig(games=1, seed=0, match_length=7)
    summary = build_dataset.build(
        config,
        dry_run=True,
        target=6,
        blocklist_dir=os.path.join(_ROOT, "data", "blocklist"),
        canary_path=os.path.join(_ROOT, "CANARY.md"),
        created="2026-07-12T00:00:00Z",
        split="pilot",
    )
    assert summary["candidates_raw"] == 9
    assert summary["records_built"] == summary["sampled"]
    assert summary["records_built"] <= 6
    # every built record validates and carries the canary + tier
    from generate import records

    for rec in summary["records"]:
        assert records.validate_record(rec) == []
        assert rec["canary"].startswith("BENCH-CANARY-")
        assert rec["tier"] in {"T1", "T2", "T3", "T4"}
        assert rec["split"] == "pilot"


def test_build_dry_run_is_deterministic():
    from generate import selfplay

    def run():
        return build_dataset.build(
            selfplay.RunConfig(games=1, seed=3, match_length=7),
            dry_run=True, target=5,
            blocklist_dir=os.path.join(_ROOT, "data", "blocklist"),
            canary_path=os.path.join(_ROOT, "CANARY.md"),
            created="2026-07-12T00:00:00Z", split="pilot",
        )

    a, b = run(), run()
    assert [r["position_id"] for r in a["records"]] == [r["position_id"] for r in b["records"]]


def test_build_main_smoke(capsys):
    rc = build_dataset.main(["--dry-run", "--target", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"records_built"' in out


# --------------------------------------------------------------------------
# verify_position
# --------------------------------------------------------------------------


def test_verify_start_position_roundtrips():
    from ids import xgid as _xgid
    from bgcore.board import Board

    xgid = _xgid.board_to_xgid(Board.starting_position([3, 1]))
    report = verify_position.verify(xgid)
    assert report["roundtrip_ok"] is True
    assert report["gnubg_ids_roundtrip_ok"] is True
    assert report["board_valid"] is True
    assert report["legal_move_count"] == 16
    assert report["phase"] == "opening-ish"
    assert report["tier_estimate"] in {"T1", "T2", "T3", "T4"}


def test_verify_cube_position_has_no_legal_moves():
    from ids import xgid as _xgid
    from bgcore.board import Board

    b = Board.starting_position()
    b.decision_type = "cube"
    report = verify_position.verify(_xgid.board_to_xgid(b))
    assert report["decision_type"] == "cube"
    assert report["legal_move_count"] == 0
    assert report["phase"] == "cube-action"


def test_verify_main_start_flag(capsys):
    rc = verify_position.main(["--start"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "round-trip lossless: True" in out
