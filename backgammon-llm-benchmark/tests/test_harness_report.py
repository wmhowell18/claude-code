"""Tests: results building + minimal schema validation + PR recompute (PLAN.md §4.4, §5.2)."""

import json
import sys
from pathlib import Path

from harness import report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import validate_pr  # noqa: E402


def _valid_results():
    return {
        "run_id": "r",
        "model": "fake/m",
        "track": "text",
        "manifest": {
            "dataset_hash": "abc",
            "prompt_version": "bench-1",
            "timestamp": "2026-07-12T00:00:00+00:00",
        },
        "aggregate": {
            "benchpr": 3.0,
            "best_move_accuracy": 0.5,
            "mean_equity_loss": 0.006,
        },
        "decisions": [
            {"position_id": "p1", "chosen": "8/5 6/5", "equity_loss": 0.0},
            {"position_id": "p2", "chosen": "24/21", "equity_loss": 0.012},
        ],
    }


def test_valid_results_passes():
    assert report.validate_results(_valid_results()) == []


def test_missing_required_field_flagged():
    obj = _valid_results()
    del obj["aggregate"]["benchpr"]
    errs = report.validate_results(obj)
    assert any("benchpr" in e for e in errs)


def test_wrong_type_flagged():
    obj = _valid_results()
    obj["aggregate"]["best_move_accuracy"] = "high"
    errs = report.validate_results(obj)
    assert any("best_move_accuracy" in e for e in errs)


def test_bad_track_enum_flagged():
    obj = _valid_results()
    obj["track"] = "telepathy"
    errs = report.validate_results(obj)
    assert any("track" in e for e in errs)


def test_decision_missing_equity_loss_flagged():
    obj = _valid_results()
    del obj["decisions"][0]["equity_loss"]
    errs = report.validate_results(obj)
    assert any("equity_loss" in e for e in errs)


def test_validate_pr_recompute_matches(tmp_path):
    obj = _valid_results()
    # make stored benchpr consistent with decisions: mean(0.0, 0.012)=0.006 -> 3.0
    obj["aggregate"]["benchpr"] = 3.0
    path = tmp_path / "res.json"
    path.write_text(json.dumps(obj))
    rows = validate_pr.recompute(path)
    assert len(rows) == 1
    assert rows[0]["match"] is True
    assert abs(rows[0]["recomputed_benchpr"] - 3.0) < 1e-9


def test_validate_pr_recompute_detects_mismatch(tmp_path):
    obj = _valid_results()
    obj["aggregate"]["benchpr"] = 99.0  # wrong
    path = tmp_path / "res.json"
    path.write_text(json.dumps(obj))
    rows = validate_pr.recompute(path)
    assert rows[0]["match"] is False


def test_validate_pr_main_exit_code(tmp_path):
    obj = _valid_results()
    obj["aggregate"]["benchpr"] = 3.0
    path = tmp_path / "res.json"
    path.write_text(json.dumps(obj))
    assert validate_pr.main([str(path)]) == 0
