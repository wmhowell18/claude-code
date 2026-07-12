"""Tests for the static leaderboard generator (site/build.py, PLAN.md §6).

The build module lives at ``site/build.py``. ``site`` collides with a stdlib
module name, so we load it by file path via importlib rather than importing
``site.build``.
"""

import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_PATH = os.path.join(REPO_ROOT, "site", "build.py")
FIXTURES_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")
FIXED_TS = "2026-07-12 00:00 UTC"


def _load_build():
    spec = importlib.util.spec_from_file_location("bench_site_build", BUILD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()


# -- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_results():
    return build.load_results(FIXTURES_DIR)


@pytest.fixture(scope="module")
def fixture_html(fixture_results):
    return build.build_html(fixture_results, FIXED_TS, synthetic=True)


@pytest.fixture(scope="module")
def fixture_board(fixture_results):
    return build.build_leaderboard_json(fixture_results, FIXED_TS)


# -- fixtures conform to the frozen schema ----------------------------------


def test_fixtures_are_present_and_synthetic(fixture_results):
    assert len(fixture_results) >= 3
    for r in fixture_results:
        assert r.get("synthetic") is True
        # schema-required top-level fields
        for key in ("run_id", "model", "track", "manifest", "aggregate", "decisions"):
            assert key in r
        assert "example/" in r["model"] or r["model"].startswith("human-panel/")


# -- valid HTML structure ---------------------------------------------------


def test_html_is_wellformed_document(fixture_html):
    assert fixture_html.startswith("<!DOCTYPE html>")
    assert "<title>" in fixture_html
    assert fixture_html.rstrip().endswith("</html>")
    # self-contained: no external network dependencies
    assert "<link" not in fixture_html
    assert "src=\"http" not in fixture_html
    assert "cdn" not in fixture_html.lower()
    assert "@import" not in fixture_html


def test_leaderboard_table_row_count(fixture_html):
    # 2 synthetic models + 1 human-panel baseline = 3 rows in the table body.
    body = fixture_html.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert body.count("<tr>") == 3


def test_benchpr_explanation_and_northstar_text(fixture_html):
    assert "BenchPR" in fixture_html
    assert "Lower is better" in fixture_html
    assert "world-class" in fixture_html


def test_synthetic_banner_present(fixture_html):
    assert "SYNTHETIC PREVIEW" in fixture_html


# -- charts (inline SVG) ----------------------------------------------------


def test_all_three_charts_render(fixture_html):
    assert fixture_html.count("<svg") >= 3  # scatter + tier bars + dumbbell


def test_scatter_reference_lines_present(fixture_html):
    for pr in ("2", "4", "8"):
        assert f'data-pr="{pr}"' in fixture_html
    assert fixture_html.count('class="refline"') == 3


def test_scatter_has_points_and_human_marker(fixture_html):
    assert 'class="pt"' in fixture_html  # model/track points
    assert 'class="human-marker"' in fixture_html  # measured human panel


def test_tier_bars_present(fixture_html):
    assert 'class="tier-bar"' in fixture_html
    for tier in ("T1", "T2", "T3", "T4"):
        assert f'data-tier="{tier}"' in fixture_html


def test_dumbbell_present(fixture_html):
    assert 'class="dot-text"' in fixture_html
    assert 'class="dot-image"' in fixture_html


# -- budget section ---------------------------------------------------------


def test_budget_section_present(fixture_html):
    assert 'id="budget"' in fixture_html
    assert "Fixed-budget track" in fixture_html


def test_budget_track_in_json(fixture_board):
    assert fixture_board["budget_track"], "expected a budget-track entry"
    entry = fixture_board["budget_track"][0]
    assert entry["benchpr_at_budget"] == 2.75
    assert entry["budget_usd"] == 10.0


# -- leaderboard.json content -----------------------------------------------


def test_leaderboard_json_ranking_order(fixture_board):
    lb = fixture_board["leaderboard"]
    assert [r["model"] for r in lb] == [
        "example/synthetic-model-a",
        "example/synthetic-model-b",
    ]
    assert [r["rank"] for r in lb] == [1, 2]
    # strictly ascending BenchPR (lower is better)
    prs = [r["benchpr"] for r in lb]
    assert prs == sorted(prs)
    assert prs[0] == 3.10


def test_leaderboard_json_track_and_cost_fields(fixture_board):
    a = fixture_board["leaderboard"][0]
    assert a["benchpr_text"] == 3.10
    assert a["benchpr_image"] == 4.80
    assert a["benchpr_ci"] == [2.80, 3.45]
    assert a["cost_usd"] is not None and a["cost_usd"] > 0
    assert a["cost_per_position"] is not None


def test_humans_separated_from_models(fixture_board):
    assert [r["model"] for r in fixture_board["humans"]] == ["human-panel/expert-panel"]
    assert all(r["model"] != "human-panel/expert-panel" for r in fixture_board["leaderboard"])


def test_meta_carries_versions(fixture_board):
    meta = fixture_board["meta"]
    assert meta["prompt_version"] == "p-2026.07.1"
    assert meta["dataset_hash"].startswith("sha256:")


# -- empty state ------------------------------------------------------------


def test_empty_state_builds_valid_page(tmp_path):
    empty_dir = tmp_path / "no_results"
    empty_dir.mkdir()
    html_out, board = build.build_site(str(empty_dir), str(tmp_path / "out"), generated=FIXED_TS)
    assert html_out.startswith("<!DOCTYPE html>")
    assert "No runs yet" in html_out
    assert board["leaderboard"] == []
    assert board["humans"] == []
    assert (tmp_path / "out" / "index.html").exists()
    assert (tmp_path / "out" / "leaderboard.json").exists()


def test_missing_results_dir_builds_empty(tmp_path):
    html_out, board = build.build_site(str(tmp_path / "does_not_exist"), str(tmp_path / "out"), generated=FIXED_TS)
    assert "No runs yet" in html_out
    assert board["leaderboard"] == []


# -- determinism ------------------------------------------------------------


def test_determinism_identical_builds(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    h1, b1 = build.build_site(FIXTURES_DIR, str(out1), generated=FIXED_TS, synthetic=True)
    h2, b2 = build.build_site(FIXTURES_DIR, str(out2), generated=FIXED_TS, synthetic=True)
    assert h1 == h2
    assert b1 == b2
    assert (out1 / "index.html").read_text() == (out2 / "index.html").read_text()
    assert (out1 / "leaderboard.json").read_text() == (out2 / "leaderboard.json").read_text()


# -- tolerance of unknown extra fields --------------------------------------


def test_tolerates_unknown_extra_fields(fixture_results):
    mutated = [dict(r) for r in fixture_results]
    mutated[0]["totally_unknown_field"] = {"nested": [1, 2, 3]}
    mutated[0]["aggregate"] = dict(mutated[0]["aggregate"], future_metric=42)
    html_out = build.build_html(mutated, FIXED_TS)
    board = build.build_leaderboard_json(mutated, FIXED_TS)
    assert html_out.startswith("<!DOCTYPE html>")
    assert board["leaderboard"], "build should still produce a ranking"


def test_written_json_is_valid(tmp_path):
    _, board = build.build_site(FIXTURES_DIR, str(tmp_path / "out"), generated=FIXED_TS, synthetic=True)
    reloaded = json.loads((tmp_path / "out" / "leaderboard.json").read_text())
    assert reloaded == board
