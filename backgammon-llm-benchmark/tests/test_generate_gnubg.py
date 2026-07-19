"""Tests: gnubg command builders + output parsers (PLAN.md §1.3, §2.1, §7).

Pure-function tests only — the subprocess shim ``gnubg.run_gnubg`` is never
called (gnubg is not installed in CI). Canned gnubg output is embedded here.
"""

import glob
import json
import os

import pytest

from bgcore.board import validate
from generate import gnubg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROLL_DIR = os.path.join(_REPO_ROOT, "rollouts", "gnubg")


# --------------------------------------------------------------------------
# command builders
# --------------------------------------------------------------------------


def test_engine_commands_set_both_seats():
    cmds = gnubg.engine_commands(plies=3)
    assert "set player 0 gnubg" in cmds
    assert "set player 1 gnubg" in cmds
    assert "set player 0 chequer evaluation plies 3" in cmds
    assert "set player 1 cube evaluation plies 3" in cmds


def test_selfplay_commands_money_vs_match():
    money = gnubg.selfplay_commands(3, seed=7)
    assert "new session" in money
    assert money.count("new game") == 3
    assert "set seed 7" in money
    assert any(c.startswith("export match mat") for c in money)

    match = gnubg.selfplay_commands(2, match_length=7)
    assert "new match 7" in match
    assert "new session" not in match


def test_selfplay_commands_validate_args():
    with pytest.raises(ValueError):
        gnubg.selfplay_commands(0)
    with pytest.raises(ValueError):
        gnubg.selfplay_commands(1, match_length=0)


def test_rollout_commands_checker_mirror_settings_table():
    s = gnubg.RolloutSettings(trials=5184, chequer_ply=3, cube_ply=2, seed=42, truncation=11)
    cmds = gnubg.rollout_commands(xgid="XGID=abc", settings=s)
    assert "set xgid XGID=abc" in cmds
    assert "set rollout trials 5184" in cmds
    assert "set rollout chequerplay plies 3" in cmds
    assert "set rollout cubedecision plies 2" in cmds
    assert "set rollout varredn on" in cmds
    assert "set rollout quasirandom on" in cmds
    assert "set rollout truncation plies 11" in cmds
    assert "set rollout seed 42" in cmds
    assert cmds[-1] == "rollout"


def test_rollout_commands_cube_uses_cube_verb():
    cmds = gnubg.rollout_commands(position_id="4HPwATDgc/ABMA", cube=True)
    assert cmds[-1] == "rollout =cube"
    assert "set board 4HPwATDgc/ABMA" in cmds


def test_rollout_settings_to_meta_matches_schema_shape():
    meta = gnubg.RolloutSettings().to_meta()
    assert meta["trials"] == 1296
    assert meta["chequer_ply"] == "2-ply"
    assert meta["variance_reduction"] is True
    assert meta["antithetic_dice"] is True
    assert meta["truncation"] == "none"
    assert meta["settings_version"] == gnubg.SETTINGS_VERSION


def test_script_is_newline_terminated():
    text = gnubg.script(["a", "b"])
    assert text == "a\nb\n"


# --------------------------------------------------------------------------
# self-play .mat parser -> per-decision boards
# --------------------------------------------------------------------------

# A legal, deterministic self-play match (moves generated from the real start
# via bgcore) including a double + take so the cube path is covered.
CANNED_MAT = """7 point match

 Game 1
 gnubg_A : 0                          gnubg_B : 0
  1) 45: 8/3 6/2                        45: 13/9 8/3
  2) Doubles => 2                       Takes
  3) 25: 8/3 8/6                        65: 13/7 9/4
  4) 14: 24/20 3/2                      21: 8/7 6/4
  5) 66: 13/7(4)                        54: 7/2 7/3
"""


def test_match_length_detection():
    assert gnubg.match_length_of(CANNED_MAT) == 7
    assert gnubg.match_length_of("Game 1\n") == 0
    # a real gnubg money export is a "0 point match" (optionally after a comment)
    assert gnubg.match_length_of('; [EventDate "2026.07.15"]\n\n 0 point match\n') == 0


def test_gnubg_hops_maps_bar_off_and_segmented_chains():
    # gnubg point numbers -> bgcore board indices (index = 25 - point);
    # bar entry is point 25, bear-off is point 0.
    assert gnubg._gnubg_hops("25/21") == [("bar", 4)]
    assert gnubg._gnubg_hops("6/0") == [(19, "off")]
    # a single checker written as shared-point segments flattens to ordered hops
    assert gnubg._gnubg_hops("24/18 18/16") == [(1, 7), (7, 9)]
    # repeated tokens expand
    assert gnubg._gnubg_hops("13/9(3)") == [(12, 16), (12, 16), (12, 16)]


def test_split_cells_by_fixed_column_not_spaces():
    # a wide 4-checker left move can leave only a single space before the right
    # cell; splitting on runs of >=2 spaces would wrongly merge them. The two
    # columns sit at fixed offsets, so we split by column instead.
    line = " 17) 22: 25/23* 25/23 14/12 12/10 22: 25/23 9/7 9/7 6/4"
    body_start = len(" 17) ")
    cells = gnubg._split_cells(line, body_start)
    assert cells == ["22: 25/23* 25/23 14/12 12/10", "22: 25/23 9/7 9/7 6/4"]
    # an empty left cell (opening move) keeps only the right cell
    opening = "  1)                             53: 8/3 6/3"
    assert gnubg._split_cells(opening, len("  1) ")) == ["53: 8/3 6/3"]


def test_parse_cell_detects_dance_and_bearoff_move():
    dance = gnubg._parse_cell("64:", seat=1)
    assert dance is not None and dance.kind == "nomove" and dance.dice == [6, 4]
    mv = gnubg._parse_cell("53: 8/3 6/3", seat=0)
    assert mv.kind == "move" and mv.dice == [5, 3]


def test_apply_gnubg_move_handles_bar_bearoff_and_hit():
    from bgcore.board import Board, validate

    # bar entry (gnubg point 25 -> board index) that hits a lone opponent blot on
    # the entry point (board index 4 == gnubg point 21). A hand-built board with
    # exactly 15 checkers per side keeps the position valid.
    pts = [0] * 26
    pts[1] = 14  # 14 mover checkers parked out of the way (+1 on the bar == 15)
    pts[4] = -1  # a lone opponent blot on the entry point
    pts[20] = -14  # the rest of the opponent's checkers
    b = Board(
        points=pts,
        bar={"x": 1, "o": 0},
        off={"x": 0, "o": 0},
        turn="x",
        dice=[4, 2],
        cube={"value": 1, "owner": "center"},
        score={"x": 0, "o": 0, "length": 0, "crawford": False},
        decision_type="checker",
    )
    b.refresh_pip()
    nb = gnubg._apply_gnubg_move(b, "25/21")  # bar/21 in bgcore terms
    validate(nb)
    assert nb.bar["x"] == 0
    assert nb.points[4] == 1  # mover now holds the point
    assert nb.bar["o"] == 1  # the blot was sent to the opponent bar


def test_parse_match_extracts_checker_and_cube_decisions():
    decisions = gnubg.parse_match(CANNED_MAT)
    # 8 checker plies + 1 cube decision
    assert len(decisions) == 9
    cubes = [d for d in decisions if d.decision_type == "cube"]
    checkers = [d for d in decisions if d.decision_type == "checker"]
    assert len(cubes) == 1
    assert len(checkers) == 8

    # every reconstructed board is a valid position
    for d in decisions:
        validate(d.board)

    # the cube decision comes before the cube turns (still centred)
    cube = cubes[0]
    assert cube.board.decision_type == "cube"
    assert cube.board.dice == []
    assert cube.board.cube == {"value": 1, "owner": "center"}


def test_parse_match_take_transfers_cube_ownership():
    decisions = gnubg.parse_match(CANNED_MAT)
    # first checker decision after the take should see the cube at value 2
    after_take = [d for d in decisions if d.decision_type == "checker" and d.move_number >= 4]
    assert after_take
    assert after_take[0].board.cube["value"] == 2
    assert after_take[0].board.cube["owner"] in ("x", "o")


def test_parse_match_is_match_play():
    decisions = gnubg.parse_match(CANNED_MAT)
    assert all(d.play_mode == "match" for d in decisions)


# --------------------------------------------------------------------------
# rollout output parsers
# --------------------------------------------------------------------------

CANNED_CHECKER_ROLLOUT = """Rollout of 24/18 13/11 (and 2 others):
1296 games rolled with Variance Reduction.
Chequer play: 2-ply, Cube decisions: 2-ply.

  1. Rollout       24/18 13/11              Eq.:  +0.1523
       0.5321 0.1450 0.0071 - 0.4679 0.1400 0.0060
         [ 0.0042]
  2. Rollout       24/18 24/23              Eq.:  +0.1401 ( -0.0122)
       0.5290 0.1420 0.0065 - 0.4710 0.1385 0.0058
         [ 0.0045]
  3. Rollout       13/11 6/1                Eq.:  +0.0980 ( -0.0543)
       0.5190 0.1350 0.0060 - 0.4810 0.1420 0.0071
         [ 0.0049]
"""


def test_parse_checker_rollout_records_every_move():
    rec = gnubg.parse_checker_rollout(
        CANNED_CHECKER_ROLLOUT, position_id="p1", xgid="XGID=x", phase="opening-ish"
    )
    assert rec["decision_type"] == "checker"
    assert rec["engine"] == "gnubg"
    moves = rec["checker"]["moves"]
    assert [m["move"] for m in moves] == ["24/18 13/11", "24/18 24/23", "13/11 6/1"]
    assert [m["rank"] for m in moves] == [1, 2, 3]
    # best move has zero error; others measured in millipoints vs best
    assert moves[0]["error_mp"] == 0.0
    assert moves[1]["error_mp"] == pytest.approx(12.2, abs=0.05)
    assert moves[2]["error_mp"] == pytest.approx(54.3, abs=0.05)
    assert moves[0]["std_err"] == 0.0042
    assert rec["best_move"] == "24/18 13/11"
    assert rec["second_best_move"] == "24/18 24/23"
    assert rec["equity_gap"] == pytest.approx(0.0122, abs=1e-6)
    assert rec["rollout_meta"]["trials"] == 1296


def test_parse_checker_rollout_needs_moves():
    with pytest.raises(ValueError):
        gnubg.parse_checker_rollout("no moves here")


CANNED_CUBE_ROLLOUT = """Cube analysis
Rollout, 1296 games, 2-ply cube decisions.

  Cubeless equity  +0.480
  Cubeful equities:
  1. No double            +0.520
  2. Double, take         +0.610
  3. Double, pass         +1.000
  Best Cube action: Double, Take
"""


def test_parse_cube_rollout_three_equities_and_errors():
    rec = gnubg.parse_cube_rollout(CANNED_CUBE_ROLLOUT, position_id="c1")
    assert rec["decision_type"] == "cube"
    cube = rec["cube"]
    assert cube["no_double_equity"] == 0.520
    assert cube["double_take_equity"] == 0.610
    assert cube["double_pass_equity"] == 1.000
    assert cube["best_action"] == "Double, Take"
    # all errors non-negative; no-double costs 90mp, wrong pass costs 390mp
    assert cube["error_mp"]["Double, Take"] == 0.0
    assert cube["error_mp"]["No double"] == pytest.approx(90.0, abs=0.05)
    assert cube["error_mp"]["Double, Pass"] == pytest.approx(390.0, abs=0.05)
    assert all(v >= 0 for v in cube["error_mp"].values())


def test_parse_cube_rollout_missing_line_raises():
    with pytest.raises(ValueError):
        gnubg.parse_cube_rollout("Cubeless equity +0.4\nNo double +0.5\n")


# -- cube_action_errors: the corrected doubler-answer scoring -----------------


def test_cube_errors_no_double_best_both_double_answers_equal():
    """Doubling is wrong (best = No double): both Double answers pay the SAME
    doubling error and |DT-DP| is NOT added (regression for the 3000-mpt bug)."""
    # bg-06f2f715c60a3064: nd=-0.52598, dt=-2.0, dp=1.0 -> R=min=-2.0, best=nd.
    best, err = gnubg.cube_action_errors(-0.52598, -2.0, 1.0)
    assert best == "No double"
    assert err["No double"] == 0.0
    assert err["Double, Take"] == pytest.approx(1474.0, abs=0.05)
    assert err["Double, Pass"] == pytest.approx(1474.0, abs=0.05)
    assert err["Double, Take"] == err["Double, Pass"]


def test_cube_errors_double_correct_take_wrong_pass_claim_adds_gap():
    """Doubling correct and the opponent should TAKE: 'Double, Take' is 0 and
    'Double, Pass' pays the response-claim penalty |DT-DP|."""
    best, err = gnubg.cube_action_errors(0.520, 0.610, 1.000)
    assert best == "Double, Take"
    assert err["Double, Take"] == 0.0
    assert err["No double"] == pytest.approx(90.0, abs=0.05)     # R - nd
    assert err["Double, Pass"] == pytest.approx(390.0, abs=0.05)  # |DT - DP|


def test_cube_errors_double_correct_pass_wrong_take_claim_adds_gap():
    """Doubling correct and the opponent should PASS (a double/pass): the wrong
    'Double, Take' claim pays |DT-DP|; 'Double, Pass' is 0."""
    # dt above dp -> R = dp, opponent passes; take-claim is the wrong response.
    best, err = gnubg.cube_action_errors(0.30, 1.20, 1.00)
    assert best == "Double, Pass"
    assert err["Double, Pass"] == 0.0
    assert err["No double"] == pytest.approx(700.0, abs=0.05)     # R(=1.0) - nd
    assert err["Double, Take"] == pytest.approx(200.0, abs=0.05)  # |DT - DP|


def test_cube_errors_borderline_toss_up_all_zero():
    """When no-double and doubling equity are essentially equal, every answer is
    ~free (a genuine toss-up), so two zeros here is legitimate."""
    best, err = gnubg.cube_action_errors(0.500, 0.500, 1.000)
    assert best == "No double"
    assert err["No double"] == 0.0
    assert err["Double, Take"] == 0.0
    assert err["Double, Pass"] == 0.0


def test_cube_errors_best_action_always_zero():
    for nd, dt, dp in [(-0.5, -2.0, 1.0), (0.52, 0.61, 1.0), (0.30, 1.20, 1.0),
                       (0.82, 1.077, 1.0), (0.95, 1.096, 1.0), (-0.06, -0.26, 1.0)]:
        best, err = gnubg.cube_action_errors(nd, dt, dp)
        assert err[best] == 0.0, (nd, dt, dp, best, err)
        assert all(v >= 0 for v in err.values())


def test_shipped_cube_rollouts_match_formula():
    """Every checked-in cube rollout's error_mp/best_action must match
    cube_action_errors(equities) — guards against the old buggy map creeping back
    (scripts/repair_cube_error_mp.py keeps them in sync)."""
    n = 0
    for path in sorted(glob.glob(os.path.join(_ROLL_DIR, "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        if rec.get("decision_type") != "cube":
            continue
        n += 1
        c = rec["cube"]
        best, err = gnubg.cube_action_errors(
            c["no_double_equity"], c["double_take_equity"], c["double_pass_equity"])
        assert c["error_mp"] == err, rec["position_id"]
        assert c["best_action"] == best, rec["position_id"]
        assert err[best] == 0.0
    assert n == 15


def test_parse_rollout_dispatch():
    checker = gnubg.parse_rollout(CANNED_CHECKER_ROLLOUT, cube=False)
    cube = gnubg.parse_rollout(CANNED_CUBE_ROLLOUT, cube=True)
    assert checker["decision_type"] == "checker"
    assert cube["decision_type"] == "cube"
