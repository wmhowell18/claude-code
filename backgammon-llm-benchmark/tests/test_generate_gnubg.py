"""Tests: gnubg command builders + output parsers (PLAN.md §1.3, §2.1, §7).

Pure-function tests only — the subprocess shim ``gnubg.run_gnubg`` is never
called (gnubg is not installed in CI). Canned gnubg output is embedded here.
"""

import pytest

from bgcore.board import validate
from generate import gnubg


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


def test_parse_rollout_dispatch():
    checker = gnubg.parse_rollout(CANNED_CHECKER_ROLLOUT, cube=False)
    cube = gnubg.parse_rollout(CANNED_CUBE_ROLLOUT, cube=True)
    assert checker["decision_type"] == "checker"
    assert cube["decision_type"] == "cube"
