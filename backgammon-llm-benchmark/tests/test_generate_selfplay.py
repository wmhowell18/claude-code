"""Tests: self-play orchestration + decision extraction (PLAN.md §2.1).

The gnubg subprocess is injected as a canned-output callable, so ``generate``
runs the whole path without spawning gnubg.
"""

import pytest

from bgcore.board import validate
from generate import selfplay
from generate.selfplay import Candidate, RunConfig


CANNED_MAT = """7 point match

 Game 1
 gnubg_A : 0                          gnubg_B : 0
  1) 45: 8/3 6/2                        45: 13/9 8/3
  2) Doubles => 2                       Takes
  3) 25: 8/3 8/6                        65: 13/7 9/4
  4) 14: 24/20 3/2                      21: 8/7 6/4
  5) 66: 13/7(4)                        54: 7/2 7/3
"""


def test_extract_candidates_attaches_phase_and_context():
    cands = selfplay.extract_candidates(CANNED_MAT, config=RunConfig(seed=5, match_length=7))
    assert len(cands) == 9
    for c in cands:
        assert isinstance(c, Candidate)
        validate(c.board)
        assert c.phase  # a phase tag was attached
        assert c.seed == 5
        assert c.play_mode == "match"
    # opening decisions classify as opening-ish
    assert cands[0].phase == "opening-ish"
    # exactly one cube decision
    assert sum(1 for c in cands if c.decision_type == "cube") == 1


def test_generate_uses_injected_runner():
    seen = {}

    def fake_runner(commands):
        seen["commands"] = commands
        return CANNED_MAT

    config = RunConfig(games=2, seed=9, match_length=7)
    cands = selfplay.generate(config, runner=fake_runner)
    assert len(cands) == 9
    # the runner received the built gnubg command script; a match run plays one
    # self-chaining match per call (games are auto-chained within the match, so a
    # single 'new game' kicks it off — loop the call for several matches).
    assert any(c == "new match 7" for c in seen["commands"])
    assert seen["commands"].count("new game") == 1


def test_build_commands_reflects_config():
    cmds = selfplay.build_commands(RunConfig(games=3, seed=1, plies=3))
    assert "new session" in cmds           # money by default
    assert cmds.count("new game") == 3
    assert "set player 0 chequer evaluation plies 3" in cmds


def test_candidate_key_is_canonical():
    cands = selfplay.extract_candidates(CANNED_MAT)
    # a candidate's dedup key equals its board's canonical key
    from bgcore.board import canonical_key

    c = cands[0]
    assert c.key() == canonical_key(c.board)


def test_merge_flattens_streams():
    a = selfplay.extract_candidates(CANNED_MAT)
    b = selfplay.extract_candidates(CANNED_MAT)
    assert len(selfplay.merge([a, b])) == len(a) + len(b)


def test_money_run_config_play_mode():
    assert RunConfig().play_mode == "money"
    assert RunConfig(match_length=7).play_mode == "match"
