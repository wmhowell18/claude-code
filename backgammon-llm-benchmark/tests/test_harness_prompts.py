"""Tests: prompt determinism + answer-format contract per track (PLAN.md §4.1-4.2)."""

import pytest

from bgcore.board import Board
from harness import prompts
from harness.prompts import PROMPT_VERSION, build_messages


def _checker_board():
    return Board.starting_position([3, 1])


def _cube_board():
    b = Board.starting_position()  # no dice -> cube decision
    assert b.decision_type == "cube"
    return b


def test_prompt_version_is_set():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_text_prompt_is_deterministic():
    b = _checker_board()
    a = build_messages(b, track="text", ascii_text="ASCII-BOARD")
    c = build_messages(b, track="text", ascii_text="ASCII-BOARD")
    assert a == c


def test_checker_contract_line_present():
    b = _checker_board()
    msgs = build_messages(b, track="text", ascii_text="X")
    user = msgs[1]["content"]
    assert "MOVE:" in user
    assert "board_json" in user
    assert "ASCII" in user or "ASCII".lower() in user.lower()
    # system states the format contract too
    assert "MOVE:" in msgs[0]["content"]


def test_cube_contract_line_present():
    b = _cube_board()
    msgs = build_messages(b, track="text", ascii_text="X")
    user = msgs[1]["content"]
    assert "ACTION:" in user
    assert "MOVE:" not in user.split("ACTION:")[1]  # nothing after the action contract label
    assert "CUBE decision" in user


def test_image_track_requires_bytes_and_builds_parts():
    b = _checker_board()
    with pytest.raises(ValueError):
        build_messages(b, track="image")  # no image bytes
    msgs = build_messages(b, track="image", image_png=b"PNGDATA")
    content = msgs[1]["content"]
    assert isinstance(content, list)
    assert any(p["type"] == "image_url" for p in content)
    # minimal text frame still carries the contract reminder
    text = "".join(p.get("text", "") for p in content if p["type"] == "text")
    assert "MOVE:" in text


def test_text_plus_image_has_both():
    b = _checker_board()
    msgs = build_messages(b, track="text+image", ascii_text="X", image_png=b"PNG")
    content = msgs[1]["content"]
    kinds = [p["type"] for p in content]
    assert "text" in kinds and "image_url" in kinds
    text = "".join(p.get("text", "") for p in content if p["type"] == "text")
    assert "board_json" in text


def test_unknown_track_rejected():
    with pytest.raises(ValueError):
        build_messages(_checker_board(), track="hologram")


def test_board_json_text_is_sorted_and_stable():
    b = _checker_board()
    one = prompts.board_json_text(b)
    two = prompts.board_json_text(b)
    assert one == two
    assert '"points"' in one
