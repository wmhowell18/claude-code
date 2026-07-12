"""Tests: intra-set dedup (incl. mirrors) + blocklist matching (PLAN.md §2.3)."""

import pytest

from bgcore.board import Board, canonical_key, flip, validate
from generate import dedup
from ids import xgid as _xgid


def _asymmetric_board():
    """A position that is NOT symmetric under color flip (so mirror != plain)."""
    pts = [0] * 26
    pts[7], pts[12], pts[15], pts[18], pts[19] = 2, 5, 3, 3, 2
    pts[20], pts[6], pts[5], pts[4], pts[2] = -2, -3, -3, -3, -4
    b = Board(points=pts, dice=[6, 3])
    b.refresh_pip()
    validate(b)
    return b


def test_asymmetric_fixture_really_is_asymmetric():
    b = _asymmetric_board()
    assert canonical_key(b) != canonical_key(flip(b))


def test_dedup_collapses_exact_duplicates():
    b = _asymmetric_board()
    out = dedup.dedup([b, b.copy(), b.copy()])
    assert len(out) == 1


def test_dedup_folds_color_mirrors_by_default():
    b = _asymmetric_board()
    mirror = flip(b)
    # plain dedup keeps both (different mover-relative keys)...
    assert len(dedup.dedup([b, mirror], mirror=False)) == 2
    # ...mirror-folded dedup treats them as one position.
    assert len(dedup.dedup([b, mirror], mirror=True)) == 1


def test_dedup_preserves_first_seen_order():
    b1 = _asymmetric_board()
    b2 = Board.starting_position([3, 1])
    out = dedup.dedup([b2, b1, b2.copy()])
    assert out[0] is b2
    assert out[1] is b1
    assert len(out) == 2


def test_blocklist_hit_normalizes_both_sides():
    b = _asymmetric_board()
    xgid = _xgid.board_to_xgid(b)
    bl = dedup.load_blocklist_data({"xgids": [xgid]})
    assert bl.contains(b)                  # exact
    assert bl.contains(flip(b))            # color-mirror of a blocked position


def test_blocklist_miss_for_unrelated_position():
    bl = dedup.load_blocklist_data({"xgids": [_xgid.board_to_xgid(_asymmetric_board())]})
    assert not bl.contains(Board.starting_position([3, 1]))


def test_blocklist_filter_splits_kept_and_rejected():
    blocked = _asymmetric_board()
    ok = Board.starting_position([3, 1])
    bl = dedup.load_blocklist_data({"xgids": [_xgid.board_to_xgid(blocked)]})
    kept, rejected = bl.filter([blocked, ok])
    assert kept == [ok]
    assert rejected == [blocked]


def test_blocklist_accepts_keys_and_entries_shapes():
    b = _asymmetric_board()
    key = dedup.mirror_key(b)
    bl_keys = dedup.load_blocklist_data({"keys": [key]})
    assert bl_keys.contains(b)
    bl_entries = dedup.load_blocklist_data(
        {"entries": [{"xgid": _xgid.board_to_xgid(b)}]}
    )
    assert bl_entries.contains(b)


def test_load_blocklist_missing_dir_is_empty():
    bl = dedup.load_blocklist("data/does-not-exist")
    assert bl.keys == set()


def test_dedup_accepts_xgid_strings_and_candidates():
    b = _asymmetric_board()
    xgid = _xgid.board_to_xgid(b)

    class Cand:
        def __init__(self, board):
            self.board = board

    out = dedup.dedup([xgid, Cand(b), b])
    assert len(out) == 1
