"""Blocklist + intra-set dedup (PLAN.md §2.3).

Two jobs:

* **Intra-set dedup** — collapse near-duplicate candidates so tiers aren't
  inflated. The dedup key is :func:`bgcore.board.canonical_key` (already
  mover-relative, so seat/turn and dice ordering don't matter) folded with the
  color/perspective mirror so a position and its color-swap map together.
* **Blocklist matching** — reject any candidate whose normalized key matches a
  known-public position (GNU BG example DBs, opening books, quiz/book positions).
  Blocklists live under ``data/blocklist/*.json``; see that directory's README
  for the file format. Both the candidate and the blocklist entries are
  normalized the same way before comparison.

A :class:`Board`, an ``XGID`` string, or an already-computed key are all
accepted wherever a "position" is expected.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from bgcore.board import Board, canonical_key, flip
from ids import xgid as _xgid


def _as_board(pos: Any) -> Board:
    if isinstance(pos, Board):
        return pos
    if isinstance(pos, str):
        return _xgid.xgid_to_board(pos)
    if isinstance(pos, dict):  # a board_json dict or a candidate-like mapping
        if "board" in pos and isinstance(pos["board"], Board):
            return pos["board"]
        return Board.from_json(pos)
    if hasattr(pos, "board"):  # selfplay.Candidate / gnubg.Decision
        return pos.board
    raise TypeError(f"cannot coerce {type(pos)!r} to a Board")


def mirror_key(board: Board) -> str:
    """Color/symmetry-normalized dedup key (PLAN.md §2.3).

    Returns the lexicographically smaller of the position's canonical key and
    its color-mirror's, so a position and its flipped (color-swapped)
    counterpart collapse to a single key.
    """
    return min(canonical_key(board), canonical_key(flip(board)))


def dedup_key(pos: Any, *, mirror: bool = True) -> str:
    """Dedup key for any accepted position type (mirror-folded by default)."""
    board = _as_board(pos)
    return mirror_key(board) if mirror else canonical_key(board)


def dedup(candidates: Iterable[Any], *, mirror: bool = True) -> list[Any]:
    """Return ``candidates`` with duplicates removed, preserving first-seen order.

    Two candidates are duplicates iff they share a :func:`dedup_key`. With
    ``mirror=True`` (default) color-mirrored positions are treated as the same.
    """
    seen: set[str] = set()
    out: list[Any] = []
    for c in candidates:
        k = dedup_key(c, mirror=mirror)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


@dataclass
class Blocklist:
    """A set of normalized keys for known-public positions to exclude."""

    keys: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)
    mirror: bool = True

    def __contains__(self, pos: Any) -> bool:
        return self.contains(pos)

    def contains(self, pos: Any) -> bool:
        """True if ``pos`` (Board / XGID / candidate) matches the blocklist."""
        return dedup_key(pos, mirror=self.mirror) in self.keys

    def add_xgid(self, xgid: str) -> None:
        self.keys.add(dedup_key(xgid, mirror=self.mirror))

    def add_board(self, board: Board) -> None:
        self.keys.add(dedup_key(board, mirror=self.mirror))

    def filter(self, candidates: Iterable[Any]) -> tuple[list[Any], list[Any]]:
        """Split candidates into ``(kept, rejected)`` by blocklist membership."""
        kept, rejected = [], []
        for c in candidates:
            (rejected if self.contains(c) else kept).append(c)
        return kept, rejected


def load_blocklist(
    path: str = "data/blocklist",
    *,
    mirror: bool = True,
) -> Blocklist:
    """Load and normalize every ``*.json`` blocklist under ``path``.

    Each file is documented in ``data/blocklist/README.md``. Missing directories
    yield an empty blocklist (Phase 0 has no lists yet).
    """
    bl = Blocklist(mirror=mirror)
    if not os.path.isdir(path):
        return bl
    for fname in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(fname, encoding="utf-8") as fh:
            data = json.load(fh)
        load_blocklist_data(data, into=bl, source=os.path.basename(fname))
    return bl


def load_blocklist_data(
    data: dict[str, Any],
    *,
    into: Blocklist | None = None,
    source: str = "<inline>",
) -> Blocklist:
    """Merge one parsed blocklist document into a :class:`Blocklist`.

    Recognized shapes (see ``data/blocklist/README.md``):

    * ``{"xgids": ["XGID=...", ...]}`` — list of raw XGID strings (normalized here).
    * ``{"keys": ["<canonical-key>", ...]}`` — pre-normalized keys.
    * ``{"entries": [{"xgid": "..."} | {"key": "..."}]}`` — annotated entries.
    """
    bl = into or Blocklist()
    added = 0
    for x in data.get("xgids", []):
        bl.add_xgid(x)
        added += 1
    for k in data.get("keys", []):
        bl.keys.add(k)
        added += 1
    for entry in data.get("entries", []):
        if "xgid" in entry:
            bl.add_xgid(entry["xgid"])
            added += 1
        elif "key" in entry:
            bl.keys.add(entry["key"])
            added += 1
    if added:
        bl.sources.append(source)
    return bl
