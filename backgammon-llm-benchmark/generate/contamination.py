"""Contamination controls: canary, hashing, redaction, findability (PLAN.md §2.4-2.5).

Pure helpers (no network, no writes) implementing the publishable anti-leakage
machinery:

* **Canary injection** — read the ``BENCH-CANARY-<uuid>`` token from ``CANARY.md``
  and stamp it into every published record (§2.5).
* **Held-out hashing** — SHA-256 of the *normalized* XGID for the published
  ``positions/heldout/hashes.json``; you cannot memorize what was only released
  as a hash (§2.5).
* **Redaction** — strip the answer-bearing content from a held-out record so only
  the hash + coarse metadata can be published (§2.5).
* **Findability report** — emit the exact search strings a human (or a
  rate-limited script) should check for a candidate before it enters the public
  split; this module performs **no** web calls itself (§2.4).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any

from bgcore.board import Board
from generate.dedup import mirror_key
from ids import gnubg_id as _gnubg
from ids import xgid as _xgid

_CANARY_RE = re.compile(r"BENCH-CANARY-[0-9a-fA-F-]{36}")

# Fields that reveal the position/answer and must be dropped before publishing a
# held-out record as anything more than a hash (PLAN.md §2.5).
_ANSWER_FIELDS = (
    "board_json",
    "ascii",
    "image_png",
    "image_svg",
    "xgid",
    "gnubg_id",
    "rollout_ref",
)


def read_canary(path: str = "CANARY.md") -> str:
    """Read the canary token from ``CANARY.md`` (PLAN.md §2.5, CANARY.md)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = _CANARY_RE.search(text)
    if not m:
        raise ValueError(f"no BENCH-CANARY-<uuid> token found in {path!r}")
    return m.group(0)


def inject_canary(record: dict[str, Any], token: str) -> dict[str, Any]:
    """Return a copy of ``record`` with the canary token stamped in."""
    out = dict(record)
    out["canary"] = token
    return out


def normalized_xgid_key(pos: Any) -> str:
    """Color/symmetry-normalized dedup key for a Board or XGID string.

    Mirror-folded (like :func:`generate.dedup.mirror_key`) so a position hashes
    identically whichever seat / color it was recorded from — consistent with the
    dedup normalization of PLAN §2.3 and robust for the membership oracle.
    """
    board = pos if isinstance(pos, Board) else _xgid.xgid_to_board(pos)
    return mirror_key(board)


def hash_xgid(pos: Any, *, algorithm: str = "sha256") -> str:
    """SHA-256 of the normalized XGID (PLAN.md §2.5).

    Normalizing first means a position stored from either seat / dice order hashes
    identically, so the published hash set is a faithful membership oracle.
    """
    key = normalized_xgid_key(pos)
    h = hashlib.new(algorithm)
    h.update(key.encode("utf-8"))
    return h.hexdigest()


def hash_record(record: dict[str, Any], *, algorithm: str = "sha256") -> str:
    """Hash a position record by its XGID (falls back to board_json)."""
    if record.get("xgid"):
        return hash_xgid(record["xgid"], algorithm=algorithm)
    if record.get("board_json"):
        board = Board.from_json(record["board_json"])
        return hash_xgid(board, algorithm=algorithm)
    raise ValueError("record has neither xgid nor board_json to hash")


def heldout_hashes(
    records: list[dict[str, Any]],
    *,
    canary: str,
    created: str | None = None,
    dataset_version: str | None = None,
    algorithm: str = "sha256",
) -> dict[str, Any]:
    """Build the published ``positions/heldout/hashes.json`` payload (PLAN.md §2.5).

    Only the hash of each held-out record (plus coarse, non-revealing tier/split
    counts) is emitted — never the positions themselves.
    """
    hashes = sorted({hash_record(r, algorithm=algorithm) for r in records})
    return {
        "canary": canary,
        "algorithm": algorithm,
        "created": created,
        "dataset_version": dataset_version,
        "count": len(hashes),
        "note": "Published hashes of the private held-out records (PLAN.md §2.5).",
        "hashes": hashes,
    }


def redact_record(
    record: dict[str, Any],
    *,
    canary: str | None = None,
    algorithm: str = "sha256",
) -> dict[str, Any]:
    """Redact a held-out record for publication: keep only non-answer metadata.

    Drops every answer-bearing field (board, renders, IDs, rollout ref) and
    replaces them with the record's SHA-256 hash, so the public artifact proves
    a position exists at a tier/split without revealing it (PLAN.md §2.5).
    """
    digest = hash_record(record, algorithm=algorithm)
    out: dict[str, Any] = {
        "hash": digest,
        "algorithm": algorithm,
        "position_id": record.get("position_id"),
        "tier": record.get("tier"),
        "phase": record.get("phase"),
        "decision_type": record.get("decision_type"),
        "play_mode": record.get("play_mode"),
        "split": record.get("split"),
        "created": record.get("created"),
        "redacted": True,
    }
    for f in _ANSWER_FIELDS:
        out.pop(f, None)
    if canary:
        out["canary"] = canary
    return {k: v for k, v in out.items() if v is not None}


@dataclass
class FindabilityReport:
    """The search strings to check for a candidate before it goes public (§2.4).

    This is a *worklist*, not a verdict: :meth:`emit` returns exact strings a
    human or a rate-limited script should search; this module makes no web calls.
    """

    xgid: str
    position_id: str
    match_id: str
    queries: list[str] = field(default_factory=list)
    note: str = (
        "Zero hits is necessary-not-sufficient evidence of unfindability "
        "(PLAN.md §2.4). Record the check date alongside results."
    )

    def emit(self) -> dict[str, Any]:
        return {
            "xgid": self.xgid,
            "gnubg_position_id": self.position_id,
            "gnubg_match_id": self.match_id,
            "queries": self.queries,
            "note": self.note,
            "web_calls_made": 0,
        }


def findability_report(pos: Any) -> FindabilityReport:
    """Build a :class:`FindabilityReport` for a Board or XGID (no web calls).

    The queries are the exact ID strings whose presence on the web would flag the
    position as findable: the XGID, the GNU BG Position ID and the Match ID, both
    bare and quoted for exact-match search.
    """
    if isinstance(pos, Board):
        board = pos
        xgid = _xgid.board_to_xgid(board)
    else:
        xgid = str(pos)
        board = _xgid.xgid_to_board(xgid)
    ids = _gnubg.board_to_ids(board)
    pid, mid = ids["position_id"], ids["match_id"]
    queries = [xgid, f'"{xgid}"', pid, f'"{pid}"', f'"{pid}" "{mid}"']
    return FindabilityReport(xgid=xgid, position_id=pid, match_id=mid, queries=queries)
