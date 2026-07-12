#!/usr/bin/env python3
"""Verify / inspect a single position by XGID (PLAN.md §7 acceptance).

A hands-on debugging tool: given an XGID it prints the reconstructed
``board_json`` and ASCII render, checks the XGID<->board round-trip is lossless,
reports the GNU BG IDs, the legal-move count, and the topology-derived phase +
taxonomy-prior tier estimate.

Usage::

    python3 scripts/verify_position.py "XGID=-a-...:0:0:1:31:0:0:0:7:10"
    python3 scripts/verify_position.py --start        # opening 3-1 example
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bgcore import moves as _moves  # noqa: E402
from bgcore.board import Board, canonical_key, validate  # noqa: E402
from generate import tiering  # noqa: E402
from ids import gnubg_id as _gnubg  # noqa: E402
from ids import xgid as _xgid  # noqa: E402
from render import ascii as _ascii  # noqa: E402


def verify(xgid: str) -> dict:
    """Return a structured verification report for an XGID."""
    board = _xgid.xgid_to_board(xgid)
    board_errors = validate(board, strict=False)

    # round-trip: board -> XGID -> board must be canonically identical
    reencoded = _xgid.board_to_xgid(board)
    roundtrip_board = _xgid.xgid_to_board(reencoded)
    roundtrip_ok = canonical_key(board) == canonical_key(roundtrip_board)

    ids = _gnubg.board_to_ids(board)
    ids_ok = canonical_key(_gnubg.ids_to_board(ids["position_id"], ids["match_id"])) == canonical_key(board)

    legal = _moves.legal_moves(board) if board.dice else []
    tr = tiering.assign_tier(board)

    return {
        "xgid_in": xgid,
        "xgid_reencoded": reencoded,
        "roundtrip_ok": roundtrip_ok,
        "gnubg_ids": ids,
        "gnubg_ids_roundtrip_ok": ids_ok,
        "board_valid": not board_errors,
        "board_errors": board_errors,
        "decision_type": board.decision_type,
        "legal_move_count": len(legal),
        "legal_moves_sample": legal[:8],
        "phase": tr.phase,
        "tier_estimate": tr.tier,
        "expert_miss_rate": tr.expert_miss_rate,
        "expected_expert_loss": tr.expected_expert_loss,
        "difficulty_source": tr.difficulty_source,
        "board_json": board.to_json(),
    }


def _print_report(xgid: str) -> int:
    report = verify(xgid)
    board = _xgid.xgid_to_board(xgid)
    print(_ascii.render(board))
    print()
    ok = report["roundtrip_ok"] and report["gnubg_ids_roundtrip_ok"] and report["board_valid"]
    print(f"XGID (re-encoded): {report['xgid_reencoded']}")
    print(f"round-trip lossless: {report['roundtrip_ok']}")
    print(f"gnubg id round-trip: {report['gnubg_ids_roundtrip_ok']}")
    print(f"  position id: {report['gnubg_ids']['position_id']}")
    print(f"  match id:    {report['gnubg_ids']['match_id']}")
    print(f"board valid: {report['board_valid']}  {report['board_errors'] or ''}")
    print(f"decision type: {report['decision_type']}   legal moves: {report['legal_move_count']}")
    if report["legal_moves_sample"]:
        print(f"  e.g. {', '.join(report['legal_moves_sample'])}")
    print(f"phase: {report['phase']}   tier estimate: {report['tier_estimate']} "
          f"(miss {report['expert_miss_rate']}, EEL {report['expected_expert_loss']} mp, "
          f"src {report['difficulty_source']})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify/inspect a position by XGID.")
    ap.add_argument("xgid", nargs="?", help="the XGID string to verify.")
    ap.add_argument("--start", action="store_true", help="use the opening 3-1 position.")
    ap.add_argument("--json", action="store_true", help="emit the full JSON report only.")
    args = ap.parse_args(argv)

    if args.start:
        xgid = _xgid.board_to_xgid(Board.starting_position([3, 1]))
    elif args.xgid:
        xgid = args.xgid
    else:
        ap.error("provide an XGID or --start")
        return 2

    if args.json:
        print(json.dumps(verify(xgid), indent=2))
        return 0
    return _print_report(xgid)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
