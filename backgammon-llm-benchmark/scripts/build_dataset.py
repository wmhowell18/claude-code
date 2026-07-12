#!/usr/bin/env python3
"""End-to-end dataset build (PLAN.md §7).

Wires the construction pipeline: self-play -> extract candidates -> dedup +
blocklist -> stratified sample -> tier -> assemble records. Ground-truth rollouts
(gnubg) are a separate, expensive step and are represented here only by a
placeholder ``rollout_ref``.

``--dry-run`` injects a small canned .mat instead of spawning gnubg, so the whole
pipeline is exercisable without the engine (which is not installed in CI). This
script prints a summary; it does **not** write into ``positions/`` or ``data/``
(that wiring lands with the real corpus in later phases).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generate import contamination, dedup, records, sample, selfplay, tiering  # noqa: E402

# A tiny, deterministic canned self-play match used by --dry-run. Legal moves
# were generated from the actual start via bgcore, and it includes a double/take
# so the cube path is exercised.
DRY_RUN_MAT = """7 point match

 Game 1
 gnubg_A : 0                          gnubg_B : 0
  1) 45: 8/3 6/2                        45: 13/9 8/3
  2) Doubles => 2                       Takes
  3) 25: 8/3 8/6                        65: 13/7 9/4
  4) 14: 24/20 3/2                      21: 8/7 6/4
  5) 66: 13/7(4)                        54: 7/2 7/3
"""


def _canned_runner(_commands: list[str]) -> str:
    return DRY_RUN_MAT


def build(
    config: selfplay.RunConfig,
    *,
    dry_run: bool,
    target: int,
    blocklist_dir: str,
    canary_path: str,
    created: str,
    split: str,
    targets: sample.SampleTargets | None = None,
) -> dict:
    """Run the pipeline and return a summary dict (no filesystem writes)."""
    runner = _canned_runner if dry_run else None
    candidates = selfplay.generate(config, runner=runner)

    before = len(candidates)
    candidates = dedup.dedup(candidates)
    bl = dedup.load_blocklist(blocklist_dir)
    kept, rejected = bl.filter(candidates)

    n = min(target, len(kept))
    sampled = sample.stratified_sample(kept, n, targets=targets, seed=config.seed)

    canary = contamination.read_canary(canary_path)
    built = []
    for c in sampled:
        tr = tiering.assign_tier(c.board)
        rec = records.build_record(
            c.board,
            tier=tr.tier,
            split=split,
            decision_type=c.decision_type,
            play_mode=c.play_mode,
            phase=tr.phase,
            expected_expert_loss=tr.expected_expert_loss,
            expert_miss_rate=tr.expert_miss_rate,
            difficulty_source=tr.difficulty_source,
            rollout_ref=f"rollouts/gnubg/{records.position_id_for(c.board)}.json",
            canary=canary,
            created=created,
        )
        records.validate_record(rec)  # raises on structural failure
        built.append(rec)

    return {
        "candidates_raw": before,
        "after_dedup": len(candidates),
        "blocklist_rejected": len(rejected),
        "sampled": len(sampled),
        "records_built": len(built),
        "strata": sample.stratum_counts(sampled),
        "tier_counts": _tier_counts(built),
        "records": built,
    }


def _tier_counts(built: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in built:
        counts[r["tier"]] = counts.get(r["tier"], 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a backgammon-LLM-benchmark dataset slice.")
    ap.add_argument("--dry-run", action="store_true", help="use canned self-play (no gnubg).")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plies", type=int, default=2)
    ap.add_argument("--match-length", type=int, default=None,
                    help="match length; omit for a money session.")
    ap.add_argument("--target", type=int, default=50, help="number of positions to sample.")
    ap.add_argument("--split", choices=["pilot", "dev", "heldout"], default="pilot")
    ap.add_argument("--blocklist-dir", default=str(_ROOT / "data" / "blocklist"))
    ap.add_argument("--canary", default=str(_ROOT / "CANARY.md"))
    ap.add_argument("--created", default="2026-07-12T00:00:00Z")
    ap.add_argument("--show-records", action="store_true", help="include full records in output.")
    args = ap.parse_args(argv)

    config = selfplay.RunConfig(
        games=args.games, seed=args.seed, plies=args.plies, match_length=args.match_length
    )
    summary = build(
        config,
        dry_run=args.dry_run,
        target=args.target,
        blocklist_dir=args.blocklist_dir,
        canary_path=args.canary,
        created=args.created,
        split=args.split,
    )
    if not args.show_records:
        summary.pop("records", None)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
