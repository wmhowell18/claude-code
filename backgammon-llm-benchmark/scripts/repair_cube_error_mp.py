#!/usr/bin/env python3
"""Recompute cube ``error_mp`` (and verify ``best_action``) for checked-in rollouts.

Some pilot cube rollout files carry a game-theoretically wrong ``error_mp`` map:
when doubling is *wrong* (best action "No double") the old formula scored
"Double, Take" as 0 and "Double, Pass" as ``dp − min(dt, dp)`` (a huge number),
instead of charging both double answers the same doubling error ``nd − min(dt, dp)``.

This script repairs each cube rollout by recomputing ``error_mp``/``best_action``
from the stored equities via the authoritative
:func:`generate.gnubg.cube_action_errors`. It **never** touches the equities (those
are real rollout output). It is **idempotent**: re-running on already-correct files
changes nothing. Run from the repo root::

    python3 scripts/repair_cube_error_mp.py            # apply the fix in place
    python3 scripts/repair_cube_error_mp.py --check    # report only, exit 1 if stale
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from generate.gnubg import cube_action_errors  # noqa: E402

ROLL_DIR = os.path.join(REPO_ROOT, "rollouts", "gnubg")


def _fmt(err: dict) -> str:
    return "ND=%.1f DT=%.1f DP=%.1f" % (
        err["No double"], err["Double, Take"], err["Double, Pass"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report only; exit non-zero if any file is stale")
    args = ap.parse_args(argv)

    rows = []
    stale = 0
    for path in sorted(glob.glob(os.path.join(ROLL_DIR, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("decision_type") != "cube":
            continue
        cube = rec.get("cube") or {}
        nd = float(cube["no_double_equity"])
        dt = float(cube["double_take_equity"])
        dp = float(cube["double_pass_equity"])
        new_best, new_err = cube_action_errors(nd, dt, dp)
        old_best = cube.get("best_action")
        old_err = cube.get("error_mp") or {}
        changed = (old_err != new_err) or (old_best != new_best)
        rows.append((rec["position_id"], nd, dt, dp, old_best, old_err, new_best, new_err, changed))
        if changed:
            stale += 1
            if not args.check:
                cube["best_action"] = new_best
                cube["error_mp"] = new_err
                with open(path, "w", encoding="utf-8") as fh:
                    # indent=2, no trailing newline — byte-identical to the
                    # generator's output so only error_mp/best_action change.
                    json.dump(rec, fh, indent=2, ensure_ascii=False)

    # before/after table
    print("position_id           nd       dt       dp     | old best / err                | new best / err")
    for pid, nd, dt, dp, ob, oe, nb, ne, ch in rows:
        mark = "  <-- CHANGED" if ch else ""
        oes = _fmt(oe) if oe else "(none)"
        print("%-20s % .4f % .4f % .4f" % (pid, nd, dt, dp))
        print("    old  %-13s %s" % (ob, oes))
        print("    new  %-13s %s%s" % (nb, _fmt(ne), mark))

    verb = "stale" if args.check else "repaired"
    print("\n%d cube rollouts scanned, %d %s." % (len(rows), stale, verb))
    if args.check and stale:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
