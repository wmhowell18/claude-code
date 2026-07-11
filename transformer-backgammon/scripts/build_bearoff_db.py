#!/usr/bin/env python3
"""Build and cache the one-sided bearoff database.

Usage:
    python scripts/build_bearoff_db.py [--max-checkers N] [--path FILE]

The full 15-checker database (54,264 positions) builds in about half a
minute and is cached (~2.5 MB) at ~/.cache/backgammon/bearoff_15.npz by
default, so this only ever needs to run once per machine.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backgammon.evaluation.bearoff import BearoffDatabase, MAX_CHECKERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-checkers", type=int, default=MAX_CHECKERS,
        help="Maximum checkers covered (default: 15 = full database)",
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Cache file path (default: ~/.cache/backgammon/bearoff_N.npz)",
    )
    args = parser.parse_args()

    t0 = time.time()
    db = BearoffDatabase.load_or_build(
        path=args.path, max_checkers=args.max_checkers
    )
    elapsed = time.time() - t0

    print(f"Bearoff database ready in {elapsed:.1f}s "
          f"(max_checkers={db.max_checkers})")
    print(f"  E[rolls] with 15 checkers stacked on the 6-point: "
          f"{db.rolls_to_bear_off((0, 0, 0, 0, 0, 15)):.3f}")
    print(f"  E[rolls] with a balanced 15-checker home board:   "
          f"{db.rolls_to_bear_off((3, 3, 3, 2, 2, 2)):.3f}")


if __name__ == "__main__":
    main()
