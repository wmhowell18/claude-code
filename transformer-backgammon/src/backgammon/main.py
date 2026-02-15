"""Command-line entrypoint for transformer-backgammon.

This module intentionally provides a small stable CLI surface so the
`backgammon` console script from ``pyproject.toml`` always resolves to a
valid callable.
"""

from __future__ import annotations

import argparse

from backgammon import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="backgammon",
        description="Transformer Backgammon CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"transformer-backgammon {__version__}",
    )
    return parser


def main() -> int:
    """CLI entrypoint used by the `backgammon` console script."""
    parser = build_parser()
    parser.parse_args()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

