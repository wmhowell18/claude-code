"""Recompute + calibrate BenchPR from a run directory (PLAN.md §4.4, §9).

Two responsibilities:

1. **Recompute (implemented now).** Read a results JSON (or a ``runs/<run_id>/``
   working tree) and recompute ``BenchPR = 500 x mean(equity_loss)`` from the
   per-decision records, checking it matches the stored aggregate. This is the
   reproducibility gate: scoring must reproduce from cache/records.

2. **Calibrate vs. XG reference games (future).** Compare BenchPR against a few
   XG-analyzed reference games to pin the exact PR constant and cube-vs-checker
   handling (PLAN §9 item 2). Stubbed with a clear message until reference data
   exists; the metric is labelled "BenchPR (PR-calibrated)" until then.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import scoring as _scoring  # noqa: E402

BENCHPR_TOLERANCE = 1e-6


def _iter_results_files(target: Path):
    """Yield results JSON paths for a file or a run directory."""
    if target.is_file():
        yield target
        return
    if target.is_dir():
        # A run working tree: recompute from scores/*.jsonl.
        scores = sorted((target / "scores").glob("*.jsonl"))
        if scores:
            for s in scores:
                yield s
            return
        # Or a results directory of *.json.
        for j in sorted(target.glob("*.json")):
            yield j
        return
    raise FileNotFoundError(target)


def _recompute_from_results(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    decisions = obj.get("decisions", [])
    losses = [float(d["equity_loss"]) for d in decisions]
    recomputed = _scoring.benchpr(losses)
    stored = obj.get("aggregate", {}).get("benchpr")
    return {
        "file": str(path),
        "model": obj.get("model"),
        "track": obj.get("track"),
        "n": len(losses),
        "recomputed_benchpr": recomputed,
        "stored_benchpr": stored,
        "match": (stored is None or abs(recomputed - float(stored)) <= BENCHPR_TOLERANCE),
    }


def _recompute_from_scores(path: Path) -> dict:
    losses = []
    model = track = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        losses.append(float(rec["equity_loss"]))
        model = model or rec.get("model")
        track = track or rec.get("track")
    return {
        "file": str(path),
        "model": model,
        "track": track,
        "n": len(losses),
        "recomputed_benchpr": _scoring.benchpr(losses),
        "stored_benchpr": None,
        "match": True,
    }


def recompute(target: Path) -> list[dict]:
    out = []
    for path in _iter_results_files(target):
        if path.suffix == ".jsonl":
            out.append(_recompute_from_scores(path))
        else:
            out.append(_recompute_from_results(path))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute/validate BenchPR from a run.")
    ap.add_argument("target", help="A results JSON file, a results dir, or a runs/<run_id>/ dir.")
    ap.add_argument("--reference", default=None,
                    help="(Future) XG-analyzed reference games to calibrate the PR constant.")
    args = ap.parse_args(argv)

    if args.reference:
        print("Calibration against XG reference games is not implemented yet "
              "(PLAN §9 item 2). Metric remains 'BenchPR (PR-calibrated)'.")

    rows = recompute(Path(args.target))
    ok = True
    for r in rows:
        status = "OK" if r["match"] else "MISMATCH"
        stored = "-" if r["stored_benchpr"] is None else f"{r['stored_benchpr']:.4f}"
        print(f"[{status}] {r['model']} [{r['track']}] "
              f"BenchPR={r['recomputed_benchpr']:.4f} (stored {stored}) n={r['n']}")
        ok = ok and r["match"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
