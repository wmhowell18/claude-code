"""Execute a benchmark run from a run config (PLAN.md §5.2).

Loads ``runs/<run>.yaml`` (or ``.json``) and drives the harness
(client/runner/scoring/report) to evaluate a model slate on a track, writing
``results/*.json`` + a per-run working tree under ``runs/<run_id>/``.

Examples::

    # Estimate request counts + build prompts, no network:
    python3 scripts/run_benchmark.py --config runs/example.yaml --dry-run

    # Real run (needs OPENROUTER_API_KEY):
    python3 scripts/run_benchmark.py --config runs/example.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running as a script (`python3 scripts/run_benchmark.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os  # noqa: E402

from harness import prompts as _prompts  # noqa: E402
from harness.runner import (  # noqa: E402
    apply_quality_gate,
    dataset_hash,
    load_config,
    load_positions,
    load_rollouts,
    run_config,
)
from bgcore.board import Board  # noqa: E402


def _resolve_dataset(config: dict, base: Path):
    ds = config.get("dataset", {}) or {}
    positions = load_positions(base / ds["positions"]) if ds.get("positions") else []
    rollouts = load_rollouts(base / ds["rollouts"]) if ds.get("rollouts") else {}
    return positions, rollouts


def _gate(config: dict, positions: list, rollouts: dict):
    """Apply the quality gate the same way run_config does. Returns (kept, excluded)."""
    gate_on = bool(config.get("quality_gate", True))
    return apply_quality_gate(positions, rollouts, enabled=gate_on)


def dry_run(config: dict, base: Path) -> dict:
    """Build prompts + estimate request counts without any network calls."""
    positions, rollouts = _resolve_dataset(config, base)
    positions, excluded = _gate(config, positions, rollouts)
    models = list(config.get("models", []))
    track = config.get("track", "text")
    sampling = config.get("sampling", {}) or {}
    n = int(sampling.get("n", 1) or 1)
    mode = sampling.get("mode", "greedy")
    max_retries = int(sampling.get("max_retries", 2) or 0)
    samples_per_pos = n if (mode == "self-consistency" and n > 1) else 1

    built = 0
    sample_prompt = None
    for pos in positions:
        board = Board.from_json(pos["board_json"])
        # Text/text+image prompts build without image bytes; image track needs a
        # render, which we skip in dry-run (count only).
        if track == "text":
            msgs = _prompts.build_messages(board, track="text", ascii_text=pos.get("ascii"))
            if sample_prompt is None:
                sample_prompt = msgs
        built += 1

    base_requests = len(models) * len(positions) * samples_per_pos
    max_requests = len(models) * len(positions) * (samples_per_pos + (max_retries if samples_per_pos == 1 else 0))
    return {
        "models": models,
        "track": track,
        "positions": len(positions),
        "prompt_version": _prompts.PROMPT_VERSION,
        "dataset_hash": dataset_hash(positions),
        "samples_per_position": samples_per_pos,
        "estimated_requests": base_requests,
        "estimated_requests_max_with_retries": max_requests,
        "prompts_built": built,
        "quality_gate": {
            "enabled": bool(config.get("quality_gate", True)),
            "excluded_count": len(excluded),
            "excluded": [{"position_id": pid, "reason": reason} for pid, reason in excluded],
        },
        "sample_prompt": sample_prompt,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the backgammon LLM benchmark.")
    ap.add_argument("--config", required=True, help="Path to a run config (.yaml or .json).")
    ap.add_argument("--base-dir", default=".", help="Repo root the dataset paths resolve against.")
    ap.add_argument("--dry-run", action="store_true", help="Build prompts + count requests; no network.")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of positions (debug).")
    ap.add_argument("--show-prompt", action="store_true", help="Print one built prompt (with --dry-run).")
    args = ap.parse_args(argv)

    base = Path(args.base_dir)
    config = load_config(args.config)

    if args.limit is not None:
        positions, rollouts = _resolve_dataset(config, base)
        # Gate first (so --limit caps eligible positions), then trim.
        positions, _excluded = _gate(config, positions, rollouts)
        positions = positions[: args.limit]
        # Inline the trimmed dataset so run_config/dry_run use it. The gate has
        # already been applied, so turn it off downstream to avoid re-gating.
        config = dict(config)
        config["dataset"] = {}
        config["quality_gate"] = False
    else:
        positions = rollouts = None

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        # Real run: fail fast with a readable message if the API key is missing,
        # rather than surfacing an auth traceback mid-run.
        print("error: OPENROUTER_API_KEY is not set. Export your OpenRouter key "
              "(or use --dry-run for a no-network estimate).", file=sys.stderr)
        return 2

    if args.dry_run:
        summary = dry_run(config, base) if positions is None else _dry_run_inline(config, positions)
        printable = dict(summary)
        prompt = printable.pop("sample_prompt", None)
        print(json.dumps(printable, indent=2))
        if args.show_prompt and prompt:
            print("\n--- sample prompt ---")
            for msg in prompt:
                content = msg["content"]
                text = content if isinstance(content, str) else "[multimodal parts]"
                print(f"[{msg['role']}]\n{text}\n")
        return 0

    results = asyncio.run(
        run_config(config, base_dir=base, positions=positions, rollouts=rollouts)
    )
    for mr in results:
        losses = [d.equity_loss for d in mr.decisions]
        benchpr = 500 * (sum(losses) / len(losses)) if losses else 0.0
        acc = sum(1 for d in mr.decisions if d.is_best) / len(mr.decisions) if mr.decisions else 0.0
        print(
            f"{mr.model} [{mr.track}]  BenchPR={benchpr:.2f}  "
            f"best-move-acc={acc:.3f}  cost=${mr.cost_usd:.4f}  n={len(mr.decisions)}"
        )
    return 0


def _dry_run_inline(config: dict, positions: list) -> dict:
    # Same as dry_run but with positions already loaded (for --limit).
    models = list(config.get("models", []))
    track = config.get("track", "text")
    sampling = config.get("sampling", {}) or {}
    n = int(sampling.get("n", 1) or 1)
    mode = sampling.get("mode", "greedy")
    samples_per_pos = n if (mode == "self-consistency" and n > 1) else 1
    return {
        "models": models,
        "track": track,
        "positions": len(positions),
        "prompt_version": _prompts.PROMPT_VERSION,
        "dataset_hash": dataset_hash(positions),
        "samples_per_position": samples_per_pos,
        "estimated_requests": len(models) * len(positions) * samples_per_pos,
        "sample_prompt": None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
