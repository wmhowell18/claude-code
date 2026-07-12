"""Async batch runner (PLAN.md §5.2).

Orchestrates a benchmark run: load a run config + the position/rollout dataset,
fan out (model x position x track) with bounded concurrency, serve from the cache
first, apply the reparse-on-unparseable policy (<= 2 re-asks, PLAN §4.6), score
each decision against its rollout, and write raw responses + parsed + scores under
``runs/<run_id>/``. Results JSON (``schema/results.schema.json``) is emitted via
:mod:`harness.report`.

Config format decision. Configs are plain data. ``.json`` is read with the stdlib
``json`` module; ``.yaml``/``.yml`` is read with PyYAML **if present**, otherwise
with the small indentation parser in this module (:func:`parse_simple_yaml`) —
enough for the flat ``runs/example.yaml`` shape. PyYAML is therefore *not* a
required dependency; JSON configs work with zero extra packages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

from bgcore.board import Board
from harness import prompts as _prompts
from harness import scoring as _scoring
from harness.cache import ResponseCache
from harness.client import ChatResult
from harness.cost import BudgetGuard, CostTracker, majority_vote
from harness.parse import ParseOutcome, parse_answer

__all__ = [
    "parse_simple_yaml",
    "load_config",
    "load_positions",
    "load_rollouts",
    "dataset_hash",
    "DecisionRecord",
    "ModelRunResult",
    "run_model",
    "run_config",
]


# ========================================================================
# Config loading
# ========================================================================


def _scalar(tok: str) -> Any:
    s = tok.strip()
    if not s:
        return None
    if (s[0] == s[-1]) and s[0] in ("'", '"') and len(s) >= 2:
        return s[1:-1]
    low = s.lower()
    if low in ("null", "none", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _strip_comment(line: str) -> str:
    # Remove a trailing ' #...' comment (not inside quotes; our configs are simple).
    in_q: str | None = None
    out: list[str] = []
    for i, ch in enumerate(line):
        if in_q:
            out.append(ch)
            if ch == in_q:
                in_q = None
            continue
        if ch in ("'", '"'):
            in_q = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out)


def parse_simple_yaml(text: str) -> Any:
    """Parse the small YAML subset used by run configs (fallback for no PyYAML).

    Supports nested block mappings, block sequences (``- item``) of scalars or
    inline ``key: value`` maps, ``#`` comments, and scalar typing
    (int/float/bool/null/quoted-string). Not a general YAML implementation.
    """
    raw_lines = text.splitlines()
    lines: list[tuple[int, str]] = []
    for raw in raw_lines:
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))

    pos = 0

    def parse_block(min_indent: int) -> Any:
        nonlocal pos
        if pos >= len(lines):
            return None
        indent, content = lines[pos]
        is_seq = content.startswith("- ") or content == "-"
        container: Any = [] if is_seq else {}
        while pos < len(lines):
            indent, content = lines[pos]
            if indent < min_indent:
                break
            if content.startswith("- ") or content == "-":
                item = content[1:].strip()
                pos += 1
                if not item:
                    container.append(parse_block(indent + 1))
                elif ":" in item and not (item[0] in ("'", '"')):
                    # inline map entry starting a sequence item
                    k, _, v = item.partition(":")
                    entry = {k.strip(): _scalar(v)}
                    # subsequent deeper lines belong to this entry
                    while pos < len(lines) and lines[pos][0] > indent:
                        sub_indent, sub = lines[pos]
                        sk, _, sv = sub.partition(":")
                        if sv.strip():
                            entry[sk.strip()] = _scalar(sv)
                            pos += 1
                        else:
                            entry[sk.strip()] = parse_block(sub_indent + 1)
                    container.append(entry)
                else:
                    container.append(_scalar(item))
                continue
            # mapping entry
            key, _, val = content.partition(":")
            key = key.strip()
            pos += 1
            if val.strip():
                container[key] = _scalar(val)
            else:
                if pos < len(lines) and lines[pos][0] > indent:
                    container[key] = parse_block(indent + 1)
                else:
                    container[key] = None
        return container

    result = parse_block(0)
    return result if result is not None else {}


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a run config (``.json`` or ``.yaml``/``.yml``)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return parse_simple_yaml(text)


# ========================================================================
# Dataset loading
# ========================================================================


def _load_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load JSON records from a file (array or JSON-lines) or a directory of .json."""
    p = Path(path)
    records: list[dict[str, Any]] = []
    if p.is_dir():
        for fp in sorted(p.glob("*.json")):
            records.extend(_load_records(fp))
        return records
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return list(data)
    if text[0] == "{":
        try:
            obj = json.loads(text)
            return [obj]
        except json.JSONDecodeError:
            pass
    # JSON-lines
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_positions(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load position records (schema/position.schema.json)."""
    return _load_records(path)


def load_rollouts(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Load rollout records keyed by ``position_id`` (schema/rollout.schema.json)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in _load_records(path):
        pid = rec.get("position_id")
        if pid:
            out[pid] = rec
    return out


def dataset_hash(positions: Sequence[dict[str, Any]]) -> str:
    """Stable SHA-256 over the position records (order-independent)."""
    keys = sorted(
        json.dumps(p, sort_keys=True, separators=(",", ":")) for p in positions
    )
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ========================================================================
# Records
# ========================================================================


@dataclass
class DecisionRecord:
    position_id: str
    track: str
    model: str
    decision_type: str
    tier: str | None
    chosen: str | None
    raw_answer: str
    is_best: bool
    equity_loss: float
    equity_loss_mp: float
    parse_failed: bool
    attempts: int
    rollout_move: str | None
    usage: dict[str, Any]
    detail: str
    votes: int = 1

    def row(self) -> dict[str, Any]:
        return {
            "equity_loss": self.equity_loss,
            "is_best": self.is_best,
            "parse_failed": self.parse_failed,
            "tier": self.tier,
            "track": self.track,
            "decision_type": self.decision_type,
        }


@dataclass
class ModelRunResult:
    model: str
    track: str
    prompt_version: str
    sampling: dict[str, Any]
    decisions: list[DecisionRecord]
    cost_usd: float
    dataset_hash: str
    budget_usd: float | None = None
    ascii_render_version: str | None = None
    image_render_version: str | None = None


# ========================================================================
# Client protocol / helpers
# ========================================================================

ClientLike = Any  # anything with an async chat(messages, *, model, ...) -> ChatResult


def _usage_from_result(res: ChatResult) -> dict[str, Any]:
    return {
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "reasoning_tokens": res.reasoning_tokens,
        "cost_usd": res.cost_usd,
    }


def _image_for(position: dict[str, Any], image_provider: Callable[[dict], bytes] | None) -> bytes | None:
    if image_provider is not None:
        return image_provider(position)
    ref = position.get("image_png")
    if ref and Path(ref).is_file():
        return Path(ref).read_bytes()
    # Lazy render as a last resort (needs the render extra / cairosvg).
    from render.raster import board_to_png  # noqa: PLC0415

    return board_to_png(Board.from_json(position["board_json"]))


# ========================================================================
# Run one model
# ========================================================================


async def run_model(
    client: ClientLike,
    positions: Sequence[dict[str, Any]],
    rollouts: dict[str, dict[str, Any]],
    *,
    model: str,
    track: str = "text",
    sampling: dict[str, Any] | None = None,
    prompt_version: str = _prompts.PROMPT_VERSION,
    cache: ResponseCache | None = None,
    budget: BudgetGuard | None = None,
    cost: CostTracker | None = None,
    image_provider: Callable[[dict], bytes] | None = None,
    concurrency: int = 8,
    raw_dir: str | os.PathLike[str] | None = None,
    noise_points: float | None = None,
    dataset_hash_str: str | None = None,
) -> ModelRunResult:
    """Evaluate one model over all positions on one track."""
    sampling = dict(sampling or {})
    temperature = sampling.get("temperature", 0)
    max_tokens = sampling.get("max_tokens")
    seed = sampling.get("seed")
    n = int(sampling.get("n", 1) or 1)
    mode = sampling.get("mode", "greedy")
    max_retries = int(sampling.get("max_retries", 2) or 0)
    cost = cost if cost is not None else CostTracker()
    budget = budget if budget is not None else BudgetGuard(None)
    sem = asyncio.Semaphore(max(1, concurrency))
    raw_root = Path(raw_dir) if raw_dir else None
    lock = asyncio.Lock()

    sampling_key = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "n": n,
        "mode": mode,
    }

    async def call_once(
        position_id: str, messages: list[dict[str, Any]], extra: Any
    ) -> ChatResult | None:
        """Cache-first single request; returns None if budget blocks it."""
        key = None
        if cache is not None:
            key = cache.key(
                model=model, prompt_version=prompt_version, position_id=position_id,
                track=track, sampling=sampling_key, extra=extra,
            )
            hit = cache.get(key)
            if hit is not None:
                return ChatResult.from_dict(hit)
        if not budget.can_spend():
            return None
        res = await client.chat(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, seed=seed,
        )
        async with lock:
            cost.add(model, res.cost_usd)
            budget.spend(res.cost_usd)
        if cache is not None and key is not None:
            cache.put(key, res.to_dict())
        return res

    async def eval_position(position: dict[str, Any]) -> DecisionRecord:
        pid = position["position_id"]
        board = Board.from_json(position["board_json"])
        rollout = rollouts.get(pid, {})
        tier = position.get("tier")
        rollout_moves = [m["move"] for m in (rollout.get("checker") or {}).get("moves", []) if m.get("move")]
        ascii_text = position.get("ascii")
        img = _image_for(position, image_provider) if track in ("image", "text+image") else None
        base_messages = _prompts.build_messages(
            board, track=track, ascii_text=ascii_text, image_png=img,
        )

        attempts = 0
        raw_answer = ""
        outcome: ParseOutcome | None = None
        votes = 1
        usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0,
                                 "reasoning_tokens": 0, "cost_usd": 0.0}

        async with sem:
            if mode == "self-consistency" and n > 1:
                samples: list[ParseOutcome] = []
                for i in range(n):
                    res = await call_once(pid, base_messages, extra={"sample": i})
                    if res is None:
                        break
                    attempts += 1
                    _accumulate_usage(usage, res)
                    oc = parse_answer(board, res.text, decision_type=board.decision_type,
                                      rollout_moves=rollout_moves)
                    samples.append(oc)
                    if not raw_answer:
                        raw_answer = res.text
                winner, tally = majority_vote(board, samples)
                if winner is not None:
                    outcome = winner
                    votes = next((c for o, c in tally if o is winner), 1)
                    raw_answer = winner.raw or raw_answer
                elif samples:
                    outcome = samples[-1]
            else:
                messages = list(base_messages)
                while attempts <= max_retries:
                    res = await call_once(pid, messages, extra={"attempt": attempts})
                    if res is None:
                        break
                    attempts += 1
                    _accumulate_usage(usage, res)
                    raw_answer = res.text
                    outcome = parse_answer(board, res.text, decision_type=board.decision_type,
                                           rollout_moves=rollout_moves)
                    if outcome.status != "unparseable":
                        break
                    if attempts > max_retries:
                        break
                    messages = list(base_messages) + [
                        {"role": "assistant", "content": res.text},
                        {"role": "user", "content": _REPARSE_REMINDER(board.decision_type)},
                    ]

        if outcome is None:
            # No response at all (budget exhausted / never issued): worst-case.
            outcome = ParseOutcome(
                "unparseable", board.decision_type,
                detail="unanswered (budget exhausted or skipped)", raw="",
            )

        score = _scoring.score_decision(rollout, board, outcome, noise_points=noise_points)

        rec = DecisionRecord(
            position_id=pid, track=track, model=model,
            decision_type=board.decision_type, tier=tier,
            chosen=score.chosen, raw_answer=raw_answer,
            is_best=score.is_best, equity_loss=score.equity_loss,
            equity_loss_mp=score.equity_loss_mp, parse_failed=score.parse_failed,
            attempts=attempts, rollout_move=outcome.rollout_move,
            usage=usage, detail=score.detail, votes=votes,
        )
        if raw_root is not None:
            _write_raw(raw_root, model, pid, rec, raw_answer)
        return rec

    results = await asyncio.gather(*(eval_position(p) for p in positions))
    # Keep input order deterministic.
    order = {p["position_id"]: i for i, p in enumerate(positions)}
    decisions = sorted(results, key=lambda r: order.get(r.position_id, 0))

    return ModelRunResult(
        model=model, track=track, prompt_version=prompt_version, sampling=sampling_key,
        decisions=decisions, cost_usd=cost.by_model().get(model, 0.0),
        dataset_hash=dataset_hash_str or "",
        budget_usd=budget.budget_usd,
        ascii_render_version=_ascii_version(), image_render_version=_image_version(),
    )


def _accumulate_usage(usage: dict[str, Any], res: ChatResult) -> None:
    usage["prompt_tokens"] += res.prompt_tokens
    usage["completion_tokens"] += res.completion_tokens
    usage["reasoning_tokens"] += res.reasoning_tokens
    usage["cost_usd"] = (usage.get("cost_usd") or 0.0) + float(res.cost_usd or 0.0)


def _REPARSE_REMINDER(decision_type: str) -> str:
    if decision_type == "cube":
        return ("Your previous reply did not end with a valid answer line. Reply again "
                "and finish with exactly one line: `ACTION: <double | no double | take | pass>`.")
    return ("Your previous reply did not end with a valid answer line. Reply again and "
            "finish with exactly one line: `MOVE: <your move in slash notation>`.")


def _write_raw(raw_root: Path, model: str, pid: str, rec: DecisionRecord, raw: str) -> None:
    d = raw_root / _slug(model)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "position_id": pid,
        "model": model,
        "raw_answer": raw,
        "chosen": rec.chosen,
        "is_best": rec.is_best,
        "equity_loss": rec.equity_loss,
        "parse_failed": rec.parse_failed,
        "attempts": rec.attempts,
        "usage": rec.usage,
        "detail": rec.detail,
    }
    (d / f"{pid}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)


def _ascii_version() -> str | None:
    try:
        from render.ascii import ASCII_RENDER_VERSION  # noqa: PLC0415

        return ASCII_RENDER_VERSION
    except Exception:  # noqa: BLE001
        return None


def _image_version() -> str | None:
    try:
        from render.svg import IMAGE_RENDER_VERSION  # noqa: PLC0415

        return IMAGE_RENDER_VERSION
    except Exception:  # noqa: BLE001
        return None


# ========================================================================
# Run a whole config
# ========================================================================


async def run_config(
    config: dict[str, Any],
    *,
    base_dir: str | os.PathLike[str] = ".",
    client: ClientLike | None = None,
    client_factory: Callable[[], ClientLike] | None = None,
    positions: Sequence[dict[str, Any]] | None = None,
    rollouts: dict[str, dict[str, Any]] | None = None,
    image_provider: Callable[[dict], bytes] | None = None,
    write_results: bool = True,
) -> list[ModelRunResult]:
    """Run every model in ``config`` and (optionally) emit results JSON.

    ``positions``/``rollouts`` may be supplied directly (tests); otherwise they are
    loaded from the ``dataset`` paths in the config. A ``client`` (or
    ``client_factory``) is used for model calls; without one a real
    :class:`harness.client.OpenRouterClient` is created.
    """
    base = Path(base_dir)
    run_id = config.get("run_id", "run")
    track = config.get("track", "text")
    sampling = dict(config.get("sampling", {}) or {})
    prompt_version = config.get("prompt_version", _prompts.PROMPT_VERSION)
    concurrency = int(config.get("concurrency", 8) or 8)
    budget_cfg = config.get("budget", {}) or {}
    budget_usd = budget_cfg.get("usd")

    ds = config.get("dataset", {}) or {}
    if positions is None:
        positions = load_positions(base / ds["positions"]) if ds.get("positions") else []
    if rollouts is None:
        rollouts = load_rollouts(base / ds["rollouts"]) if ds.get("rollouts") else {}

    ds_hash = dataset_hash(positions)
    run_dir = base / "runs" / run_id
    cache = ResponseCache(run_dir / "cache")
    results_dir = base / config.get("results_dir", "results")

    owns_client = client is None
    if client is None:
        if client_factory is not None:
            client = client_factory()
        else:
            from harness.client import OpenRouterClient  # noqa: PLC0415

            client = OpenRouterClient()

    from harness import report as _report  # noqa: PLC0415

    out: list[ModelRunResult] = []
    try:
        for model in config.get("models", []):
            budget = BudgetGuard(float(budget_usd)) if budget_usd is not None else BudgetGuard(None)
            cost = CostTracker()
            mr = await run_model(
                client, positions, rollouts, model=model, track=track,
                sampling=sampling, prompt_version=prompt_version, cache=cache,
                budget=budget, cost=cost, image_provider=image_provider,
                concurrency=concurrency, raw_dir=run_dir / "raw",
                dataset_hash_str=ds_hash,
            )
            out.append(mr)
            _write_scores(run_dir, mr)
            if write_results:
                results_dir.mkdir(parents=True, exist_ok=True)
                obj = _report.build_results(mr, run_id=run_id)
                path = results_dir / f"{run_id}__{_slug(model)}__{track}.json"
                path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        if owns_client and hasattr(client, "aclose"):
            maybe = client.aclose()
            if asyncio.iscoroutine(maybe):
                await maybe
    return out


def _write_scores(run_dir: Path, mr: ModelRunResult) -> None:
    d = run_dir / "scores"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_slug(mr.model)}__{mr.track}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in mr.decisions:
            fh.write(json.dumps(_record_to_dict(rec), sort_keys=True) + "\n")


def _record_to_dict(rec: DecisionRecord) -> dict[str, Any]:
    return {
        "position_id": rec.position_id,
        "track": rec.track,
        "model": rec.model,
        "decision_type": rec.decision_type,
        "tier": rec.tier,
        "chosen": rec.chosen,
        "raw_answer": rec.raw_answer,
        "is_best": rec.is_best,
        "equity_loss": rec.equity_loss,
        "equity_loss_mp": rec.equity_loss_mp,
        "parse_failed": rec.parse_failed,
        "attempts": rec.attempts,
        "rollout_move": rec.rollout_move,
        "usage": rec.usage,
        "detail": rec.detail,
        "votes": rec.votes,
    }
