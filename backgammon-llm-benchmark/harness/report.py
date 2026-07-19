"""Results emitter (PLAN.md §5.2, §6).

Turns a completed model run (:class:`harness.runner.ModelRunResult`) into the
results JSON consumed by the static site, conforming to
``schema/results.schema.json``: aggregate metrics (BenchPR + CI, best-move
accuracy, per-tier/track/decision-type roll-ups), cost totals, budget-track
fields, and a reproducibility manifest (prompt/render versions, dataset hash,
timestamp). It also validates the emitted object against the schema with a small
built-in structural checker, so no ``jsonschema`` dependency is required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness import scoring as _scoring

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from harness.runner import DecisionRecord, ModelRunResult

__all__ = [
    "BENCHPR_LABEL",
    "build_results",
    "aggregate_from_records",
    "validate_results",
    "load_results_schema",
    "write_results",
]

# Until validated against >=3 XG-analyzed games (PLAN §4.4, docs/SCORING.md).
BENCHPR_LABEL = "BenchPR (PR-calibrated)"

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "results.schema.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def aggregate_from_records(records: list["DecisionRecord"]) -> dict[str, Any]:
    """Aggregate per PLAN §4.4 from decision records."""
    return _scoring.aggregate(rec.row() for rec in records)


def _decision_json(rec: "DecisionRecord") -> dict[str, Any]:
    return {
        "position_id": rec.position_id,
        "chosen": rec.chosen if rec.chosen is not None else "",
        "raw_answer": rec.raw_answer,
        "is_best": rec.is_best,
        "equity_loss": rec.equity_loss,
        "parse_failed": rec.parse_failed,
        "usage": {
            "prompt_tokens": int(rec.usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(rec.usage.get("completion_tokens", 0) or 0),
            "reasoning_tokens": int(rec.usage.get("reasoning_tokens", 0) or 0),
            "cost_usd": float(rec.usage.get("cost_usd", 0.0) or 0.0),
        },
    }


def build_results(
    run: "ModelRunResult",
    *,
    run_id: str | None = None,
    include_decisions: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a results object conforming to ``schema/results.schema.json``.

    ``include_decisions=False`` yields an aggregate-only file (held-out split; the
    ``decisions`` array is emitted empty as the schema requires the key).
    """
    agg = aggregate_from_records(run.decisions)
    rid = run_id or f"{run.model}:{run.track}"

    aggregate: dict[str, Any] = {
        "benchpr": agg["benchpr"],
        "benchpr_label": BENCHPR_LABEL,
        "benchpr_ci95": agg["benchpr_ci95"],
        "best_move_accuracy": agg["best_move_accuracy"],
        "mean_equity_loss": agg["mean_equity_loss"],
        "cost_usd": run.cost_usd,
        "n": agg["n"],
        "parse_failures": agg["parse_failures"],
        "per_tier": agg["per_tier"],
        "per_track": agg["per_track"],
        "per_decision_type": agg["per_decision_type"],
    }
    if run.budget_usd is not None:
        # Budget track: BenchPR achieved within budget is the run's BenchPR
        # (unanswered positions were already scored worst-case by the runner).
        aggregate["benchpr_at_budget"] = agg["benchpr"]

    sampling = run.sampling or {}
    manifest: dict[str, Any] = {
        "dataset_hash": run.dataset_hash,
        "prompt_version": run.prompt_version,
        "timestamp": timestamp or _now_iso(),
        "sampling": {
            "temperature": sampling.get("temperature", 0),
            "n": int(sampling.get("n", 1) or 1),
            "mode": sampling.get("mode", "greedy"),
        },
    }
    if run.ascii_render_version:
        manifest["ascii_render_version"] = run.ascii_render_version
    if run.image_render_version:
        manifest["image_render_version"] = run.image_render_version
    if run.budget_usd is not None:
        manifest["budget_usd"] = run.budget_usd
    if run.quality_gate is not None:
        # Additive record of the eligibility gate (PLAN §3.2): which positions the
        # runner dropped and why, so results stay auditable/reproducible.
        manifest["quality_gate"] = run.quality_gate

    decisions = [_decision_json(r) for r in run.decisions] if include_decisions else []

    return {
        "run_id": rid,
        "model": run.model,
        "track": run.track,
        "manifest": manifest,
        "aggregate": aggregate,
        "decisions": decisions,
    }


def write_results(
    run: "ModelRunResult",
    out_dir: str | Path,
    *,
    run_id: str | None = None,
    include_decisions: bool = True,
) -> Path:
    """Serialize results JSON to ``out_dir`` and return the path."""
    obj = build_results(run, run_id=run_id, include_decisions=include_decisions)
    errs = validate_results(obj)
    if errs:
        raise ValueError("results failed schema validation: " + "; ".join(errs))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from harness.runner import _slug  # noqa: PLC0415

    path = out / f"{_slug(str(run_id or run.model))}__{run.track}.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ========================================================================
# Minimal JSON-Schema structural validator (no jsonschema dependency)
# ========================================================================


def load_results_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def _type_ok(value: Any, typ: str) -> bool:
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    py = _TYPE_MAP.get(typ)
    return isinstance(value, py) if py else True


def _validate(value: Any, schema: dict[str, Any], path: str, errs: list[str]) -> None:
    typ = schema.get("type")
    if typ and not _type_ok(value, typ):
        errs.append(f"{path or '<root>'}: expected {typ}, got {type(value).__name__}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if typ == "object" and isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{path}: missing required '{req}'")
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in value:
                _validate(value[k], sub, f"{path}.{k}" if path else k, errs)
        add = schema.get("additionalProperties")
        if isinstance(add, dict):
            for k, v in value.items():
                if k not in props:
                    _validate(v, add, f"{path}.{k}" if path else k, errs)
    if typ == "array" and isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: array shorter than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: array longer than maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]", errs)


def validate_against_schema(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of structural violations (empty = valid).

    Supports the subset of JSON Schema our contracts use: ``type``, ``required``,
    ``properties``, ``items``, ``enum``, ``additionalProperties`` (as a schema),
    ``minItems``/``maxItems``. Not a full validator — a dependency-free gate.
    """
    errs: list[str] = []
    _validate(instance, schema, "", errs)
    return errs


def validate_results(obj: Any) -> list[str]:
    """Validate a results object against ``schema/results.schema.json``."""
    return validate_against_schema(obj, load_results_schema())
