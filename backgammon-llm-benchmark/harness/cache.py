"""Content-addressed response cache (PLAN.md §5.2).

Disk cache under ``runs/<run_id>/cache/``, one JSON file per key. The key is a
SHA-256 over the tuple that fully determines a model call —
``(model, prompt_version, position_id, track, sampling_params)`` — so a rerun
with the same inputs is free and scoring is reproducible offline. Cached values
are plain dicts (typically a :meth:`harness.client.ChatResult.to_dict`), so the
cache stays JSON-serialisable and engine-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

__all__ = ["cache_key", "ResponseCache"]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(
    *,
    model: str,
    prompt_version: str,
    position_id: str,
    track: str,
    sampling: dict[str, Any] | None = None,
    extra: Any = None,
) -> str:
    """SHA-256 hex digest of the call-identifying tuple."""
    payload = {
        "model": model,
        "prompt_version": prompt_version,
        "position_id": position_id,
        "track": track,
        "sampling": sampling or {},
        "extra": extra,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    """File-per-key JSON cache rooted at a directory (created on demand)."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        # Shard by the first two hex chars to keep directories small.
        return self.root / key[:2] / f"{key}.json"

    def key(self, **kw: Any) -> str:
        return cache_key(**kw)

    def has(self, key: str) -> bool:
        return self._path(key).is_file()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, sort_keys=True)
        os.replace(tmp, path)

    def get_or_none(self, **kw: Any) -> tuple[str, dict[str, Any] | None]:
        """Return ``(key, cached_value_or_None)`` for the identifying kwargs."""
        k = self.key(**kw)
        return k, self.get(k)
