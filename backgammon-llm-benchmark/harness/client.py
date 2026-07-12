"""Async OpenRouter client (PLAN.md §5.1-5.2).

Thin httpx-based wrapper over OpenRouter's OpenAI-compatible endpoint with
retries/backoff. Captures per-request ``usage`` (prompt/completion/reasoning
tokens + reported dollar cost), which feeds the cost axis and budget track.

The client speaks the ``/chat/completions`` shape: a list of messages where each
message's ``content`` is either a plain string (text) or a list of *parts*
(``{"type": "text", ...}`` / ``{"type": "image_url", ...}``) so text, image, and
text+image tracks all go through one code path. It asks OpenRouter for usage
accounting (``"usage": {"include": true}``) so every result carries token counts
and the authoritative dollar cost.

Nothing here performs real network I/O at import time; tests drive it with an
``httpx.MockTransport`` (pass ``transport=...``) and no API key is required for
those. A real run reads ``OPENROUTER_API_KEY`` from the environment.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Awaitable, Callable

import httpx

__all__ = [
    "OPENROUTER_BASE_URL",
    "ChatResult",
    "ClientError",
    "AuthError",
    "RateLimitError",
    "ServerError",
    "APIConnectionError",
    "BadRequestError",
    "BadResponseError",
    "image_data_url",
    "image_part",
    "text_part",
    "user_message",
    "OpenRouterClient",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# -- error types ----------------------------------------------------------


class ClientError(Exception):
    """Base class for all client errors."""


class AuthError(ClientError):
    """401/403 — missing or rejected API key. Not retried."""


class RateLimitError(ClientError):
    """429 — rate limited beyond the retry budget."""


class ServerError(ClientError):
    """5xx — upstream error beyond the retry budget."""


class APIConnectionError(ClientError):
    """Network / timeout error beyond the retry budget."""


class BadRequestError(ClientError):
    """4xx (other than auth) — malformed request; not retried."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class BadResponseError(ClientError):
    """200 but the body was not a parseable chat completion."""


# -- result type ----------------------------------------------------------


@dataclass
class ChatResult:
    """A single chat-completion result with usage/cost accounting.

    ``cost_usd`` is OpenRouter's authoritative reported cost for the request
    (``None`` if the provider did not report one). ``latency_s`` is wall-clock
    time for the (final, successful) HTTP request. ``raw`` keeps the decoded JSON
    for debugging / re-parsing.
    """

    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    latency_s: float
    raw_id: str | None
    model: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResult":
        fields = {
            "text",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cost_usd",
            "latency_s",
            "raw_id",
            "model",
            "raw",
        }
        return cls(**{k: data.get(k) for k in fields})


# -- message helpers ------------------------------------------------------


def image_data_url(png_bytes: bytes, *, mime: str = "image/png") -> str:
    """Encode raw image bytes as a base64 ``data:`` URL for an image part."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_part(png_bytes: bytes, *, mime: str = "image/png") -> dict[str, Any]:
    """Build an OpenAI/OpenRouter ``image_url`` content part from bytes."""
    return {"type": "image_url", "image_url": {"url": image_data_url(png_bytes, mime=mime)}}


def text_part(text: str) -> dict[str, Any]:
    """Build a ``text`` content part."""
    return {"type": "text", "text": text}


def user_message(text: str | None = None, images: list[bytes] | None = None) -> dict[str, Any]:
    """Build a user message; plain string content for text-only, else a parts list."""
    if not images:
        return {"role": "user", "content": text or ""}
    parts: list[dict[str, Any]] = []
    if text:
        parts.append(text_part(text))
    for img in images:
        parts.append(image_part(img))
    return {"role": "user", "content": parts}


# -- client ---------------------------------------------------------------

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class OpenRouterClient:
    """Async OpenRouter chat-completions client (httpx).

    Parameters
    ----------
    api_key:
        Bearer token; falls back to ``OPENROUTER_API_KEY``. May be omitted when a
        mock ``transport`` is supplied (tests).
    transport:
        Optional ``httpx`` transport (e.g. ``httpx.MockTransport``) for tests.
    max_retries:
        Number of *retries* (in addition to the first attempt) on 429/5xx/timeout.
    backoff_base / backoff_cap:
        Exponential backoff: ``min(cap, base * 2**attempt)`` seconds, unless the
        response carries a ``Retry-After`` header.
    sleep:
        Injectable async sleep (tests pass a no-op to run fast).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        referer: str | None = None,
        title: str | None = "backgammon-llm-benchmark",
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep or asyncio.sleep
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter attribution headers (optional but recommended).
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- request building -------------------------------------------------

    @staticmethod
    def build_payload(
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        usage: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if usage:
            payload["usage"] = {"include": True}
        if extra_body:
            payload.update(extra_body)
        return payload

    # -- backoff ----------------------------------------------------------

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        return min(self.backoff_cap, self.backoff_base * (2 ** attempt))

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        val = resp.headers.get("retry-after")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    # -- main call --------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        usage: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        """POST one chat completion, retrying transient failures, and return a
        typed :class:`ChatResult`. Raises a :class:`ClientError` subclass when the
        retry budget is exhausted or the request is rejected."""
        payload = self.build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            usage=usage,
            extra_body=extra_body,
        )
        attempt = 0
        while True:
            t0 = perf_counter()
            try:
                resp = await self._client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.max_retries:
                    raise APIConnectionError(f"connection failed: {exc!r}") from exc
                await self._sleep(self._delay(attempt, None))
                attempt += 1
                continue

            status = resp.status_code
            if status == 200:
                return self._parse_response(resp, model=model, latency=perf_counter() - t0)
            if status in (401, 403):
                raise AuthError(f"HTTP {status}: {resp.text[:300]}")
            if status in _RETRYABLE_STATUS:
                if attempt >= self.max_retries:
                    if status == 429:
                        raise RateLimitError(f"rate limited after {attempt + 1} tries")
                    raise ServerError(f"HTTP {status} after {attempt + 1} tries")
                await self._sleep(self._delay(attempt, self._retry_after(resp)))
                attempt += 1
                continue
            # Any other 4xx: not retryable.
            raise BadRequestError(status, resp.text)

    # -- response parsing -------------------------------------------------

    @staticmethod
    def _parse_response(resp: httpx.Response, *, model: str, latency: float) -> ChatResult:
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise BadResponseError(f"non-JSON body: {exc!r}") from exc
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise BadResponseError(f"missing choices: {data!r}"[:300]) from exc

        content = message.get("content")
        if isinstance(content, list):
            # Some providers return parts; concatenate any text parts.
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = content or ""

        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens", usage.get("reasoning_tokens", 0)) or 0
        cost = usage.get("cost")
        cost_usd = float(cost) if cost is not None else None

        return ChatResult(
            text=text,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            reasoning_tokens=int(reasoning or 0),
            cost_usd=cost_usd,
            latency_s=latency,
            raw_id=data.get("id"),
            model=data.get("model", model),
            raw=data,
        )
