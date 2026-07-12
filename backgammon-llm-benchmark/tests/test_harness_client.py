"""Tests: async OpenRouter client on a mocked httpx transport (PLAN.md §5.1-5.2)."""

import asyncio
import base64
import json

import httpx
import pytest

from harness.client import (
    AuthError,
    BadRequestError,
    OpenRouterClient,
    RateLimitError,
    image_data_url,
    image_part,
    user_message,
)


def _run(coro):
    return asyncio.run(coro)


def _ok_response(request, content="MOVE: 8/5 6/5", cost=0.0012, reasoning=3):
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "gen-123",
            "model": body["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 7,
                "cost": cost,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
        },
    )


def test_chat_success_captures_usage_and_cost():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("Authorization")
        return _ok_response(request)

    client = OpenRouterClient(api_key="secret", transport=httpx.MockTransport(handler))
    res = _run(client.chat([{"role": "user", "content": "hi"}], model="a/b",
                           temperature=0, max_tokens=64, seed=7))
    _run(client.aclose())

    assert res.text == "MOVE: 8/5 6/5"
    assert res.finish_reason == "stop"
    assert res.prompt_tokens == 42 and res.completion_tokens == 7
    assert res.reasoning_tokens == 3
    assert res.cost_usd == pytest.approx(0.0012)
    assert res.raw_id == "gen-123"
    # request wiring: usage include + passthrough params + auth header
    assert seen["body"]["usage"] == {"include": True}
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["max_tokens"] == 64
    assert seen["body"]["seed"] == 7
    assert seen["auth"] == "Bearer secret"


def test_retry_after_429_then_success():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return _ok_response(request)

    async def noop(_):  # fast, deterministic backoff
        return None

    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler),
                              backoff_base=0.0, sleep=noop)
    res = _run(client.chat([{"role": "user", "content": "hi"}], model="a/b"))
    _run(client.aclose())
    assert calls["n"] == 2
    assert res.text.startswith("MOVE:")


def test_429_exhausts_retries_raises_ratelimit():
    def handler(request):
        return httpx.Response(429, json={"error": "nope"})

    async def noop(_):
        return None

    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler),
                              max_retries=2, backoff_base=0.0, sleep=noop)
    with pytest.raises(RateLimitError):
        _run(client.chat([{"role": "user", "content": "hi"}], model="a/b"))
    _run(client.aclose())


def test_auth_error_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(AuthError):
        _run(client.chat([{"role": "user", "content": "hi"}], model="a/b"))
    _run(client.aclose())
    assert calls["n"] == 1


def test_bad_request_not_retried():
    def handler(request):
        return httpx.Response(400, text="bad model")

    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(BadRequestError):
        _run(client.chat([{"role": "user", "content": "hi"}], model="a/b"))
    _run(client.aclose())


def test_image_message_payload_shape():
    png = b"\x89PNG\r\n\x1a\nFAKEDATA"
    part = image_part(png)
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == png

    msg = user_message("look at this", images=[png])
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    kinds = [p["type"] for p in msg["content"]]
    assert kinds == ["text", "image_url"]


def test_client_sends_image_parts_over_the_wire():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok_response(request)

    png = b"PNGBYTES"
    client = OpenRouterClient(api_key="k", transport=httpx.MockTransport(handler))
    msgs = [user_message("what play?", images=[png])]
    _run(client.chat(msgs, model="a/b"))
    _run(client.aclose())
    content = seen["body"]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == image_data_url(png)
