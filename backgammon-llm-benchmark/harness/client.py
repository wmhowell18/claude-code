"""Async OpenRouter client (PLAN.md §5.1-5.2).

Thin httpx-based wrapper over OpenRouter's OpenAI-compatible endpoint with
retries/backoff. Captures per-request ``usage`` (prompt/completion/reasoning
tokens + reported dollar cost), which feeds the cost axis and budget track.
"""
