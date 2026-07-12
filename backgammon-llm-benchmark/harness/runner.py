"""Async batch runner (PLAN.md §5.2).

Task pool over (model x position x track) with concurrency caps. Resumable via
the cache so interrupted runs continue without repaying. Reads a run config
from ``runs/<run>.yaml`` and writes a reproducibility manifest per run.
"""
