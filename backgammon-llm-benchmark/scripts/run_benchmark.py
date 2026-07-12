"""Execute a benchmark run from a run config (PLAN.md §5.2).

Loads ``runs/<run>.yaml`` and drives the harness (client/runner/scoring/report)
to evaluate a model slate on a track, writing results/*.json + a manifest.
"""
