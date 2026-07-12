"""End-to-end dataset build (PLAN.md §7).

Orchestrates the pipeline: generate self-play -> sample -> dedup/blocklist ->
rollout ground truth -> tier -> emit position records + manifests.
"""
