"""Content-addressed response cache (PLAN.md §5.2).

Keyed by (model, prompt_version, position_id, sampling_params). Makes runs
idempotent/resumable and ensures we never pay twice for the same model call.
"""
