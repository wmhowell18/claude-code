"""Shared JIT compilation cache for neural network inference.

Provides a single cache so that the same compiled inference function
is reused across training (self_play) and evaluation (network_agent, search).
"""

import jax

# Module-level cache keyed by apply_fn identity
_jit_inference_cache = {}


def get_jit_inference(apply_fn):
    """Get or create a JIT-compiled inference function.

    Caches the JIT-compiled function by apply_fn identity so it's
    only compiled once per model. The compiled function takes (params, x)
    and returns the raw tuple from the model's forward pass.

    Args:
        apply_fn: The model's apply function (e.g., state.apply_fn).

    Returns:
        JIT-compiled function: (params, x) -> (equity, policy, cube, attn).
    """
    fn_id = id(apply_fn)
    if fn_id not in _jit_inference_cache:
        @jax.jit
        def _jit_infer(params, x):
            return apply_fn({'params': params}, x, training=False)
        _jit_inference_cache[fn_id] = _jit_infer
    return _jit_inference_cache[fn_id]
