"""Transformer neural network architecture."""

from backgammon.network.network import (
    # Model classes
    BackgammonTransformer,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    ValueHead,
    PolicyHead,

    # Configuration presets
    small_transformer_config,
    medium_transformer_config,
    large_transformer_config,

    # Network utilities
    init_network,
    count_parameters,
    parameter_stats,

    # Forward pass
    forward,
    forward_batch,
)

__all__ = [
    # Model classes
    "BackgammonTransformer",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "ValueHead",
    "PolicyHead",

    # Configuration presets
    "small_transformer_config",
    "medium_transformer_config",
    "large_transformer_config",

    # Network utilities
    "init_network",
    "count_parameters",
    "parameter_stats",

    # Forward pass
    "forward",
    "forward_batch",
]
