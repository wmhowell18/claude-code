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

    # Loss functions
    equity_loss,
    mse_equity_loss,
    policy_loss,
    total_loss,

    # Training
    create_train_state,
    train_step,
    eval_step,
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

    # Loss functions
    "equity_loss",
    "mse_equity_loss",
    "policy_loss",
    "total_loss",

    # Training
    "create_train_state",
    "train_step",
    "eval_step",
]
