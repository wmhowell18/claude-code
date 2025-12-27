"""Board encoding for neural networks."""

from backgammon.encoding.encoder import (
    # Configuration presets
    raw_encoding_config,
    minimal_encoding_config,
    standard_encoding_config,
    rich_encoding_config,

    # Main encoding functions
    encode_board,
    encode_boards,
    decode_board,
    extract_position_features,
    feature_dimension,

    # Encoding variants
    encode_raw,
    encode_one_hot,
    encode_geometric,
    encode_strategic,
    encode_full,

    # Dice encoding
    encode_dice,
    dice_to_embedding_id,

    # Equity encoding
    outcome_to_equity,
    equity_to_array,
    array_to_equity,

    # Preprocessing
    normalize_features,
    canonicalize_board,
    augment_position,

    # Batch utilities
    stack_encoded_boards,
)

__all__ = [
    # Configuration presets
    "raw_encoding_config",
    "minimal_encoding_config",
    "standard_encoding_config",
    "rich_encoding_config",

    # Main encoding functions
    "encode_board",
    "encode_boards",
    "decode_board",
    "extract_position_features",
    "feature_dimension",

    # Encoding variants
    "encode_raw",
    "encode_one_hot",
    "encode_geometric",
    "encode_strategic",
    "encode_full",

    # Dice encoding
    "encode_dice",
    "dice_to_embedding_id",

    # Equity encoding
    "outcome_to_equity",
    "equity_to_array",
    "array_to_equity",

    # Preprocessing
    "normalize_features",
    "canonicalize_board",
    "augment_position",

    # Batch utilities
    "stack_encoded_boards",
]
