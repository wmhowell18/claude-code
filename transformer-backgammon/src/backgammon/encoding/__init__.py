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

from backgammon.encoding.action_encoder import (
    # Action encoding
    encode_move_to_action,
    create_action_mask,
    create_move_to_action_map,
    encode_move_to_one_hot,
    create_uniform_policy,
    select_move_from_policy,
    get_action_space_size,
    analyze_move_collisions,

    # Alternative encodings
    encode_move_structured,
    decode_move_structured,
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

    # Action encoding
    "encode_move_to_action",
    "create_action_mask",
    "create_move_to_action_map",
    "encode_move_to_one_hot",
    "create_uniform_policy",
    "select_move_from_policy",
    "get_action_space_size",
    "analyze_move_collisions",
    "encode_move_structured",
    "decode_move_structured",
]
