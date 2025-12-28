"""Board encoding for neural network input.

This module implements various encoding strategies for converting board states
into neural network input features. Following the philosophy of letting the
transformer learn representations, we provide both minimal (raw) encodings
and richer encodings with hand-crafted features.
"""

from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from backgammon.core.types import (
    Board,
    Player,
    Point,
    Dice,
    Equity,
    GameOutcome,
    EncodedBoard,
    EncodingConfig,
)
from backgammon.core.dice import ALL_DICE_ROLLS


# ==============================================================================
# ENCODING CONFIGURATION PRESETS
# ==============================================================================


def raw_encoding_config() -> EncodingConfig:
    """Pure raw encoding - most general, let transformer learn everything.

    Returns just [our_checkers, opp_checkers] for each position.
    This is the recommended starting point.

    Returns:
        EncodingConfig with feature_dim=2
    """
    return EncodingConfig(
        use_one_hot_counts=False,
        include_geometric_features=False,
        include_strategic_features=False,
        include_dice_encoding=False,
        feature_dim=2,
    )


def minimal_encoding_config() -> EncodingConfig:
    """Minimal encoding similar to TD-Gammon.

    Raw counts plus basic geometric features (distance to home).

    Returns:
        EncodingConfig with feature_dim=5
    """
    return EncodingConfig(
        use_one_hot_counts=False,
        include_geometric_features=True,
        include_strategic_features=False,
        include_dice_encoding=False,
        feature_dim=5,
    )


def standard_encoding_config() -> EncodingConfig:
    """Standard encoding with one-hot counts and geometric features.

    More expressive but still relatively general.
    Similar to Jacob Hilton's approach.

    Returns:
        EncodingConfig with feature_dim=35 (32 one-hot + 3 geometric)
    """
    return EncodingConfig(
        use_one_hot_counts=True,
        include_geometric_features=True,
        include_strategic_features=False,
        include_dice_encoding=False,
        feature_dim=35,
    )


def rich_encoding_config() -> EncodingConfig:
    """Rich encoding with all features including strategic ones.

    Most hand-crafted features but potentially most sample-efficient.

    Returns:
        EncodingConfig with feature_dim=45 (32 one-hot + 3 geometric + 10 strategic)
    """
    return EncodingConfig(
        use_one_hot_counts=True,
        include_geometric_features=True,
        include_strategic_features=True,
        include_dice_encoding=False,
        feature_dim=45,
    )


# ==============================================================================
# FEATURE DIMENSION CALCULATION
# ==============================================================================


def feature_dimension(config: EncodingConfig) -> int:
    """Calculate feature dimension for a given encoding config.

    Args:
        config: Encoding configuration

    Returns:
        Number of features per position
    """
    dim = 0

    # Base encoding: checker counts
    if config.use_one_hot_counts:
        dim += 32  # 16 for our checkers + 16 for opponent
    else:
        dim += 2  # our_checkers, opp_checkers

    # Geometric features
    if config.include_geometric_features:
        dim += 3  # distance_to_home, is_home_board, is_opp_home

    # Strategic features
    if config.include_strategic_features:
        dim += 10  # Various strategic indicators

    return dim


# ==============================================================================
# RAW ENCODING
# ==============================================================================


def encode_raw(board: Board, point: Point) -> NDArray[np.float32]:
    """Encode a single position with just checker counts.

    This is the most general encoding - let the transformer learn everything.

    Args:
        board: Current board state
        point: Which position to encode (0=bar, 1-24=points, 25=off)

    Returns:
        Array of shape (2,) containing [our_checkers, opp_checkers]
    """
    player = board.player_to_move

    our_checkers = board.get_checkers(player, point)
    opp_checkers = board.get_checkers(player.opponent(), point)

    # Normalize to [0, 1] range (max 15 checkers on a point)
    return np.array([our_checkers / 15.0, opp_checkers / 15.0], dtype=np.float32)


# ==============================================================================
# ONE-HOT ENCODING
# ==============================================================================


def encode_one_hot(board: Board, point: Point) -> NDArray[np.float32]:
    """Encode a single position with one-hot checker counts.

    More expressive than raw encoding but still general.

    Args:
        board: Current board state
        point: Which position to encode

    Returns:
        Array of shape (32,) containing one-hot encodings
        [our_checkers_onehot (16), opp_checkers_onehot (16)]
    """
    player = board.player_to_move

    our_checkers = board.get_checkers(player, point)
    opp_checkers = board.get_checkers(player.opponent(), point)

    # One-hot encode counts (0-15)
    our_onehot = np.zeros(16, dtype=np.float32)
    opp_onehot = np.zeros(16, dtype=np.float32)

    our_onehot[min(our_checkers, 15)] = 1.0
    opp_onehot[min(opp_checkers, 15)] = 1.0

    return np.concatenate([our_onehot, opp_onehot])


# ==============================================================================
# GEOMETRIC ENCODING
# ==============================================================================


def encode_geometric(board: Board, point: Point) -> NDArray[np.float32]:
    """Encode position with raw features plus geometric information.

    Adds positional context that might help the network learn faster.

    Args:
        board: Current board state
        point: Which position to encode

    Returns:
        Array of shape (5,) containing:
        [our_checkers, opp_checkers, distance_to_home, is_home_board, is_opp_home]
    """
    player = board.player_to_move

    # Start with raw encoding
    raw = encode_raw(board, point)

    # Calculate geometric features
    if point == 0:  # Bar
        distance_to_home = 1.0  # Normalized: furthest from home
        is_home_board = 0.0
        is_opp_home = 0.0
    elif point == 25:  # Off
        distance_to_home = 0.0  # Home!
        is_home_board = 0.0
        is_opp_home = 0.0
    else:
        # For White: home is 1-6, opponent home is 19-24
        # For Black: home is 19-24, opponent home is 1-6
        if player == Player.WHITE:
            distance_to_home = point / 24.0  # Normalize to [0, 1]
            is_home_board = 1.0 if point <= 6 else 0.0
            is_opp_home = 1.0 if point >= 19 else 0.0
        else:  # Black
            distance_to_home = (25 - point) / 24.0
            is_home_board = 1.0 if point >= 19 else 0.0
            is_opp_home = 1.0 if point <= 6 else 0.0

    return np.concatenate([raw, [distance_to_home, is_home_board, is_opp_home]])


# ==============================================================================
# STRATEGIC ENCODING
# ==============================================================================


def encode_strategic(board: Board, point: Point) -> NDArray[np.float32]:
    """Encode position with geometric features plus strategic indicators.

    Adds hand-crafted strategic features. Most informative but least general.

    Args:
        board: Current board state
        point: Which position to encode

    Returns:
        Array of shape (~15,) containing geometric + strategic features
    """
    player = board.player_to_move

    # Start with geometric encoding
    geometric = encode_geometric(board, point)

    # Calculate strategic features
    our_checkers = board.get_checkers(player, point)
    opp_checkers = board.get_checkers(player.opponent(), point)

    # Is this a blot? (single checker vulnerable to hit)
    is_blot = 1.0 if our_checkers == 1 else 0.0

    # Is this an anchor? (2+ checkers in opponent's home board)
    if player == Player.WHITE:
        is_anchor = 1.0 if our_checkers >= 2 and point >= 19 else 0.0
    else:
        is_anchor = 1.0 if our_checkers >= 2 and point <= 6 else 0.0

    # Is this part of a prime? (consecutive made points)
    is_prime = 0.0
    if our_checkers >= 2 and 1 <= point <= 24:
        # Check if adjacent points are also made
        adjacent_made = 0
        for adj in [point - 1, point + 1]:
            if 1 <= adj <= 24 and board.get_checkers(player, adj) >= 2:
                adjacent_made += 1
        is_prime = adjacent_made / 2.0

    # Opponent blot here? (hitting opportunity)
    opp_blot = 1.0 if opp_checkers == 1 else 0.0

    # Made point? (2+ checkers, safe)
    is_made = 1.0 if our_checkers >= 2 else 0.0

    # Advanced anchor? (made point in opponent territory)
    is_advanced_anchor = 0.0
    if our_checkers >= 2:
        if player == Player.WHITE and point >= 13:
            is_advanced_anchor = 1.0
        elif player == Player.BLACK and point <= 12:
            is_advanced_anchor = 1.0

    # Stack height (normalized)
    stack_height = our_checkers / 15.0

    # Opponent stack height
    opp_stack_height = opp_checkers / 15.0

    # Is this the bar?
    is_bar = 1.0 if point == 0 else 0.0

    # Is this off?
    is_off = 1.0 if point == 25 else 0.0

    strategic_features = np.array([
        is_blot,
        is_anchor,
        is_prime,
        opp_blot,
        is_made,
        is_advanced_anchor,
        stack_height,
        opp_stack_height,
        is_bar,
        is_off,
    ], dtype=np.float32)

    return np.concatenate([geometric, strategic_features])


# ==============================================================================
# FULL ENCODING (CONFIGURABLE)
# ==============================================================================


def encode_full(config: EncodingConfig, board: Board, point: Point) -> NDArray[np.float32]:
    """Encode position with all requested features based on config.

    Args:
        config: Encoding configuration
        board: Current board state
        point: Which position to encode

    Returns:
        Feature array of dimension specified by config.feature_dim
    """
    # Start with base encoding
    if config.use_one_hot_counts:
        features = encode_one_hot(board, point)
    else:
        features = encode_raw(board, point)

    # Add geometric features if requested
    if config.include_geometric_features:
        geometric = encode_geometric(board, point)
        # Extract just the geometric part (last 3 elements)
        features = np.concatenate([features, geometric[-3:]])

    # Add strategic features if requested
    if config.include_strategic_features:
        strategic = encode_strategic(board, point)
        # Extract just the strategic part (last 10 elements)
        features = np.concatenate([features, strategic[-10:]])

    return features


# ==============================================================================
# POSITION FEATURE EXTRACTION
# ==============================================================================


def extract_position_features(
    config: EncodingConfig,
    board: Board,
    point: Point
) -> NDArray[np.float32]:
    """Extract features for a single position.

    This is the main entry point for encoding a single position.

    Args:
        config: Encoding configuration
        board: Current board state
        point: Which position to encode (0-25)

    Returns:
        Feature array of dimension config.feature_dim
    """
    return encode_full(config, board, point)


# ==============================================================================
# BOARD ENCODING
# ==============================================================================


def encode_board(config: EncodingConfig, board: Board) -> EncodedBoard:
    """Encode a single board state into neural network input format.

    Args:
        config: Encoding configuration
        board: Board state to encode

    Returns:
        EncodedBoard with batch_size=1
    """
    # Extract features for all 26 positions
    position_features = np.zeros((1, 26, config.feature_dim), dtype=np.float32)

    for point in range(26):
        position_features[0, point] = extract_position_features(config, board, point)

    return EncodedBoard(
        position_features=position_features,
        dice_features=None,
        batch_size=1,
        sequence_length=26,
        feature_dim=config.feature_dim,
    )


def encode_boards(config: EncodingConfig, boards: List[Board]) -> EncodedBoard:
    """Encode a batch of boards.

    Args:
        config: Encoding configuration
        boards: List of board states to encode

    Returns:
        EncodedBoard with batch_size=len(boards)
    """
    batch_size = len(boards)
    position_features = np.zeros((batch_size, 26, config.feature_dim), dtype=np.float32)

    for i, board in enumerate(boards):
        for point in range(26):
            position_features[i, point] = extract_position_features(config, board, point)

    return EncodedBoard(
        position_features=position_features,
        dice_features=None,
        batch_size=batch_size,
        sequence_length=26,
        feature_dim=config.feature_dim,
    )


def decode_board(config: EncodingConfig, encoded: EncodedBoard) -> Optional[Board]:
    """Decode board from network representation (for debugging).

    Note: This only works for raw encoding, not one-hot or other variants.

    Args:
        config: Encoding configuration used for encoding
        encoded: Encoded board representation

    Returns:
        Decoded board if possible, None if encoding is not invertible
    """
    if config.use_one_hot_counts or config.include_strategic_features:
        # Can't decode from one-hot or strategic features
        return None

    # Only works for raw encoding
    if encoded.batch_size != 1:
        return None

    from backgammon.core.board import empty_board

    board = empty_board()

    # Decode each position
    for point in range(26):
        features = encoded.position_features[0, point]

        # De-normalize checker counts (first 2 features)
        our_checkers = int(round(features[0] * 15.0))
        opp_checkers = int(round(features[1] * 15.0))

        board.set_checkers(board.player_to_move, point, our_checkers)
        board.set_checkers(board.player_to_move.opponent(), point, opp_checkers)

    return board


# ==============================================================================
# DICE ENCODING
# ==============================================================================


def encode_dice(dice: Dice) -> NDArray[np.float32]:
    """Encode dice roll as features.

    Args:
        dice: Dice roll (die1, die2)

    Returns:
        One-hot encoding of shape (12,): [die1_onehot (6), die2_onehot (6)]
    """
    die1_onehot = np.zeros(6, dtype=np.float32)
    die2_onehot = np.zeros(6, dtype=np.float32)

    die1_onehot[dice[0] - 1] = 1.0
    die2_onehot[dice[1] - 1] = 1.0

    return np.concatenate([die1_onehot, die2_onehot])


def dice_to_embedding_id(dice: Dice) -> int:
    """Convert dice to embedding ID (0-20).

    Maps each of the 21 unique dice outcomes to a unique integer.

    Args:
        dice: Dice roll (die1, die2)

    Returns:
        Integer in range [0, 20]
    """
    # Canonicalize dice first (ensure die1 <= die2)
    canonical = tuple(sorted(dice))

    try:
        return ALL_DICE_ROLLS.index(canonical)
    except ValueError:
        raise ValueError(f"Invalid dice roll: {dice}")


# ==============================================================================
# EQUITY ENCODING
# ==============================================================================


def outcome_to_equity(outcome: GameOutcome, player: Player) -> Equity:
    """Convert game outcome to equity from a player's perspective.

    Args:
        outcome: Game outcome
        player: Which player's perspective

    Returns:
        Equity representing the outcome as probabilities
    """
    # Initialize all probabilities to 0
    equity = Equity(
        win_normal=0.0,
        win_gammon=0.0,
        win_backgammon=0.0,
        lose_gammon=0.0,
        lose_backgammon=0.0,
    )

    # Check if this player won
    did_win = outcome.winner == player

    if did_win:
        # We won - set appropriate win probability to 1.0
        if outcome.points == 1:
            equity.win_normal = 1.0
        elif outcome.points == 2:
            equity.win_gammon = 1.0
        else:  # points == 3
            equity.win_backgammon = 1.0
    else:
        # We lost - set appropriate loss probability to 1.0
        if outcome.points == 1:
            # Normal loss (lose_normal is implicit)
            pass  # All zeros is correct
        elif outcome.points == 2:
            equity.lose_gammon = 1.0
        else:  # points == 3
            equity.lose_backgammon = 1.0

    return equity


def equity_to_array(equity: Equity) -> NDArray[np.float32]:
    """Encode equity as array for training.

    Args:
        equity: Equity probabilities

    Returns:
        Array of shape (5,) containing equity components
    """
    return equity.to_array()


def array_to_equity(arr: NDArray[np.float32]) -> Equity:
    """Decode equity from array.

    Args:
        arr: Array of shape (5,) containing equity components

    Returns:
        Equity object
    """
    return Equity.from_array(arr)


# ==============================================================================
# NORMALIZATION AND PREPROCESSING
# ==============================================================================


def normalize_features(features: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize features to mean=0, std=1.

    Args:
        features: Array of shape (..., feature_dim)

    Returns:
        Normalized features of same shape
    """
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)

    # Avoid division by zero
    std = np.where(std < 1e-6, 1.0, std)

    return (features - mean) / std


def canonicalize_board(board: Board) -> Board:
    """Standardize board representation.

    Always represent from perspective of player to move.
    This is already handled by Board.player_to_move, so this is a no-op.

    Args:
        board: Board state

    Returns:
        Canonicalized board (same as input)
    """
    return board


def augment_position(board: Board) -> List[Board]:
    """Create equivalent positions through data augmentation.

    For backgammon, the main augmentation is color flipping.

    Args:
        board: Original board state

    Returns:
        List of equivalent boards (including original)
    """
    from backgammon.core.board import flip_board

    # Original + flipped (swap colors and mirror)
    return [board, flip_board(board)]


# ==============================================================================
# BATCH UTILITIES
# ==============================================================================


def stack_encoded_boards(boards: List[EncodedBoard]) -> EncodedBoard:
    """Stack multiple encoded boards into a single batch.

    Args:
        boards: List of encoded boards (each with batch_size=1)

    Returns:
        Single EncodedBoard with combined batch
    """
    if not boards:
        raise ValueError("Cannot stack empty list of boards")

    # Check all have same feature_dim
    feature_dim = boards[0].feature_dim
    for b in boards:
        if b.feature_dim != feature_dim:
            raise ValueError("All boards must have same feature_dim")

    # Stack position features
    position_features = np.concatenate(
        [b.position_features for b in boards],
        axis=0
    )

    # Stack dice features if present
    dice_features = None
    if boards[0].dice_features is not None:
        dice_features = np.concatenate(
            [b.dice_features for b in boards if b.dice_features is not None],
            axis=0
        )

    return EncodedBoard(
        position_features=position_features,
        dice_features=dice_features,
        batch_size=len(boards),
        sequence_length=26,
        feature_dim=feature_dim,
    )
