"""Tests for encoder module."""

import pytest
import numpy as np
from backgammon.core.board import initial_board, empty_board
from backgammon.core.types import Player, GameOutcome, Equity
from backgammon.encoding.encoder import (
    # Configuration presets
    raw_encoding_config,
    minimal_encoding_config,
    standard_encoding_config,
    rich_encoding_config,
    feature_dimension,

    # Encoding functions
    encode_raw,
    encode_one_hot,
    encode_geometric,
    encode_strategic,
    encode_full,
    extract_position_features,

    # Board encoding
    encode_board,
    encode_boards,
    decode_board,

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


class TestEncodingConfigs:
    """Tests for encoding configuration presets."""

    def test_raw_config(self):
        """Test raw encoding configuration."""
        config = raw_encoding_config()
        assert config.feature_dim == 2
        assert not config.use_one_hot_counts
        assert not config.include_geometric_features
        assert not config.include_strategic_features

    def test_minimal_config(self):
        """Test minimal encoding configuration."""
        config = minimal_encoding_config()
        assert config.feature_dim == 5
        assert not config.use_one_hot_counts
        assert config.include_geometric_features
        assert not config.include_strategic_features

    def test_standard_config(self):
        """Test standard encoding configuration."""
        config = standard_encoding_config()
        assert config.feature_dim == 35
        assert config.use_one_hot_counts
        assert config.include_geometric_features
        assert not config.include_strategic_features

    def test_rich_config(self):
        """Test rich encoding configuration."""
        config = rich_encoding_config()
        assert config.feature_dim == 45
        assert config.use_one_hot_counts
        assert config.include_geometric_features
        assert config.include_strategic_features

    def test_feature_dimension(self):
        """Test feature dimension calculation."""
        assert feature_dimension(raw_encoding_config()) == 2
        assert feature_dimension(minimal_encoding_config()) == 5
        assert feature_dimension(standard_encoding_config()) == 35
        assert feature_dimension(rich_encoding_config()) == 45


class TestRawEncoding:
    """Tests for raw encoding."""

    def test_encode_raw_empty_point(self):
        """Test raw encoding of empty point."""
        board = empty_board()
        features = encode_raw(board, 1)

        assert features.shape == (2,)
        assert features[0] == 0.0  # No our checkers
        assert features[1] == 0.0  # No opponent checkers

    def test_encode_raw_with_checkers(self):
        """Test raw encoding with checkers."""
        board = empty_board()
        board.white_checkers[6] = 5
        board.black_checkers[6] = 3
        board.player_to_move = Player.WHITE

        features = encode_raw(board, 6)

        assert features.shape == (2,)
        assert abs(features[0] - 5/15) < 1e-6  # White's 5 checkers
        assert abs(features[1] - 3/15) < 1e-6  # Black's 3 checkers

    def test_encode_raw_initial_position(self):
        """Test raw encoding at initial position."""
        board = initial_board()

        # White's 24-point has 2 checkers
        features = encode_raw(board, 24)
        assert abs(features[0] - 2/15) < 1e-6

        # Black's 1-point has 2 checkers
        features = encode_raw(board, 1)
        assert abs(features[1] - 2/15) < 1e-6


class TestOneHotEncoding:
    """Tests for one-hot encoding."""

    def test_encode_one_hot_empty(self):
        """Test one-hot encoding of empty point."""
        board = empty_board()
        features = encode_one_hot(board, 1)

        assert features.shape == (32,)
        # First element of each one-hot should be 1 (0 checkers)
        assert features[0] == 1.0
        assert features[16] == 1.0
        assert np.sum(features) == 2.0

    def test_encode_one_hot_with_checkers(self):
        """Test one-hot encoding with checkers."""
        board = empty_board()
        board.white_checkers[6] = 5
        board.black_checkers[6] = 3
        board.player_to_move = Player.WHITE

        features = encode_one_hot(board, 6)

        assert features.shape == (32,)
        assert features[5] == 1.0  # Our 5 checkers
        assert features[16 + 3] == 1.0  # Opponent's 3 checkers
        assert np.sum(features) == 2.0


class TestGeometricEncoding:
    """Tests for geometric encoding."""

    def test_encode_geometric_bar(self):
        """Test geometric encoding of bar."""
        board = empty_board()
        features = encode_geometric(board, 0)

        assert features.shape == (5,)
        assert features[2] == 1.0  # distance_to_home (furthest)
        assert features[3] == 0.0  # is_home_board
        assert features[4] == 0.0  # is_opp_home

    def test_encode_geometric_home_board(self):
        """Test geometric encoding of home board."""
        board = empty_board()
        board.player_to_move = Player.WHITE

        # White's home is 1-6
        features = encode_geometric(board, 1)
        assert features.shape == (5,)
        assert features[3] == 1.0  # is_home_board
        assert features[4] == 0.0  # is_opp_home

    def test_encode_geometric_opponent_home(self):
        """Test geometric encoding of opponent's home."""
        board = empty_board()
        board.player_to_move = Player.WHITE

        # Black's home (white's opponent) is 19-24
        features = encode_geometric(board, 20)
        assert features.shape == (5,)
        assert features[3] == 0.0  # Not our home
        assert features[4] == 1.0  # is_opp_home

    def test_encode_geometric_off(self):
        """Test geometric encoding of off position."""
        board = empty_board()
        features = encode_geometric(board, 25)

        assert features.shape == (5,)
        assert features[2] == 0.0  # distance_to_home (home!)


class TestStrategicEncoding:
    """Tests for strategic encoding."""

    def test_encode_strategic_blot(self):
        """Test strategic encoding detects blots."""
        board = empty_board()
        board.white_checkers[8] = 1  # Single checker = blot
        board.player_to_move = Player.WHITE

        features = encode_strategic(board, 8)

        # is_blot should be 1.0 (first strategic feature)
        assert features[5] == 1.0  # 5 geometric + is_blot

    def test_encode_strategic_made_point(self):
        """Test strategic encoding detects made points."""
        board = empty_board()
        board.white_checkers[6] = 2  # 2+ checkers = made point
        board.player_to_move = Player.WHITE

        features = encode_strategic(board, 6)

        # is_made should be 1.0
        assert features[9] == 1.0  # 5 geometric + blot + anchor + prime + opp_blot + is_made

    def test_encode_strategic_bar(self):
        """Test strategic encoding of bar."""
        board = empty_board()
        features = encode_strategic(board, 0)

        # is_bar should be 1.0 (second to last strategic feature)
        assert features[-2] == 1.0

    def test_encode_strategic_off(self):
        """Test strategic encoding of off."""
        board = empty_board()
        features = encode_strategic(board, 25)

        # is_off should be 1.0 (last strategic feature)
        assert features[-1] == 1.0


class TestFullEncoding:
    """Tests for full configurable encoding."""

    def test_encode_full_raw(self):
        """Test full encoding with raw config."""
        config = raw_encoding_config()
        board = initial_board()

        features = encode_full(config, board, 1)
        assert features.shape == (config.feature_dim,)
        assert features.shape == (2,)

    def test_encode_full_minimal(self):
        """Test full encoding with minimal config."""
        config = minimal_encoding_config()
        board = initial_board()

        features = encode_full(config, board, 1)
        assert features.shape == (config.feature_dim,)
        assert features.shape == (5,)

    def test_encode_full_standard(self):
        """Test full encoding with standard config."""
        config = standard_encoding_config()
        board = initial_board()

        features = encode_full(config, board, 1)
        assert features.shape == (config.feature_dim,)
        assert features.shape == (35,)

    def test_encode_full_rich(self):
        """Test full encoding with rich config."""
        config = rich_encoding_config()
        board = initial_board()

        features = encode_full(config, board, 1)
        assert features.shape == (config.feature_dim,)
        assert features.shape == (45,)


class TestBoardEncoding:
    """Tests for board encoding."""

    def test_encode_board_single(self):
        """Test encoding a single board."""
        config = raw_encoding_config()
        board = initial_board()

        encoded = encode_board(config, board)

        assert encoded.batch_size == 1
        assert encoded.sequence_length == 26
        assert encoded.feature_dim == 2
        assert encoded.position_features.shape == (1, 26, 2)

    def test_encode_board_all_positions(self):
        """Test that all 26 positions are encoded."""
        config = raw_encoding_config()
        board = initial_board()

        encoded = encode_board(config, board)

        # Check that we have features for all positions
        for point in range(26):
            features = encoded.position_features[0, point]
            assert features.shape == (2,)

    def test_encode_boards_batch(self):
        """Test encoding a batch of boards."""
        config = raw_encoding_config()
        boards = [initial_board(), empty_board(), initial_board()]

        encoded = encode_boards(config, boards)

        assert encoded.batch_size == 3
        assert encoded.sequence_length == 26
        assert encoded.feature_dim == 2
        assert encoded.position_features.shape == (3, 26, 2)

    def test_decode_board_raw(self):
        """Test decoding a raw encoded board."""
        config = raw_encoding_config()
        board = initial_board()

        encoded = encode_board(config, board)
        decoded = decode_board(config, encoded)

        assert decoded is not None
        # Check a few key positions
        assert decoded.white_checkers[24] == 2
        assert decoded.black_checkers[1] == 2

    def test_decode_board_onehot_fails(self):
        """Test that decoding one-hot fails gracefully."""
        config = standard_encoding_config()
        board = initial_board()

        encoded = encode_board(config, board)
        decoded = decode_board(config, encoded)

        # Should return None for non-invertible encodings
        assert decoded is None


class TestDiceEncoding:
    """Tests for dice encoding."""

    def test_encode_dice(self):
        """Test dice encoding."""
        dice = (3, 5)
        encoded = encode_dice(dice)

        assert encoded.shape == (12,)
        assert encoded[2] == 1.0  # die1=3 (index 2)
        assert encoded[6 + 4] == 1.0  # die2=5 (index 4)
        assert np.sum(encoded) == 2.0

    def test_encode_dice_doubles(self):
        """Test encoding doubles."""
        dice = (4, 4)
        encoded = encode_dice(dice)

        assert encoded.shape == (12,)
        assert encoded[3] == 1.0  # die1=4
        assert encoded[6 + 3] == 1.0  # die2=4

    def test_dice_to_embedding_id(self):
        """Test dice to embedding ID conversion."""
        # First roll (1,1) should be ID 0
        assert dice_to_embedding_id((1, 1)) == 0

        # (1,2) should be ID 1
        assert dice_to_embedding_id((1, 2)) == 1
        assert dice_to_embedding_id((2, 1)) == 1  # Canonical

        # Last roll (6,6) should be ID 20
        assert dice_to_embedding_id((6, 6)) == 20

    def test_dice_to_embedding_id_all(self):
        """Test all 21 dice outcomes have unique IDs."""
        from backgammon.core.dice import ALL_DICE_ROLLS

        ids = set()
        for dice in ALL_DICE_ROLLS:
            dice_id = dice_to_embedding_id(dice)
            assert dice_id not in ids
            assert 0 <= dice_id <= 20
            ids.add(dice_id)

        assert len(ids) == 21


class TestEquityEncoding:
    """Tests for equity encoding."""

    def test_outcome_to_equity_white_wins_normal(self):
        """Test converting normal white win to equity."""
        outcome = GameOutcome(winner=Player.WHITE, points=1)
        equity = outcome_to_equity(outcome, Player.WHITE)

        assert equity.win_normal == 1.0
        assert equity.win_gammon == 0.0
        assert equity.win_backgammon == 0.0
        assert equity.lose_gammon == 0.0
        assert equity.lose_backgammon == 0.0

    def test_outcome_to_equity_gammon(self):
        """Test converting gammon to equity."""
        outcome = GameOutcome(winner=Player.WHITE, points=2)
        equity = outcome_to_equity(outcome, Player.WHITE)

        assert equity.win_gammon == 1.0
        assert equity.win_normal == 0.0

    def test_outcome_to_equity_backgammon(self):
        """Test converting backgammon to equity."""
        outcome = GameOutcome(winner=Player.BLACK, points=3)
        equity = outcome_to_equity(outcome, Player.WHITE)

        # White lost backgammon
        assert equity.lose_backgammon == 1.0

    def test_outcome_to_equity_lose_normal(self):
        """Test converting normal loss to equity."""
        outcome = GameOutcome(winner=Player.BLACK, points=1)
        equity = outcome_to_equity(outcome, Player.WHITE)

        # All zeros = normal loss
        assert equity.win_normal == 0.0
        assert equity.win_gammon == 0.0
        assert equity.lose_gammon == 0.0
        assert equity.lose_backgammon == 0.0

    def test_equity_array_roundtrip(self):
        """Test equity array encoding/decoding roundtrip."""
        equity = Equity(
            win_normal=0.3,
            win_gammon=0.2,
            win_backgammon=0.1,
            lose_gammon=0.15,
            lose_backgammon=0.05,
        )

        arr = equity_to_array(equity)
        assert arr.shape == (5,)

        equity2 = array_to_equity(arr)
        assert abs(equity2.win_normal - equity.win_normal) < 1e-6
        assert abs(equity2.win_gammon - equity.win_gammon) < 1e-6
        assert abs(equity2.win_backgammon - equity.win_backgammon) < 1e-6
        assert abs(equity2.lose_gammon - equity.lose_gammon) < 1e-6
        assert abs(equity2.lose_backgammon - equity.lose_backgammon) < 1e-6


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_normalize_features(self):
        """Test feature normalization."""
        # Create some features with non-zero mean/std
        features = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float32)

        normalized = normalize_features(features)

        # Should have mean close to 0 and std close to 1
        mean = np.mean(normalized, axis=0)
        std = np.std(normalized, axis=0)

        assert np.allclose(mean, 0.0, atol=1e-6)
        assert np.allclose(std, 1.0, atol=1e-6)

    def test_normalize_features_constant(self):
        """Test normalizing constant features doesn't divide by zero."""
        features = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)

        # Should not raise error
        normalized = normalize_features(features)
        assert normalized.shape == features.shape

    def test_canonicalize_board(self):
        """Test board canonicalization."""
        board = initial_board()
        canonical = canonicalize_board(board)

        # Should be same board (already canonical)
        assert canonical.player_to_move == board.player_to_move

    def test_augment_position(self):
        """Test position augmentation."""
        board = initial_board()
        augmented = augment_position(board)

        assert len(augmented) == 2  # Original + flipped
        assert augmented[0] is board  # First is original


class TestBatchUtilities:
    """Tests for batch utilities."""

    def test_stack_encoded_boards(self):
        """Test stacking encoded boards."""
        config = raw_encoding_config()
        boards = [
            encode_board(config, initial_board()),
            encode_board(config, empty_board()),
        ]

        stacked = stack_encoded_boards(boards)

        assert stacked.batch_size == 2
        assert stacked.sequence_length == 26
        assert stacked.feature_dim == 2
        assert stacked.position_features.shape == (2, 26, 2)

    def test_stack_encoded_boards_empty(self):
        """Test stacking empty list raises error."""
        with pytest.raises(ValueError):
            stack_encoded_boards([])

    def test_stack_encoded_boards_mismatched_dims(self):
        """Test stacking boards with different feature dims raises error."""
        board1 = encode_board(raw_encoding_config(), initial_board())
        board2 = encode_board(minimal_encoding_config(), initial_board())

        with pytest.raises(ValueError):
            stack_encoded_boards([board1, board2])


class TestExtractPositionFeatures:
    """Tests for extract_position_features."""

    def test_extract_position_features_raw(self):
        """Test extracting position features with raw config."""
        config = raw_encoding_config()
        board = initial_board()

        features = extract_position_features(config, board, 1)
        assert features.shape == (2,)

    def test_extract_position_features_standard(self):
        """Test extracting position features with standard config."""
        config = standard_encoding_config()
        board = initial_board()

        features = extract_position_features(config, board, 1)
        assert features.shape == (35,)

    def test_extract_position_features_all_points(self):
        """Test extracting features for all points."""
        config = raw_encoding_config()
        board = initial_board()

        # Should work for all 26 positions
        for point in range(26):
            features = extract_position_features(config, board, point)
            assert features.shape == (2,)
            assert features.dtype == np.float32


class TestGlobalBoardFeatures:
    """Tests for global board-level features."""

    def test_global_features_shape(self):
        """Test that global features have correct shape."""
        from backgammon.encoding.encoder import encode_global_board_features, GLOBAL_FEATURE_DIM
        board = initial_board()
        features = encode_global_board_features(board)
        assert features.shape == (GLOBAL_FEATURE_DIM,)
        assert features.dtype == np.float32

    def test_global_features_initial_position(self):
        """Test global features for the initial position."""
        from backgammon.encoding.encoder import encode_global_board_features
        board = initial_board()
        features = encode_global_board_features(board)

        # Pip counts should be equal for initial position
        our_pip = features[0]
        opp_pip = features[1]
        assert abs(our_pip - opp_pip) < 0.01  # Equal pip counts

        # Pip diff should be ~0
        pip_diff = features[2]
        assert abs(pip_diff) < 0.01

        # Should be in contact
        is_contact = features[3]
        assert is_contact == 1.0

        # No checkers borne off
        our_off = features[5]
        opp_off = features[6]
        assert our_off == 0.0
        assert opp_off == 0.0

    def test_global_features_race_position(self):
        """Test global features for a race position."""
        from backgammon.encoding.encoder import encode_global_board_features
        from backgammon.core.board import race_position
        board = race_position()
        features = encode_global_board_features(board)

        # Should NOT be in contact
        is_contact = features[3]
        assert is_contact == 0.0

    def test_global_feature_dim_constant(self):
        """Test the GLOBAL_FEATURE_DIM constant."""
        from backgammon.encoding.encoder import GLOBAL_FEATURE_DIM
        assert GLOBAL_FEATURE_DIM == 8
