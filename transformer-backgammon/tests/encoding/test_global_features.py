"""Tests for global encoding features (contact, pip count, home board, primes)."""

import pytest
import numpy as np

from backgammon.core.board import (
    Board,
    initial_board,
    empty_board,
)
from backgammon.core.types import Player
from backgammon.encoding.encoder import (
    compute_global_features,
    _detect_contact,
    _count_home_board_points,
    _longest_prime,
    enhanced_encoding_config,
    full_encoding_config,
    encode_board,
    feature_dimension,
)


# ==============================================================================
# Contact detection tests
# ==============================================================================


class TestContactDetection:
    def test_initial_board_is_contact(self):
        board = initial_board()
        assert _detect_contact(board, Player.WHITE) == 1.0

    def test_pure_race(self):
        """No contact when all our checkers are ahead of opponent's."""
        board = empty_board()
        # White checkers in home board (1-6)
        board.set_checkers(Player.WHITE, 1, 5)
        board.set_checkers(Player.WHITE, 2, 5)
        board.set_checkers(Player.WHITE, 3, 5)
        # Black checkers in their home board (19-24)
        board.set_checkers(Player.BLACK, 22, 5)
        board.set_checkers(Player.BLACK, 23, 5)
        board.set_checkers(Player.BLACK, 24, 5)
        board.player_to_move = Player.WHITE

        assert _detect_contact(board, Player.WHITE) == 0.0

    def test_bar_means_contact(self):
        """Checker on bar always means contact."""
        board = empty_board()
        board.set_checkers(Player.WHITE, 0, 1)  # On bar
        board.set_checkers(Player.WHITE, 1, 14)
        board.set_checkers(Player.BLACK, 24, 15)
        board.player_to_move = Player.WHITE

        assert _detect_contact(board, Player.WHITE) == 1.0


# ==============================================================================
# Home board points tests
# ==============================================================================


class TestHomeboardPoints:
    def test_initial_board_white(self):
        board = initial_board()
        # Initial: White has 2 on point 6, so 1 made point
        count = _count_home_board_points(board, Player.WHITE)
        assert count == 1  # Only point 6 is made

    def test_full_home_board(self):
        board = empty_board()
        for pt in range(1, 7):
            board.set_checkers(Player.WHITE, pt, 2)
        board.set_checkers(Player.WHITE, 25, 3)  # Remaining 3 off
        assert _count_home_board_points(board, Player.WHITE) == 6

    def test_empty_home_board(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 12, 15)
        assert _count_home_board_points(board, Player.WHITE) == 0

    def test_blots_dont_count(self):
        board = empty_board()
        for pt in range(1, 7):
            board.set_checkers(Player.WHITE, pt, 1)  # Single checkers
        board.set_checkers(Player.WHITE, 25, 9)
        assert _count_home_board_points(board, Player.WHITE) == 0

    def test_black_home_board(self):
        board = empty_board()
        board.set_checkers(Player.BLACK, 19, 3)
        board.set_checkers(Player.BLACK, 20, 3)
        board.set_checkers(Player.BLACK, 21, 3)
        board.set_checkers(Player.BLACK, 22, 3)
        board.set_checkers(Player.BLACK, 23, 2)
        board.set_checkers(Player.BLACK, 24, 1)
        assert _count_home_board_points(board, Player.BLACK) == 5  # 19-23 made (2+ checkers)


# ==============================================================================
# Prime detection tests
# ==============================================================================


class TestPrimeDetection:
    def test_no_prime(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 1, 2)
        board.set_checkers(Player.WHITE, 3, 2)  # Gap at 2
        board.set_checkers(Player.WHITE, 25, 11)
        assert _longest_prime(board, Player.WHITE) == 1

    def test_two_point_prime(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 5, 3)
        board.set_checkers(Player.WHITE, 6, 3)
        board.set_checkers(Player.WHITE, 25, 9)
        assert _longest_prime(board, Player.WHITE) == 2

    def test_six_point_prime(self):
        board = empty_board()
        for pt in range(7, 13):
            board.set_checkers(Player.WHITE, pt, 2)
        board.set_checkers(Player.WHITE, 25, 3)
        assert _longest_prime(board, Player.WHITE) == 6

    def test_prime_capped_at_six(self):
        """Even if more than 6 consecutive points, cap at 6."""
        board = empty_board()
        # 7 consecutive points (not realistic but tests cap)
        for pt in range(1, 8):
            board.set_checkers(Player.WHITE, pt, 2)
        board.set_checkers(Player.WHITE, 25, 1)
        assert _longest_prime(board, Player.WHITE) == 6

    def test_empty_board(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 25, 15)
        assert _longest_prime(board, Player.WHITE) == 0


# ==============================================================================
# Global features integration
# ==============================================================================


class TestComputeGlobalFeatures:
    def test_shape(self):
        board = initial_board()
        features = compute_global_features(board)
        assert features.shape == (8,)
        assert features.dtype == np.float32

    def test_all_finite(self):
        board = initial_board()
        features = compute_global_features(board)
        assert np.all(np.isfinite(features))

    def test_values_in_range(self):
        """All global features should be in [0, 1]."""
        board = initial_board()
        features = compute_global_features(board)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_bearoff_progress_initial(self):
        """No checkers borne off at start."""
        board = initial_board()
        features = compute_global_features(board)
        assert features[7] == 0.0  # our_bearoff_progress

    def test_pip_count_features(self):
        """Pip count features should be positive for initial position."""
        board = initial_board()
        features = compute_global_features(board)
        assert features[1] > 0.0  # our_pip_norm
        assert features[2] > 0.0  # opp_pip_norm


# ==============================================================================
# Encoding config tests
# ==============================================================================


class TestEnhancedEncoding:
    def test_feature_dim(self):
        config = enhanced_encoding_config()
        assert config.feature_dim == 10  # 2 raw + 8 global
        assert feature_dimension(config) == 10

    def test_encode_board(self):
        config = enhanced_encoding_config()
        board = initial_board()
        encoded = encode_board(config, board)
        assert encoded.position_features.shape == (1, 26, 10)

    def test_global_features_broadcast(self):
        """Global features should be the same across all positions."""
        config = enhanced_encoding_config()
        board = initial_board()
        encoded = encode_board(config, board)
        # Last 8 features should be identical for all 26 positions
        for i in range(1, 26):
            np.testing.assert_array_almost_equal(
                encoded.position_features[0, 0, 2:],  # Global features of pos 0
                encoded.position_features[0, i, 2:],  # Global features of pos i
            )


class TestFullEncoding:
    def test_feature_dim(self):
        config = full_encoding_config()
        assert config.feature_dim == 53
        assert feature_dimension(config) == 53

    def test_encode_board(self):
        config = full_encoding_config()
        board = initial_board()
        encoded = encode_board(config, board)
        assert encoded.position_features.shape == (1, 26, 53)
