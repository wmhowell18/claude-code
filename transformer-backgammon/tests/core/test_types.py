"""Tests for core type definitions."""

import pytest
import numpy as np
from backgammon.core.types import (
    Board,
    Player,
    MoveStep,
    Equity,
    GameOutcome,
    EncodedBoard,
)


class TestPlayer:
    """Tests for Player enum."""

    def test_opponent(self):
        """Test opponent() method."""
        assert Player.WHITE.opponent() == Player.BLACK
        assert Player.BLACK.opponent() == Player.WHITE

    def test_string_representation(self):
        """Test string conversion."""
        assert str(Player.WHITE) == "white"
        assert str(Player.BLACK) == "black"


class TestBoard:
    """Tests for Board dataclass."""

    def test_board_creation(self):
        """Test creating an empty board."""
        board = Board()
        assert len(board.white_checkers) == 26
        assert len(board.black_checkers) == 26
        assert board.player_to_move == Player.WHITE

    def test_get_set_checkers(self):
        """Test getting and setting checker counts."""
        board = Board()

        # Set white checkers
        board.set_checkers(Player.WHITE, 6, 5)
        assert board.get_checkers(Player.WHITE, 6) == 5

        # Set black checkers
        board.set_checkers(Player.BLACK, 13, 2)
        assert board.get_checkers(Player.BLACK, 13) == 2

    def test_board_copy(self):
        """Test board copying."""
        board1 = Board()
        board1.set_checkers(Player.WHITE, 1, 3)

        board2 = board1.copy()
        board2.set_checkers(Player.WHITE, 1, 5)

        # Original should be unchanged
        assert board1.get_checkers(Player.WHITE, 1) == 3
        assert board2.get_checkers(Player.WHITE, 1) == 5

    def test_invalid_checker_count(self):
        """Test that invalid checker counts raise errors."""
        board = Board()
        with pytest.raises(AssertionError):
            board.set_checkers(Player.WHITE, 1, 16)  # Too many checkers


class TestMoveStep:
    """Tests for MoveStep dataclass."""

    def test_valid_move_step(self):
        """Test creating a valid move step."""
        step = MoveStep(from_point=24, to_point=20, die_used=4, hits_opponent=False)
        assert step.from_point == 24
        assert step.to_point == 20
        assert step.die_used == 4
        assert not step.hits_opponent

    def test_hitting_move(self):
        """Test move that hits opponent."""
        step = MoveStep(from_point=8, to_point=5, die_used=3, hits_opponent=True)
        assert step.hits_opponent

    def test_invalid_points(self):
        """Test that invalid points raise errors."""
        with pytest.raises(AssertionError):
            MoveStep(from_point=26, to_point=20, die_used=4)  # Invalid from_point

        with pytest.raises(AssertionError):
            MoveStep(from_point=20, to_point=-1, die_used=4)  # Invalid to_point

    def test_invalid_die(self):
        """Test that invalid die values raise errors."""
        with pytest.raises(AssertionError):
            MoveStep(from_point=20, to_point=14, die_used=7)  # Invalid die


class TestEquity:
    """Tests for Equity dataclass."""

    def test_equity_creation(self):
        """Test creating equity."""
        equity = Equity(
            win_normal=0.4,
            win_gammon=0.1,
            win_backgammon=0.05,
            lose_gammon=0.1,
            lose_backgammon=0.05,
        )
        assert equity.win_normal == 0.4
        # lose_normal should be computed
        assert abs(equity.lose_normal - 0.3) < 1e-6

    def test_expected_value(self):
        """Test expected value calculation."""
        # Certain win
        equity = Equity(
            win_normal=1.0,
            win_gammon=0.0,
            win_backgammon=0.0,
            lose_gammon=0.0,
            lose_backgammon=0.0,
        )
        assert abs(equity.expected_value() - 1.0) < 1e-6

        # Certain loss
        equity = Equity(
            win_normal=0.0,
            win_gammon=0.0,
            win_backgammon=0.0,
            lose_gammon=0.0,
            lose_backgammon=0.0,
        )
        assert abs(equity.expected_value() - (-1.0)) < 1e-6

        # Gammon
        equity = Equity(
            win_normal=0.0,
            win_gammon=1.0,
            win_backgammon=0.0,
            lose_gammon=0.0,
            lose_backgammon=0.0,
        )
        assert abs(equity.expected_value() - 2.0) < 1e-6

    def test_to_from_array(self):
        """Test conversion to/from numpy array."""
        equity = Equity(
            win_normal=0.3,
            win_gammon=0.2,
            win_backgammon=0.1,
            lose_gammon=0.15,
            lose_backgammon=0.05,
        )

        arr = equity.to_array()
        assert arr.shape == (5,)
        assert arr.dtype == np.float32

        equity2 = Equity.from_array(arr)
        assert abs(equity2.win_normal - equity.win_normal) < 1e-6
        assert abs(equity2.win_gammon - equity.win_gammon) < 1e-6


class TestGameOutcome:
    """Tests for GameOutcome dataclass."""

    def test_normal_win(self):
        """Test normal win."""
        outcome = GameOutcome(winner=Player.WHITE, points=1)
        assert outcome.winner == Player.WHITE
        assert outcome.points == 1
        assert not outcome.is_gammon()
        assert not outcome.is_backgammon()

    def test_gammon(self):
        """Test gammon."""
        outcome = GameOutcome(winner=Player.BLACK, points=2)
        assert outcome.is_gammon()
        assert not outcome.is_backgammon()

    def test_backgammon(self):
        """Test backgammon."""
        outcome = GameOutcome(winner=Player.WHITE, points=3)
        assert outcome.is_gammon()
        assert outcome.is_backgammon()

    def test_invalid_points(self):
        """Test that invalid points raise errors."""
        with pytest.raises(AssertionError):
            GameOutcome(winner=Player.WHITE, points=0)

        with pytest.raises(AssertionError):
            GameOutcome(winner=Player.WHITE, points=4)


class TestEncodedBoard:
    """Tests for EncodedBoard dataclass."""

    def test_encoded_board_creation(self):
        """Test creating an encoded board."""
        # Create dummy features [batch=2, seq_len=26, feature_dim=10]
        features = np.random.randn(2, 26, 10).astype(np.float32)

        encoded = EncodedBoard(position_features=features)
        assert encoded.batch_size == 2
        assert encoded.sequence_length == 26
        assert encoded.feature_dim == 10

    def test_encoded_board_with_dice(self):
        """Test encoded board with dice features."""
        features = np.random.randn(1, 26, 5).astype(np.float32)
        dice_features = np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float32)  # One-hot for die=1

        encoded = EncodedBoard(
            position_features=features,
            dice_features=dice_features,
        )
        assert encoded.dice_features is not None
        assert encoded.dice_features.shape == (1, 6)
