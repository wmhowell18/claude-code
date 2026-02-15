"""Tests for race equity formula (Effective Pip Count)."""

import pytest
import numpy as np

from backgammon.core.board import (
    Board,
    initial_board,
    empty_board,
)
from backgammon.core.types import Player
from backgammon.evaluation.race import (
    effective_pip_count,
    race_equity,
    is_race_position,
)


class TestEffectivePipCount:
    def test_all_off_is_zero(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 25, 15)
        assert effective_pip_count(board, Player.WHITE) == 0.0

    def test_positive_for_checkers_on_board(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 1, 2)
        board.set_checkers(Player.WHITE, 25, 13)
        epc = effective_pip_count(board, Player.WHITE)
        assert epc > 0

    def test_higher_pips_worse(self):
        """Player with checkers further from home should have higher EPC."""
        board1 = empty_board()
        board1.set_checkers(Player.WHITE, 1, 5)
        board1.set_checkers(Player.WHITE, 25, 10)

        board2 = empty_board()
        board2.set_checkers(Player.WHITE, 12, 5)
        board2.set_checkers(Player.WHITE, 25, 10)

        epc1 = effective_pip_count(board1, Player.WHITE)
        epc2 = effective_pip_count(board2, Player.WHITE)
        assert epc2 > epc1  # Further away = higher EPC

    def test_includes_corrections(self):
        """EPC should be higher than raw pip count due to corrections."""
        board = empty_board()
        board.set_checkers(Player.WHITE, 6, 5)
        board.set_checkers(Player.WHITE, 5, 5)
        board.set_checkers(Player.WHITE, 4, 5)

        from backgammon.core.board import pip_count
        raw = pip_count(board, Player.WHITE)
        epc = effective_pip_count(board, Player.WHITE)
        assert epc > raw  # Corrections add to raw pip count


class TestRaceEquity:
    def test_equal_position_near_zero(self):
        """Symmetric positions should have near-zero equity."""
        board = empty_board()
        board.set_checkers(Player.WHITE, 1, 5)
        board.set_checkers(Player.WHITE, 2, 5)
        board.set_checkers(Player.WHITE, 3, 5)
        board.set_checkers(Player.BLACK, 22, 5)
        board.set_checkers(Player.BLACK, 23, 5)
        board.set_checkers(Player.BLACK, 24, 5)

        eq = race_equity(board, Player.WHITE)
        assert abs(eq) < 0.3  # Should be roughly even

    def test_big_lead_positive(self):
        """Big pip count lead should give positive equity."""
        board = empty_board()
        # White almost done: just 2 checkers on point 1
        board.set_checkers(Player.WHITE, 1, 2)
        board.set_checkers(Player.WHITE, 25, 13)
        # Black way behind: 15 checkers on points 20-22
        board.set_checkers(Player.BLACK, 20, 5)
        board.set_checkers(Player.BLACK, 21, 5)
        board.set_checkers(Player.BLACK, 22, 5)

        # White is way ahead (2 pips vs ~315 pips)
        eq = race_equity(board, Player.WHITE)
        assert eq > 0.5

    def test_already_won(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 25, 15)
        board.set_checkers(Player.BLACK, 24, 15)
        eq = race_equity(board, Player.WHITE)
        assert eq == 1.0

    def test_already_lost(self):
        """If opponent has borne off all checkers, we've lost."""
        board = empty_board()
        board.set_checkers(Player.WHITE, 1, 15)
        board.set_checkers(Player.BLACK, 25, 15)  # Black borne off = Black won
        eq = race_equity(board, Player.WHITE)
        assert eq == -1.0

    def test_range(self):
        """Equity should be in [-1, 1]."""
        board = initial_board()
        eq = race_equity(board, Player.WHITE)
        assert -1.0 <= eq <= 1.0

    def test_opponent_perspective_negates(self):
        """Equity from opponent's perspective should be roughly negated."""
        board = empty_board()
        board.set_checkers(Player.WHITE, 1, 5)
        board.set_checkers(Player.WHITE, 2, 5)
        board.set_checkers(Player.WHITE, 3, 5)
        board.set_checkers(Player.BLACK, 20, 5)
        board.set_checkers(Player.BLACK, 21, 5)
        board.set_checkers(Player.BLACK, 22, 5)

        eq_white = race_equity(board, Player.WHITE)
        eq_black = race_equity(board, Player.BLACK)
        # Not exactly negated due to asymmetry in corrections, but close
        assert abs(eq_white + eq_black) < 0.1


class TestIsRacePosition:
    def test_initial_not_race(self):
        board = initial_board()
        assert is_race_position(board) is False

    def test_pure_race(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 1, 5)
        board.set_checkers(Player.WHITE, 2, 5)
        board.set_checkers(Player.WHITE, 3, 5)
        board.set_checkers(Player.BLACK, 22, 5)
        board.set_checkers(Player.BLACK, 23, 5)
        board.set_checkers(Player.BLACK, 24, 5)
        assert is_race_position(board) is True

    def test_bar_not_race(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 0, 1)  # On bar
        board.set_checkers(Player.WHITE, 1, 14)
        board.set_checkers(Player.BLACK, 24, 15)
        assert is_race_position(board) is False

    def test_contact_not_race(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 12, 15)
        board.set_checkers(Player.BLACK, 10, 15)
        # White on 12, Black on 10 -> contact (White needs to pass Black)
        assert is_race_position(board) is False

    def test_all_off_is_race(self):
        """If all checkers are borne off, it's trivially a race."""
        board = empty_board()
        board.set_checkers(Player.WHITE, 25, 15)
        board.set_checkers(Player.BLACK, 25, 15)  # Both borne off
        assert is_race_position(board) is True
