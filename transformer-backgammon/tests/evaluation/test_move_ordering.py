"""Tests for move ordering heuristics and transposition table."""

import pytest
import numpy as np

from backgammon.core.board import (
    Board,
    initial_board,
    empty_board,
    generate_legal_moves,
    apply_move,
)
from backgammon.core.types import Player, MoveStep
from backgammon.evaluation.search import (
    score_move_heuristic,
    order_moves,
    TranspositionTable,
    _board_hash,
)


# ==============================================================================
# Move ordering heuristic tests
# ==============================================================================


class TestScoreMoveHeuristic:
    """Tests for score_move_heuristic."""

    def test_empty_move_scores_zero(self):
        board = initial_board()
        score = score_move_heuristic(board, Player.WHITE, ())
        assert score == 0.0

    def test_hitting_move_scores_higher(self):
        """A move that hits an opponent blot should score higher."""
        # Set up a board where white can hit a black blot
        board = empty_board()
        board.set_checkers(Player.WHITE, 6, 5)
        board.set_checkers(Player.WHITE, 8, 3)
        board.set_checkers(Player.WHITE, 13, 5)
        board.set_checkers(Player.WHITE, 24, 2)  # white on 24
        board.set_checkers(Player.BLACK, 22, 1)  # black blot on 22
        board.set_checkers(Player.BLACK, 1, 2)
        board.set_checkers(Player.BLACK, 12, 5)
        board.set_checkers(Player.BLACK, 17, 3)
        board.set_checkers(Player.BLACK, 19, 4)
        board.player_to_move = Player.WHITE

        # Move that hits: 24 -> 22 (hitting blot)
        hit_move = (MoveStep(from_point=24, to_point=22, die_used=2, hits_opponent=True),)
        # Move that doesn't hit: 13 -> 11
        no_hit_move = (MoveStep(from_point=13, to_point=11, die_used=2, hits_opponent=False),)

        hit_score = score_move_heuristic(board, Player.WHITE, hit_move)
        no_hit_score = score_move_heuristic(board, Player.WHITE, no_hit_move)

        assert hit_score > no_hit_score

    def test_bearoff_scores_high(self):
        """Bearing off checkers should score high."""
        board = empty_board()
        board.set_checkers(Player.WHITE, 1, 2)
        board.set_checkers(Player.WHITE, 2, 2)
        board.set_checkers(Player.WHITE, 3, 3)
        board.set_checkers(Player.WHITE, 4, 3)
        board.set_checkers(Player.WHITE, 5, 3)
        board.set_checkers(Player.WHITE, 6, 2)
        board.set_checkers(Player.BLACK, 19, 5)
        board.set_checkers(Player.BLACK, 20, 5)
        board.set_checkers(Player.BLACK, 21, 5)
        board.player_to_move = Player.WHITE

        # Move that bears off: 1 -> off
        bearoff_move = (MoveStep(from_point=1, to_point=25, die_used=1, hits_opponent=False),)
        # Move that doesn't bear off: 6 -> 5
        normal_move = (MoveStep(from_point=6, to_point=5, die_used=1, hits_opponent=False),)

        bearoff_score = score_move_heuristic(board, Player.WHITE, bearoff_move)
        normal_score = score_move_heuristic(board, Player.WHITE, normal_move)

        assert bearoff_score > normal_score


class TestOrderMoves:
    """Tests for order_moves."""

    def test_single_move(self):
        board = initial_board()
        moves = [()]
        ordered = order_moves(board, Player.WHITE, moves)
        assert ordered == moves

    def test_empty_moves(self):
        board = initial_board()
        ordered = order_moves(board, Player.WHITE, [])
        assert ordered == []

    def test_preserves_all_moves(self):
        """All moves should still be present after ordering."""
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        ordered = order_moves(board, Player.WHITE, moves)
        assert len(ordered) == len(moves)
        assert set(id(m) for m in ordered) == set(id(m) for m in ordered)

    def test_returns_list(self):
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        ordered = order_moves(board, Player.WHITE, moves)
        assert isinstance(ordered, list)


# ==============================================================================
# Transposition table tests
# ==============================================================================


class TestTranspositionTable:
    """Tests for TranspositionTable."""

    def test_empty_lookup_returns_none(self):
        tt = TranspositionTable()
        board = initial_board()
        assert tt.lookup(board, 0) is None

    def test_store_and_lookup(self):
        tt = TranspositionTable()
        board = initial_board()
        tt.store(board, 0, 0.5)
        assert tt.lookup(board, 0) == pytest.approx(0.5)

    def test_different_ply_levels(self):
        """Values stored at different ply levels should not collide."""
        tt = TranspositionTable()
        board = initial_board()
        tt.store(board, 0, 0.5)
        tt.store(board, 1, 0.8)
        assert tt.lookup(board, 0) == pytest.approx(0.5)
        assert tt.lookup(board, 1) == pytest.approx(0.8)

    def test_different_boards(self):
        """Different boards should get different entries."""
        tt = TranspositionTable()
        board1 = initial_board()
        board2 = empty_board()
        board2.set_checkers(Player.WHITE, 25, 15)
        board2.set_checkers(Player.BLACK, 25, 15)

        tt.store(board1, 0, 0.5)
        tt.store(board2, 0, -0.3)

        assert tt.lookup(board1, 0) == pytest.approx(0.5)
        assert tt.lookup(board2, 0) == pytest.approx(-0.3)

    def test_eviction(self):
        """Table should evict entries when max_size is exceeded."""
        tt = TranspositionTable(max_size=10)
        # Fill table beyond max_size
        for i in range(20):
            board = empty_board()
            board.set_checkers(Player.WHITE, 25, 15)
            board.set_checkers(Player.BLACK, i % 25, 15 if i % 25 != 25 else 0)
            tt.store(board, 0, float(i))

        assert len(tt) <= 10

    def test_clear(self):
        tt = TranspositionTable()
        board = initial_board()
        tt.store(board, 0, 0.5)
        tt.clear()
        assert len(tt) == 0
        assert tt.lookup(board, 0) is None

    def test_len(self):
        tt = TranspositionTable()
        assert len(tt) == 0
        board = initial_board()
        tt.store(board, 0, 0.5)
        assert len(tt) == 1


class TestBoardHash:
    """Tests for board hashing."""

    def test_same_board_same_hash(self):
        board1 = initial_board()
        board2 = initial_board()
        assert _board_hash(board1) == _board_hash(board2)

    def test_different_boards_different_hash(self):
        board1 = initial_board()
        board2 = empty_board()
        board2.set_checkers(Player.WHITE, 25, 15)
        board2.set_checkers(Player.BLACK, 25, 15)
        assert _board_hash(board1) != _board_hash(board2)

    def test_player_to_move_affects_hash(self):
        """Same checker layout but different player to move should hash differently."""
        board1 = initial_board()
        board1.player_to_move = Player.WHITE
        board2 = initial_board()
        board2.player_to_move = Player.BLACK
        assert _board_hash(board1) != _board_hash(board2)
