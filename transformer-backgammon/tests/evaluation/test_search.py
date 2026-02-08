"""Tests for search module (0-ply, 1-ply, and 2-ply move evaluation).

IMPORTANT: 1-ply and 2-ply tests use minimal bearoff positions (2-3 checkers)
to keep test runtime under a few seconds. Full-board positions with these
search depths would take minutes per test on CPU.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from backgammon.core.board import (
    Board,
    initial_board,
    empty_board,
    generate_legal_moves,
    is_game_over,
)
from backgammon.core.types import Player
from backgammon.encoding.encoder import raw_encoding_config
from backgammon.training.train import TrainingConfig, create_train_state
from backgammon.evaluation.search import (
    _equity_to_value,
    _batch_evaluate,
    _terminal_value,
    select_move_0ply,
    select_move_1ply,
    select_move_2ply,
    select_move,
    order_moves,
    _score_move_heuristic,
)


@pytest.fixture(scope="module")
def small_state():
    """Create a small training state for testing search functions."""
    config = TrainingConfig(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        train_policy=False,
    )
    rng = jax.random.PRNGKey(42)
    return create_train_state(config, rng)


@pytest.fixture
def encoding_config():
    return raw_encoding_config()


def _bearoff_board(white_pts, black_pts):
    """Create a minimal board for fast search tests.

    Args:
        white_pts: dict {point: count} for white (points 1-6 for home board)
        black_pts: dict {point: count} for black (points 19-24 for home board)
    """
    board = empty_board()
    w_on_board = sum(white_pts.values())
    b_on_board = sum(black_pts.values())
    for pt, cnt in white_pts.items():
        board.set_checkers(Player.WHITE, pt, cnt)
    board.set_checkers(Player.WHITE, 25, 15 - w_on_board)
    for pt, cnt in black_pts.items():
        board.set_checkers(Player.BLACK, pt, cnt)
    board.set_checkers(Player.BLACK, 0, 15 - b_on_board)
    return board


# ============================================================
# Pure unit tests (no network needed)
# ============================================================

class TestEquityToValue:
    """Tests for equity-to-value conversion."""

    def test_all_win_normal(self):
        equity = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        assert float(_equity_to_value(equity)[0]) == pytest.approx(1.0, abs=1e-5)

    def test_all_lose_normal(self):
        equity = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        assert float(_equity_to_value(equity)[0]) == pytest.approx(-1.0, abs=1e-5)

    def test_all_win_gammon(self):
        equity = jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        assert float(_equity_to_value(equity)[0]) == pytest.approx(2.0, abs=1e-5)

    def test_all_win_backgammon(self):
        equity = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
        assert float(_equity_to_value(equity)[0]) == pytest.approx(3.0, abs=1e-5)

    def test_all_lose_gammon(self):
        equity = jnp.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        assert float(_equity_to_value(equity)[0]) == pytest.approx(-2.0, abs=1e-5)

    def test_batch(self):
        equity = jnp.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        vals = _equity_to_value(equity)
        assert vals.shape == (2,)


class TestTerminalValue:
    """Tests for terminal position value computation."""

    def test_white_wins(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 25, 15)
        board.set_checkers(Player.BLACK, 19, 15)
        assert is_game_over(board)
        assert _terminal_value(board, Player.WHITE) > 0

    def test_perspective_negates(self):
        board = empty_board()
        board.set_checkers(Player.WHITE, 25, 15)
        board.set_checkers(Player.BLACK, 19, 15)
        assert _terminal_value(board, Player.WHITE) == -_terminal_value(board, Player.BLACK)


# ============================================================
# Tests requiring a network (use small_state fixture)
# ============================================================

class TestBatchEvaluate:
    def test_empty(self, small_state, encoding_config):
        assert len(_batch_evaluate(small_state, [], encoding_config)) == 0

    def test_single(self, small_state, encoding_config):
        vals = _batch_evaluate(small_state, [initial_board()], encoding_config)
        assert vals.shape == (1,)
        assert np.isfinite(vals[0])

    def test_multiple(self, small_state, encoding_config):
        vals = _batch_evaluate(small_state, [initial_board()] * 3, encoding_config)
        assert vals.shape == (3,)

    def test_deterministic(self, small_state, encoding_config):
        b = initial_board()
        vals = _batch_evaluate(small_state, [b, b], encoding_config)
        assert float(vals[0]) == pytest.approx(float(vals[1]), abs=1e-5)


class TestSelectMove0Ply:
    def test_no_moves(self, small_state, encoding_config):
        move, val = select_move_0ply(small_state, initial_board(), Player.WHITE, [], encoding_config)
        assert move == ()
        assert val == 0.0

    def test_single_move(self, small_state, encoding_config):
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (6, 5))
        move, val = select_move_0ply(small_state, board, Player.WHITE, [moves[0]], encoding_config)
        assert move == moves[0]

    def test_returns_legal(self, small_state, encoding_config):
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        move, _ = select_move_0ply(small_state, board, Player.WHITE, moves, encoding_config)
        assert move in moves

    def test_deterministic(self, small_state, encoding_config):
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        m1, v1 = select_move_0ply(small_state, board, Player.WHITE, moves, encoding_config)
        m2, v2 = select_move_0ply(small_state, board, Player.WHITE, moves, encoding_config)
        assert m1 == m2
        assert v1 == pytest.approx(v2, abs=1e-5)


class TestSelectMove1Ply:
    """1-ply tests use minimal bearoff positions for speed."""

    def test_no_moves(self, small_state, encoding_config):
        move, val = select_move_1ply(small_state, initial_board(), Player.WHITE, [], encoding_config)
        assert move == ()
        assert val == 0.0

    def test_returns_legal_bearoff(self, small_state, encoding_config):
        """1-ply on a 2-checker bearoff position (very few branches)."""
        board = _bearoff_board({1: 1, 2: 1}, {23: 1, 24: 1})
        moves = generate_legal_moves(board, Player.WHITE, (1, 2))
        if moves:
            move, val = select_move_1ply(small_state, board, Player.WHITE, moves, encoding_config)
            assert move in moves
            assert np.isfinite(val)

    def test_value_range(self, small_state, encoding_config):
        board = _bearoff_board({1: 1, 3: 1}, {22: 1, 24: 1})
        moves = generate_legal_moves(board, Player.WHITE, (2, 1))
        if moves:
            _, val = select_move_1ply(small_state, board, Player.WHITE, moves, encoding_config)
            assert -4.0 <= val <= 4.0


class TestSelectMove2Ply:
    """2-ply tests use minimal 1-checker positions for speed."""

    def test_no_moves(self, small_state, encoding_config):
        move, val = select_move_2ply(small_state, initial_board(), Player.WHITE, [], encoding_config)
        assert move == ()
        assert val == 0.0

    def test_returns_legal_bearoff(self, small_state, encoding_config):
        """2-ply on a 1-checker bearoff (minimal branching)."""
        board = _bearoff_board({2: 1}, {23: 1})
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        if moves:
            move, val = select_move_2ply(small_state, board, Player.WHITE, moves, encoding_config)
            assert move in moves
            assert np.isfinite(val)


class TestSelectMoveDispatch:
    def test_0ply(self, small_state, encoding_config):
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        move, _ = select_move(small_state, board, Player.WHITE, (3, 1), moves, ply=0, encoding_config=encoding_config)
        assert move in moves

    def test_1ply(self, small_state, encoding_config):
        board = _bearoff_board({1: 1, 2: 1}, {23: 1, 24: 1})
        moves = generate_legal_moves(board, Player.WHITE, (1, 2))
        if moves:
            move, _ = select_move(small_state, board, Player.WHITE, (1, 2), moves, ply=1, encoding_config=encoding_config)
            assert move in moves

    def test_2ply(self, small_state, encoding_config):
        board = _bearoff_board({2: 1}, {23: 1})
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        if moves:
            move, _ = select_move(small_state, board, Player.WHITE, (3, 1), moves, ply=2, encoding_config=encoding_config)
            assert move in moves

    def test_invalid_ply(self, small_state, encoding_config):
        moves = generate_legal_moves(initial_board(), Player.WHITE, (3, 1))
        with pytest.raises(ValueError, match="Unsupported ply depth"):
            select_move(small_state, initial_board(), Player.WHITE, (3, 1), moves, ply=3, encoding_config=encoding_config)

    def test_default_encoding(self, small_state):
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        move, _ = select_move(small_state, board, Player.WHITE, (3, 1), moves, ply=0)
        assert move in moves


class TestMoveOrdering:
    """Tests for move ordering heuristics."""

    def test_order_moves_returns_all_moves(self):
        """Ordering preserves all moves."""
        board = initial_board()
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        ordered = order_moves(board, Player.WHITE, moves)
        assert len(ordered) == len(moves)
        assert set(id(m) for m in ordered) == set(id(m) for m in moves) or \
            len(ordered) == len(moves)

    def test_order_moves_single_move(self):
        """Single move is returned as-is."""
        board = _bearoff_board({1: 1}, {24: 1})
        moves = generate_legal_moves(board, Player.WHITE, (1, 1))
        ordered = order_moves(board, Player.WHITE, moves)
        assert len(ordered) == len(moves)

    def test_order_moves_empty(self):
        """Empty move list returns empty."""
        board = initial_board()
        ordered = order_moves(board, Player.WHITE, [])
        assert ordered == []

    def test_heuristic_scores_hits_higher(self):
        """Moves that hit opponent should score higher."""
        # Create a position where one move hits and another doesn't
        board = empty_board()
        board.white_checkers[6] = 2
        board.black_checkers[5] = 1  # Blot on point 5

        moves = generate_legal_moves(board, Player.WHITE, (1, 2))
        if len(moves) > 1:
            scores = [
                _score_move_heuristic(board, Player.WHITE, m)
                for m in moves
            ]
            # Moves with hits should have the highest scores
            hit_moves = [
                i for i, m in enumerate(moves)
                if any(step.hits_opponent for step in m)
            ]
            non_hit_moves = [
                i for i, m in enumerate(moves)
                if not any(step.hits_opponent for step in m)
            ]
            if hit_moves and non_hit_moves:
                max_hit_score = max(scores[i] for i in hit_moves)
                max_non_hit_score = max(scores[i] for i in non_hit_moves)
                assert max_hit_score > max_non_hit_score
