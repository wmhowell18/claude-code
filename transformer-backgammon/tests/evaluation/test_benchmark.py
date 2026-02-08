"""Tests for benchmark and evaluation infrastructure."""

import pytest
import jax
import numpy as np

from backgammon.core.board import initial_board
from backgammon.core.types import Player
from backgammon.evaluation.agents import random_agent, pip_count_agent
from backgammon.training.train import TrainingConfig, create_train_state
from backgammon.evaluation.benchmark import (
    EvalResult,
    evaluate_agents,
    evaluate_against_baselines,
    BenchmarkPosition,
    get_benchmark_positions,
    compute_equity_error,
    EvalHistory,
    run_evaluation_checkpoint,
)


@pytest.fixture(scope="module")
def small_state():
    """Create a small training state for benchmark tests."""
    config = TrainingConfig(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        train_policy=False,
    )
    rng = jax.random.PRNGKey(42)
    return create_train_state(config, rng)


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_win_rate(self):
        """Test win rate computation."""
        result = EvalResult(
            agent_name="Test", opponent_name="Opp",
            num_games=10, wins=6, losses=3, draws=1,
            gammons_won=1, gammons_lost=0,
            backgammons_won=0, backgammons_lost=0,
            avg_game_length=50.0,
        )
        assert result.win_rate == pytest.approx(0.6)

    def test_win_rate_no_games(self):
        """Win rate with 0 games should be 0."""
        result = EvalResult(
            agent_name="Test", opponent_name="Opp",
            num_games=0, wins=0, losses=0, draws=0,
            gammons_won=0, gammons_lost=0,
            backgammons_won=0, backgammons_lost=0,
            avg_game_length=0.0,
        )
        assert result.win_rate == 0.0

    def test_ppg(self):
        """Test points per game computation."""
        result = EvalResult(
            agent_name="Test", opponent_name="Opp",
            num_games=10, wins=5, losses=5, draws=0,
            gammons_won=2, gammons_lost=0,
            backgammons_won=0, backgammons_lost=0,
            avg_game_length=50.0,
        )
        # wins=5, gammons_won=2 -> 5 + 2 = 7 points won
        # losses=5, gammons_lost=0 -> 5 points lost
        # ppg = (7 - 5) / 10 = 0.2
        assert result.ppg == pytest.approx(0.2)

    def test_summary(self):
        """Test summary string generation."""
        result = EvalResult(
            agent_name="Test", opponent_name="Opp",
            num_games=10, wins=6, losses=3, draws=1,
            gammons_won=1, gammons_lost=0,
            backgammons_won=0, backgammons_lost=0,
            avg_game_length=50.0,
        )
        summary = result.summary()
        assert "Test" in summary
        assert "Opp" in summary
        assert "60.0%" in summary


class TestEvaluateAgents:
    """Tests for agent evaluation."""

    def test_random_vs_random(self):
        """Random vs random should be roughly 50/50."""
        agent = random_agent(seed=42)
        opponent = random_agent(seed=123)
        result = evaluate_agents(
            agent, opponent, num_games=20, rng=np.random.default_rng(0)
        )
        assert result.num_games == 20
        assert result.wins + result.losses + result.draws == 20
        # Random vs random shouldn't be too far from 50%
        assert 0.1 <= result.win_rate <= 0.9

    def test_pip_count_beats_random(self):
        """Pip count should generally beat random."""
        agent = pip_count_agent()
        opponent = random_agent(seed=42)
        result = evaluate_agents(
            agent, opponent, num_games=30, rng=np.random.default_rng(0)
        )
        # Pip count should win more than it loses
        assert result.wins >= result.losses

    def test_balanced_colors(self):
        """Games should be split between colors."""
        agent = random_agent(seed=42)
        opponent = random_agent(seed=123)
        result = evaluate_agents(
            agent, opponent, num_games=10, rng=np.random.default_rng(0)
        )
        assert result.num_games == 10


class TestBenchmarkPositions:
    """Tests for benchmark position suite."""

    def test_positions_exist(self):
        """Should return a non-empty list of positions."""
        positions = get_benchmark_positions()
        assert len(positions) > 0

    def test_position_structure(self):
        """Each position should have required fields."""
        positions = get_benchmark_positions()
        for pos in positions:
            assert isinstance(pos, BenchmarkPosition)
            assert pos.name
            assert pos.board is not None
            assert pos.category
            assert pos.description

    def test_categories_diverse(self):
        """Positions should cover multiple categories."""
        positions = get_benchmark_positions()
        categories = {p.category for p in positions}
        # Should have at least 3 different categories
        assert len(categories) >= 3

    def test_some_have_expected_equity(self):
        """Some positions should have known expected equity."""
        positions = get_benchmark_positions()
        with_equity = [p for p in positions if p.expected_equity is not None]
        assert len(with_equity) >= 2

    def test_standard_opening_is_included(self):
        """Standard opening should be in the suite."""
        positions = get_benchmark_positions()
        names = [p.name for p in positions]
        assert "standard_opening" in names

    def test_boards_are_valid(self):
        """All boards should be valid (no assertion errors)."""
        positions = get_benchmark_positions()
        for pos in positions:
            board = pos.board
            # Basic validity checks
            assert len(board.white_checkers) == 26
            assert len(board.black_checkers) == 26
            assert sum(board.white_checkers) == 15
            assert sum(board.black_checkers) == 15


class TestEquityError:
    """Tests for equity error computation."""

    def test_with_known_positions(self, small_state):
        """Should compute error for positions with known equity."""
        positions = get_benchmark_positions()
        errors = compute_equity_error(small_state, positions)

        assert 'mae' in errors
        assert 'rmse' in errors
        assert 'max_error' in errors
        assert 'n_evaluated' in errors
        assert errors['n_evaluated'] > 0
        assert errors['mae'] >= 0
        assert errors['rmse'] >= errors['mae']

    def test_per_position_details(self, small_state):
        """Should return per-position error details."""
        positions = get_benchmark_positions()
        errors = compute_equity_error(small_state, positions)

        assert 'per_position' in errors
        for name, predicted, expected, err in errors['per_position']:
            assert isinstance(name, str)
            assert np.isfinite(predicted)
            assert np.isfinite(expected)
            assert err >= 0

    def test_no_known_positions(self, small_state):
        """Should handle case with no known equities."""
        positions = [
            BenchmarkPosition(
                name="unknown",
                board=initial_board(),
                category="test",
                description="No known equity",
                expected_equity=None,
            )
        ]
        errors = compute_equity_error(small_state, positions)
        assert errors['n_evaluated'] == 0
        assert errors['mae'] == 0.0


class TestEvalHistory:
    """Tests for evaluation history tracking."""

    def test_empty_history(self):
        """Empty history should produce valid summary."""
        history = EvalHistory()
        summary = history.summary()
        assert "No evaluation history" in summary

    def test_add_entry(self):
        """Should record entries correctly."""
        history = EvalHistory()

        result = EvalResult(
            agent_name="Test", opponent_name="Random",
            num_games=10, wins=6, losses=3, draws=1,
            gammons_won=0, gammons_lost=0,
            backgammons_won=0, backgammons_lost=0,
            avg_game_length=50.0,
        )

        history.add(step=100, games_played=500, results={"Random": result})
        assert len(history.entries) == 1
        assert history.entries[0]['step'] == 100
        assert history.entries[0]['wr_vs_Random'] == pytest.approx(0.6)

    def test_summary_format(self):
        """Summary should include column headers and data."""
        history = EvalHistory()

        result = EvalResult(
            agent_name="Test", opponent_name="Random",
            num_games=10, wins=6, losses=3, draws=1,
            gammons_won=0, gammons_lost=0,
            backgammons_won=0, backgammons_lost=0,
            avg_game_length=50.0,
        )

        history.add(step=100, games_played=500, results={"Random": result})
        summary = history.summary()
        assert "Step" in summary
        assert "100" in summary


class TestEvaluationCheckpoint:
    """Tests for full evaluation checkpoint.

    NOTE: These tests play actual games with the neural network agent,
    which is slow on CPU (~30s per game on full board). We use very
    small game counts and test the structure rather than statistical
    properties.
    """

    def test_evaluate_agents_neural_vs_random(self, small_state):
        """Neural agent can play against random (2 games only)."""
        from backgammon.evaluation.network_agent import create_neural_agent
        agent = create_neural_agent(small_state, temperature=0.0, name="Test", ply=0)
        opponent = random_agent(seed=99)
        result = evaluate_agents(agent, opponent, num_games=2, rng=np.random.default_rng(42))
        assert result.num_games == 2
        assert result.wins + result.losses + result.draws == 2

    def test_eval_history_with_equity(self):
        """EvalHistory records equity errors correctly."""
        history = EvalHistory()
        result = EvalResult(
            agent_name="Test", opponent_name="Random",
            num_games=10, wins=6, losses=3, draws=1,
            gammons_won=0, gammons_lost=0,
            backgammons_won=0, backgammons_lost=0,
            avg_game_length=50.0,
        )
        equity_errors = {'mae': 0.5, 'rmse': 0.7}
        history.add(step=100, games_played=500, results={"Random": result},
                    equity_errors=equity_errors)
        assert history.entries[0]['equity_mae'] == 0.5
        assert history.entries[0]['equity_rmse'] == 0.7
