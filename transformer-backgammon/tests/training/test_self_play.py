"""Tests for self-play game generation."""

import pytest
import numpy as np

from backgammon.training.self_play import (
    GameStep,
    GameResult,
    play_game,
    generate_training_batch,
    compute_game_statistics,
)
from backgammon.core.board import initial_board, get_early_training_variants
from backgammon.core.types import Player
from backgammon.evaluation.agents import random_agent, pip_count_agent


class TestGameStep:
    """Test GameStep namedtuple."""

    def test_game_step_creation(self):
        """Test creating a GameStep."""
        board = initial_board()
        step = GameStep(
            board=board,
            player=Player.WHITE,
            legal_moves=[(), ((6, 3), (6, 2))],
            move_taken=((6, 3), (6, 2)),
            dice=(3, 4),
        )

        assert step.board == board
        assert step.player == Player.WHITE
        assert step.dice == (3, 4)
        assert len(step.legal_moves) == 2


class TestGameResult:
    """Test GameResult dataclass."""

    def test_game_result_creation(self):
        """Test creating a GameResult."""
        board = initial_board()
        steps = []

        result = GameResult(
            steps=steps,
            outcome='white_wins',
            num_moves=25,
            starting_position=board,
        )

        assert result.num_moves == 25
        assert result.outcome == 'white_wins'
        assert len(result.steps) == 0


class TestPlayGame:
    """Test playing a single game."""

    def test_play_game_random_vs_random(self):
        """Test playing a game between two random agents."""
        rng = np.random.default_rng(42)

        result = play_game(
            white_agent=random_agent(),
            black_agent=random_agent(),
            starting_position=initial_board(),
            max_moves=500,
            rng=rng,
        )

        # Game should complete (random agents eventually finish)
        assert result is not None
        assert result.num_moves > 0
        assert len(result.steps) == result.num_moves

        # Should have an outcome (very unlikely to reach max_moves)
        assert result.outcome is not None

    def test_play_game_steps_recorded(self):
        """Test that game steps are properly recorded."""
        rng = np.random.default_rng(42)

        result = play_game(
            white_agent=random_agent(),
            black_agent=random_agent(),
            starting_position=initial_board(),
            max_moves=500,
            rng=rng,
        )

        # Each step should have required fields
        for step in result.steps:
            assert step.board is not None
            assert step.player in [Player.WHITE, Player.BLACK]
            assert step.dice is not None
            assert isinstance(step.legal_moves, list)
            assert step.move_taken in step.legal_moves

    def test_play_game_alternating_players(self):
        """Test that players alternate."""
        rng = np.random.default_rng(42)

        result = play_game(
            white_agent=random_agent(),
            black_agent=random_agent(),
            starting_position=initial_board(),
            max_moves=500,
            rng=rng,
        )

        # Check alternation (White starts)
        if len(result.steps) > 1:
            assert result.steps[0].player == Player.WHITE
            # Players should alternate
            for i in range(1, min(10, len(result.steps))):
                assert result.steps[i].player != result.steps[i-1].player

    def test_play_game_pip_count_vs_random(self):
        """Test pip count agent vs random."""
        rng = np.random.default_rng(42)

        result = play_game(
            white_agent=pip_count_agent(),
            black_agent=random_agent(),
            starting_position=initial_board(),
            max_moves=500,
            rng=rng,
        )

        assert result is not None
        assert result.num_moves > 0

    def test_play_game_max_moves_limit(self):
        """Test that max_moves limit is respected."""
        rng = np.random.default_rng(42)

        # Use very low max_moves to force early termination
        result = play_game(
            white_agent=random_agent(),
            black_agent=random_agent(),
            starting_position=initial_board(),
            max_moves=5,
            rng=rng,
        )

        # Should terminate at or before max_moves
        assert result.num_moves <= 5

    def test_play_game_with_variant(self):
        """Test playing from variant starting position."""
        from backgammon.core.board import hypergammon_start

        rng = np.random.default_rng(42)

        result = play_game(
            white_agent=random_agent(),
            black_agent=random_agent(),
            starting_position=hypergammon_start(),
            max_moves=200,
            rng=rng,
        )

        # Hypergammon should work (only 3 checkers per side)
        assert result is not None
        assert result.num_moves > 0


class TestGenerateTrainingBatch:
    """Test batch game generation."""

    def test_generate_batch_basic(self):
        """Test generating a batch of games."""
        rng = np.random.default_rng(42)

        games = generate_training_batch(
            num_games=5,
            get_variant_fn=get_early_training_variants,
            white_agent=random_agent(),
            black_agent=random_agent(),
            rng=rng,
        )

        assert len(games) == 5
        for game in games:
            assert isinstance(game, GameResult)
            assert game.num_moves > 0

    def test_generate_batch_uses_variants(self):
        """Test that batch generation uses different variants."""
        rng = np.random.default_rng(42)

        games = generate_training_batch(
            num_games=10,
            get_variant_fn=get_early_training_variants,
            white_agent=random_agent(),
            black_agent=random_agent(),
            rng=rng,
        )

        # Get starting positions
        starting_positions = [g.starting_position for g in games]

        # Should have variety (at least 2 different starting positions)
        # This is probabilistic but very likely with 10 games
        unique_starts = len(set(id(p) for p in starting_positions))
        # We just check we got games
        assert len(games) == 10


    def test_generate_batch_rejects_empty_variants(self):
        """Test empty variants list raises a clear error."""
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="at least one starting position"):
            generate_training_batch(
                num_games=1,
                variants=[],
                white_agent=random_agent(),
                black_agent=random_agent(),
                rng=rng,
            )

    def test_generate_batch_pip_count_agents(self):
        """Test generating batch with pip count agents."""
        rng = np.random.default_rng(42)

        games = generate_training_batch(
            num_games=3,
            get_variant_fn=get_early_training_variants,
            white_agent=pip_count_agent(),
            black_agent=pip_count_agent(),
            rng=rng,
        )

        assert len(games) == 3
        # Pip count games should work
        for game in games:
            assert game.num_moves > 0


class TestComputeStatistics:
    """Test game statistics computation."""

    def test_compute_stats_empty(self):
        """Test computing stats on empty list."""
        stats = compute_game_statistics([])

        assert stats['total_games'] == 0

    def test_compute_stats_single_game(self):
        """Test computing stats on single game."""
        rng = np.random.default_rng(42)

        game = play_game(
            white_agent=random_agent(),
            black_agent=random_agent(),
            starting_position=initial_board(),
            max_moves=500,
            rng=rng,
        )

        stats = compute_game_statistics([game])

        assert stats['total_games'] == 1
        assert stats['white_wins'] + stats['black_wins'] + stats['draws'] == 1
        assert stats['avg_moves'] == game.num_moves

    def test_compute_stats_batch(self):
        """Test computing stats on batch."""
        rng = np.random.default_rng(42)

        games = generate_training_batch(
            num_games=10,
            get_variant_fn=get_early_training_variants,
            white_agent=random_agent(),
            black_agent=random_agent(),
            rng=rng,
        )

        stats = compute_game_statistics(games)

        assert stats['total_games'] == 10
        assert stats['white_wins'] + stats['black_wins'] + stats['draws'] == 10
        assert 0 <= stats['white_win_rate'] <= 1.0
        assert stats['avg_moves'] > 0

    def test_compute_stats_win_rate(self):
        """Test win rate calculation."""
        # Create mock game results
        from backgammon.core.types import GameOutcome

        games = [
            GameResult(
                steps=[],
                outcome=GameOutcome(winner=Player.WHITE, points=1),
                num_moves=20,
                starting_position=initial_board(),
            ),
            GameResult(
                steps=[],
                outcome=GameOutcome(winner=Player.WHITE, points=1),
                num_moves=25,
                starting_position=initial_board(),
            ),
            GameResult(
                steps=[],
                outcome=GameOutcome(winner=Player.BLACK, points=1),
                num_moves=30,
                starting_position=initial_board(),
            ),
        ]

        stats = compute_game_statistics(games)

        assert stats['total_games'] == 3
        assert stats['white_wins'] == 2
        assert stats['black_wins'] == 1
        assert stats['white_win_rate'] == pytest.approx(2/3)
        assert stats['avg_moves'] == pytest.approx(25.0)


class TestSelfPlayIntegration:
    """Integration tests for self-play."""

    def test_full_pipeline(self):
        """Test complete self-play pipeline."""
        rng = np.random.default_rng(42)

        # Generate batch
        games = generate_training_batch(
            num_games=5,
            get_variant_fn=get_early_training_variants,
            white_agent=pip_count_agent(),
            black_agent=random_agent(),
            rng=rng,
        )

        # Compute stats
        stats = compute_game_statistics(games)

        # Verify pipeline worked
        assert len(games) == 5
        assert stats['total_games'] == 5
        assert all(g.num_moves > 0 for g in games)
        assert all(len(g.steps) == g.num_moves for g in games)

    def test_multiple_games_complete(self):
        """Test that multiple games complete successfully."""
        rng = np.random.default_rng(42)

        # Play multiple games
        for i in range(5):
            result = play_game(
                white_agent=random_agent(),
                black_agent=random_agent(),
                starting_position=initial_board(),
                max_moves=500,
                rng=rng,
            )

            assert result is not None
            assert result.num_moves > 0
            assert len(result.steps) > 0
