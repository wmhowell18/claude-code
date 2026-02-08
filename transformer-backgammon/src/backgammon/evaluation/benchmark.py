"""Benchmarking and evaluation infrastructure for backgammon AI.

Provides:
- Win rate evaluation against baseline agents
- Benchmark position suite with known properties
- Equity error metrics
- Training progress tracking via periodic evaluation callbacks
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from flax.training import train_state

from backgammon.core.board import (
    Board,
    initial_board,
    apply_move,
    generate_legal_moves,
    is_game_over,
    pip_count,
)
from backgammon.core.types import Player, GameOutcome
from backgammon.evaluation.agents import Agent, random_agent, pip_count_agent
from backgammon.evaluation.network_agent import create_neural_agent
from backgammon.training.self_play import play_game


# ==============================================================================
# WIN RATE EVALUATION
# ==============================================================================


@dataclass
class EvalResult:
    """Result of evaluating an agent against a baseline."""
    agent_name: str
    opponent_name: str
    num_games: int
    wins: int
    losses: int
    draws: int
    gammons_won: int
    gammons_lost: int
    backgammons_won: int
    backgammons_lost: int
    avg_game_length: float

    @property
    def win_rate(self) -> float:
        """Win rate as fraction [0, 1]."""
        if self.num_games == 0:
            return 0.0
        return self.wins / self.num_games

    @property
    def ppg(self) -> float:
        """Points per game (positive = agent winning more points)."""
        if self.num_games == 0:
            return 0.0
        points_won = self.wins + self.gammons_won + self.backgammons_won * 2
        points_lost = self.losses + self.gammons_lost + self.backgammons_lost * 2
        return (points_won - points_lost) / self.num_games

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"{self.agent_name} vs {self.opponent_name}: "
            f"{self.win_rate:.1%} ({self.wins}W/{self.losses}L/{self.draws}D "
            f"in {self.num_games}g, "
            f"gammons: {self.gammons_won}/{self.gammons_lost}, "
            f"ppg: {self.ppg:+.3f}, "
            f"avg len: {self.avg_game_length:.0f})"
        )


def evaluate_agents(
    agent: Agent,
    opponent: Agent,
    num_games: int = 100,
    starting_position: Optional[Board] = None,
    rng: Optional[np.random.Generator] = None,
) -> EvalResult:
    """Evaluate an agent against an opponent over many games.

    Plays half the games as white and half as black to eliminate
    first-move advantage bias.

    Args:
        agent: Agent to evaluate.
        opponent: Opponent agent.
        num_games: Total number of games (split evenly between colors).
        starting_position: Starting position (defaults to standard).
        rng: Random number generator.

    Returns:
        EvalResult with detailed statistics.
    """
    if rng is None:
        rng = np.random.default_rng()
    if starting_position is None:
        starting_position = initial_board()

    wins = 0
    losses = 0
    draws = 0
    gammons_won = 0
    gammons_lost = 0
    backgammons_won = 0
    backgammons_lost = 0
    total_moves = 0
    games_per_side = num_games // 2

    for i in range(num_games):
        # Alternate colors: even = agent is white, odd = agent is black
        if i < games_per_side:
            result = play_game(agent, opponent, starting_position.copy(), rng=rng)
            agent_color = Player.WHITE
        else:
            result = play_game(opponent, agent, starting_position.copy(), rng=rng)
            agent_color = Player.BLACK

        total_moves += result.num_moves

        if result.outcome is None:
            draws += 1
            continue

        agent_won = result.outcome.winner == agent_color
        pts = result.outcome.points

        if agent_won:
            wins += 1
            if pts == 2:
                gammons_won += 1
            elif pts == 3:
                backgammons_won += 1
        else:
            losses += 1
            if pts == 2:
                gammons_lost += 1
            elif pts == 3:
                backgammons_lost += 1

    return EvalResult(
        agent_name=agent.name,
        opponent_name=opponent.name,
        num_games=num_games,
        wins=wins,
        losses=losses,
        draws=draws,
        gammons_won=gammons_won,
        gammons_lost=gammons_lost,
        backgammons_won=backgammons_won,
        backgammons_lost=backgammons_lost,
        avg_game_length=total_moves / max(num_games, 1),
    )


def evaluate_against_baselines(
    state: train_state.TrainState,
    num_games: int = 100,
    ply: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, EvalResult]:
    """Evaluate a neural network agent against standard baselines.

    Runs evaluation against:
    - Random agent
    - Pip count agent

    Args:
        state: Network training state.
        num_games: Games per opponent (split by color).
        ply: Search depth for the neural agent (0, 1, or 2).
        rng: Random number generator.

    Returns:
        Dict mapping opponent name to EvalResult.
    """
    if rng is None:
        rng = np.random.default_rng()

    agent = create_neural_agent(state, temperature=0.0, name=f"Neural-{ply}ply", ply=ply)

    results = {}
    for opponent in [random_agent(seed=None), pip_count_agent()]:
        result = evaluate_agents(agent, opponent, num_games, rng=rng)
        results[opponent.name] = result

    return results


# ==============================================================================
# BENCHMARK POSITIONS
# ==============================================================================


@dataclass
class BenchmarkPosition:
    """A benchmark position with known properties.

    Attributes:
        name: Descriptive name for the position.
        board: The board state.
        category: Position category (opening, contact, race, bearoff, etc.).
        description: What makes this position interesting.
        expected_winner: Expected winner if known (None if unclear).
        expected_equity: Approximate equity if known (from reference engine).
    """
    name: str
    board: Board
    category: str
    description: str
    expected_winner: Optional[Player] = None
    expected_equity: Optional[float] = None


def _make_board(
    white_points: Dict[int, int],
    black_points: Dict[int, int],
    player_to_move: Player = Player.WHITE,
) -> Board:
    """Helper to create a board from point->count dicts.

    Args:
        white_points: {point: count} for white checkers.
        black_points: {point: count} for black checkers.
        player_to_move: Who moves next.

    Returns:
        Board state.
    """
    board = Board()
    for pt, count in white_points.items():
        board.set_checkers(Player.WHITE, pt, count)
    for pt, count in black_points.items():
        board.set_checkers(Player.BLACK, pt, count)
    board.player_to_move = player_to_move
    return board


def get_benchmark_positions() -> List[BenchmarkPosition]:
    """Get a curated suite of benchmark positions.

    These positions cover different game phases and strategic themes.
    They can be used to track whether the network is learning
    correct positional evaluation.

    Returns:
        List of BenchmarkPosition objects.
    """
    positions = []

    # --- Opening positions ---
    positions.append(BenchmarkPosition(
        name="standard_opening",
        board=initial_board(),
        category="opening",
        description="Standard starting position. Roughly equal.",
        expected_equity=0.0,
    ))

    # --- Race positions ---
    # Simple race: white is slightly ahead in pip count
    positions.append(BenchmarkPosition(
        name="race_white_ahead",
        board=_make_board(
            white_points={6: 5, 5: 4, 4: 3, 3: 2, 2: 1},
            black_points={19: 5, 20: 4, 21: 3, 22: 2, 23: 1},
        ),
        category="race",
        description="Pure race, white is 3 pips ahead (63 vs 66).",
        expected_winner=Player.WHITE,
    ))

    # Even race position
    positions.append(BenchmarkPosition(
        name="race_even",
        board=_make_board(
            white_points={6: 5, 5: 5, 4: 5},
            black_points={19: 5, 20: 5, 21: 5},
        ),
        category="race",
        description="Symmetric pure race, equal pip counts.",
        expected_equity=0.0,
    ))

    # --- Bearoff positions ---
    # White about to bear off, black far behind
    positions.append(BenchmarkPosition(
        name="bearoff_winning",
        board=_make_board(
            white_points={1: 3, 2: 3, 3: 3, 25: 6},  # 9 on board, 6 off
            black_points={19: 5, 20: 5, 21: 5},
        ),
        category="bearoff",
        description="White bearing off, well ahead. Strong white advantage.",
        expected_winner=Player.WHITE,
    ))

    # Close bearoff
    positions.append(BenchmarkPosition(
        name="bearoff_close",
        board=_make_board(
            white_points={1: 2, 2: 2, 3: 2, 25: 9},  # 6 left
            black_points={24: 2, 23: 2, 22: 2, 0: 9},  # 6 left (using 0 for off since black bears off high)
        ),
        category="bearoff",
        description="Both sides nearly done bearing off. Close contest.",
    ))

    # --- Contact positions ---
    # Back game: white has anchors deep in black's home
    positions.append(BenchmarkPosition(
        name="contact_back_game",
        board=_make_board(
            white_points={24: 2, 23: 2, 6: 5, 5: 3, 4: 3},
            black_points={1: 2, 12: 5, 17: 3, 19: 5},
        ),
        category="contact",
        description="White playing a back game with two deep anchors.",
    ))

    # Blitz position: one side attacking aggressively
    positions.append(BenchmarkPosition(
        name="contact_blitz",
        board=_make_board(
            white_points={6: 4, 5: 3, 4: 3, 3: 3, 8: 2},
            black_points={0: 2, 19: 5, 20: 3, 21: 3, 12: 2},  # 2 on bar
        ),
        category="contact",
        description="White blitzing: strong home board, black has 2 on bar.",
        expected_winner=Player.WHITE,
    ))

    # --- Prime positions ---
    positions.append(BenchmarkPosition(
        name="prime_6point",
        board=_make_board(
            white_points={6: 3, 5: 3, 4: 2, 3: 2, 2: 2, 1: 3},
            black_points={24: 2, 19: 5, 12: 5, 17: 3},
        ),
        category="prime",
        description="White has a full 6-point prime. Black trapped.",
        expected_winner=Player.WHITE,
    ))

    # --- Gammon threat positions ---
    positions.append(BenchmarkPosition(
        name="gammon_threat",
        board=_make_board(
            white_points={6: 5, 5: 3, 4: 2, 3: 3, 25: 2},  # white has 2 off
            black_points={1: 3, 2: 2, 12: 5, 17: 3, 24: 2},  # black hasn't escaped
        ),
        category="gammon",
        description="White may gammon: bearing off while black stuck in white's home.",
        expected_winner=Player.WHITE,
    ))

    # --- Anchor positions ---
    positions.append(BenchmarkPosition(
        name="mutual_holding",
        board=_make_board(
            white_points={24: 2, 13: 5, 8: 3, 6: 5},
            black_points={1: 2, 12: 5, 17: 3, 19: 5},
        ),
        category="contact",
        description="Mutual holding game. Both sides have deep anchors.",
        expected_equity=0.0,
    ))

    return positions


# ==============================================================================
# EQUITY ERROR METRICS
# ==============================================================================


def compute_equity_error(
    state: train_state.TrainState,
    positions: List[BenchmarkPosition],
) -> Dict[str, float]:
    """Compute equity prediction error on benchmark positions.

    For positions with known expected_equity, computes the network's
    equity estimate and reports error metrics.

    Args:
        state: Network training state.
        positions: Benchmark positions to evaluate.

    Returns:
        Dict with error metrics:
            - 'mae': Mean absolute error
            - 'rmse': Root mean squared error
            - 'max_error': Maximum absolute error
            - 'n_evaluated': Number of positions with known equity
            - 'per_position': List of (name, predicted, expected, error)
    """
    from backgammon.evaluation.search import _batch_evaluate, _equity_to_value
    from backgammon.encoding.encoder import raw_encoding_config

    encoding_config = raw_encoding_config()

    # Filter to positions with known equity
    known = [(p, p.expected_equity) for p in positions if p.expected_equity is not None]

    if not known:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'max_error': 0.0,
            'n_evaluated': 0,
            'per_position': [],
        }

    boards = [p.board for p, _ in known]
    expected = np.array([eq for _, eq in known])

    # Get network predictions
    predicted = _batch_evaluate(state, boards, encoding_config)

    errors = np.abs(predicted - expected)

    per_position = []
    for i, (pos, exp_eq) in enumerate(known):
        per_position.append((
            pos.name,
            float(predicted[i]),
            exp_eq,
            float(errors[i]),
        ))

    return {
        'mae': float(np.mean(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'max_error': float(np.max(errors)),
        'n_evaluated': len(known),
        'per_position': per_position,
    }


# ==============================================================================
# TRAINING EVALUATION CALLBACK
# ==============================================================================


@dataclass
class EvalHistory:
    """Tracks evaluation results over training."""
    entries: List[Dict] = field(default_factory=list)

    def add(self, step: int, games_played: int, results: Dict[str, EvalResult],
            equity_errors: Optional[Dict[str, float]] = None):
        """Record evaluation results at a training checkpoint.

        Args:
            step: Training step number.
            games_played: Total games played so far.
            results: Win rate results per opponent.
            equity_errors: Equity error metrics (optional).
        """
        entry = {
            'step': step,
            'games_played': games_played,
        }
        for opp_name, result in results.items():
            entry[f'wr_vs_{opp_name}'] = result.win_rate
            entry[f'ppg_vs_{opp_name}'] = result.ppg
        if equity_errors:
            entry['equity_mae'] = equity_errors.get('mae', 0.0)
            entry['equity_rmse'] = equity_errors.get('rmse', 0.0)
        self.entries.append(entry)

    def summary(self) -> str:
        """Multi-line summary of evaluation history."""
        if not self.entries:
            return "No evaluation history."

        lines = ["Step  | Games  | vs Random | vs PipCount | Eq MAE"]
        lines.append("-" * 58)

        for e in self.entries:
            wr_rand = e.get('wr_vs_Random', 0.0)
            wr_pip = e.get('wr_vs_PipCount', 0.0)
            eq_mae = e.get('equity_mae', 0.0)
            lines.append(
                f"{e['step']:5d} | {e['games_played']:6d} | "
                f"{wr_rand:8.1%} | {wr_pip:10.1%} | {eq_mae:.4f}"
            )

        return "\n".join(lines)


def run_evaluation_checkpoint(
    state: train_state.TrainState,
    step: int,
    games_played: int,
    eval_history: EvalHistory,
    num_eval_games: int = 50,
    ply: int = 0,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict[str, EvalResult]:
    """Run a full evaluation checkpoint during training.

    Evaluates against baselines, computes equity errors on benchmark
    positions, and records results to the evaluation history.

    Args:
        state: Current network training state.
        step: Training step number.
        games_played: Total games played.
        eval_history: History tracker to record results.
        num_eval_games: Games per opponent for win rate evaluation.
        ply: Search depth for evaluation.
        rng: Random number generator.
        verbose: Print results to console.

    Returns:
        Dict of EvalResult per opponent.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Win rate evaluation
    results = evaluate_against_baselines(
        state, num_games=num_eval_games, ply=ply, rng=rng
    )

    # Equity error on benchmark positions
    positions = get_benchmark_positions()
    equity_errors = compute_equity_error(state, positions)

    # Record to history
    eval_history.add(step, games_played, results, equity_errors)

    if verbose:
        print(f"\n--- Evaluation at step {step} ({games_played} games) ---")
        for result in results.values():
            print(f"  {result.summary()}")
        if equity_errors['n_evaluated'] > 0:
            print(f"  Equity error (MAE): {equity_errors['mae']:.4f} "
                  f"(RMSE: {equity_errors['rmse']:.4f}, "
                  f"max: {equity_errors['max_error']:.4f})")
            for name, pred, expected, err in equity_errors['per_position']:
                print(f"    {name}: predicted={pred:+.3f} expected={expected:+.3f} err={err:.3f}")
        print()

    return results
