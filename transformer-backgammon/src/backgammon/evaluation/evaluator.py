"""Periodic evaluation of model strength during training.

Measures win rate against reference agents (random, pip count) at
configurable intervals. Logs results to the metrics system so you
can track whether the model is actually improving over training.

Usage in training loop:
    evaluator = TrainingEvaluator(eval_every_n_batches=50)
    ...
    # Inside training loop:
    eval_metrics = evaluator.maybe_evaluate(batch_num, state)
    if eval_metrics:
        metrics_logger.log_metrics(eval_metrics, step=batch_num, prefix="eval/")
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from flax.training import train_state

from backgammon.core.board import initial_board
from backgammon.core.types import Player
from backgammon.evaluation.agents import Agent, random_agent, pip_count_agent
from backgammon.evaluation.network_agent import create_neural_agent


@dataclass
class EvalConfig:
    """Configuration for periodic evaluation."""

    # How often to evaluate (in training batches)
    eval_every_n_batches: int = 50

    # Number of games per evaluation (per opponent)
    games_vs_random: int = 50
    games_vs_pip_count: int = 50

    # Whether to also evaluate 1-ply (slower but more informative)
    eval_1ply: bool = False
    games_1ply_vs_random: int = 20

    # Random seed for reproducible evaluations
    seed: int = 7777


def evaluate_vs_opponent(
    agent: Agent,
    opponent: Agent,
    num_games: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Play games and return detailed win statistics.

    Plays half as white, half as black to eliminate first-move bias.

    Args:
        agent: The agent being evaluated.
        opponent: The reference opponent.
        num_games: Total games to play (split evenly between colors).
        rng: Random number generator.

    Returns:
        Dictionary with win_rate, gammon_rate, avg_moves, etc.
    """
    # Lazy import to avoid circular dependency (evaluation <-> training)
    from backgammon.training.self_play import play_game

    wins = 0
    gammons = 0
    backgammons = 0
    losses = 0
    loss_gammons = 0
    total = 0
    total_moves = 0

    half = num_games // 2

    for i in range(num_games):
        if i < half:
            # Agent as white
            result = play_game(agent, opponent, initial_board(), rng=rng)
            agent_color = Player.WHITE
        else:
            # Agent as black
            result = play_game(opponent, agent, initial_board(), rng=rng)
            agent_color = Player.BLACK

        total += 1
        total_moves += result.num_moves

        if result.outcome is not None:
            if result.outcome.winner == agent_color:
                wins += 1
                if result.outcome.points >= 2:
                    gammons += 1
                if result.outcome.points >= 3:
                    backgammons += 1
            else:
                losses += 1
                if result.outcome.points >= 2:
                    loss_gammons += 1

    win_rate = wins / total if total > 0 else 0.0
    return {
        "win_rate": win_rate,
        "gammon_rate": gammons / total if total > 0 else 0.0,
        "backgammon_rate": backgammons / total if total > 0 else 0.0,
        "loss_gammon_rate": loss_gammons / total if total > 0 else 0.0,
        "avg_moves": total_moves / total if total > 0 else 0.0,
        "games_played": total,
    }


class TrainingEvaluator:
    """Periodically evaluates model strength during training.

    Call maybe_evaluate() each batch — it returns metrics only when
    it's time to evaluate, otherwise returns None.

    Tracks history so you can see strength progression.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()

        # Create reference agents (fixed seeds for reproducibility)
        self._random_agent = random_agent(seed=self.config.seed)
        self._pip_count_agent = pip_count_agent()

        # History of evaluations
        self.history = []

    def maybe_evaluate(
        self,
        batch_num: int,
        state: train_state.TrainState,
    ) -> Optional[Dict[str, float]]:
        """Evaluate if it's time, otherwise return None.

        Args:
            batch_num: Current training batch number.
            state: Current model training state.

        Returns:
            Dictionary of eval metrics, or None if not time yet.
        """
        if batch_num == 0:
            return None
        if batch_num % self.config.eval_every_n_batches != 0:
            return None

        return self.evaluate(state, batch_num)

    def evaluate(
        self,
        state: train_state.TrainState,
        batch_num: int = 0,
    ) -> Dict[str, float]:
        """Run full evaluation suite against reference agents.

        Args:
            state: Current model training state.
            batch_num: Current batch number (for logging).

        Returns:
            Dictionary of all evaluation metrics.
        """
        t0 = time.time()
        rng = np.random.default_rng(self.config.seed)

        # Create greedy neural agent (temperature=0 for deterministic eval)
        neural_agent = create_neural_agent(state, temperature=0.0, name="Eval-0ply")

        metrics = {}

        # Evaluate vs random
        if self.config.games_vs_random > 0:
            vs_random = evaluate_vs_opponent(
                neural_agent, self._random_agent,
                self.config.games_vs_random, rng,
            )
            metrics["vs_random_wr"] = vs_random["win_rate"]
            metrics["vs_random_gammon_rate"] = vs_random["gammon_rate"]
            metrics["vs_random_avg_moves"] = vs_random["avg_moves"]

        # Evaluate vs pip count
        if self.config.games_vs_pip_count > 0:
            vs_pip = evaluate_vs_opponent(
                neural_agent, self._pip_count_agent,
                self.config.games_vs_pip_count, rng,
            )
            metrics["vs_pip_wr"] = vs_pip["win_rate"]
            metrics["vs_pip_gammon_rate"] = vs_pip["gammon_rate"]
            metrics["vs_pip_avg_moves"] = vs_pip["avg_moves"]

        # Evaluate 1-ply vs random (optional, slower)
        if self.config.eval_1ply and self.config.games_1ply_vs_random > 0:
            neural_1ply = create_neural_agent(
                state, temperature=0.0, name="Eval-1ply", ply=1,
            )
            vs_random_1ply = evaluate_vs_opponent(
                neural_1ply, self._random_agent,
                self.config.games_1ply_vs_random, rng,
            )
            metrics["vs_random_1ply_wr"] = vs_random_1ply["win_rate"]

        elapsed = time.time() - t0
        metrics["eval_time_s"] = elapsed

        # Store in history
        self.history.append({
            "batch_num": batch_num,
            **metrics,
        })

        return metrics

    def print_summary(self):
        """Print a summary table of evaluation history."""
        if not self.history:
            print("No evaluations yet.")
            return

        print()
        print("Evaluation History:")
        print("-" * 70)
        header = f"{'Batch':>6s}  {'vs Random':>10s}  {'vs PipCount':>11s}"
        if any("vs_random_1ply_wr" in h for h in self.history):
            header += f"  {'1-ply vs Rnd':>12s}"
        header += f"  {'Time':>6s}"
        print(header)
        print("-" * 70)

        for h in self.history:
            line = f"{h['batch_num']:6d}"
            line += f"  {h.get('vs_random_wr', 0):10.1%}"
            line += f"  {h.get('vs_pip_wr', 0):11.1%}"
            if "vs_random_1ply_wr" in h:
                line += f"  {h['vs_random_1ply_wr']:12.1%}"
            line += f"  {h.get('eval_time_s', 0):5.1f}s"
            print(line)
        print("-" * 70)

        # Show trend
        if len(self.history) >= 2:
            first_wr = self.history[0].get("vs_random_wr", 0)
            last_wr = self.history[-1].get("vs_random_wr", 0)
            delta = last_wr - first_wr
            direction = "improving" if delta > 0.02 else "declining" if delta < -0.02 else "stable"
            print(f"Trend: {first_wr:.1%} -> {last_wr:.1%} ({direction})")
