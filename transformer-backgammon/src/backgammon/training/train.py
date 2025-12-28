"""Main training loop with curriculum learning and checkpointing.

Implements:
- Curriculum learning (early → mid → late training phases)
- Warmstart with pip count agent
- Self-play data generation
- Checkpointing and metrics logging
- GPU-optimized JAX/Flax training
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state, checkpoints

from backgammon.core.board import (
    get_early_training_variants,
    get_mid_training_variants,
    get_late_training_variants,
)
from backgammon.evaluation.agents import pip_count_agent
from backgammon.training.self_play import generate_training_batch, compute_game_statistics
from backgammon.network.network import BackgammonTransformer


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Training phases (curriculum learning)
    warmstart_games: int = 500  # Pip count vs pip count
    early_phase_games: int = 1000  # Simplified variants dominant
    mid_phase_games: int = 3000  # Mixed variants
    late_phase_games: int = 10000  # Full complexity

    # Batch and checkpoint settings
    games_per_batch: int = 32
    checkpoint_every_n_batches: int = 100
    log_every_n_batches: int = 10

    # Model settings
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1

    # Optimizer settings
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Random seed
    seed: int = 42


@dataclass
class TrainingMetrics:
    """Metrics for a training batch."""
    phase: str
    batch_num: int
    total_games: int

    # Game statistics
    white_win_rate: float
    avg_moves: float
    gammons: int
    backgammons: int

    # Training metrics
    loss: float
    accuracy: float
    learning_rate: float

    # Timing
    batch_time: float
    games_per_second: float


class TrainingPhase:
    """Training phase manager."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.total_games_played = 0

    def get_current_phase(self) -> Tuple[str, callable]:
        """Get current training phase and variant function.

        Returns:
            Tuple of (phase_name, variant_function)
        """
        n = self.total_games_played

        if n < self.config.warmstart_games:
            return "warmstart", get_early_training_variants
        elif n < self.config.warmstart_games + self.config.early_phase_games:
            return "early", get_early_training_variants
        elif n < (self.config.warmstart_games + self.config.early_phase_games +
                  self.config.mid_phase_games):
            return "mid", get_mid_training_variants
        else:
            return "late", get_late_training_variants

    def should_use_pip_count_warmstart(self) -> bool:
        """Check if we should use pip count agents (warmstart phase)."""
        return self.total_games_played < self.config.warmstart_games


def create_train_state(config: TrainingConfig, rng: jax.random.PRNGKey) -> train_state.TrainState:
    """Create initial training state with model and optimizer.

    Args:
        config: Training configuration
        rng: JAX random key

    Returns:
        Initialized training state
    """
    # Initialize model
    model = BackgammonTransformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    )

    # Dummy input for initialization (26-dim board encoding)
    dummy_input = jnp.zeros((1, 26))
    variables = model.init(rng, dummy_input, train=False)

    # Learning rate schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=100000,  # Total training steps
    )

    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=schedule),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
    )


def save_checkpoint(state: train_state.TrainState, config: TrainingConfig, step: int):
    """Save training checkpoint.

    Args:
        state: Current training state
        config: Training configuration
        step: Current step number
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoints.save_checkpoint(
        ckpt_dir=str(checkpoint_dir),
        target=state,
        step=step,
        keep=5,  # Keep last 5 checkpoints
    )


def save_metrics(metrics: TrainingMetrics, config: TrainingConfig):
    """Save training metrics to log file.

    Args:
        metrics: Metrics to save
        config: Training configuration
    """
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training_log.jsonl"

    with open(log_file, 'a') as f:
        f.write(json.dumps(asdict(metrics)) + '\n')


def train(config: Optional[TrainingConfig] = None):
    """Main training loop.

    Args:
        config: Training configuration (uses defaults if None)
    """
    if config is None:
        config = TrainingConfig()

    # Initialize random state
    rng = np.random.default_rng(config.seed)
    jax_rng = jax.random.PRNGKey(config.seed)

    # Create training state
    state = create_train_state(config, jax_rng)

    # Training phase manager
    phase_manager = TrainingPhase(config)

    # Create agents
    pip_agent = pip_count_agent()
    # TODO: Create neural network agent wrapper

    batch_num = 0

    print(f"🎲 Starting training with {config.warmstart_games} warmstart games")
    print(f"📊 Early: {config.early_phase_games}, Mid: {config.mid_phase_games}, Late: {config.late_phase_games}")
    print(f"💾 Checkpoints: {config.checkpoint_dir}, Logs: {config.log_dir}")
    print()

    while True:
        batch_start = time.time()

        # Get current phase
        phase_name, get_variants_fn = phase_manager.get_current_phase()
        use_warmstart = phase_manager.should_use_pip_count_warmstart()

        # Select agents
        if use_warmstart:
            white_agent = pip_agent
            black_agent = pip_agent
        else:
            # TODO: Use neural network agents
            white_agent = pip_agent  # Placeholder
            black_agent = pip_agent

        # Generate training batch
        games = generate_training_batch(
            num_games=config.games_per_batch,
            get_variant_fn=get_variants_fn,
            white_agent=white_agent,
            black_agent=black_agent,
            rng=rng,
        )

        # Compute statistics
        stats = compute_game_statistics(games)

        # Update counts
        phase_manager.total_games_played += len(games)
        batch_num += 1

        # TODO: Train neural network on batch
        # For now, just track statistics

        batch_time = time.time() - batch_start

        # Create metrics
        metrics = TrainingMetrics(
            phase=phase_name,
            batch_num=batch_num,
            total_games=phase_manager.total_games_played,
            white_win_rate=stats['white_win_rate'],
            avg_moves=stats['avg_moves'],
            gammons=stats['gammons'],
            backgammons=stats['backgammons'],
            loss=0.0,  # TODO: Real loss
            accuracy=0.0,  # TODO: Real accuracy
            learning_rate=config.learning_rate,  # TODO: Current LR
            batch_time=batch_time,
            games_per_second=len(games) / batch_time,
        )

        # Logging
        if batch_num % config.log_every_n_batches == 0:
            print(f"[{phase_name:8s}] Batch {batch_num:4d} | "
                  f"Games: {phase_manager.total_games_played:5d} | "
                  f"WR: {stats['white_win_rate']:.3f} | "
                  f"Moves: {stats['avg_moves']:.1f} | "
                  f"Speed: {metrics.games_per_second:.1f} games/s")

            save_metrics(metrics, config)

        # Checkpointing
        if batch_num % config.checkpoint_every_n_batches == 0:
            save_checkpoint(state, config, batch_num)
            print(f"💾 Checkpoint saved at batch {batch_num}")

        # Check if training complete
        total_target = (config.warmstart_games + config.early_phase_games +
                       config.mid_phase_games + config.late_phase_games)
        if phase_manager.total_games_played >= total_target:
            print(f"\n✅ Training complete! {phase_manager.total_games_played} games played")
            save_checkpoint(state, config, batch_num)
            break


if __name__ == "__main__":
    # Run training with default config
    config = TrainingConfig(
        warmstart_games=100,  # Small for testing
        early_phase_games=200,
        mid_phase_games=300,
        late_phase_games=400,
        games_per_batch=16,
        checkpoint_every_n_batches=50,
        log_every_n_batches=5,
    )

    train(config)
