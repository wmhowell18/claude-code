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
from backgammon.core.types import TransformerConfig
from backgammon.encoding.action_encoder import get_action_space_size
from backgammon.evaluation.agents import pip_count_agent
from backgammon.evaluation.network_agent import create_neural_agent
from backgammon.training.self_play import generate_training_batch, compute_game_statistics
from backgammon.training.replay_buffer import ReplayBuffer
from backgammon.training.losses import train_step
from backgammon.training.metrics import MetricsLogger
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

    # Model settings (transformer architecture)
    # Default: smaller model for faster iteration (colleague's recommendation)
    embed_dim: int = 128  # Start small, scale up after validation
    num_heads: int = 8
    num_layers: int = 4  # Reduced from 6 for faster training
    ff_dim: int = 512  # 4x embed_dim
    dropout_rate: float = 0.1

    # Training mode
    train_policy: bool = False  # If False, value-only training (simpler)

    # Optimizer settings
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Replay buffer settings
    replay_buffer_size: int = 100_000
    replay_buffer_min_size: int = 1_000
    training_batch_size: int = 256  # Neural network training batch size
    train_steps_per_game_batch: int = 4  # Training steps per game batch

    # Agent settings
    neural_agent_temperature: float = 0.3  # Exploration during self-play

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
    # Create transformer config
    transformer_config = TransformerConfig(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout_rate=config.dropout_rate,
        input_feature_dim=2,  # Raw encoding has 2 features per position
        use_policy_head=config.train_policy,  # Enable policy prediction based on config
        num_actions=get_action_space_size() if config.train_policy else 0,
    )

    # Initialize model
    model = BackgammonTransformer(config=transformer_config)

    # Dummy input for initialization (26 positions x 2 features = 52 dims)
    # Note: The network expects raw features, not flattened encoding
    dummy_input = jnp.zeros((1, 26, 2))
    variables = model.init(rng, dummy_input, training=False)

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
    """Main training loop with curriculum learning and self-play.

    Args:
        config: Training configuration (uses defaults if None)
    """
    if config is None:
        config = TrainingConfig()

    # Initialize random state
    rng = np.random.default_rng(config.seed)
    jax_rng = jax.random.PRNGKey(config.seed)

    # Create training state
    print("🔧 Initializing model and optimizer...")
    state = create_train_state(config, jax_rng)

    # Create replay buffer
    print("💾 Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        min_size=config.replay_buffer_min_size,
        eviction_policy='fifo',
    )

    # Create metrics logger
    print("📊 Setting up metrics logging...")
    metrics_logger = MetricsLogger(
        log_dir=config.log_dir,
        run_name="backgammon_training",
        use_tensorboard=False,  # Optional: enable if tensorboard installed
        use_wandb=False,  # Optional: enable if W&B configured
        console_interval=config.log_every_n_batches,
    )

    # Log hyperparameters
    metrics_logger.log_hyperparams(asdict(config))

    # Training phase manager
    phase_manager = TrainingPhase(config)

    # Create agents
    pip_agent = pip_count_agent()

    batch_num = 0
    total_train_steps = 0

    print(f"\n🎲 Starting training:")
    print(f"   Warmstart: {config.warmstart_games} games (pip count vs pip count)")
    print(f"   Early:     {config.early_phase_games} games (simplified variants)")
    print(f"   Mid:       {config.mid_phase_games} games (mixed variants)")
    print(f"   Late:      {config.late_phase_games} games (full complexity)")
    print(f"\n💾 Checkpoints: {config.checkpoint_dir}")
    print(f"📊 Logs: {config.log_dir}\n")

    try:
        while True:
            batch_start = time.time()

            # Get current phase
            phase_name, get_variants_fn = phase_manager.get_current_phase()
            use_warmstart = phase_manager.should_use_pip_count_warmstart()

            # Select agents for self-play
            if use_warmstart:
                # Warmstart: pip count vs pip count
                white_agent = pip_agent
                black_agent = pip_agent
            else:
                # Self-play: neural network vs itself (with exploration)
                neural_agent = create_neural_agent(
                    state=state,
                    temperature=config.neural_agent_temperature,
                    name="NeuralNet",
                )
                white_agent = neural_agent
                black_agent = neural_agent

            # Generate training batch through self-play
            games = generate_training_batch(
                num_games=config.games_per_batch,
                get_variant_fn=get_variants_fn,
                white_agent=white_agent,
                black_agent=black_agent,
                rng=rng,
            )

            # Add games to replay buffer
            for game in games:
                replay_buffer.add_game(game)

            # Compute game statistics
            stats = compute_game_statistics(games)

            # Update phase counts
            phase_manager.total_games_played += len(games)
            batch_num += 1

            # Train neural network (if buffer is ready)
            train_loss = 0.0
            train_acc = 0.0

            if replay_buffer.is_ready():
                # Run multiple training steps per game batch
                for _ in range(config.train_steps_per_game_batch):
                    # Sample training batch from replay buffer
                    train_batch = replay_buffer.sample_batch(config.training_batch_size)

                    # Generate RNG for dropout
                    jax_rng, step_rng = jax.random.split(jax_rng)

                    # Training step (gradient update)
                    state, step_metrics = train_step(state, train_batch, step_rng)

                    # Accumulate metrics
                    train_loss += float(step_metrics['total_loss'])
                    # Note: We don't have accuracy easily available from train_step
                    # Could add it if needed

                    total_train_steps += 1

                # Average over training steps
                train_loss /= config.train_steps_per_game_batch

            batch_time = time.time() - batch_start

            # Get current learning rate
            # For simplicity, just use the peak learning rate from config
            # (actual LR varies with warmup/cosine schedule)
            current_lr = config.learning_rate

            # Create metrics
            metrics = TrainingMetrics(
                phase=phase_name,
                batch_num=batch_num,
                total_games=phase_manager.total_games_played,
                white_win_rate=stats['white_win_rate'],
                avg_moves=stats['avg_moves'],
                gammons=stats['gammons'],
                backgammons=stats['backgammons'],
                loss=train_loss,
                accuracy=train_acc,
                learning_rate=current_lr,
                batch_time=batch_time,
                games_per_second=len(games) / batch_time,
            )

            # Log metrics
            metrics_logger.log_metrics({
                'loss': train_loss,
                'white_win_rate': stats['white_win_rate'],
                'avg_moves': stats['avg_moves'],
                'learning_rate': current_lr,
                'games_per_second': metrics.games_per_second,
                'replay_buffer_size': len(replay_buffer),
                'total_train_steps': total_train_steps,
            }, step=batch_num, prefix=f"{phase_name}/")

            # Console logging
            if batch_num % config.log_every_n_batches == 0:
                buffer_status = f"{len(replay_buffer)}/{replay_buffer.max_size}"
                print(f"[{phase_name:8s}] Batch {batch_num:4d} | "
                      f"Games: {phase_manager.total_games_played:5d} | "
                      f"Loss: {train_loss:.4f} | "
                      f"WR: {stats['white_win_rate']:.3f} | "
                      f"Moves: {stats['avg_moves']:.1f} | "
                      f"Buffer: {buffer_status} | "
                      f"Speed: {metrics.games_per_second:.1f} g/s")

                # Save metrics to file
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
                print(f"   Total training steps: {total_train_steps}")
                print(f"   Final loss: {train_loss:.4f}")

                # Save final checkpoint
                save_checkpoint(state, config, batch_num)

                # Save summary
                metrics_logger.save_summary({
                    'total_games': phase_manager.total_games_played,
                    'total_batches': batch_num,
                    'total_train_steps': total_train_steps,
                    'final_loss': train_loss,
                    'final_learning_rate': current_lr,
                })

                break

    finally:
        # Clean up metrics logger
        metrics_logger.close()


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
