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
from backgammon.evaluation.agents import pip_count_agent, greedy_pip_count_agent
from backgammon.evaluation.network_agent import create_neural_agent
from backgammon.training.self_play import (
    generate_training_batch,
    compute_game_statistics,
    play_games_batched,
)
from backgammon.training.replay_buffer import ReplayBuffer
from backgammon.training.losses import train_step, compute_metrics
from backgammon.training.metrics import MetricsLogger
from backgammon.network.network import BackgammonTransformer
from backgammon.evaluation.benchmark import (
    EvalHistory,
    run_evaluation_checkpoint,
)


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

    # Compute dtype for forward pass (None=float32, 'bfloat16' for TPU)
    compute_dtype: Optional[str] = None

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
    temperature_start: float = 1.0  # Starting temperature for exploration schedule
    temperature_end: float = 0.1  # Final temperature for exploration schedule
    use_temperature_schedule: bool = True  # Enable temperature decay over training

    # TD(lambda) settings
    td_lambda: float = 0.7  # TD(lambda) parameter (0=TD(0), 1=MC, 0.7=recommended)
    use_td_lambda: bool = True  # Enable TD(lambda) targets (requires neural self-play)

    # Position weighting
    use_position_weighting: bool = True  # Weight positions by importance for sampling

    # Validation and early stopping
    validation_fraction: float = 0.1  # Fraction of games used for validation
    early_stopping_patience: int = 5  # Stop after N eval checkpoints without improvement
    use_early_stopping: bool = True  # Enable early stopping

    # Evaluation settings
    eval_every_n_batches: int = 50  # Run evaluation checkpoint every N batches
    eval_num_games: int = 50  # Games per opponent during evaluation
    eval_ply: int = 0  # Search depth for evaluation (0 is fast, 1 is slow)

    # Game length limits
    max_moves: int = 200  # Max moves per game before declaring draw

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Random seed
    seed: int = 42


def v6e_quick_training_config() -> TrainingConfig:
    """Quick training config optimized for TPU v6e-1 (single chip).

    Uses small model (~500K params) with bfloat16 for efficient TPU usage.
    Designed for initial validation runs that cost minimal TPU credits.
    Total: ~2,500 games.
    """
    return TrainingConfig(
        # Small curriculum for quick validation
        warmstart_games=200,
        early_phase_games=500,
        mid_phase_games=800,
        late_phase_games=1000,

        # Small model (~500K params) — fits easily in 16GB HBM
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout_rate=0.1,

        # Larger game batch to better amortize TPU dispatch overhead
        # during batched inference and increase games/sec on v6e.
        training_batch_size=512,
        games_per_batch=128,
        train_steps_per_game_batch=4,

        # Value-only training (simpler, faster for validation)
        train_policy=False,

        # bfloat16 for ~2x speedup on v6e
        compute_dtype='bfloat16',

        # Optimizer
        learning_rate=3e-4,
        warmup_steps=200,
        max_grad_norm=1.0,

        # Replay buffer (smaller for quick run)
        replay_buffer_size=50_000,
        replay_buffer_min_size=500,

        # More frequent checkpoints for short run
        checkpoint_every_n_batches=50,
        log_every_n_batches=5,
        eval_every_n_batches=25,
        eval_num_games=30,

        # TD(lambda) for better targets
        use_td_lambda=True,
        td_lambda=0.7,

        # Paths
        checkpoint_dir="checkpoints_v6e",
        log_dir="logs_v6e",

        seed=42,
    )


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

    def get_current_temperature(self) -> float:
        """Get current exploration temperature based on training progress.

        Linearly decays from temperature_start to temperature_end over
        the course of training. During warmstart, returns temperature_start.

        Returns:
            Current temperature value.
        """
        if not self.config.use_temperature_schedule:
            return self.config.neural_agent_temperature

        total_target = (self.config.warmstart_games + self.config.early_phase_games +
                        self.config.mid_phase_games + self.config.late_phase_games)
        # Progress through non-warmstart training
        neural_games = max(0, self.total_games_played - self.config.warmstart_games)
        neural_total = total_target - self.config.warmstart_games
        if neural_total <= 0:
            return self.config.temperature_start

        progress = min(1.0, neural_games / neural_total)
        return self.config.temperature_start + progress * (
            self.config.temperature_end - self.config.temperature_start
        )


def create_train_state(config: TrainingConfig, rng: jax.random.PRNGKey) -> train_state.TrainState:
    """Create initial training state with model and optimizer.

    Args:
        config: Training configuration
        rng: JAX random key

    Returns:
        Initialized training state
    """
    # Resolve compute dtype
    dtype = None
    if config.compute_dtype == 'bfloat16':
        dtype = jnp.bfloat16

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
        dtype=dtype,
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


def save_checkpoint(
    state: train_state.TrainState,
    config: TrainingConfig,
    step: int,
    is_best: bool = False,
):
    """Save training checkpoint.

    Args:
        state: Current training state
        config: Training configuration
        step: Current step number
        is_best: If True, also save as best model checkpoint
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoints.save_checkpoint(
        ckpt_dir=str(checkpoint_dir),
        target=state,
        step=step,
        keep=5,  # Keep last 5 checkpoints
        overwrite=True,  # Allow overwriting existing checkpoints
    )

    if is_best:
        best_dir = checkpoint_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        checkpoints.save_checkpoint(
            ckpt_dir=str(best_dir),
            target=state,
            step=step,
            keep=1,  # Only keep the single best
            overwrite=True,
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

    # Create replay buffer (training)
    print("💾 Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        min_size=config.replay_buffer_min_size,
        eviction_policy='fifo',
        use_position_weighting=config.use_position_weighting,
    )

    # Create validation buffer (for early stopping)
    val_buffer = ReplayBuffer(
        max_size=max(1000, config.replay_buffer_size // 10),
        min_size=100,
        eviction_policy='fifo',
    ) if config.validation_fraction > 0 else None

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

    # Evaluation history
    eval_history = EvalHistory()
    eval_rng = np.random.default_rng(config.seed + 1000)

    # Early stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_step = 0

    # LR plateau detection state
    lr_plateau_best = float('inf')
    lr_plateau_counter = 0
    lr_plateau_patience = 3  # Warn after 3 eval checkpoints without improvement

    # Create agents
    # Use greedy pip count for warmstart (fast: pip count only, no heuristic scans)
    # Full pip count agent is too slow for bulk game generation
    warmstart_agent = greedy_pip_count_agent()
    pip_agent = pip_count_agent()  # Used for evaluation

    batch_num = 0
    total_train_steps = 0

    # Cache generated starting variants per phase to avoid rebuilding
    # identical Board objects every batch (helps TPU game throughput).
    phase_variants_cache = {}

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

            # Generate training batch through self-play
            record_values = config.use_td_lambda and not use_warmstart

            if phase_name not in phase_variants_cache:
                phase_variants_cache[phase_name] = get_variants_fn()
            variants = phase_variants_cache[phase_name]

            if use_warmstart:
                # Warmstart: greedy pip count vs greedy pip count (fast, sequential)
                games = generate_training_batch(
                    num_games=config.games_per_batch,
                    white_agent=warmstart_agent,
                    black_agent=warmstart_agent,
                    variants=variants,
                    rng=rng,
                    record_value_estimates=False,
                    max_moves=config.max_moves,
                )
            else:
                # Neural self-play: batched simulation (dramatically faster)
                # Plays all games simultaneously with JIT-compiled batched
                # inference, reducing ~7800 TPU dispatches to ~120 per batch.
                current_temp = phase_manager.get_current_temperature()
                games = play_games_batched(
                    num_games=config.games_per_batch,
                    state=state,
                    variants=variants,
                    temperature=current_temp,
                    max_moves=config.max_moves,
                    rng=rng,
                    record_value_estimates=record_values,
                )

            # Add games to replay buffer (with TD(lambda) targets if enabled)
            # Split between training and validation buffers
            td_lambda_param = config.td_lambda if record_values else None
            for game in games:
                if (val_buffer is not None and
                        rng.random() < config.validation_fraction):
                    val_buffer.add_game(game, td_lambda=td_lambda_param)
                else:
                    replay_buffer.add_game(game, td_lambda=td_lambda_param)

            # Compute game statistics
            stats = compute_game_statistics(games)

            # Update phase counts
            phase_manager.total_games_played += len(games)
            batch_num += 1

            # Train neural network (if buffer is ready)
            train_loss = 0.0
            train_acc = 0.0
            max_grad_norm_seen = 0.0
            grad_clips = 0

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

                    # Track gradient clipping diagnostics
                    gn = float(step_metrics.get('grad_norm', 0.0))
                    max_grad_norm_seen = max(max_grad_norm_seen, gn)
                    if gn > config.max_grad_norm:
                        grad_clips += 1

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
                'max_grad_norm': max_grad_norm_seen,
                'grad_clips': grad_clips,
            }, step=batch_num, prefix=f"{phase_name}/")

            # Console logging
            if batch_num % config.log_every_n_batches == 0:
                buffer_status = f"{len(replay_buffer)}/{replay_buffer.max_size}"
                temp_str = f"Temp: {phase_manager.get_current_temperature():.2f} | " if not use_warmstart else ""
                grad_str = f" | GradNorm: {max_grad_norm_seen:.2f}" if max_grad_norm_seen > 0 else ""
                clip_str = f" (clipped {grad_clips}x)" if grad_clips > 0 else ""
                print(f"[{phase_name:8s}] Batch {batch_num:4d} | "
                      f"Games: {phase_manager.total_games_played:5d} | "
                      f"Loss: {train_loss:.4f} | "
                      f"WR: {stats['white_win_rate']:.3f} | "
                      f"Moves: {stats['avg_moves']:.1f} | "
                      f"{temp_str}"
                      f"Buffer: {buffer_status} | "
                      f"Speed: {metrics.games_per_second:.1f} g/s"
                      f"{grad_str}{clip_str}")

                # Save metrics to file
                save_metrics(metrics, config)

            # Evaluation checkpoint
            if batch_num % config.eval_every_n_batches == 0 and replay_buffer.is_ready():
                run_evaluation_checkpoint(
                    state=state,
                    step=batch_num,
                    games_played=phase_manager.total_games_played,
                    eval_history=eval_history,
                    num_eval_games=config.eval_num_games,
                    ply=config.eval_ply,
                    rng=eval_rng,
                    verbose=True,
                )

                # Validation loss for early stopping
                if val_buffer is not None and val_buffer.is_ready():
                    val_batch = val_buffer.sample_batch(
                        min(config.training_batch_size, len(val_buffer))
                    )
                    val_metrics = compute_metrics(state, val_batch)
                    val_loss = val_metrics['loss']

                    if batch_num % config.log_every_n_batches == 0:
                        print(f"  Val loss: {val_loss:.4f} "
                              f"(best: {best_val_loss:.4f}, "
                              f"patience: {patience_counter}/{config.early_stopping_patience})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state_step = batch_num
                        # Save best model checkpoint
                        save_checkpoint(state, config, batch_num, is_best=True)
                    else:
                        patience_counter += 1

                    # LR plateau detection (diagnostic)
                    if val_loss < lr_plateau_best:
                        lr_plateau_best = val_loss
                        lr_plateau_counter = 0
                    else:
                        lr_plateau_counter += 1
                        if lr_plateau_counter >= lr_plateau_patience:
                            print(f"  ⚠ Val loss plateaued for {lr_plateau_counter} "
                                  f"evals (consider reducing LR)")

                    metrics_logger.log_metrics({
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'lr_plateau_counter': lr_plateau_counter,
                    }, step=batch_num, prefix="validation/")

                    if (config.use_early_stopping and
                            patience_counter >= config.early_stopping_patience):
                        print(f"\n⏹ Early stopping triggered at batch {batch_num}")
                        print(f"  Best val loss: {best_val_loss:.4f} at step {best_state_step}")
                        print(f"  No improvement for {patience_counter} eval checkpoints")
                        break

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

                # Final evaluation
                print("\n📊 Final evaluation:")
                run_evaluation_checkpoint(
                    state=state,
                    step=batch_num,
                    games_played=phase_manager.total_games_played,
                    eval_history=eval_history,
                    num_eval_games=config.eval_num_games,
                    ply=config.eval_ply,
                    rng=eval_rng,
                    verbose=True,
                )

                print("\n📈 Evaluation history:")
                print(eval_history.summary())

                # Save summary
                metrics_logger.save_summary({
                    'total_games': phase_manager.total_games_played,
                    'total_batches': batch_num,
                    'total_train_steps': total_train_steps,
                    'final_loss': train_loss,
                    'final_learning_rate': current_lr,
                    'eval_history': eval_history.entries,
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
