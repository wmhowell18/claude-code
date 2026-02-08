"""Training module for backgammon AI.

Includes:
- Self-play game generation
- Curriculum learning (warmstart → early → mid → late)
- Checkpointing and metrics logging
- GPU-optimized training with JAX/Flax
"""

from backgammon.training.self_play import (
    GameStep,
    GameResult,
    play_game,
    generate_training_batch,
    compute_game_statistics,
)

from backgammon.training.train import (
    TrainingConfig,
    TrainingMetrics,
    TrainingPhase,
    train,
    create_train_state,
    save_checkpoint,
    save_metrics,
)

from backgammon.training.losses import (
    compute_policy_loss,
    compute_equity_loss,
    compute_combined_loss,
    train_step,
    prepare_training_batch,
    compute_metrics,
)

from backgammon.training.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
)

from backgammon.training.metrics import (
    MetricsLogger,
    MetricsAggregator,
)

__all__ = [
    # Self-play
    "GameStep",
    "GameResult",
    "play_game",
    "generate_training_batch",
    "compute_game_statistics",
    # Training
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingPhase",
    "train",
    "create_train_state",
    "save_checkpoint",
    "save_metrics",
    # Loss functions
    "compute_policy_loss",
    "compute_equity_loss",
    "compute_combined_loss",
    "train_step",
    "prepare_training_batch",
    "compute_metrics",
    # Replay buffer
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    # Metrics logging
    "MetricsLogger",
    "MetricsAggregator",
]
