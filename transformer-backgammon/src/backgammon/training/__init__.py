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
]
