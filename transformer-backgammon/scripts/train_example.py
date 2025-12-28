#!/usr/bin/env python3
"""Example training script showing how to use the training module.

This demonstrates:
1. Curriculum learning across training phases
2. Pip count warmstart
3. Checkpointing and logging
4. Self-play data generation

Usage:
    python scripts/train_example.py
"""

from backgammon.training import TrainingConfig, train


def main():
    """Run training with example configuration."""

    # Configure training for a quick test run
    config = TrainingConfig(
        # Curriculum phases (small for demo)
        warmstart_games=100,      # Pip count vs pip count
        early_phase_games=200,    # Hypergammon-heavy
        mid_phase_games=300,      # Mixed variants
        late_phase_games=400,     # Full complexity

        # Batch settings
        games_per_batch=16,       # Games per training batch
        checkpoint_every_n_batches=50,  # Save every 50 batches
        log_every_n_batches=5,    # Log every 5 batches

        # Model architecture
        d_model=256,              # Transformer hidden dimension
        num_heads=8,              # Attention heads
        num_layers=6,             # Transformer layers
        dropout_rate=0.1,

        # Optimizer
        learning_rate=3e-4,
        warmup_steps=100,
        max_grad_norm=1.0,

        # Paths
        checkpoint_dir="checkpoints",
        log_dir="logs",

        # Reproducibility
        seed=42,
    )

    print("=" * 60)
    print("  Backgammon Transformer Training")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  • Warmstart:  {config.warmstart_games} games (pip count)")
    print(f"  • Early:      {config.early_phase_games} games (hypergammon-heavy)")
    print(f"  • Mid:        {config.mid_phase_games} games (mixed)")
    print(f"  • Late:       {config.late_phase_games} games (full complexity)")
    print(f"  • Total:      {config.warmstart_games + config.early_phase_games + config.mid_phase_games + config.late_phase_games} games")
    print()
    print(f"  • Model:      {config.num_layers} layers, {config.d_model} dim, {config.num_heads} heads")
    print(f"  • Batch size: {config.games_per_batch} games")
    print(f"  • LR:         {config.learning_rate} (warmup: {config.warmup_steps} steps)")
    print()
    print("=" * 60)
    print()

    # Run training
    train(config)


if __name__ == "__main__":
    main()
