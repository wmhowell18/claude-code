#!/usr/bin/env python3
"""Training script optimized for TPU v6e-1 (single chip).

Usage on a v6e-1 TPU VM:
    # 1. SSH into your TPU VM
    # 2. Install the package
    pip install -e .

    # 3. Run training (defaults to small model, bfloat16, ~2500 games)
    python scripts/train_v6e.py

    # Or with custom settings:
    python scripts/train_v6e.py --games 5000 --batch-size 1024
    python scripts/train_v6e.py --games-per-batch 192
    python scripts/train_v6e.py --model-size medium --dtype float32

Environment:
    Requires JAX with TPU support:
        pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"""

import argparse
import sys
import time


def verify_tpu():
    """Verify TPU is available and print device info."""
    import jax
    import jax.numpy as jnp

    backend = jax.default_backend()
    devices = jax.devices()

    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {backend}")
    print(f"Devices: {len(devices)}")
    for d in devices:
        print(f"  {d}")

    if backend != 'tpu':
        print("\nWARNING: Not running on TPU. Performance will be significantly worse.")
        print("To use TPU, ensure JAX TPU is installed and you're on a TPU VM.")
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != 'y':
            sys.exit(1)

    # Quick matmul test to verify bfloat16 works
    x = jnp.ones((256, 256), dtype=jnp.bfloat16)
    y = jnp.dot(x, x)
    y.block_until_ready()
    print(f"\nbfloat16 matmul test: OK (result dtype: {y.dtype})")

    return backend, devices


def main():
    parser = argparse.ArgumentParser(description='Train backgammon on TPU v6e-1')
    parser.add_argument('--games', type=int, default=None,
                        help='Total training games (default: 2500 for quick run)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size (default: 512)')
    parser.add_argument('--games-per-batch', type=int, default=None,
                        help='Self-play games generated per batch (default: v6e preset)')
    parser.add_argument('--model-size', choices=['small', 'medium'], default='small',
                        help='Model size preset (default: small)')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16',
                        help='Compute dtype (default: bfloat16)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_v6e',
                        help='Checkpoint directory')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip TPU verification')
    args = parser.parse_args()

    print("=" * 60)
    print("  Backgammon Transformer - TPU v6e-1 Training")
    print("=" * 60)
    print()

    # Step 1: Verify TPU
    if not args.skip_verify:
        verify_tpu()
    print()

    # Step 2: Configure training
    from backgammon.training.train import v6e_quick_training_config, train

    config = v6e_quick_training_config()

    # Override with CLI args
    config.training_batch_size = args.batch_size
    config.checkpoint_dir = args.checkpoint_dir
    if args.games_per_batch is not None:
        config.games_per_batch = args.games_per_batch

    if args.dtype == 'float32':
        config.compute_dtype = None
    else:
        config.compute_dtype = 'bfloat16'

    if args.model_size == 'medium':
        config.embed_dim = 128
        config.num_heads = 8
        config.num_layers = 4
        config.ff_dim = 512

    if args.games is not None:
        # Distribute games across phases proportionally
        total = args.games
        config.warmstart_games = max(100, total // 12)
        config.early_phase_games = total // 5
        config.mid_phase_games = total * 3 // 10
        config.late_phase_games = (
            total - config.warmstart_games - config.early_phase_games - config.mid_phase_games
        )

    total_games = (config.warmstart_games + config.early_phase_games +
                   config.mid_phase_games + config.late_phase_games)

    print(f"Configuration:")
    print(f"  Model: {config.num_layers}L / {config.embed_dim}d / {config.num_heads}H")
    print(f"  Compute dtype: {args.dtype}")
    print(f"  Batch size: {config.training_batch_size}")
    print(f"  Games per self-play batch: {config.games_per_batch}")
    print(f"  Total games: {total_games}")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print()

    # Step 3: Train
    t0 = time.time()
    train(config)
    elapsed = time.time() - t0

    print(f"\nTotal training time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
