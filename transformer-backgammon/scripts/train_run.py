#!/usr/bin/env python3
"""Training launcher with preset configurations.

Provides named presets for different training scales, from quick smoke tests
to multi-hour production runs. Each preset tunes curriculum phases, model
size, optimizer schedule, and evaluation frequency for its target scale.

Usage:
    # List available presets
    python scripts/train_run.py --list

    # Run a preset (interactive selection if omitted)
    python scripts/train_run.py --preset poc-15k
    python scripts/train_run.py --preset full-50k
    python scripts/train_run.py --preset long-200k

    # Override specific settings
    python scripts/train_run.py --preset poc-15k --dtype float32
    python scripts/train_run.py --preset full-50k --checkpoint-dir /mnt/data/checkpoints
    python scripts/train_run.py --preset poc-15k --skip-verify
"""

import argparse
import sys
import textwrap
import time


# ── Preset definitions ────────────────────────────────────────────────────

PRESETS = {
    "smoke": {
        "description": "Pipeline validation (~5 min). Tiny model, 500 games.",
        "total_games": 500,
        "model": {"embed_dim": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 256},
        "training": {
            "training_batch_size": 256,
            "games_per_batch": 64,
            "train_steps_per_game_batch": 4,
            "replay_buffer_size": 10_000,
            "replay_buffer_min_size": 200,
            "learning_rate": 3e-4,
            "warmup_steps": 5,
            "decay_steps": 40,
        },
        "eval": {
            "checkpoint_every_n_batches": 25,
            "log_every_n_batches": 5,
            "eval_every_n_batches": 10,
            "eval_num_games": 20,
        },
        "paths": {"checkpoint_dir": "checkpoints_smoke", "log_dir": "logs_smoke"},
    },
    "quick-2.5k": {
        "description": "Quick validation (~15-30 min). Small model, 2,500 games.",
        "total_games": 2500,
        "model": {"embed_dim": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 256},
        "training": {
            "training_batch_size": 512,
            "games_per_batch": 128,
            "train_steps_per_game_batch": 4,
            "replay_buffer_size": 50_000,
            "replay_buffer_min_size": 500,
            "learning_rate": 3e-4,
            "warmup_steps": 10,
            "decay_steps": 80,
        },
        "eval": {
            "checkpoint_every_n_batches": 50,
            "log_every_n_batches": 5,
            "eval_every_n_batches": 25,
            "eval_num_games": 30,
        },
        "paths": {"checkpoint_dir": "checkpoints_quick", "log_dir": "logs_quick"},
    },
    "dev-5k": {
        "description": "Development run (~30-60 min). Small model, 5,000 games.",
        "total_games": 5000,
        "model": {"embed_dim": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 256},
        "training": {
            "training_batch_size": 512,
            "games_per_batch": 128,
            "train_steps_per_game_batch": 4,
            "replay_buffer_size": 50_000,
            "replay_buffer_min_size": 500,
            "learning_rate": 3e-4,
            "warmup_steps": 20,
            "decay_steps": 160,
        },
        "eval": {
            "checkpoint_every_n_batches": 50,
            "log_every_n_batches": 5,
            "eval_every_n_batches": 25,
            "eval_num_games": 40,
        },
        "paths": {"checkpoint_dir": "checkpoints_dev", "log_dir": "logs_dev"},
    },
    "poc-15k": {
        "description": "Proof-of-concept (~1-2 hr). Medium model, 15,000 games.",
        "total_games": 15000,
        "model": {"embed_dim": 128, "num_heads": 8, "num_layers": 4, "ff_dim": 512},
        "training": {
            "training_batch_size": 256,
            "games_per_batch": 64,
            "train_steps_per_game_batch": 4,
            "replay_buffer_size": 100_000,
            "replay_buffer_min_size": 1000,
            "learning_rate": 3e-4,
            "warmup_steps": 500,
            "decay_steps": 1000,
        },
        "eval": {
            "checkpoint_every_n_batches": 100,
            "log_every_n_batches": 10,
            "eval_every_n_batches": 50,
            "eval_num_games": 50,
        },
        "paths": {"checkpoint_dir": "checkpoints_poc", "log_dir": "logs_poc"},
    },
    "full-50k": {
        "description": "Full training (~4-6 hr). Medium model, 50,000 games.",
        "total_games": 50000,
        "model": {"embed_dim": 128, "num_heads": 8, "num_layers": 4, "ff_dim": 512},
        "training": {
            "training_batch_size": 256,
            "games_per_batch": 64,
            "train_steps_per_game_batch": 4,
            "replay_buffer_size": 200_000,
            "replay_buffer_min_size": 2000,
            "learning_rate": 3e-4,
            "warmup_steps": 1000,
            "decay_steps": 3200,
        },
        "eval": {
            "checkpoint_every_n_batches": 200,
            "log_every_n_batches": 20,
            "eval_every_n_batches": 100,
            "eval_num_games": 50,
        },
        "paths": {"checkpoint_dir": "checkpoints_full", "log_dir": "logs_full"},
    },
    "long-200k": {
        "description": "Long training (~24+ hr). Large model, 200,000 games.",
        "total_games": 200000,
        "model": {"embed_dim": 256, "num_heads": 16, "num_layers": 8, "ff_dim": 1024},
        "training": {
            "training_batch_size": 512,
            "games_per_batch": 128,
            "train_steps_per_game_batch": 4,
            "replay_buffer_size": 500_000,
            "replay_buffer_min_size": 5000,
            "learning_rate": 1e-4,
            "warmup_steps": 2000,
            "decay_steps": 6400,
        },
        "eval": {
            "checkpoint_every_n_batches": 500,
            "log_every_n_batches": 50,
            "eval_every_n_batches": 200,
            "eval_num_games": 100,
        },
        "paths": {"checkpoint_dir": "checkpoints_long", "log_dir": "logs_long"},
    },
}

# Estimated parameter counts for display
MODEL_PARAMS = {
    (64, 4, 2, 256): "~500K",
    (128, 8, 4, 512): "~2M",
    (256, 16, 8, 1024): "~10M",
}


def estimate_params(model_cfg):
    key = (model_cfg["embed_dim"], model_cfg["num_heads"],
           model_cfg["num_layers"], model_cfg["ff_dim"])
    return MODEL_PARAMS.get(key, "unknown")


# ── Display ───────────────────────────────────────────────────────────────

def print_presets():
    """Print a formatted table of available presets."""
    print()
    print("Available training presets:")
    print("=" * 72)
    fmt = "  {:<14s} {:>7s} games   {:>6s} params   {}"
    print(fmt.format("PRESET", "", "", "DESCRIPTION"))
    print("  " + "-" * 68)
    for name, cfg in PRESETS.items():
        g = cfg["total_games"]
        p = estimate_params(cfg["model"])
        desc = cfg["description"].split(".")[0]  # First sentence only
        print(fmt.format(name, f"{g:,}", p, desc))
    print()


def print_config_summary(name, cfg, dtype):
    """Print a summary of the chosen configuration."""
    m = cfg["model"]
    t = cfg["training"]
    total = cfg["total_games"]
    params = estimate_params(m)

    print()
    print("=" * 60)
    print(f"  Training Preset: {name}")
    print(f"  {cfg['description']}")
    print("=" * 60)
    print()
    print(f"  Model:       {m['num_layers']}L / {m['embed_dim']}d / {m['num_heads']}H / "
          f"{m['ff_dim']}ff  ({params} params)")
    print(f"  Dtype:       {dtype}")
    print(f"  Games:       {total:,} (neural self-play, full variants)")
    print(f"  Batch:       {t['training_batch_size']} train / "
          f"{t['games_per_batch']} self-play")
    print(f"  LR:          {t['learning_rate']:.0e} (warmup {t['warmup_steps']}, "
          f"decay {t['decay_steps']} steps)")
    print(f"  Replay:      {t['replay_buffer_size']:,} capacity")
    print(f"  Checkpoints: {cfg['paths']['checkpoint_dir']}")
    print(f"  Logs:        {cfg['paths']['log_dir']}")
    print()


def interactive_select():
    """Let the user pick a preset interactively."""
    names = list(PRESETS.keys())
    print()
    print("Select a training preset:")
    print()
    for i, name in enumerate(names, 1):
        cfg = PRESETS[name]
        g = cfg["total_games"]
        p = estimate_params(cfg["model"])
        print(f"  [{i}] {name:<14s}  {g:>7,} games, {p:>6s} params — "
              f"{cfg['description']}")
    print()
    while True:
        try:
            choice = input(f"Enter choice [1-{len(names)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number between 1 and {len(names)}.")


# ── Build config ──────────────────────────────────────────────────────────

def build_config(preset_name, dtype="bfloat16", checkpoint_dir=None, log_dir=None):
    """Build a TrainingConfig from a preset name with optional overrides."""
    from backgammon.training.train import TrainingConfig

    cfg = PRESETS[preset_name]
    m = cfg["model"]
    t = cfg["training"]
    e = cfg["eval"]
    p = cfg["paths"]

    config = TrainingConfig(
        # Total games (no curriculum — pure self-play)
        total_games=cfg["total_games"],

        # Model
        embed_dim=m["embed_dim"],
        num_heads=m["num_heads"],
        num_layers=m["num_layers"],
        ff_dim=m["ff_dim"],
        dropout_rate=0.1,

        # Training mode
        train_policy=False,  # Value-only (policy head is non-functional)
        compute_dtype="bfloat16" if dtype == "bfloat16" else None,

        # Optimizer
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        decay_steps=t["decay_steps"],
        max_grad_norm=1.0,
        ema_decay=0.999,

        # Replay buffer
        replay_buffer_size=t["replay_buffer_size"],
        replay_buffer_min_size=t["replay_buffer_min_size"],
        training_batch_size=t["training_batch_size"],
        games_per_batch=t["games_per_batch"],
        train_steps_per_game_batch=t["train_steps_per_game_batch"],

        # TD(lambda)
        use_td_lambda=True,
        td_lambda=0.7,

        # 1-ply lookahead for stronger self-play training signal
        use_1ply_selfplay=True,
        lookahead_top_k=8,

        # Data augmentation
        use_position_weighting=True,
        use_color_flip_augmentation=True,

        # Evaluation & checkpointing
        checkpoint_every_n_batches=e["checkpoint_every_n_batches"],
        log_every_n_batches=e["log_every_n_batches"],
        eval_every_n_batches=e["eval_every_n_batches"],
        eval_num_games=e["eval_num_games"],

        # Early stopping
        use_early_stopping=True,
        early_stopping_patience=5,
        validation_fraction=0.1,

        # Paths
        checkpoint_dir=checkpoint_dir or p["checkpoint_dir"],
        log_dir=log_dir or p["log_dir"],

        seed=42,
    )
    return config


# ── Hardware check ────────────────────────────────────────────────────────

def verify_hardware():
    """Detect and verify available hardware."""
    import jax
    import jax.numpy as jnp

    backend = jax.default_backend()
    devices = jax.devices()

    print(f"JAX version: {jax.__version__}")
    print(f"Backend:     {backend}")
    print(f"Devices:     {len(devices)}")
    for d in devices:
        print(f"  {d}")

    if backend == "tpu":
        # Verify bfloat16 on TPU
        x = jnp.ones((256, 256), dtype=jnp.bfloat16)
        y = jnp.dot(x, x)
        y.block_until_ready()
        print(f"bfloat16 matmul: OK")
    elif backend == "gpu":
        print("Running on GPU. bfloat16 should work on modern GPUs (A100+).")
    else:
        print("WARNING: Running on CPU. Training will be very slow.")
        print("Consider using --dtype float32 for CPU.")

    return backend


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train backgammon transformer with preset configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            presets:
              smoke       Pipeline validation (~5 min, 500 games, ~500K params)
              quick-2.5k  Quick validation (~15-30 min, 2,500 games, ~500K params)
              dev-5k      Development run (~30-60 min, 5,000 games, ~500K params)
              poc-15k     Proof-of-concept (~1-2 hr, 15,000 games, ~2M params)
              full-50k    Full training (~4-6 hr, 50,000 games, ~2M params)
              long-200k   Long training (~24+ hr, 200,000 games, ~10M params)
        """),
    )
    parser.add_argument(
        "--preset", type=str, choices=list(PRESETS.keys()), default=None,
        help="Training preset to use (interactive selection if omitted)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available presets and exit",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="bfloat16",
        help="Compute dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Override log directory",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip hardware verification",
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        print_presets()
        return

    # Select preset
    if args.preset is None:
        preset_name = interactive_select()
    else:
        preset_name = args.preset

    cfg = PRESETS[preset_name]

    # Print header
    print()
    print("=" * 60)
    print("  Backgammon Transformer — Training Launcher")
    print("=" * 60)

    # Hardware check
    if not args.skip_verify:
        print()
        verify_hardware()

    # Build config and print summary
    config = build_config(
        preset_name,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    print_config_summary(preset_name, cfg, args.dtype)

    # Confirm for long runs
    tg = cfg["total_games"]
    if tg >= 50_000:
        resp = input(f"This run has {tg:,} games and may take hours. Continue? [Y/n] ")
        if resp.strip().lower() == "n":
            print("Aborted.")
            return

    # Train
    from backgammon.training.train import train

    t0 = time.time()
    train(config)
    elapsed = time.time() - t0

    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print()
    print(f"Training complete. Total time: {hrs}h {mins}m ({elapsed:.0f}s)")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: {config.log_dir}")


if __name__ == "__main__":
    main()
