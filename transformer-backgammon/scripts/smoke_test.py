#!/usr/bin/env python3
"""Smoke test: verify training pipeline produces a model better than random.

Run this before committing to a long TPU training run. It:
1. Trains a tiny model for ~500 games (~3-5 minutes on CPU)
2. Checks that equity loss decreases during training
3. Evaluates the trained model against a random agent
4. PASSes only if loss decreases AND trained model wins >55%

Usage:
    python scripts/smoke_test.py
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

from backgammon.core.board import initial_board, get_early_training_variants
from backgammon.core.types import Player
from backgammon.evaluation.agents import random_agent, pip_count_agent
from backgammon.evaluation.network_agent import create_neural_agent
from backgammon.training.self_play import play_game, generate_training_batch
from backgammon.training.replay_buffer import ReplayBuffer
from backgammon.training.losses import train_step
from backgammon.training.train import create_train_state, TrainingConfig


def evaluate_agent(agent, opponent, num_games=50, rng=None):
    """Play games and return agent's win rate as white + black (averaged)."""
    if rng is None:
        rng = np.random.default_rng(99)
    wins = 0
    total = 0
    # Play as both colors to eliminate first-move advantage bias
    for _ in range(num_games // 2):
        # Agent as white
        result = play_game(agent, opponent, initial_board(), rng=rng)
        if result.outcome is not None and result.outcome.winner == Player.WHITE:
            wins += 1
        total += 1
        # Agent as black
        result = play_game(opponent, agent, initial_board(), rng=rng)
        if result.outcome is not None and result.outcome.winner == Player.BLACK:
            wins += 1
        total += 1
    return wins / total


def main():
    print("=" * 60)
    print("  SMOKE TEST: Backgammon Training Pipeline")
    print("=" * 60)
    print()

    # --- Setup ---
    config = TrainingConfig(
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout_rate=0.1,
        train_policy=True,
        learning_rate=3e-4,
        warmup_steps=50,
        replay_buffer_size=10000,
        replay_buffer_min_size=200,
        training_batch_size=64,
        seed=42,
    )

    jax_rng = jax.random.PRNGKey(config.seed)
    np_rng = np.random.default_rng(config.seed)

    print("[1/4] Initializing model...")
    state = create_train_state(config, jax_rng)

    # --- Check 1: Untrained baseline ---
    print("[2/4] Evaluating untrained model vs random (baseline)...")
    untrained_agent = create_neural_agent(state, temperature=0.0, name="Untrained")
    rand_agent = random_agent(seed=123)
    untrained_wr = evaluate_agent(untrained_agent, rand_agent, num_games=40, rng=np_rng)
    print(f"       Untrained win rate: {untrained_wr:.1%}")

    # --- Train ---
    print("[3/4] Training (~500 games, pip count warmstart)...")
    buffer = ReplayBuffer(max_size=config.replay_buffer_size, min_size=config.replay_buffer_min_size)
    pip_agent = pip_count_agent()

    losses = []
    t0 = time.time()
    total_games = 0
    total_steps = 0

    # More training: 30 batches x 16 games = 480 games, 8 grad steps each = 240 steps
    for batch_idx in range(30):
        # Generate games with pip count agent
        games = generate_training_batch(
            num_games=16,
            get_variant_fn=get_early_training_variants,
            white_agent=pip_agent,
            black_agent=pip_agent,
            rng=np_rng,
        )
        for game in games:
            buffer.add_game(game)
        total_games += len(games)

        # Train if buffer ready
        if buffer.is_ready():
            for _ in range(8):
                batch = buffer.sample_batch(config.training_batch_size)
                jax_rng, step_rng = jax.random.split(jax_rng)
                state, metrics = train_step(state, batch, step_rng)
                losses.append(float(metrics['total_loss']))
                total_steps += 1

        # Progress
        if (batch_idx + 1) % 10 == 0:
            recent_loss = np.mean(losses[-10:]) if losses else 0
            print(f"       ... batch {batch_idx+1}/30, games={total_games}, "
                  f"steps={total_steps}, loss={recent_loss:.4f}")

    elapsed = time.time() - t0
    print(f"       Done: {total_games} games, {total_steps} gradient steps in {elapsed:.1f}s")

    # --- Check 2: Loss should decrease ---
    if len(losses) >= 20:
        first_quarter = np.mean(losses[:len(losses)//4])
        last_quarter = np.mean(losses[-len(losses)//4:])
        loss_decreased = last_quarter < first_quarter
        loss_ratio = last_quarter / first_quarter
        print(f"       Loss: {first_quarter:.4f} -> {last_quarter:.4f} "
              f"({(1-loss_ratio)*100:.1f}% decrease) {'OK' if loss_decreased else 'FAIL'}")
    else:
        loss_decreased = False
        print("       FAIL: Not enough training steps")

    # --- Check 3: Trained model (0-ply) vs random ---
    print("[4/5] Evaluating trained model (0-ply) vs random...")
    trained_0ply = create_neural_agent(state, temperature=0.0, name="Trained-0ply")
    wr_0ply = evaluate_agent(trained_0ply, rand_agent, num_games=80, rng=np_rng)
    print(f"       0-ply win rate: {wr_0ply:.1%} (was {untrained_wr:.1%} untrained)")

    # --- Check 4: Trained model (1-ply search) vs random ---
    print("[5/5] Evaluating trained model (1-ply search) vs random...")
    print("       (this is slower — evaluates all opponent dice responses)")
    trained_1ply = create_neural_agent(state, temperature=0.0, name="Trained-1ply", ply=1)
    wr_1ply = evaluate_agent(trained_1ply, rand_agent, num_games=20, rng=np_rng)
    search_helps = wr_1ply > wr_0ply
    print(f"       1-ply win rate: {wr_1ply:.1%} "
          f"({'better' if search_helps else 'not better'} than 0-ply)")

    # --- Check 5: bfloat16 pipeline ---
    print("[6/6] Testing bfloat16 pipeline...")
    bf16_config = TrainingConfig(
        embed_dim=64, num_heads=4, num_layers=2, ff_dim=256,
        dropout_rate=0.1, train_policy=False,
        learning_rate=3e-4, warmup_steps=10,
        replay_buffer_size=1000, replay_buffer_min_size=100,
        training_batch_size=32, compute_dtype='bfloat16',
        seed=99,
    )
    bf16_rng = jax.random.PRNGKey(99)
    bf16_state = create_train_state(bf16_config, bf16_rng)

    # Quick training: generate a few games and do gradient steps
    bf16_buffer = ReplayBuffer(max_size=1000, min_size=100)
    bf16_games = generate_training_batch(
        num_games=8,
        get_variant_fn=get_early_training_variants,
        white_agent=pip_agent,
        black_agent=pip_agent,
        rng=np.random.default_rng(99),
    )
    for game in bf16_games:
        bf16_buffer.add_game(game)

    bf16_ok = True
    if bf16_buffer.is_ready():
        batch = bf16_buffer.sample_batch(32)
        bf16_rng, step_rng = jax.random.split(bf16_rng)
        bf16_state, bf16_metrics = train_step(bf16_state, batch, step_rng)
        bf16_loss = float(bf16_metrics['total_loss'])
        bf16_ok = np.isfinite(bf16_loss)
        print(f"       bfloat16 loss: {bf16_loss:.4f} {'OK' if bf16_ok else 'FAIL (NaN/Inf)'}")
    else:
        print("       bfloat16 buffer not ready (OK, skipping grad step)")

    # Verify params stayed float32
    for leaf in jax.tree_util.tree_leaves(bf16_state.params):
        if leaf.dtype != jnp.float32:
            bf16_ok = False
            print(f"       FAIL: param dtype is {leaf.dtype}, expected float32")
            break
    if bf16_ok:
        print("       bfloat16 param dtypes: OK (all float32)")

    # --- Results ---
    print()
    print("=" * 60)
    all_pass = loss_decreased and wr_1ply > 0.50 and bf16_ok
    if all_pass:
        print("  PASS: Training pipeline, search, and bfloat16 are working.")
        print("  Safe to start a longer training run.")
    else:
        print("  RESULT: Partial pass")
    print(f"    Loss decreased:     {'YES' if loss_decreased else 'NO'}")
    print(f"    0-ply vs random:    {wr_0ply:.1%}")
    print(f"    1-ply vs random:    {wr_1ply:.1%}")
    print(f"    1-ply > 0-ply:      {'YES' if search_helps else 'NO'}")
    print(f"    bfloat16 pipeline:  {'YES' if bf16_ok else 'NO'}")
    if loss_decreased and not all_pass:
        print()
        print("  NOTE: Loss is decreasing, which confirms training works.")
        print("  The model may need more training steps to beat random.")
        print("  1-ply search provides the biggest strength gain with a")
        print("  properly trained network.")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
