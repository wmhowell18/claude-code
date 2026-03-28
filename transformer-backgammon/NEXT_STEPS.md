# Next Steps: Small Proof-of-Concept Training Run

Granular, ordered steps to validate the training pipeline and get a small model working before committing to a large run.

> **Architecture update (Mar 2026)**: The transformer now uses modern best practices: RMSNorm (not LayerNorm), SwiGLU activation (not GELU), pre-norm (not post-norm), AdamW (not Adam), EMA weight averaging for evaluation, and color-flip data augmentation. muP and schedule-free optimizer are available as options. A Stochastic MuZero network architecture is also available in `network/muzero.py` but not yet wired into training.

---

## Step 1: Run Smoke Test

Verify the bug fixes don't break anything.

```bash
cd transformer-backgammon
python scripts/smoke_test.py
```

**Expected**: Loss decreases, trained model beats random at >55%, bfloat16 works.

**If it fails**: Check error message. Most likely cause is a shape mismatch from the encoding changes (2->10 features). Fix and re-run.

---

## Step 2: Run Quick Training (2,500 games)

Use the existing v6e quick config for a fast proof-of-concept:

```python
from backgammon.training.train import train, v6e_quick_training_config

config = v6e_quick_training_config()
train(config)
```

This runs: 200 warmstart + 500 early + 800 mid + 1000 late = 2,500 games.
Small model: 2 layers, 64 dim, 4 heads (~500K params).

**What to look for**:
- [ ] Loss decreases over time (most important signal)
- [ ] Loss continues decreasing through all 4 curriculum phases
- [ ] Win rate vs random improves at eval checkpoints
- [ ] No NaN/Inf in loss or gradients
- [ ] Games/second is reasonable (>10 on GPU, >50 on TPU)

---

## Step 3: Evaluate the Trained Model

After training completes, evaluate more thoroughly:

```python
from backgammon.evaluation.benchmark import run_evaluation_checkpoint

# The training loop runs this automatically, but you can re-run with more games:
run_evaluation_checkpoint(
    state=state,
    step=0,
    games_played=2500,
    eval_history=eval_history,
    num_eval_games=100,  # More games for tighter estimate
    ply=0,
    rng=rng,
    verbose=True,
)
```

**Success criteria for 2,500 games**:
- [ ] Beats random agent >70% of the time at 0-ply
- [ ] Beats random agent >85% at 1-ply
- [ ] Equity error on benchmark positions is decreasing

---

## Step 4: Scale to 5K Games (Same Small Model)

If Step 3 passes, double the training:

```python
config = v6e_quick_training_config()
config.warmstart_games = 500
config.early_phase_games = 1000
config.mid_phase_games = 1500
config.late_phase_games = 2000
# Total: 5,000 games
train(config)
```

**What to look for**:
- [ ] Win rate vs random at 0-ply reaches >80%
- [ ] Win rate vs pip count agent starts climbing (even >45% is a good sign)
- [ ] Loss hasn't plateaued (still decreasing at end of training)

---

## Step 5: Compare Architectures (Optional but Recommended)

Before scaling up, verify the transformer is actually helping. Train the same 5K games with:

**A) Tiny transformer** (current: 2 layers, 64 dim, ~500K params):
```python
config.embed_dim = 64
config.num_heads = 4
config.num_layers = 2
```

**B) Slightly larger transformer** (4 layers, 128 dim, ~2M params):
```python
config.embed_dim = 128
config.num_heads = 8
config.num_layers = 4
```

Compare win rates and equity errors. The larger model should be better -- if not, there may be a remaining training issue.

---

## Step 6: Scale to 15K Games (Medium Model)

This is the real proof-of-concept. Use the medium model:

```python
config = TrainingConfig(
    warmstart_games=1000,
    early_phase_games=3000,
    mid_phase_games=5000,
    late_phase_games=6000,
    # Total: 15,000 games

    embed_dim=128,
    num_heads=8,
    num_layers=4,
    ff_dim=512,
    dropout_rate=0.1,

    train_policy=False,
    compute_dtype='bfloat16',

    learning_rate=3e-4,
    warmup_steps=500,
    max_grad_norm=1.0,

    replay_buffer_size=100_000,
    replay_buffer_min_size=1000,
    training_batch_size=256,
    games_per_batch=64,
    train_steps_per_game_batch=4,

    use_td_lambda=True,
    td_lambda=0.7,

    checkpoint_every_n_batches=100,
    log_every_n_batches=10,
    eval_every_n_batches=50,
    eval_num_games=50,

    checkpoint_dir="checkpoints_poc",
    log_dir="logs_poc",
    seed=42,
)
train(config)
```

**Success criteria for 15K games**:
- [ ] Beats random >90% at 0-ply
- [ ] Beats pip count agent >55% at 0-ply
- [ ] Beats pip count agent >65% at 1-ply
- [ ] Benchmark equity MAE < 0.15

---

## Step 7: Analyze Results

After the 15K game run, inspect the training logs:

```python
# Read training logs
import json
with open("logs_poc/training_log.jsonl") as f:
    entries = [json.loads(line) for line in f]

# Plot loss curve
losses = [e['loss'] for e in entries]
# Look for: steady decrease, no sudden jumps, no plateau

# Plot win rate curve
win_rates = [e['white_win_rate'] for e in entries]
# Look for: starting near 50%, staying near 50% (balanced self-play)
```

**Key questions**:
- Did the loss plateau? If so, at what game count?
- Is the model still improving at 15K games? (If yes, scale further)
- Did early stopping trigger? (If so, why? May need more data diversity)
- Is there a quality jump when switching curriculum phases?

---

## Step 8: Decision Point

Based on the 15K game results:

**If the model beats pip count at >55% (0-ply)**:
The pipeline works. Proceed to a 50K+ game run with the same config. This is the path to a real backgammon bot.

**If the model doesn't beat pip count**:
Something is still wrong. Debug by:
1. Check if loss is actually decreasing (training signal exists)
2. Check if equity predictions on benchmark positions are improving
3. Try a plain MLP instead of transformer to rule out architecture issues
4. Try pure Monte Carlo targets (disable TD-lambda) to rule out target issues

---

## Quick Reference: Config Sizes

| Size | Layers | Dim | Params | Games | Time (GPU) |
|------|--------|-----|--------|-------|------------|
| Tiny | 2 | 64 | ~500K | 2.5K | ~10 min |
| Small | 2 | 64 | ~500K | 5K | ~20 min |
| Medium | 4 | 128 | ~2M | 15K | ~1-2 hr |
| Full | 4 | 128 | ~2M | 50K | ~4-6 hr |
| Large | 8 | 256 | ~10M | 200K | ~24+ hr |
