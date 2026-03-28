# Transformer Backgammon

A GPU/TPU-optimized backgammon AI using transformer neural networks, built with JAX/Flax.

## Overview

This project builds a competitive backgammon engine using modern deep learning:

- **Transformer architecture** with RMSNorm, SwiGLU, pre-norm, and optional muP scaling
- **TD(lambda) learning** with self-play, experience replay, and curriculum training
- **JAX/Flax** for GPU/TPU acceleration with bfloat16 mixed precision
- **N-ply lookahead** (0/1/2-ply) with batched dice averaging

The long-term goal is to reach and surpass XtremeGammon-level play by leveraging GPU parallelism, modern architectures, and scale. See [BEATING_XG.md](BEATING_XG.md) for the full competitive analysis.

## Architecture

```
Input (Board State: 26 positions x 10 features)
    |
Encoder (per-position raw features + broadcast global features)
    |                  [pip counts, contact, primes, bearoff]
    |
Transformer Encoder (configurable depth/width)
  - Pre-norm with RMSNorm
  - Multi-head Self-Attention (optional muP scaling)
  - SwiGLU Feed-Forward Networks
  - Residual Connections
    |
Global Mean Pooling
    |
Output Heads
  - Value Head  -> 5-dim equity (softmax: P(win), P(win gammon), ...)
  - Policy Head -> Move probabilities (optional, disabled for now)
  - Cube Head   -> Double/take/pass decisions (optional)
```

**Model sizes:**

| Config | Layers | Dim | Heads | Params |
|--------|--------|-----|-------|--------|
| Small  | 2      | 64  | 4     | ~500K  |
| Medium | 4      | 128 | 8     | ~2M    |
| Large  | 8      | 256 | 16    | ~10M   |

A Stochastic MuZero architecture (`network/muzero.py`) is also included for MCTS-based planning with dice as chance nodes.

### Board Representation

The board is a sequence of 26 tokens: `[bar] + [points 1-24] + [off]`. Each token carries a 10-dimensional feature vector (2 raw checker counts + 8 global features broadcast to every position). Self-attention learns strategic relationships between positions.

### Training Pipeline

- **Curriculum learning**: warmstart (pip count opponent) -> early -> mid -> late (full self-play)
- **TD(lambda)** targets with renormalization and Monte Carlo fallback
- **Experience replay** with position weighting and FIFO eviction
- **EMA** weight averaging for smoother evaluation
- **AdamW** with warmup + cosine decay (or schedule-free optimizer)
- **Color-flip augmentation** for 2x effective data

## Current Status

The core engine, network, training pipeline, and evaluation framework are complete. We are preparing for the first training run.

**What's working:**
- Full game engine with move generation, doubling cube, match play
- Transformer network with value/policy/cube heads
- TD(lambda) training with curriculum learning and self-play
- 0/1/2-ply search with batched dice averaging
- Checkpoint save/restore, metrics logging, early stopping
- Benchmark position evaluation and agent comparison

**What's not yet wired:**
- Policy head (disabled; hash collisions in action encoding need a better scheme)
- GnuBG interface for head-to-head evaluation
- Bearoff database (major gap vs XG)
- 3-ply search

See [TODO.md](TODO.md) for the full feature list and [PERIODIC_REVIEW.md](PERIODIC_REVIEW.md) for the latest code review.

## Getting Started

### Install

```bash
cd transformer-backgammon
pip install -e ".[dev]"
```

For GPU/TPU support, install the appropriate JAX variant:
```bash
# CPU only (default)
pip install jax jaxlib

# CUDA 12
pip install jax[cuda12]

# TPU (on Google Cloud)
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Verify

```bash
pytest tests/ -q
python scripts/smoke_test.py
```

### Train

```bash
# Quick validation run (~2,500 games, ~10 min on GPU)
python scripts/train_v6e.py

# Or from Python:
from backgammon.training.train import train, v6e_quick_training_config
train(v6e_quick_training_config())
```

See [NEXT_STEPS.md](NEXT_STEPS.md) for the step-by-step training plan from smoke test through scaling.

### Evaluate

```bash
# Run benchmark evaluation
python scripts/evaluate.py --checkpoint checkpoints_v6e/ --games 100
```

## Project Structure

```
transformer-backgammon/
├── src/backgammon/
|   ├── core/              # Game engine: board, dice, types, cube, match play
|   ├── encoding/          # Board -> features: raw (2-dim), enhanced (10-dim)
|   |                      #   + action encoder for policy head
|   ├── network/           # Transformer: attention, SwiGLU, value/policy/cube heads
|   |                      #   + Stochastic MuZero world model
|   ├── training/          # Training: self-play, TD(lambda), replay buffer, losses
|   |                      #   + curriculum, metrics, checkpointing
|   ├── evaluation/        # Agents, search (0/1/2-ply), benchmarks
|   └── utils/             # Config, logging
├── tests/                 # Mirrors src/ structure
├── scripts/               # smoke_test.py, train_v6e.py, evaluate.py
├── configs/               # Training configs (YAML)
├── benchmarks/            # Benchmark positions for evaluation
└── pyproject.toml         # Package config and dependencies
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for module dependencies and data flow.

## Configuration

Key training parameters (see `TrainingConfig` in `training/train.py`):

```python
# Model
embed_dim = 128          # Embedding dimension
num_layers = 4           # Transformer blocks
num_heads = 8            # Attention heads
input_feature_dim = 10   # 2 raw + 8 global features

# Training
learning_rate = 3e-4     # Peak LR (with warmup + cosine decay)
decay_steps = 100000     # Cosine decay total steps
training_batch_size = 256
replay_buffer_size = 100_000
td_lambda = 0.7          # TD(lambda) parameter

# Curriculum
warmstart_games = 500    # vs pip count agent
early_phase_games = 1000
mid_phase_games = 3000
late_phase_games = 10000
```

## Comparison to XtremeGammon

| Aspect | XtremeGammon | This Project |
|--------|--------------|--------------|
| **Platform** | CPU-optimized | GPU/TPU-optimized |
| **Evaluation** | Sequential 2-3 ply | Batched 0-2 ply (3-ply planned) |
| **Network** | Shallow MLP (~20K params) | Transformer (500K-10M params) |
| **Architecture** | LayerNorm, GELU | RMSNorm, SwiGLU, pre-norm, muP |
| **Features** | Hand-crafted (many) | Learned + 8 global features |
| **Training** | Years of refinement | TD(lambda) + EMA + AdamW + curriculum |
| **Endgame** | Bearoff databases | Neural only (databases planned) |
| **Speed** | Very fast per position | 10-100x faster batched on GPU |

The key bet is that GPU batching (512+ positions simultaneously) compensates for per-position latency, and that transformers can learn strategic features that hand-engineering misses. See [BEATING_XG.md](BEATING_XG.md) for the full path forward.

## Roadmap

| Milestone | Status |
|-----------|--------|
| Core game engine | Done |
| Transformer network | Done |
| Training pipeline (TD-lambda, curriculum, replay) | Done |
| 0/1/2-ply search with dice averaging | Done |
| Doubling cube + match play | Done |
| First training run (2.5K games) | Next |
| Scale to 15K games, beat pip count agent | Planned |
| GnuBG interface for evaluation | Planned |
| Bearoff database | Planned |
| 3-ply search | Planned |
| Scale to 200K+ games | Planned |

## References

- [TD-Gammon (Tesauro, 1995)](https://www.bkgm.com/articles/tesauro/tdl.html) - Self-play TD-learning for backgammon
- [Jacob Hilton's Backgammon](https://github.com/jacobhilton/backgammon) - TD(0) + dice averaging, 2.5K games to 41% vs GNU BG
- [Stochastic MuZero (2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1) - AlphaZero-style approach for stochastic games
- [Pgx: GPU-Accelerated Game Simulators](https://github.com/sotetsuk/pgx) - JAX-based game environments
- [GNU Backgammon](https://www.gnu.org/software/gnubg/) - Open-source backgammon engine

## License

MIT

## Contributing

Contributions welcome! See [TODO.md](TODO.md) for open items. Key areas:

- Training at scale (longer runs, hyperparameter sweeps)
- Bearoff database implementation
- GnuBG integration for evaluation
- Architecture experiments (MoE, deeper models)
- Encoding improvements (blot vulnerability, escape features)
