# Transformer Backgammon

A GPU-optimized backgammon AI using transformer neural networks, built with JAX/Flax.

## Project Overview

This project implements a modern, GPU-efficient backgammon bot using:
- **Transformer architecture** for position evaluation
- **TD-learning** with experience replay (inspired by TD-Gammon and Jacob Hilton's work)
- **JAX/Flax** for GPU acceleration
- **N-ply lookahead** with dice averaging for move selection

### Goals

1. **GPU Efficiency**: Leverage modern GPU parallelization to evaluate thousands of positions simultaneously
2. **General-Purpose Architecture**: Minimal hand-crafted features - let the transformer learn strategy
3. **Strong Play**: Target 80-90% of XtremeGammon's strength at 2-5× speed
4. **Open Source**: Provide a modern reference implementation for backgammon AI research

## Architecture

```
Input (Board State)
    ↓
Encoder (26 positions → feature vectors)
    ↓
Transformer Encoder
  - Multi-head Self-Attention (learns positional relationships)
  - Feed-Forward Networks
  - Layer Normalization
  - Residual Connections
    ↓
Global Pooling (sequence → fixed representation)
    ↓
Output Heads
  - Value Head → Equity (win/gammon/backgammon probabilities)
  - Policy Head → Move probabilities (optional)
```

### Key Design Decisions

**1. Board Representation**
- 26 positions as sequence: [bar] + [points 1-24] + [off]
- Each position = token with features
- Self-attention learns relationships between positions

**2. Encoding Options** (configurable)
- Raw: Just checker counts (most general)
- One-hot: Discrete representation
- Geometric: Add positional info (distance to home, etc.)
- Strategic: Include features like blots, anchors (optional)

**3. Training Approach**
- Self-play game generation
- TD(0) learning with dice averaging (Jacob Hilton's insight)
- Experience replay buffer
- GPU-batched training

**4. Move Selection**
- 0-ply: Direct network evaluation
- 1-ply: Evaluate all moves, average over 21 opponent dice rolls
- 2-ply: Extend to opponent responses (GPU-parallelized)

## File Structure

This repository contains **interface specifications** (`.mli` files) that define the architecture:

```
transformer-backgammon/
├── types.mli           # Core type definitions
├── board.mli           # Board representation and game rules
├── encoder.mli         # Board encoding for neural networks
├── network.mli         # Transformer architecture
├── training.mli        # Training loop and experience replay
├── evaluation.mli      # Position evaluation and move selection
├── config.mli          # Configuration management
├── main.mli            # Entry points and CLI
└── README.md           # This file
```

## Implementation Roadmap

### Phase 1: Core Game Engine (Week 1-2)
- [ ] Implement `board.ml` - game rules, move generation
- [ ] Write comprehensive tests
- [ ] Benchmark move generation speed

### Phase 2: Neural Network (Week 3-4)
- [ ] Implement transformer in JAX/Flax (`network.ml`)
- [ ] Board encoding (`encoder.ml`)
- [ ] Test forward/backward passes

### Phase 3: Training Pipeline (Week 5-6)
- [ ] Self-play game generation
- [ ] Experience replay buffer
- [ ] Training loop
- [ ] Checkpointing and logging

### Phase 4: Evaluation (Week 7-8)
- [ ] N-ply search implementation
- [ ] Dice averaging optimization
- [ ] Benchmark against random/greedy baselines

### Phase 5: Scaling (Week 9-12)
- [ ] GPU optimization (profiling, batching)
- [ ] Large-scale training (100k+ games)
- [ ] Evaluation vs GNU Backgammon

### Phase 6: Advanced Features (Future)
- [ ] MCTS integration
- [ ] Doubling cube strategy
- [ ] Opening book
- [ ] Endgame databases

## Key Insights from Literature

### TD-Gammon (Tesauro, 1995)
- Self-play TD-learning achieves master-level play
- Hand-crafted features improve from intermediate → master level
- ~1.5M games needed for convergence

### Jacob Hilton's Implementation
- TD(0) with dice averaging works well
- 1-ply lookahead: evaluate all moves, average over opponent dice
- 2,500 games → 41% vs GNU BG (grandmaster level)
- Experience replay + Adam optimizer

### Stochastic MuZero (2022)
- AlphaZero-style approach CAN work for backgammon
- Matched GNU BG Grandmaster with learned model
- Shows general-purpose approaches are viable

### Our Innovation
- **Transformers** for learning positional relationships
- **GPU parallelization** for massive batching
- **Modern architectures** (deeper, wider than historical approaches)
- **Scale** - train for longer with more compute

## Comparison to XtremeGammon

| Aspect | XtremeGammon | This Project |
|--------|--------------|--------------|
| **Platform** | CPU-optimized | GPU-optimized |
| **Evaluation** | Sequential 2-3 ply | Batched 2-3 ply |
| **Network** | Shallow (~20k params) | Deep transformer (1M-50M params) |
| **Features** | Hand-crafted | Learned (transformer) |
| **Training** | Years of refinement | Modern RL + scale |
| **Speed** | Very fast on CPU | 10-100× faster on GPU (batched) |

## Getting Started

### Prerequisites
```bash
pip install jax jaxlib flax optax
pip install numpy pytest
pip install wandb  # optional: for experiment tracking
```

### Usage (once implemented)
```bash
# Train from scratch
python main.py train --config configs/baseline.json

# Evaluate against GNU BG
python main.py evaluate --model checkpoints/best.pkl --opponent gnubg

# Play interactively
python main.py play --model checkpoints/best.pkl

# Run benchmarks
python main.py benchmark --model checkpoints/best.pkl --suite xg_positions
```

## Configuration

See `config.mli` for all configuration options. Key parameters:

```python
# Transformer architecture
num_layers: 6          # Transformer blocks
embed_dim: 256         # Embedding dimension
num_heads: 8           # Attention heads
ff_dim: 1024          # Feed-forward hidden size

# Training
batch_size: 128
learning_rate: 1e-4
replay_buffer_size: 50000
games_per_iteration: 100

# Search
ply_depth: 1          # 0, 1, or 2
average_dice: true    # Use all 21 dice outcomes
```

## Performance Targets

| Milestone | Games Trained | Win Rate vs GNU BG | Status |
|-----------|---------------|-------------------|--------|
| Random baseline | 0 | ~5% | ⬜ |
| Learns basic rules | 100 | ~15% | ⬜ |
| Intermediate | 1,000 | ~25% | ⬜ |
| Advanced | 10,000 | ~35% | ⬜ |
| Expert (Jacob Hilton) | 2,500 | ~41% | ⬜ Target |
| Master | 100,000 | ~45% | ⬜ Stretch |

## References

- [TD-Gammon (Tesauro, 1995)](https://www.bkgm.com/articles/tesauro/tdl.html)
- [Jacob Hilton's Backgammon](https://github.com/jacobhilton/backgammon)
- [Stochastic MuZero (2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1)
- [Pgx: GPU-Accelerated Game Simulators](https://github.com/sotetsuk/pgx)
- [GNU Backgammon](https://www.gnu.org/software/gnubg/)

## License

MIT (to be determined)

## Contributing

Contributions welcome! This is a learning project - beginner-friendly.

Areas where help is needed:
- Game engine optimization
- Transformer architecture experiments
- Training at scale
- Benchmarking and evaluation
- Documentation

## Acknowledgments

- Gerald Tesauro for TD-Gammon
- Jacob Hilton for the OCaml implementation
- DeepMind for AlphaGo/AlphaZero/MuZero
- GNU Backgammon team
