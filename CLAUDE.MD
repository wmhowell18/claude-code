# Claude Code Context: Transformer Backgammon Project

## Project Overview

This repository contains experiments with Claude Code, specifically focusing on a **GPU/TPU-optimized backgammon AI** using transformer neural networks built with JAX/Flax.

### Main Subproject: transformer-backgammon

A modern backgammon bot leveraging:
- **Transformer architecture** for position evaluation
- **TD-learning** with experience replay (inspired by TD-Gammon and Jacob Hilton)
- **JAX/Flax** for GPU/TPU acceleration
- **N-ply lookahead** with dice averaging for move selection

## Repository Structure

```
claude-code/
├── README.md                          # Minimal root readme
└── transformer-backgammon/            # Main backgammon AI project
    ├── src/                           # Source code (Python)
    ├── tests/                         # Test suites
    ├── scripts/                       # Training/evaluation scripts
    ├── *.mli                          # Interface specifications (design docs)
    ├── colab_tpu_training.ipynb      # Google Colab TPU training notebook
    ├── PROJECT_STRUCTURE.md           # Detailed architecture and sprint plan
    ├── README.md                      # Comprehensive project documentation
    └── SETUP.md                       # Setup instructions
```

## Current Status

**Phase:** Interface design and early implementation

The project currently has:
- ✅ Complete interface specifications (.mli files) for all major components
- ✅ Google Colab TPU training notebook
- ✅ Project structure and sprint plan
- ⬜ Core game engine implementation (Sprint 1)
- ⬜ Neural network implementation (Sprints 2-3)
- ⬜ Training pipeline (Sprints 4-6)

## Key Technical Details

### Architecture
- **Input:** 26 positions as sequence: [bar] + [points 1-24] + [off]
- **Transformer:** Multi-head self-attention with 6-12 layers, 256-512 dimensions
- **Output:** Value head (equity) and optional policy head
- **Training:** Self-play with TD(0) learning and experience replay

### Technology Stack
- **ML Framework:** JAX/Flax (GPU/TPU optimized)
- **Language:** Python 3.8+
- **Testing:** pytest
- **Dependencies:** jax, jaxlib, flax, optax, numpy

### Performance Targets
| Milestone | Games | Win Rate vs GNU BG | Status |
|-----------|-------|-------------------|--------|
| Jacob Hilton baseline | 2,500 | ~41% | Target |
| Medium transformer | 25,000 | ~43% | Stretch |
| Large transformer | 100,000 | ~45% | Stretch |

## Important Files

### Documentation
- `transformer-backgammon/README.md` - Main project documentation, architecture, goals
- `transformer-backgammon/PROJECT_STRUCTURE.md` - Implementation roadmap with 8 sprints
- `transformer-backgammon/SETUP.md` - Setup and installation instructions

### Interface Specifications (.mli files)
These define the module contracts:
- `types.mli` - Core type definitions
- `board.mli` - Board representation and game rules
- `encoder.mli` - Board encoding for neural networks
- `network.mli` - Transformer architecture
- `training.mli` - Training loop and experience replay
- `evaluation.mli` - Position evaluation and move selection
- `config.mli` - Configuration management
- `main.mli` - Entry points and CLI

### Training Resources
- `colab_tpu_training.ipynb` - Ready-to-use Colab notebook for TPU training

## Development Guidelines

### When Working on This Project

1. **Follow the Sprint Plan:** Reference `PROJECT_STRUCTURE.md` for implementation order
2. **Test-Driven:** Write tests for each module before/during implementation
3. **GPU Optimization:** Design for batched operations and GPU parallelization
4. **Minimal Features:** Avoid hand-crafted features - let the transformer learn

### Code Organization
- **Core logic:** `src/core/` (types, board, dice)
- **Encoding:** `src/encoding/` (feature extraction)
- **Network:** `src/network/` (transformer architecture)
- **Training:** `src/training/` (self-play, replay buffer, optimizer)
- **Evaluation:** `src/evaluation/` (search, benchmarks)
- **Utilities:** `src/utils/` (config, checkpointing, logging)

### Key Design Principles

1. **GPU-First:** Batch everything, use JIT compilation
2. **Simple:** Prefer learned features over hand-crafted ones
3. **Modern:** Use latest JAX/Flax patterns and optimizations
4. **Tested:** Comprehensive unit and integration tests
5. **Reproducible:** Track experiments, checkpoints, and configs

## Quick Start Commands

### Development (when implemented)
```bash
# Install dependencies
cd transformer-backgammon
pip install -r requirements.txt

# Run tests
pytest tests/

# Train a model
python main.py train --config configs/baseline.json

# Evaluate against GNU BG
python main.py evaluate --model checkpoints/best.pkl --opponent gnubg

# Play interactively
python main.py play --model checkpoints/best.pkl
```

### Google Colab
Click the "Open in Colab" badge in `transformer-backgammon/README.md` for free TPU training.

## Context for Claude

### What This Project Is
A research/learning project to build a competitive backgammon AI using modern transformer architectures and GPU acceleration. The goal is to reach 80-90% of XtremeGammon's strength at 2-5× speed using learned features instead of hand-crafted ones.

### What This Project Isn't
- Not production-ready (yet)
- Not focused on CPU optimization (GPU/TPU only)
- Not using traditional shallow networks (modern transformers)

### Common Tasks You Might Help With
1. **Implementation:** Converting .mli interface specs to .py implementations
2. **Testing:** Writing comprehensive test suites for game logic and neural networks
3. **Optimization:** Profiling and optimizing GPU performance
4. **Training:** Setting up and monitoring training runs
5. **Evaluation:** Comparing against baselines (random, GNU BG)
6. **Documentation:** Explaining architecture decisions and implementation details

### Things to Watch Out For
1. **JAX Gotchas:** JAX requires pure functions, careful with side effects
2. **Memory:** GPU memory is limited, design for efficient batching
3. **Testing:** Game logic has many edge cases (bearing off, hitting, etc.)
4. **Performance:** Backgammon requires evaluating thousands of positions/sec
5. **Correctness:** Game rules must be perfect for valid training

## References

- [TD-Gammon (Tesauro, 1995)](https://www.bkgm.com/articles/tesauro/tdl.html) - Original TD-learning backgammon bot
- [Jacob Hilton's Backgammon](https://github.com/jacobhilton/backgammon) - OCaml implementation achieving 41% vs GNU BG
- [Stochastic MuZero (2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1) - AlphaZero-style approach for backgammon
- [GNU Backgammon](https://www.gnu.org/software/gnubg/) - Reference implementation for evaluation

## License

MIT (to be determined)

---

**Last Updated:** 2026-01-10
**Project Status:** Interface design complete, implementation in progress
**Current Sprint:** Sprint 1 - Core Game Engine
