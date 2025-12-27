# Project Structure

Detailed architecture and implementation plan for transformer-backgammon.

## Directory Structure

```
transformer-backgammon/
├── src/
│   ├── core/
│   │   ├── types.py           # Core type definitions (dataclasses)
│   │   ├── board.py           # Board representation and game rules
│   │   └── dice.py            # Dice utilities
│   │
│   ├── encoding/
│   │   ├── encoder.py         # Board → features encoding
│   │   ├── features.py        # Feature extraction functions
│   │   └── presets.py         # Encoding presets (raw, one-hot, etc.)
│   │
│   ├── network/
│   │   ├── transformer.py     # Main transformer architecture
│   │   ├── attention.py       # Multi-head attention implementation
│   │   ├── layers.py          # Transformer blocks, FFN, etc.
│   │   ├── heads.py           # Value and policy heads
│   │   └── configs.py         # Network configuration presets
│   │
│   ├── training/
│   │   ├── trainer.py         # Main training loop
│   │   ├── replay_buffer.py   # Experience replay implementation
│   │   ├── selfplay.py        # Self-play game generation
│   │   ├── optimizer.py       # Optimizer wrappers
│   │   └── metrics.py         # Training metrics and logging
│   │
│   ├── evaluation/
│   │   ├── evaluator.py       # Position evaluation
│   │   ├── search.py          # N-ply lookahead
│   │   ├── mcts.py            # Monte Carlo Tree Search (optional)
│   │   └── benchmarks.py      # Benchmark suites
│   │
│   └── utils/
│       ├── config.py          # Configuration management
│       ├── checkpointing.py   # Save/load models
│       ├── logging.py         # Logging utilities
│       └── visualization.py   # Attention visualization, etc.
│
├── tests/
│   ├── test_board.py          # Board logic tests
│   ├── test_encoder.py        # Encoding tests
│   ├── test_network.py        # Network tests
│   ├── test_training.py       # Training tests
│   └── test_evaluation.py     # Evaluation tests
│
├── configs/
│   ├── test.json              # Small config for testing
│   ├── baseline.json          # Jacob Hilton replica
│   ├── medium.json            # Medium transformer
│   └── large.json             # Large-scale training
│
├── benchmarks/
│   ├── gnu_bg_positions.json  # GNU BG benchmark positions
│   ├── xg_positions.json      # XG expert positions
│   └── expert_games.json      # Human expert games
│
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── play.py                # Interactive play
│   ├── analyze.py             # Game analysis
│   └── benchmark.py           # Run benchmarks
│
├── notebooks/
│   ├── 01_game_engine.ipynb   # Game engine exploration
│   ├── 02_encoding.ipynb      # Encoding experiments
│   ├── 03_network.ipynb       # Network architecture
│   └── 04_training.ipynb      # Training visualization
│
├── docs/
│   ├── architecture.md        # Detailed architecture docs
│   ├── training_guide.md      # How to train models
│   ├── evaluation_guide.md    # How to evaluate
│   └── api.md                 # API documentation
│
├── checkpoints/               # Saved models (gitignored)
├── logs/                      # Training logs (gitignored)
├── data/                      # Training data (gitignored)
│
├── main.py                    # Main CLI entry point
├── setup.py                   # Package setup
├── requirements.txt           # Python dependencies
├── README.md                  # Main readme
└── PROJECT_STRUCTURE.md       # This file
```

## Module Dependencies

```
types.py (no dependencies)
  ↓
board.py ← dice.py
  ↓
encoder.py ← features.py
  ↓
network/transformer.py ← attention.py, layers.py, heads.py
  ↓
evaluation/evaluator.py ← search.py
  ↓
training/trainer.py ← replay_buffer.py, selfplay.py, optimizer.py
  ↓
main.py (CLI)
```

## Implementation Order

### Sprint 1: Core Game Engine (Week 1)
**Goal**: Working backgammon game logic

```
1. types.py
   - Define Board, Move, Dice types
   - Core data structures

2. dice.py
   - Dice rolling
   - All 21 dice outcomes
   - Dice utilities

3. board.py
   - Board initialization
   - Move generation
   - Move application
   - Game state queries

4. tests/test_board.py
   - Test all game rules
   - Edge cases
   - Performance benchmarks
```

**Deliverable**: `python -m pytest tests/test_board.py` passes

---

### Sprint 2: Encoding (Week 2)
**Goal**: Convert boards to neural network inputs

```
1. features.py
   - Raw encoding
   - One-hot encoding
   - Geometric features
   - Strategic features

2. encoder.py
   - Main encoding pipeline
   - Batching utilities
   - Equity encoding/decoding

3. presets.py
   - Encoding configurations
   - Feature dimension calculations

4. tests/test_encoder.py
   - Test all encoding variants
   - Verify shapes
   - Benchmark speed
```

**Deliverable**: Can encode 1000 boards/sec

---

### Sprint 3: Network Architecture (Week 3-4)
**Goal**: Working transformer in JAX

```
1. network/layers.py
   - Basic dense layers
   - Layer normalization
   - Dropout

2. network/attention.py
   - Multi-head attention
   - Positional encoding

3. network/heads.py
   - Value head (equity)
   - Policy head (optional)

4. network/transformer.py
   - Full transformer encoder
   - Forward pass
   - Parameter initialization

5. network/configs.py
   - Small, medium, large configs

6. tests/test_network.py
   - Test forward pass
   - Test gradient flow
   - Test parameter count
```

**Deliverable**: Forward pass on batch of 128 boards

---

### Sprint 4: Training Pipeline (Week 5-6)
**Goal**: Train a network via self-play

```
1. training/replay_buffer.py
   - Circular buffer
   - Sampling
   - Game → examples conversion

2. training/selfplay.py
   - Single game simulation
   - Parallel game generation
   - GPU-batched evaluation

3. training/optimizer.py
   - Adam optimizer wrapper
   - Learning rate schedules

4. training/trainer.py
   - Main training loop
   - Checkpointing
   - Logging

5. training/metrics.py
   - Loss tracking
   - Performance metrics
   - Wandb integration

6. tests/test_training.py
   - Test replay buffer
   - Test training step
   - Test convergence (toy problem)
```

**Deliverable**: Train small network for 100 games

---

### Sprint 5: Evaluation (Week 7-8)
**Goal**: Strong move selection

```
1. evaluation/evaluator.py
   - Position evaluation
   - Batched evaluation
   - Equity → expected value

2. evaluation/search.py
   - 0-ply (greedy)
   - 1-ply (dice averaging)
   - 2-ply (opponent response)
   - Pruning optimizations

3. evaluation/benchmarks.py
   - Load benchmark positions
   - Run evaluation
   - Compare to known best moves

4. tests/test_evaluation.py
   - Test search correctness
   - Test performance
   - Test vs baselines
```

**Deliverable**: 1-ply search evaluates 1000 pos/sec

---

### Sprint 6: Integration (Week 9-10)
**Goal**: End-to-end training and evaluation

```
1. main.py
   - CLI argument parsing
   - Dispatch to train/eval/play

2. scripts/train.py
   - Training script
   - Config loading
   - Experiment tracking

3. scripts/evaluate.py
   - Evaluation against baselines
   - GNU BG integration

4. scripts/play.py
   - Interactive play
   - Human vs AI

5. utils/config.py
   - Config management
   - Validation

6. utils/checkpointing.py
   - Save/load utilities

7. utils/logging.py
   - Structured logging
```

**Deliverable**: `python main.py train --config configs/baseline.json`

---

### Sprint 7: Optimization (Week 11-12)
**Goal**: GPU performance optimization

```
1. Profile everything
   - Identify bottlenecks
   - CPU vs GPU time
   - Memory usage

2. Optimize encoding
   - JIT compile
   - Vectorize operations

3. Optimize network
   - Batch size tuning
   - Mixed precision training

4. Optimize search
   - Parallel move evaluation
   - Efficient batching

5. Benchmark
   - Positions/second
   - Games/minute
   - Training throughput
```

**Deliverable**: 10× faster than naive implementation

---

### Sprint 8: Scale Up (Week 13-16)
**Goal**: Train strong models

```
1. Train baseline (Jacob Hilton replica)
   - 5×400 network
   - 2,500 games
   - Target: 41% vs GNU BG

2. Train medium transformer
   - 6 layers, 256 dim
   - 25,000 games
   - Target: 43% vs GNU BG

3. Train large transformer
   - 12 layers, 512 dim
   - 100,000 games
   - Target: 45% vs GNU BG

4. Hyperparameter tuning
   - Learning rate
   - Batch size
   - Network architecture

5. Analysis
   - Visualize attention
   - Analyze mistakes
   - Compare to XG
```

**Deliverable**: Model competitive with GNU BG

---

## Data Flow

### Training Loop
```
1. Generate self-play games (GPU-parallelized)
   → game_records

2. Convert games to training examples
   → training_examples

3. Add to replay buffer
   → replay_buffer

4. Sample minibatch
   → batch

5. Encode boards
   → encoded_boards [batch, 26, features]

6. Forward pass (transformer)
   → network_output

7. Compute loss
   → loss

8. Backward pass
   → gradients

9. Update parameters (Adam)
   → new_params

10. Repeat
```

### Evaluation Flow
```
1. Load board state
   → board

2. Generate legal moves
   → legal_moves

3. For each move:
   a. Apply move → next_board
   b. For each of 21 opponent dice:
      i. Generate opponent moves
      ii. Evaluate resulting positions (batched)
      iii. Compute expected equity

4. Select move with best expected equity
   → best_move
```

## Testing Strategy

### Unit Tests
- Each module has comprehensive tests
- Test edge cases, invalid inputs
- Performance benchmarks

### Integration Tests
- End-to-end training
- End-to-end evaluation
- Config loading and validation

### Regression Tests
- Benchmark positions (known correct moves)
- Performance regressions
- Numerical stability

### Performance Tests
- Positions/second benchmarks
- Memory usage profiling
- GPU utilization

## Deployment

### Model Export
```
1. Train model
   → checkpoints/model_v1.pkl

2. Convert to inference format
   → models/model_v1_inference.pkl

3. Optional: Export to ONNX
   → models/model_v1.onnx

4. Optional: Export to TFLite
   → models/model_v1.tflite
```

### Inference Server
```
FastAPI server with endpoints:
- POST /evaluate - Evaluate position
- POST /select_move - Get best move
- POST /analyze_game - Analyze game
```

## Metrics to Track

### Training Metrics
- Equity loss (cross-entropy)
- Training games played
- Positions trained on
- Training throughput (games/min)

### Evaluation Metrics
- Win rate vs random
- Win rate vs GNU BG (0-ply, 1-ply, 2-ply)
- Error rate on benchmark positions
- Equity prediction accuracy

### Performance Metrics
- Positions evaluated/second
- GPU utilization
- Memory usage
- Batch size achieved

## Next Steps

1. ✅ Define interfaces (.mli files)
2. ⬜ Implement core game engine
3. ⬜ Implement encoding
4. ⬜ Implement transformer
5. ⬜ Implement training loop
6. ⬜ Implement evaluation
7. ⬜ Integration and testing
8. ⬜ Optimization
9. ⬜ Scale up training
10. ⬜ Compare to XG/GNU BG
