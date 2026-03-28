# Periodic Code Review Log

---

## 2026-03-28 — Deep Dive Code Review (Agent Swarm, 6 Reviewers)

**Goal:** Validate training readiness before first small training run.

**Scope:** Full codebase review across 6 subsystems: core game engine, encoding, network architecture, training pipeline, evaluation system, dependencies/config.

### CRITICAL — Must Fix Before Training

1. **Broken imports in `network/__init__.py`** — Imports `equity_loss`, `train_step`, etc. from `network.network`, but they live in `training/losses.py` and `training/train.py`. Will `ImportError` at import time.

2. **Stray `@dataclass` decorator in `types.py` (line 419)** — Bare decorator above a comment block before `GameOutcome`. Harmless (stacks idempotently) but signals possible parse issues.

3. **Global features not wired into training** — `train.py` hardcodes `input_feature_dim=2`, ignoring the 8-dim global features (pip counts, contact, primes, bearoff). Per TODO.md, "the single highest-impact improvement available."

4. **TD targets not renormalized after clipping** — After clipping to [0,1], the 5-dim probability vector may not sum to 1.0, causing cross-entropy instability.

### HIGH PRIORITY — Fix Before Production Run

5. **Action encoder hash collisions** — 1024 hash buckets with 100+ legal moves (doubles) guarantees collisions. Non-blocking with `train_policy=False`.

6. **Cube drop awards wrong points** — `apply_cube_action(PASS)` reads post-double cube value instead of pre-double. Awards 2x correct points on cube drops.

7. **Cosine decay `decay_steps=100000` hardcoded** — v6e quick config only runs ~80 gradient steps. LR never leaves warmup. Acceptable for validation, must fix for production.

8. **`equity_weight=0.5` halves gradients in value-only mode** — Configured LR of 3e-4 effectively behaves like 1.5e-4.

9. **Value head softmax vs target compatibility** — 5-output equity head uses `nn.softmax` (outputs sum to 1.0). Must verify TD targets are compatible (normal loss = all zeros won't work with softmax).

10. **No checkpoint restoration** — `train()` saves checkpoints but never restores. Multi-hour runs can't resume.

### WARNINGS — Should Fix Soon

| # | Area | Issue |
|---|------|-------|
| 11 | Core | Missing "must use higher die" rule |
| 12 | Core | Move dedup sorts steps, potentially losing hit-order significance |
| 13 | Core | `pip_count` includes borne-off checkers at 25 pips |
| 14 | Core | Spurious partial moves in `_generate_moves_recursive` (10-100x bloat for doubles) |
| 15 | Network | muP half-implemented — missing init scaling and per-layer LR |
| 16 | Network | No final RMSNorm after transformer stack |
| 17 | Network | Conditional dropout creation fragile with Flax tracing |
| 18 | Encoding | `encode_boards` is pure Python double loop — training bottleneck |
| 19 | Encoding | Two different `compute_global_features` functions with different feature sets |
| 20 | Encoding | `select_move_from_policy` division by zero when all probs are 0.0 |
| 21 | Eval | Inconsistent gammon counting between evaluator.py (cumulative) and benchmark.py (exclusive) |
| 22 | Eval | 2-ply search redundant 0-ply pass, no cross-move batching |
| 23 | Eval | Non-reproducible baseline evaluation (unseeded random agent) |
| 24 | Training | LR logged from separate schedule object — misleading with schedule-free optimizer |
| 25 | Training | Color-flip augmentation doesn't flip legal_moves/move_taken |
| 26 | Config | `requirements.txt` stale vs `pyproject.toml` |

### INFO — Nice to Have

- No test coverage for CubeHead, muP, MuZero, or `play_games_batched`
- `forward_batch` is just an alias for `forward`
- `canonicalize_board` is a no-op
- Strategic features contain redundant info (stack height = raw feature)
- MD5 for transposition table hash (slower than needed)
- Duplicated equity-to-value conversion in network_agent.py and search.py

### Recommended Action Plan

**Minimum for v6e quick validation run (2,500 games):**
1. Fix broken imports in `network/__init__.py`
2. Ensure `train_policy=False` in config
3. Run `smoke_test.py` first

**For meaningful training signal:**
4. Wire global features into `input_feature_dim`
5. Add TD target renormalization after clipping
6. Set `equity_weight=1.0` for value-only mode

**Defer to next session:**
- Cube drop scoring, higher die rule, muP, checkpoint restoration, encoding vectorization, 2-ply batching
