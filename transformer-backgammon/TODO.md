# Backgammon Transformer — Feature Roadmap & TODO

> **Status**: Search, benchmarking, training improvements, and doubling cube complete. 0/1/2-ply search with batch evaluation, move ordering with progressive deepening, transposition table, and TD(lambda) training implemented. Win rate tracking, benchmark positions, position weighting, validation splits, and early stopping integrated into training loop. Race equity formula for pure race positions. Full doubling cube with match play, match equity tables, cubeful equity, and cube decision network head.
>
> **Current Maturity**: ~7/10 for competitive play. Solid game engine + neural training + search + benchmarking + TD(lambda) + exploration schedule + global encoding features + race evaluation + training infrastructure (position weighting, validation, early stopping) + doubling cube + match play. Missing GnuBG interface, MCTS, and rollout-based training.
>
> **Known Issues (Feb 2026)**: Code review identified several correctness bugs requiring attention before long training runs: policy head is non-functional (hash collisions + dummy targets), global features are implemented but not wired into training (input_feature_dim hardcoded to 2), TD targets are not renormalized before cross-entropy loss, and training telemetry is misleading (train_acc always 0.0, LR logged as constant peak). See "KNOWN ISSUES" section below.

---

## How to Use This Document

Items are grouped by priority tier. Within each tier, items are roughly ordered by impact. Each item has:
- **Status**: `[ ]` todo, `[x]` done, `[~]` in progress, `[-]` skipped/deferred
- **Effort**: S (hours), M (days), L (weeks)
- **Impact**: How much this improves the bot's playing strength

---

## KNOWN ISSUES: Fix Before Long Runs

*Correctness bugs and silent failures identified in code review (Feb 2026). These directly affect
training quality and telemetry — address before committing to multi-hour TPU runs.*

- [ ] **101. `train_acc` always 0.0** — `train.py` initializes `train_acc = 0.0` and never
  updates it during the training loop. `compute_metrics()` computes accuracy but is only called for
  validation. Training logs always show 0.0 accuracy, making this metric useless for monitoring
  convergence. Fix: call `compute_metrics()` on a training minibatch at eval checkpoints, or remove
  the metric. (Effort: S, Impact: diagnostic)

- [ ] **102. LR logged as constant peak LR** — `train.py:547-549` hardcodes
  `current_lr = config.learning_rate` instead of reading the actual scheduled value from optax
  state. The warmup and cosine decay schedule is running correctly, but logs always show the peak
  LR — making LR curves completely uninformative for debugging schedule issues. Fix: extract actual
  LR from the optax state via `state.opt_state`. (Effort: S, Impact: diagnostic)

- [ ] **103. TD targets not renormalized before cross-entropy** — `td_lambda.py` clips each
  component of the 6-dim target to [0, 1] elementwise, then converts to 5-dim and passes to
  `compute_equity_loss`. The result may not sum to 1 (or may sum near 0 for large eligibility
  traces), which collapses gradient signal for those samples and can cause cross-entropy instability.
  Fix: after clipping, renormalize the 5-dim target to sum to 1 (or to skip samples where sum < ε).
  (Effort: S, Impact: training quality)

- [ ] **104. Policy head non-functional — disable until fixed** — Two compounding issues: (1)
  `action_encoder.py` maps moves to 1024 slots via a hash with frequent collisions for the 20-100+
  legal moves in typical positions, making one-hot targets ambiguous; (2) `replay_buffer.py:
  sample_batch` returns pre-allocated dummy zero arrays for policy targets and all-true masks
  regardless of actual position. The policy head receives zero meaningful gradient. Training with
  `train_policy=True` wastes compute and may destabilize the value head. Fix: disable policy head
  for value-only training, then address item 59 (precomputed action lookup) to fix the root cause.
  (Effort: M, Impact: large — fixes training stability and removes wasted compute)

- [ ] **105. Global features not wired into training** — Items 25-28, 31-32 are marked complete,
  but `train.py:282` hardcodes `input_feature_dim=2` (raw 2-feature encoding only). The 8-dim
  global features implemented in `encode_global_board_features()` — pip counts, contact detection,
  prime length, bearoff progress, race equity — are never seen by the network during training. The
  `enhanced_encoding_config()` and `full_encoding_config()` presets exist but are unused. Fix:
  switch training to use `full_encoding_config()` and update `input_feature_dim` accordingly. This
  is the single highest-impact improvement available without new code. (Effort: S, Impact: large)

- [ ] **106. Dead `train_step` in `network.py`** — `network.py:769` defines a second
  `@jax.jit train_step` with a different loss formulation: no gradient clipping, raw cross-entropy
  without the combined loss wrapper. The main training loop imports `train_step` from `losses.py`.
  The `network.py` version is dead code, but risks being imported by mistake and producing silently
  wrong behavior (different gradients, no clipping). Fix: delete the dead function and the
  associated dead helpers (`equity_loss`, `mse_equity_loss`, `policy_loss`, `total_loss`,
  `create_train_state`, `eval_step`). (Effort: S, Impact: safety/clarity)

- [ ] **107. Stale `TrainingConfig` in `types.py`** — `types.py` contains a `TrainingConfig`
  dataclass that conflicts with the real `TrainingConfig` in `train.py`. The one in `types.py` is
  an older, simpler version that is not used anywhere in the training loop. Risks confusion and
  wrong-import bugs. Fix: delete it. (Effort: S, Impact: clarity)

---

## TIER 1: Critical for Competitive Play

*These items represent the gap between "toy project" and "functional bot". Do these first.*

### Search & Lookahead (the #1 gap)

- [x] **1. 1-ply lookahead with dice averaging** — Evaluate all legal moves, for each apply the move, then average the equity over all 21 dice rolls. This is the single biggest strength multiplier (~10x). See Jacob Hilton's approach. (Effort: M, Impact: massive) *(Feb 2025)*
- [x] **2. 2-ply search** — After your move, consider opponent's best response across all their dice rolls. (Effort: M, Impact: large) *(Feb 2026)*
- [x] **3. Batch evaluation of multiple positions** — Feed all candidate resulting positions into the network in one GPU forward pass instead of one-by-one. Critical for search speed. (Effort: S, Impact: large for speed) *(Feb 2025)*
- [x] **4. Move ordering heuristics + progressive deepening** — Heuristic-based move scoring (hits, bearoffs, made points, blots) for ordering. 2-ply search uses progressive deepening: evaluate all moves at 0-ply, then only top-k candidates at 2-ply. `order_moves()` and `select_move_2ply_pruned()` in `evaluation/search.py`. (Effort: S, Impact: moderate) *(Feb 2026)*
- [x] **5. Transposition table / position cache** — MD5-based board hashing with configurable cache size. Stores position evaluations keyed by board state + ply depth. LRU-style eviction. (Effort: M, Impact: moderate) *(Feb 2026)*
- [x] **6. Parallel move evaluation** — Batch all candidate resulting positions into one network call. Complementary to item 3. (Effort: S, Impact: large for speed) *(Feb 2025)*

### Doubling Cube (essential for real backgammon)

- [x] **7. Cube state tracking in Board** — CubeState, CubeOwner, CubeAction types in `core/types.py`. Full cube module at `core/cube.py` with can_double, apply_cube_action, legal_cube_actions. (Effort: S, Impact: foundational) *(Feb 2026)*
- [x] **8. Cube decision network** — CubeHead in `network/network.py` with 4-output (no_double, double, take, pass). Optional via `use_cube_head` config. Cube state encoding in `core/cube.py`. (Effort: M, Impact: large) *(Feb 2026)*
- [x] **9. Match equity tables** — Kazaross/Rockwell 15x15 MET in `core/cube.py`. Match state tracking with Crawford rule. (Effort: S, Impact: large for match play) *(Feb 2026)*
- [x] **10. Cube-aware equity calculation** — Cubeful equity for money and match play. should_double/should_take decisions. evaluate_cube_decision quality metrics. (Effort: S, Impact: large) *(Feb 2026)*

### Evaluation & Benchmarking (can't improve what you can't measure)

- [ ] **11. GnuBG interface** — Play games against gnubg, compare equity estimates position-by-position. (Effort: M, Impact: critical for development)
- [x] **12. Benchmark position suite** — 10 curated positions covering opening, race, bearoff, contact, prime, gammon threat categories with known equities. (Effort: M, Impact: critical) *(Feb 2026)*
- [x] **13. Win rate tracking over training** — Periodic evaluation callback in training loop: win rate vs random & pip count agents, plus equity error on benchmark positions. (Effort: S, Impact: large) *(Feb 2026)*
- [ ] **14. Move accuracy metric** — % of moves matching gnubg's top choice at 2-ply. (Effort: S, Impact: large)
- [x] **15. Equity error metric** — MAE/RMSE/max-error of network equity vs expected equity on benchmark positions. Computed per-position with detailed breakdown. (Effort: S, Impact: large) *(Feb 2026)*

---

## TIER 2: Major Strength Improvements

*Do these after Tier 1 is working. Each one meaningfully improves play quality.*

### Training Methodology

- [x] **16. TD(lambda) returns** — Full TD(lambda) implementation following TD-Gammon (Tesauro 1995). Records network equity estimates during self-play, computes targets using backward eligibility traces in fixed-perspective (White) 6-dim equity space. Configurable lambda (default 0.7). Integrated into replay buffer and training loop. (Effort: M, Impact: large) *(Feb 2026)*
- [ ] **17. N-step bootstrapping** — Use V(s') from the network for truncated episodes instead of waiting for game end. (Effort: M, Impact: large)
- [ ] **18. Rollout-based training targets** — 1-ply rollout equity gives much better targets than raw game outcome. Key technique from gnubg. (Effort: L, Impact: large)
- [x] **19. Exploration schedule** — Linear temperature decay from 1.0 → 0.1 over training. Configurable start/end temperatures. Logged per-batch. Warmstart phase uses fixed temperature. (Effort: S, Impact: moderate) *(Feb 2026)*
- [ ] **20. Opponent diversity** — Periodically play against older snapshots and baseline agents, not just self. Prevents forgetting. (Effort: M, Impact: moderate)
- [x] **21. Position weighting** — Weight positions by game progress (late game = higher weight) and equity uncertainty (50/50 positions = higher weight). Integrated into replay buffer with weighted sampling. (Effort: S, Impact: moderate) *(Feb 2026)*
- [ ] **22. More warmstart games** — Current 500 games is thin. Consider 2000-5000 pip count warmstart games. (Effort: S, Impact: small)
- [x] **23. Performance-tied LR scheduling** — Validation loss plateau detection with warnings. Logs plateau counter to metrics. Used alongside cosine schedule and early stopping. (Effort: S, Impact: moderate) *(Feb 2026)*
- [x] **24. Gradient clipping diagnostics** — Log max gradient norm and clip frequency per batch. Console output shows when clipping activates. Tracked in metrics. (Effort: S, Impact: small) *(Feb 2026)*

### Encoding Improvements

- [x] **25. Contact vs race detection** — Binary feature indicating whether checkers are still in contact. `is_past_contact()` in `core/board.py` and encoded in `encode_global_board_features()`. (Effort: S, Impact: large) *(Feb 2026)*
- [x] **26. Pip count as input feature** — Normalized pip counts (our/opp/diff) in `encode_global_board_features()`. Part of global features broadcast to all positions. (Effort: S, Impact: moderate) *(Feb 2026)*
- [x] **27. Home board control features** — `home_board_points_made()` in `core/board.py`, encoded as fraction (0-1) in global features. (Effort: S, Impact: moderate) *(Feb 2026)*
- [x] **28. Prime detection** — Length of longest prime for each player, capped at 6, normalized. Part of global features. (Effort: S, Impact: moderate) *(Feb 2026)*
- [ ] **29. Blot vulnerability encoding** — Which checkers are within direct/indirect hitting range. (Effort: M, Impact: moderate)
- [ ] **30. Escape features** — How many dice rolls let a trapped/back checker escape. (Effort: M, Impact: moderate)
- [x] **31. Race equity estimate** — `race_equity_estimate()` using adjusted pip count / Keith count formula in `core/board.py`. Also available standalone in `evaluation/race.py`. Encoded in global features. (Effort: S, Impact: small) *(Feb 2026)*
- [x] **32. Bearoff progress features** — Fraction of checkers borne off for each player in `encode_global_board_features()`. (Effort: S, Impact: small) *(Feb 2026)*

### Architecture

- [ ] **33. Architecture ablation study** — Compare transformer vs MLP vs CNN on same data. Transformer may be overkill for 26-position sequence. (Effort: L, Impact: informational)
- [ ] **34. Attention visualization** — Render attention maps to see what positions the network focuses on. Debugging tool. (Effort: S, Impact: informational)
- [ ] **35. Learned temperature on equity softmax** — Let the network learn how "sharp" its equity distribution should be. (Effort: S, Impact: small)

---

## TIER 3: Serious Bot Features

*These make the bot competitive in real play scenarios.*

### Match Play

- [x] **36. Match score tracking** — MatchState in `core/types.py` and `core/cube.py` with update_match_score, is_match_over, match_winner. (Effort: S, Impact: foundational) *(Feb 2026)*
- [ ] **37. Score-dependent strategy** — Different play when leading vs trailing in match. (Effort: M, Impact: large for match play)
- [x] **38. Match equity table implementation** — Kazaross/Rockwell 15×15 table in `core/cube.py`. See item 9. (Effort: S, Impact: large for match play) *(Feb 2026)*
- [x] **39. Cube handling in match context** — can_double_in_match, cubeful equity for match play, encode_match_state. Gammon-go/too-good logic via should_double/should_take. (Effort: M, Impact: large) *(Feb 2026)*
- [x] **40. Crawford rule and post-Crawford** — Crawford/post-Crawford state tracking, is_crawford_game check disables doubling. (Effort: S, Impact: moderate) *(Feb 2026)*

### Endgame

- [ ] **41. Bearoff database (<=6 checkers)** — Precomputed perfect play for simplified endgame positions. (Effort: L, Impact: large for endgame)
- [x] **42. Race equity formula** — Effective Pip Count / Keith count for pure race evaluation. Implemented in `evaluation/race.py` and `core/board.py` with EPC corrections and sigmoid equity mapping. (Effort: S, Impact: moderate) *(Feb 2026)*
- [ ] **43. One-sided bearoff database** — No contact, just bear off optimally. (Effort: M, Impact: moderate)
- [ ] **44. Contact endgame tablebases** — Perfect play for simple contact positions. (Effort: L, Impact: moderate)

### MCTS / Advanced Search

- [ ] **45. MCTS implementation** — Stub exists in network_agent.py, not implemented. (Effort: L, Impact: large)
- [ ] **46. MCTS with neural network prior** — AlphaZero-style: use policy head as prior, value head for leaf evaluation. (Effort: L, Impact: large)
- [ ] **47. Rollout policy for MCTS leaf evaluation** — Lightweight policy for fast rollouts. (Effort: M, Impact: moderate)
- [ ] **48. Progressive widening** — Gradually expand MCTS tree to manage branching factor. (Effort: M, Impact: moderate)
- [ ] **49. Time management** — Allocate more search time to critical/complex positions. (Effort: S, Impact: moderate)

### Training Infrastructure

- [ ] **50. Distributed training** — Scale across multiple GPUs/TPUs with data parallelism. (Effort: L, Impact: large for speed)
- [ ] **51. Async game generation** — Generate self-play games in parallel with gradient updates. (Effort: M, Impact: large for speed)
- [ ] **52. Hyperparameter search** — Grid or Bayesian optimization over architecture, LR, batch size, etc. (Effort: M, Impact: moderate)
- [x] **53. Train/validation/test splits** — Games randomly split between training and validation buffers (configurable fraction, default 10%). Validation loss computed at eval checkpoints. (Effort: S, Impact: moderate) *(Feb 2026)*
- [x] **54. Early stopping** — Training stops when validation loss hasn't improved for N consecutive eval checkpoints (configurable patience, default 5). Saves best model automatically. (Effort: S, Impact: moderate) *(Feb 2026)*
- [x] **55. Best-model tracking** — Saves best model checkpoint (by validation loss) to a dedicated `checkpoints/best/` directory, separate from rolling checkpoints. (Effort: S, Impact: small) *(Feb 2026)*

---

## TIER 4: Polish & Performance

*Optimization and developer experience improvements.*

### Speed & Efficiency

- [ ] **56. JIT compile evaluation pipeline** — Not just train_step, also inference and search. (Effort: S, Impact: moderate)
- [ ] **108. Search encoding path not vectorized** — `search.py:_encode_boards_batch` uses a
  double Python loop: 26 `extract_position_features` calls per board. The fast vectorized path
  used in `self_play.py` directly slices numpy arrays without Python-level iteration. Every 1-ply
  or 2-ply evaluation checkpoint call is ~26× slower than necessary for encoding alone. Fix:
  replace `_encode_boards_batch` with vectorized numpy slicing matching the `_encode_boards_fast()`
  pattern in `self_play.py`. (Effort: S, Impact: moderate — significant if eval_ply >= 1)
- [ ] **109. 2-ply search makes k×21 separate network calls** — `select_move_2ply` evaluates each
  legal move sequentially via `evaluate_move_2ply()`, which internally calls
  `_evaluate_position_1ply` for each of 21 opponent dice outcomes individually. For k=20 legal
  moves that is 420 separate network calls. By contrast, `select_move_1ply` correctly batches all
  boards from all moves and all dice into a single call. 2-ply evaluation during training
  checkpoints is therefore 20-50× slower than it should be. Fix: restructure 2-ply outer loop to
  collect all (move, dice_roll) result boards first, run one batched call, then assign values. More
  fundamental than item 56. (Effort: M, Impact: large)
- [ ] **110. Replay buffer sampling via Python list comprehension** — `replay_buffer.py:
  sample_batch` gathers encoded boards with `[self._encoded_boards[i] for i in indices]`.
  `_encoded_boards` is a Python list, not a numpy array, so vectorized indexing is unavailable.
  For batch_size=512, this is 512 individual Python list lookups. Fix: convert `_encoded_boards`
  to a pre-allocated numpy array at init, enabling `self._encoded_boards_arr[indices]` with a
  single vectorized op. (Effort: S, Impact: small-moderate at buffer sizes >50K)
- [ ] **57. Optimize move generation** — Currently pure Python loops. Could use numpy vectorization or Cython. (Effort: M, Impact: moderate)
- [ ] **58. Legal move caching within a turn** — Don't regenerate legal moves for same board+dice. (Effort: S, Impact: small)
- [ ] **59. Precomputed action-to-move lookup** — Replace hash-based encoding with direct lookup table. (Effort: S, Impact: small)
- [x] **60. Batch self-play** — `play_games_batched()` in `self_play.py` plays N games simultaneously, collecting all boards from all active games into a single batched network call per step. JIT-cached inference with padding for variable game counts. (Effort: L, Impact: large for speed) *(Feb 2026)*
- [x] **61. TPU-specific optimizations** — bfloat16 mixed precision, v6e-1 training config preset, batch sizing for single-chip TPU. Multi-chip pmap/sharding deferred until needed. (Effort: M, Impact: moderate on TPU) *(Feb 2026)*
- [x] **62. Mixed precision training** — bfloat16 compute with float32 params via Flax dtype/param_dtype. LayerNorm and softmax kept in float32 for stability. (Effort: S, Impact: moderate) *(Feb 2026)*

### Data & Knowledge

- [ ] **63. Opening book** — Pre-computed best moves for first 3-4 moves from gnubg/XtremeGammon. (Effort: S, Impact: moderate)
- [ ] **64. Pre-training on expert games** — Train on gnubg rollout data or master-level human games before self-play. (Effort: M, Impact: large)
- [ ] **65. Synthetic position generation** — Generate underrepresented board states for training diversity. (Effort: M, Impact: moderate)
- [ ] **66. Data augmentation via board flipping** — Already partially implemented in encoder.py, not used in training loop. Wire it up. (Effort: S, Impact: small)
- [ ] **67. Position classification labels** — Label positions as race/contact/holding/back game/blitz/prime. Useful for per-category evaluation. (Effort: M, Impact: informational)
- [ ] **68. Disagreement training** — Find positions where network and gnubg disagree, train harder on those. (Effort: M, Impact: moderate)

### Evaluation Extensions

- [ ] **69. Error categorization** — Classify mistakes as tactical (blunder) vs strategic (wrong plan). (Effort: M, Impact: informational)
- [ ] **70. Per-phase evaluation** — Measure opening, middlegame, endgame accuracy separately. (Effort: S, Impact: informational)
- [ ] **71. Gammon rate accuracy** — Does the network correctly predict gammon/backgammon probability? (Effort: S, Impact: moderate)
- [ ] **72. Win-rate confidence intervals** — Statistical significance of benchmark results. Need enough games. (Effort: S, Impact: informational)
- [ ] **73. Elo rating system** — Track Elo against baselines over training. (Effort: M, Impact: informational)
- [ ] **74. Live play interface** — Even text-based CLI for human testing. (Effort: M, Impact: moderate for testing)
- [ ] **75. Regression testing** — Ensure new models don't lose strength on specific position types. (Effort: M, Impact: moderate)

---

## TIER 5: Championship-Level Features

*For when you're pushing toward top-tier play.*

### Advanced Training

- [ ] **76. Self-play against past snapshots** — Play against frozen copies of earlier versions to avoid catastrophic forgetting. (Effort: M, Impact: moderate)
- [ ] **77. Population-based training** — Maintain multiple agents with different hyperparameters, breed the best. (Effort: L, Impact: moderate)
- [ ] **78. Weakness-targeted curriculum** — Automatically train more on position types where the bot loses most equity. (Effort: M, Impact: moderate)
- [ ] **79. Phase-specific networks** — Separate specialized networks for contact, race, and bearoff. (Effort: L, Impact: moderate)
- [ ] **80. Ensemble of networks** — Average equity from multiple models for more robust evaluation. (Effort: M, Impact: moderate)
- [ ] **81. Knowledge distillation** — Train smaller "fast" network from large "slow" one for real-time play. (Effort: M, Impact: moderate)
- [ ] **82. Dirichlet noise injection** — Add noise to prior during self-play (AlphaZero technique) for exploration. (Effort: S, Impact: small)
- [ ] **83. Resign detection** — Network knows when position is hopeless, saves training time on decided games. (Effort: S, Impact: small)
- [ ] **84. Symmetry exploitation** — Explicitly use black/white symmetry and board reflection in training. (Effort: S, Impact: small)

### Competitive Play

- [ ] **85. FIBS interface** — Connect to First Internet Backgammon Server for online play. (Effort: L, Impact: large for testing)
- [ ] **86. Match format support** — 1-point, 3-point, 5-point, 7-point matches. (Effort: S, Impact: moderate)
- [ ] **87. Chouette support** — Multi-player variant. (Effort: M, Impact: niche)
- [ ] **88. Jacoby rule** — Gammons don't count if cube not turned (money game variant). (Effort: S, Impact: small)
- [x] **89. Beaver/raccoon support** — Beaver action implemented in `core/cube.py` apply_cube_action. (Effort: S, Impact: niche) *(Feb 2026)*
- [ ] **90. Automatic doubling support** — House rule variant. (Effort: S, Impact: niche)

---

## TIER 6: Nice-to-Have

*Quality of life, UI, deployment.*

- [ ] **91. Web UI** — Play against the bot in a browser. (Effort: L, Impact: demo)
- [ ] **92. Position analysis tool** — Show equity, best moves, error magnitude for any position. (Effort: M, Impact: moderate)
- [ ] **93. Game annotation** — Mark blunders, missed doubles, questionable plays in a completed game. (Effort: M, Impact: moderate)
- [ ] **94. Training dashboard** — Real-time loss curves, win rates, position counts in a web UI. (Effort: M, Impact: convenience)
- [ ] **95. Model export** — ONNX or TFLite for deployment on mobile/edge. (Effort: M, Impact: deployment)
- [ ] **96. Adjustable skill level** — Play at different strengths for practice/teaching. (Effort: S, Impact: nice-to-have)
- [ ] **97. Opening theory extraction** — Analyze what opening moves the trained bot prefers and why. (Effort: S, Impact: informational)
- [ ] **98. Position difficulty estimation** — How hard is this position to evaluate correctly? (Effort: M, Impact: informational)
- [ ] **99. Explainability features** — Why did the bot choose this move? Feature attribution. (Effort: L, Impact: nice-to-have)
- [ ] **100. Game replay viewer** — Load and step through recorded games. (Effort: M, Impact: nice-to-have)

---

## Recommended Order of Attack

1. ~~**Items 1-3, 6** (1-ply search + batch eval)~~ — DONE (search.py)
2. ~~**Items 12, 13, 15** (benchmarking)~~ — DONE (benchmark.py + training loop integration)
3. ~~**Items 4-5** (move ordering + transposition table)~~ — DONE (search.py: progressive deepening, TranspositionTable)
4. ~~**Item 16** (TD(lambda))~~ — DONE (td_lambda.py + self_play.py + replay_buffer.py)
5. ~~**Items 19, 25-28** (exploration schedule + encoding improvements)~~ — DONE (encoder.py + train.py)
6. ~~**Items 7-10** (doubling cube)~~ — DONE (core/cube.py + types + network cube head)
7. **Fix items 103-105** (TD target normalization, disable policy head, wire global features) — fix these before any long training run, as they directly affect gradient signal and convergence
8. **Longer training run** with the improved pipeline — see where the model plateaus
8. **Items 17-18** (N-step bootstrapping + rollout targets) — even better training signal
9. **Items 45-46** (MCTS) — sophisticated search for strongest play

---

## Completed Items (for reference)

- [x] **Fix value target corruption** — `replay_buffer.py:66` compared `GameOutcome` to string `'white_wins'`, always False. All targets were -1.0. Fixed to use `outcome_to_equity()`. (Feb 2025)
- [x] **Fix value loss semantic mismatch** — `losses.py` used MSE on a scalar weighted sum vs scalar target. Fixed to use cross-entropy on 5-dim equity distribution. (Feb 2025)
- [x] **Fix nondeterministic move encoding** — `network_agent.py:146` used `hash(move)` (Python-randomized). Fixed to use deterministic `encode_move_to_action()`. (Feb 2025)
- [x] **Implement prepare_training_batch** — Was returning dummy zeros. Now properly encodes boards and computes equity targets. (Feb 2025)
- [x] **Add smoke test script** — `scripts/smoke_test.py` validates loss decreases before committing to long training runs. (Feb 2025)
- [x] **1-ply lookahead with dice averaging** — `evaluation/search.py` with batch GPU evaluation. (Feb 2025)
- [x] **Batch position evaluation** — Single GPU forward pass for all candidate positions. (Feb 2025)
- [x] **2-ply search** — Opponent uses 1-ply response. `evaluate_move_2ply` and `select_move_2ply` in search.py. (Feb 2026)
- [x] **Benchmark position suite** — 10 curated positions in `evaluation/benchmark.py` covering opening, race, bearoff, contact, prime, gammon. (Feb 2026)
- [x] **Win rate tracking over training** — `run_evaluation_checkpoint()` callback integrated into training loop. Evaluates vs random & pip count agents at configurable intervals. (Feb 2026)
- [x] **Equity error metric** — MAE/RMSE/max-error on benchmark positions with per-position breakdown. (Feb 2026)
- [x] **Move ordering + progressive deepening** — Heuristic-based move scoring for ordering. 2-ply uses 0-ply pre-screening to select top-k candidates. (Feb 2026)
- [x] **Transposition table** — MD5-based position caching with configurable size and LRU eviction. (Feb 2026)
- [x] **TD(lambda) returns** — Full implementation with 6-dim equity, fixed-perspective TD computation, backward eligibility traces. Integrated into self-play, replay buffer, and training loop. (Feb 2026)
- [x] **Exploration schedule** — Linear temperature decay (1.0→0.1) over training for self-play exploration. (Feb 2026)
- [x] **Global encoding features** — Contact detection, pip counts, home board control, prime detection, and bearoff progress. 8 global features broadcast to all 26 positions. New `enhanced_encoding_config()` and `full_encoding_config()` presets. (Feb 2026)
- [x] **Bearoff progress features** — Checkers borne off as a normalized feature. Part of global features (feature index 7). (Feb 2026)
- [x] **Race equity formula** — Effective Pip Count with positional corrections (gap, crossover, bar, wastage) and sigmoid mapping to equity. Standalone evaluation for pure race positions in `evaluation/race.py`. (Feb 2026)
- [x] **Position weighting** — Replay buffer weighted sampling based on game progress and equity uncertainty. Late-game and uncertain positions sampled more frequently. (Feb 2026)
- [x] **Train/validation splits** — Random game-level split between training and validation buffers. Validation loss tracked at evaluation checkpoints. (Feb 2026)
- [x] **Early stopping** — Patience-based early stopping on validation loss. Best model checkpoint saved automatically. (Feb 2026)
- [x] **Fix circular import** — Lazy import of `play_game` in `evaluator.py` to break `evaluation <-> training` circular dependency. (Feb 2026)
- [x] **Best-model tracking** — Dedicated `checkpoints/best/` directory for best validation loss model. (Feb 2026)
- [x] **LR plateau detection** — Warns when validation loss plateaus for consecutive eval checkpoints. (Feb 2026)
- [x] **Gradient clipping diagnostics** — Per-batch max grad norm and clip frequency tracking in console and metrics. (Feb 2026)
- [x] **Fix value-only agent crash** — 0-ply `NeuralNetworkAgent` now falls back to value-based search when policy head is disabled. (Feb 2026)
- [x] **Doubling cube types & state** — CubeState, CubeOwner, CubeAction in `core/types.py`. Full cube module at `core/cube.py`. (Feb 2026)
- [x] **Cube decision network head** — CubeHead (4-output) in `network/network.py` with `use_cube_head` config. (Feb 2026)
- [x] **Match equity tables** — Kazaross/Rockwell 15×15 MET in `core/cube.py`. (Feb 2026)
- [x] **Cubeful equity calculation** — Money and match cubeful equity, should_double/should_take, evaluate_cube_decision. (Feb 2026)
- [x] **Match play** — MatchState with score tracking, Crawford/post-Crawford rule, match equity lookups. (Feb 2026)
- [x] **Cube encoding** — encode_cube_state (4-dim) and encode_match_state (5-dim) for network input. (Feb 2026)
- [x] **Global board features** — `encode_global_board_features()` with 8-dim vector: pip counts, contact detection, race equity, bearoff progress, home board control. (Feb 2026)
- [x] **Contact vs race detection** — `is_past_contact()` in `core/board.py`. (Feb 2026)
- [x] **Race equity formula** — `race_equity_estimate()` using adjusted pip count / Keith count. (Feb 2026)
- [x] **Move ordering heuristics** — `order_moves()` and `select_move_2ply_pruned()` for top-K pruning at 2-ply. (Feb 2026)
