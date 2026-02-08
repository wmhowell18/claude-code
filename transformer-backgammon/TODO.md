# Backgammon Transformer — Feature Roadmap & TODO

> **Status**: Search, benchmarking, and doubling cube complete. 0/1/2-ply search with batch evaluation. Win rate tracking and benchmark positions in training loop. Full doubling cube with match play, match equity tables, cubeful equity, and cube decision network head.
>
> **Current Maturity**: ~6/10 for competitive play. Solid game engine + neural training + search + benchmarking + doubling cube + match play. Missing advanced training techniques (TD(lambda)), GnuBG interface, and MCTS.

---

## How to Use This Document

Items are grouped by priority tier. Within each tier, items are roughly ordered by impact. Each item has:
- **Status**: `[ ]` todo, `[x]` done, `[~]` in progress, `[-]` skipped/deferred
- **Effort**: S (hours), M (days), L (weeks)
- **Impact**: How much this improves the bot's playing strength

---

## TIER 1: Critical for Competitive Play

*These items represent the gap between "toy project" and "functional bot". Do these first.*

### Search & Lookahead (the #1 gap)

- [x] **1. 1-ply lookahead with dice averaging** — Evaluate all legal moves, for each apply the move, then average the equity over all 21 dice rolls. This is the single biggest strength multiplier (~10x). See Jacob Hilton's approach. (Effort: M, Impact: massive) *(Feb 2025)*
- [x] **2. 2-ply search** — After your move, consider opponent's best response across all their dice rolls. (Effort: M, Impact: large) *(Feb 2026)*
- [x] **3. Batch evaluation of multiple positions** — Feed all candidate resulting positions into the network in one GPU forward pass instead of one-by-one. Critical for search speed. (Effort: S, Impact: large for speed) *(Feb 2025)*
- [ ] **4. Move ordering heuristics** — Evaluate most promising moves first to enable pruning. Sort by pip count improvement, hits, etc. (Effort: S, Impact: moderate)
- [ ] **5. Transposition table / position cache** — Avoid re-evaluating identical positions across search. Hash board state → equity. (Effort: M, Impact: moderate)
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

- [ ] **16. TD(lambda) returns** — Intermediate positions get discounted credit from future outcomes, not just final result. Much better training signal. (Effort: M, Impact: large)
- [ ] **17. N-step bootstrapping** — Use V(s') from the network for truncated episodes instead of waiting for game end. (Effort: M, Impact: large)
- [ ] **18. Rollout-based training targets** — 1-ply rollout equity gives much better targets than raw game outcome. Key technique from gnubg. (Effort: L, Impact: large)
- [ ] **19. Exploration schedule** — Decay self-play temperature over training (e.g., 1.0 → 0.1). Currently fixed at 0.3. (Effort: S, Impact: moderate)
- [ ] **20. Opponent diversity** — Periodically play against older snapshots and baseline agents, not just self. Prevents forgetting. (Effort: M, Impact: moderate)
- [ ] **21. Position weighting** — Weight endgame and critical (high-equity-error) positions more heavily in training. (Effort: S, Impact: moderate)
- [ ] **22. More warmstart games** — Current 500 games is thin. Consider 2000-5000 pip count warmstart games. (Effort: S, Impact: small)
- [ ] **23. Performance-tied LR scheduling** — Reduce LR when validation equity error plateaus, not just cosine decay. (Effort: S, Impact: moderate)
- [ ] **24. Gradient clipping diagnostics** — Log when gradient clipping activates, how often, and magnitude. (Effort: S, Impact: small)

### Encoding Improvements

- [ ] **25. Contact vs race detection** — Binary or continuous feature indicating whether pieces are still in contact. Fundamentally different strategies apply. (Effort: S, Impact: large)
- [ ] **26. Pip count as input feature** — Normalized total pip count for each player. Cheap and informative. (Effort: S, Impact: moderate)
- [ ] **27. Home board control features** — How many home board points are made, how close to bearing off. (Effort: S, Impact: moderate)
- [ ] **28. Prime detection** — Length and location of longest prime for each player. (Effort: S, Impact: moderate)
- [ ] **29. Blot vulnerability encoding** — Which checkers are within direct/indirect hitting range. (Effort: M, Impact: moderate)
- [ ] **30. Escape features** — How many dice rolls let a trapped/back checker escape. (Effort: M, Impact: moderate)
- [ ] **31. Race equity estimate** — Pure pip count equity formula as input feature for race positions. (Effort: S, Impact: small)
- [ ] **32. Bearoff progress features** — Number of checkers already borne off for each player. (Effort: S, Impact: small)

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
- [ ] **42. Race equity formula** — Effective Pip Count / Keith count for pure race evaluation. (Effort: S, Impact: moderate)
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
- [ ] **53. Train/validation/test splits** — Proper held-out evaluation to detect overfitting. (Effort: S, Impact: moderate)
- [ ] **54. Early stopping** — Stop training when validation equity error stops improving. (Effort: S, Impact: moderate)
- [ ] **55. Best-model tracking** — Keep the checkpoint with best validation metric, not just latest. (Effort: S, Impact: small)

---

## TIER 4: Polish & Performance

*Optimization and developer experience improvements.*

### Speed & Efficiency

- [ ] **56. JIT compile evaluation pipeline** — Not just train_step, also inference and search. (Effort: S, Impact: moderate)
- [ ] **57. Optimize move generation** — Currently pure Python loops. Could use numpy vectorization or Cython. (Effort: M, Impact: moderate)
- [ ] **58. Legal move caching within a turn** — Don't regenerate legal moves for same board+dice. (Effort: S, Impact: small)
- [ ] **59. Precomputed action-to-move lookup** — Replace hash-based encoding with direct lookup table. (Effort: S, Impact: small)
- [ ] **60. Batch self-play** — Play N games simultaneously, step all games forward in parallel. (Effort: L, Impact: large for speed)
- [ ] **61. TPU-specific optimizations** — pmap, sharding, proper batch sizing for TPU topology. (Effort: M, Impact: moderate on TPU)
- [ ] **62. Mixed precision training** — bfloat16 on TPU for ~2x speedup. (Effort: S, Impact: moderate)

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
3. ~~**Items 7-10** (doubling cube)~~ — DONE (core/cube.py + types + network cube head)
4. **Longer training run** with the fixed pipeline — see where the model plateaus
5. **Items 16-18** (TD(lambda) + rollout targets) — dramatically better training signal
6. **Items 25-28** (better encoding) — more signal for the network to learn from
7. **Items 45-46** (MCTS) — sophisticated search for strongest play

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
- [x] **Fix value-only agent crash** — 0-ply `NeuralNetworkAgent` now falls back to value-based search when policy head is disabled. (Feb 2026)
- [x] **Doubling cube types & state** — CubeState, CubeOwner, CubeAction in `core/types.py`. Full cube module at `core/cube.py`. (Feb 2026)
- [x] **Cube decision network head** — CubeHead (4-output) in `network/network.py` with `use_cube_head` config. (Feb 2026)
- [x] **Match equity tables** — Kazaross/Rockwell 15×15 MET in `core/cube.py`. (Feb 2026)
- [x] **Cubeful equity calculation** — Money and match cubeful equity, should_double/should_take, evaluate_cube_decision. (Feb 2026)
- [x] **Match play** — MatchState with score tracking, Crawford/post-Crawford rule, match equity lookups. (Feb 2026)
- [x] **Cube encoding** — encode_cube_state (4-dim) and encode_match_state (5-dim) for network input. (Feb 2026)
