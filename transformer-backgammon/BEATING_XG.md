# Road to Beating XG: What It Will Take

An honest assessment of what this project needs to surpass Xtreme Gammon, the strongest backgammon engine, whose core approach hasn't been fundamentally challenged in nearly 20 years.

---

## Understanding the Target

XG's strength comes from the combination of several well-tuned components, not any single breakthrough:

- **Neural network**: Relatively small (~160 hidden units, ~20K parameters) but trained on millions of self-play positions over years with meticulous hyperparameter tuning
- **3-ply search with rollouts**: Evaluates positions by playing out hundreds of games from each candidate move, not just raw network evaluation
- **Comprehensive bearoff databases**: Perfect play for all endgame positions with 6 or fewer checkers remaining (~54K positions for one-sided, much larger for two-sided)
- **Opening book**: Pre-computed best moves for the first ~3 moves from exhaustive analysis
- **Cube model**: Extremely accurate double/take/pass decisions refined over years
- **Race equity model**: Near-perfect evaluation of non-contact racing positions

XG's estimated Elo in the backgammon community is roughly 2200+. GNU Backgammon at its strongest settings is perhaps 50-100 Elo below. The gap between "good neural net + 2-ply" and "XG-level" is significant but not insurmountable.

## Where We Are Today

After the bug fixes just landed:

| Component | Status | XG Parity? |
|-----------|--------|------------|
| Game engine | Complete | Yes |
| Neural network | Transformer, untrained | Unknown |
| Training pipeline | TD(lambda) + curriculum | Comparable approach |
| Search | 0/1/2-ply with batching | Behind (XG uses 3-ply + rollouts) |
| Board encoding | 10-dim enhanced | Reasonable |
| Bearoff database | None | Major gap |
| Opening book | None | Minor gap |
| Cube decisions | Infrastructure only | Major gap |
| Race model | Pip count formula | Behind |
| Evaluation framework | Benchmark positions | Behind (no GnuBG interface) |

## The Path Forward, in Phases

### Phase 1: Establish a Baseline (now - weeks)

**Goal**: Complete a successful training run and measure where we stand.

This is where we are right now. The bug fixes should unblock training. The immediate priorities:

1. **Complete a 10K-15K game training run** with the fixed pipeline. Measure win rate vs random and vs pip count agent. This is the very first milestone.

2. **GnuBG interface** (TODO #11). Without this, we're flying blind. We need position-by-position equity comparison against a known-strong engine. GnuBG's command-line interface makes this straightforward: pipe positions in, parse evaluations out. This is the single most important development infrastructure investment.

3. **Move accuracy metric** (TODO #14). What percentage of our moves match GnuBG's top choice? This is more diagnostic than win rate because it reveals *where* we're losing equity.

**Exit criteria**: Win rate vs GnuBG at 0-ply is measurable and improving over training.

### Phase 2: Get the Neural Network Right (weeks - months)

**Goal**: Network equity estimates within 0.02 MAE of GnuBG on benchmark positions.

This is the core challenge. The transformer architecture is unproven for backgammon, and the question is whether it can match or exceed the shallow networks used by XG/GnuBG. Arguments in favor: transformers excel at learning relationships in structured sequences, and a 26-position backgammon board IS a structured sequence where long-range dependencies matter (e.g., a prime at points 4-9 affects the value of a blot at point 20). Arguments against: backgammon positions have much less structure than language/images, and the overhead of attention may not be worth it vs a well-tuned MLP.

Key work:

1. **Architecture ablation** (TODO #33). Train identical pipelines with:
   - The current transformer (4 layers, 128 dim)
   - A plain MLP (similar parameter count)
   - A larger transformer (8 layers, 256 dim)

   If the transformer doesn't clearly beat the MLP on equity accuracy, switch. Don't be sentimental about the architecture. TD-Gammon used a simple 2-layer network and reached world-class play. The architecture matters less than training quality.

2. **Training scale**. TD-Gammon needed 1.5M games for its best results. Jacob Hilton reached 41% vs GnuBG with just 2,500 games, but with a very clean pipeline. Plan for:
   - 50K games: Should beat pip count agent consistently
   - 500K games: Should approach intermediate GnuBG level
   - 2M+ games: Should be competitive with GnuBG at comparable ply

3. **Rollout-based training targets** (TODO #18). This is the single biggest training improvement available. Instead of training on raw game outcomes (noisy) or TD(lambda) estimates (bootstrapped from a weak network), generate targets by:
   - For each position, play out 100+ games from that position using 1-ply search
   - Use the average outcome as the training target
   - This provides much higher quality targets at the cost of ~100x more compute per training position

   GnuBG's training is fundamentally based on this idea. It's why GnuBG can reach its level of play with a small network.

4. **Encoding refinement**. The current 10-dim encoding is a good start, but consider:
   - Blot exposure features (TODO #29): Which checkers can be hit and with how many dice combinations?
   - Escape features (TODO #30): How many rolls let a trapped checker escape?
   - These are the kind of features that TD-Gammon found were worth ~10% improvement in play quality when Tesauro added them for version 3.0

5. **Opponent diversity** (TODO #20). Self-play alone leads to blind spots. Periodically play against:
   - Frozen snapshots of earlier network versions
   - GnuBG at various levels (beginner, intermediate)
   - Random and pip-count agents (to prevent forgetting basics)

**Exit criteria**: Network at 0-ply evaluates benchmark positions within 0.03 equity of GnuBG. Win rate vs GnuBG at comparable search depth reaches 35-40%.

### Phase 3: Search Depth and Endgame (months)

**Goal**: Match GnuBG's playing strength through better search and perfect endgame play.

1. **3-ply search** with proper batching. The current 2-ply search makes k*21 separate network calls (TODO #109). Fix this first, then extend to 3-ply. With batched evaluation and move ordering, 3-ply should be tractable at ~1-2 seconds per move on GPU.

2. **Bearoff database** (TODO #41). This is non-negotiable for championship-level play. The endgame is where equity errors compound, and perfect bearoff play eliminates an entire class of errors.
   - One-sided bearoff: ~54K positions for 15 checkers on points 1-6. Store exact win/gammon probabilities. This alone significantly improves endgame play.
   - Two-sided bearoff: Much larger but can be computed for positions with <=6 checkers total on each side. This handles the final phase of most games perfectly.
   - Contact bearoff: Positions with checkers still in contact but bearing off. More complex, larger database.

3. **Rollout-based move selection**. For critical positions (especially cube decisions), don't just use the network value -- play out 128+ games per candidate move and average the results. This is how XG achieves its highest accuracy. It's slow, but it's accurate, and with GPU batching we can rollout much faster than XG does on CPU.

4. **Time management** (TODO #49). Not all positions need the same search depth. Simple forced-move positions need zero thought. Complex middlegame positions with multiple reasonable moves benefit from deeper search. Implement a simple heuristic: more search for positions where the top 2-3 moves have similar evaluations.

**Exit criteria**: Win rate vs GnuBG at comparable thinking time reaches 45-48%. Endgame play is essentially perfect in pure bearoff positions.

### Phase 4: Cube Mastery (months)

**Goal**: Double/take/pass decisions within 0.01 equity of XG.

Cube play is roughly 20-30% of backgammon skill. A bot that plays checkers perfectly but makes poor cube decisions will still lose significant equity.

1. **Cube training with self-play**. The CubeHead infrastructure exists. Train it on:
   - Self-play games where cube decisions are recorded
   - Positions labeled by GnuBG's cube analysis
   - Match equity table-aware training (correct cube decisions depend on the match score)

2. **Cubeful equity integration**. The network currently predicts cubeless equity (what would happen if no cube existed). True cube play requires cubeful equity: "what is this position worth given that doubling, taking, and passing are options?" This is more complex because it's recursive (the cube value depends on future cube decisions).

3. **Match play training** (TODO #37). Cube decisions are score-dependent. At 2-away 2-away, you should double very aggressively. At Crawford, you can't double at all. The network needs to understand match context. Options:
   - Train separate cube models for different match scores
   - Include match score as input features (encode_match_state already exists)
   - Use the match equity table for exact cube decisions at all scores

**Exit criteria**: Cube decision agreement with XG exceeds 95% across a diverse set of 1000+ positions at various match scores.

### Phase 5: Surpassing XG (months - year)

**Goal**: Consistent positive expected equity against XG in money games and matches.

This is the hardest phase because XG is a moving target in terms of effective strength, and the final few percentage points of improvement are the most expensive.

1. **Massive training scale**. XG benefited from years of continuous training. Plan for 5M-10M+ self-play games with rollout targets. This is primarily a compute budget question. On a single TPU v6e, estimate:
   - Simple self-play: ~100 games/second = 1M games in ~3 hours
   - With 1-ply rollout targets: ~1 game/second = 1M games in ~12 days
   - With 3-ply rollout targets: Much slower, but much higher quality

   The key insight is that compute efficiency matters more than raw compute. The transformer's batched GPU inference should give a significant throughput advantage over XG's CPU-based training.

2. **MCTS integration** (TODO #45-46). AlphaZero-style MCTS with the neural network as both prior and value estimator. This was shown to match GnuBG by the Stochastic MuZero paper (2022). The key adaptation for backgammon: the stochastic dice roll means each node in the tree has a chance node (21 possible dice outcomes) before the move selection node. This increases the branching factor but is handled by the chance-node extension to MCTS.

3. **Population-based training** (TODO #77). Train multiple agents with different hyperparameters simultaneously. The strongest survive and cross-pollinate. This is how OpenAI Five and similar projects found good hyperparameters without manual search.

4. **Weakness-targeted curriculum** (TODO #78). Once we can measure per-position-type accuracy against XG:
   - Identify position categories where we lose the most equity
   - Generate synthetic positions of those types
   - Train harder on our weakest areas
   - This is how human backgammon players improve: study your leaks

5. **Ensemble methods** (TODO #80). Average predictions from 3-5 independently trained networks. This reduces variance and typically gains 20-50 Elo for free.

**Exit criteria**: Positive expected value against XG in a 1000+ game money session. Match win rate >50% in 7-point matches.

## The Key Bets

This project makes several bets relative to XG's approach:

| Bet | Rationale | Risk |
|-----|-----------|------|
| Transformer > shallow MLP | Attention over board positions captures strategic relationships | May be overkill; MLP might be equally good with less compute |
| GPU batching > CPU speed | Evaluate 1000 positions simultaneously instead of one at a time | Single-position latency matters for interactive play |
| Modern RL > classical TD | TD(lambda) + curriculum + replay buffer should converge faster | More complex pipeline, more things to go wrong |
| Scale > tuning | Larger model + more training should overcome tuning disadvantage | XG had years of careful tuning; we're starting fresh |
| Open architecture > secrets | Transparent design allows community contribution | XG's proprietary tuning may be hard to replicate |

The strongest bet is GPU batching. XG evaluates one position at a time on CPU. With a GPU, we can evaluate 512 positions in the time XG evaluates 1. This means our 2-ply search with GPU batching may be effectively faster than XG's 3-ply on CPU, depending on the position. This speed advantage compounds during training (more games per hour) and during play (deeper effective search).

## What Specifically Would XG Not Expect?

XG was designed in an era before:
- **Transformers** (2017). Attention mechanisms that can learn which board relationships matter.
- **Large-scale self-play** (AlphaGo/AlphaZero, 2016-2017). Training purely from self-play without human game data.
- **Stochastic MCTS** (MuZero, 2019-2022). Tree search that handles randomness natively.
- **TPU/GPU training at scale** (2020s). Training throughput that would have taken years on 2005-era hardware.

The most promising angle of attack is not to out-tune XG on its own terms (shallow network + CPU search), but to use fundamentally different tools:
- A transformer that learns positional relationships XG's network can't represent
- MCTS that explores the game tree more intelligently than fixed-depth search
- GPU throughput that allows deeper rollouts in less wall-clock time
- Modern training techniques (TD(lambda), curriculum, replay) that converge faster

## Realistic Timeline

| Milestone | Estimated Time | Confidence |
|-----------|---------------|------------|
| First successful training run | Days | High |
| Beat pip count agent consistently | 1-2 weeks | High |
| 25% win rate vs GnuBG (0-ply) | 1-2 months | Medium |
| 40% win rate vs GnuBG (1-ply) | 2-4 months | Medium |
| Match GnuBG strength | 4-8 months | Medium-Low |
| Exceed GnuBG | 6-12 months | Low-Medium |
| Challenge XG | 12-24 months | Low |
| Beat XG consistently | 18-36 months | Speculative |

The timeline is heavily dependent on compute budget and whether the transformer architecture proves effective. If the transformer doesn't work well, switching to a proven MLP architecture could save months. If it works well, the attention mechanism could provide insights into backgammon strategy that no existing engine has.

## Minimum Viable Improvements to Beat XG

If forced to prioritize ruthlessly, the absolute minimum path to beating XG:

1. **Working training pipeline** (just fixed)
2. **Rollout-based training targets** (10x better training signal)
3. **Bearoff database** (perfect endgame = no free equity for opponent)
4. **3-ply search with GPU batching** (deeper search = better play)
5. **GnuBG interface** (can't improve what you can't measure)
6. **2M+ training games** (raw scale)
7. **Cube training on labeled data** (20-30% of the game)

Everything else is optimization, polish, or hedging bets. These 7 items, done well, should be sufficient to at least match XG's level. Beating it consistently requires either a fundamentally better evaluation function (the transformer bet) or fundamentally better search (the MCTS bet), or both.
