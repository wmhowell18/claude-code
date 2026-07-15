# Backgammon LLM Benchmark — Founding Design Doc (PLAN.md)

**Status:** Draft v1 · **Created:** 2026-07-12 · **Owner:** wesleyhowell7

A benchmark that measures how well large language models play backgammon —
checker plays *and* cube decisions — scored against **GNU Backgammon (gnubg)
rollout** ground truth, using a human-comparable error metric (an XG-PR-like
score). The
benchmark ships two tracks (text and image), difficulty tiers, a
contamination-resistant private position set, and a static leaderboard that
plots model skill against dollar cost with a top-human "north-star" line.

This document is opinionated. Each section states a **default recommendation**
in bold and notes alternatives briefly. A second agent should be able to
scaffold the repo from the [Repository Structure](#8-repository-structure)
section and start Phase 0.

---

## Decisions log

- **2026-07-15 — Ground-truth engine is GNU Backgammon (gnubg), permanently.**
  gnubg rollouts are the authoritative source of truth for every position, in
  all phases — not an interim stand-in for eXtreme Gammon (XG). This resolves
  Open Question §9.1 (the XG batch-analysis / GUI-automation options are moot)
  and §9.3 (the match-equity table is gnubg's default MET, which must match the
  truth engine). XG rollouts are **not** the end state; at most they are an
  optional future cross-check on a handful of marquee positions. **BenchPR stays
  calibrated to XG's published PR scale** (§4.4) — that is a property of the
  *metric formula* for human comparability, and is independent of which engine
  produces the equities; it remains open (§9.2) only in the sense of pinning the
  exact constant.

---

## 0. Goals and non-goals

**Goals**

- A reproducible benchmark of LLM backgammon skill, reported as a single
  human-comparable number (PR-like error rate) plus best-move accuracy.
- Resistance to pretraining contamination so scores reflect *reasoning*, not
  memorization.
- Two modality tracks (text, image) sharing one position set and one scoring
  pipeline.
- A cost-aware leaderboard: skill vs. dollars, plus a fixed-budget ranking.

**Non-goals (v1)**

- Full-game play / self-play against the models (we evaluate single decisions
  from fixed positions, which is how backgammon strength is actually measured).
- Training or fine-tuning models.
- Being a general "reasoning" benchmark — this is domain-specific and proud
  of it.

---

## 1. Position dataset design

### 1.1 Representations (every position ships in all of these)

Each position is one JSON record with multiple redundant encodings so we can
serve any track from a single source of truth.

| Field | Format | Role |
|-------|--------|------|
| `xgid` | XGID string | **Canonical ID** and dedup key |
| `gnubg_id` | GNU BG Position ID + Match ID | Secondary ID / interop with GNU BG |
| `board_json` | structured JSON (see below) | Machine-readable, unambiguous |
| `ascii` | fixed-width ASCII render | Text-track prompt payload |
| `image_png` | rendered PNG (from SVG) | Image-track prompt payload |
| `image_svg` | source SVG | Regenerate PNG at any DPI |

**Canonical ID: XGID.** It is compact, encodes board + cube + dice + score +
turn, and is a widely-recognized notation (XG's native currency, and readable by
GNU BG via ID conversion). We use it as the primary key and the dedup hash
source; this is a notation choice and is independent of the ground-truth engine
(gnubg, §1.3).

> Note the residual risk (see §2.5): XGID *format* is likely present in
> pretraining even if our specific positions are not. That is fine — we are not
> hiding the notation, only the position-to-answer mapping.

**Secondary ID: GNU BG Position ID.** Free/open tooling (GNU BG) can parse it,
which matters because GNU BG is our ground-truth rollout engine (§1.3, §7). Store
both the Position ID and Match ID.

**Structured JSON (`board_json`)** — the unambiguous form, e.g.:

```json
{
  "points": [0, 2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5,
             -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0],
  "bar":  {"x": 0, "o": 0},
  "off":  {"x": 0, "o": 0},
  "turn": "x",
  "dice": [3, 1],
  "cube": {"value": 1, "owner": "center"},
  "score": {"x": 0, "o": 0, "length": 7, "crawford": false},
  "pip":  {"x": 167, "o": 167},
  "decision_type": "checker"
}
```

Conventions: indices 1–24 are points from the mover's perspective (index 0 =
mover's bar entry area / off is separate), positive counts are the mover (`x`),
negative are the opponent (`o`), mirroring the canonical-mover convention used
in the sibling `transformer-backgammon` engine. **The JSON perspective is always
"player to move,"** which removes color ambiguity.

**ASCII render** — a fixed, versioned template (monospace, 2-high point stacks
with counts, bar in the middle, pip counts + cube + dice + score in a header).
Deterministic given `board_json`. Version the template string
(`ascii_render_version`) so a template change is traceable.

### 1.2 Image rendering

**Default: SVG authored programmatically → rasterized to PNG** (e.g.,
`svgwrite`/hand-built SVG → `cairosvg` or `resvg`). SVG source is committed so
images regenerate deterministically and losslessly at any resolution.

One **consistent house style** (single theme in v1; a second theme is a possible
robustness ablation later). Every image **must legibly show**:

- The 24 points with checkers (distinct colors + count numerals on tall stacks).
- Bar and off/bear-off trays with counts.
- **Dice** (the roll to be played) for checker decisions.
- **Cube** value and position (centered / owned by which side).
- **Match score and match length** (and Crawford flag) — or "money" label.
- **Pip counts** for both sides.
- Point numbering (1–24) and a clear "player on roll" indicator.

Rationale: we are testing play skill, not board-reading OCR puzzles, so all
decision-relevant state is visible. (A later ablation can hide pip counts to
test spatial reasoning.) Render at a fixed target (e.g., 1024×768 PNG) plus keep
SVG for retina/zoom. Store `image_render_version`.

### 1.3 Ground truth (source of truth = GNU BG rollouts)

Ground truth is a **full rollout** per position, not a static evaluation.
**GNU Backgammon (gnubg) is the authoritative engine, permanently** (Decisions
log, 2026-07-15; §9.1). XG rollouts are not the end state — at most an optional
future cross-check on a few marquee positions.

**Standardized rollout settings (engine = GNU BG):**

| Setting | Default |
|---------|---------|
| Engine | GNU BG rollout (authoritative) |
| Checker play depth | GNU BG "world-class" / 2-ply chequer for rollouts |
| Cube decisions in rollout | GNU BG world-class cube evaluation |
| Trials | 1296 (money-safe minimum) → **standard 1296, deep tier 5184** |
| Truncation | none for short; adaptive/truncated for race positions |
| Variance reduction | ON (GNU BG variance reduction / quasi-random dice) |
| Duplicated (antithetic) dice | ON |
| Seed | recorded per position for reproducibility |

We record the settings blob (`rollout_meta`) with every position so a reader
knows exactly how truth was produced, and so we can re-roll if we upgrade the
engine.

**What we store per position:**

For **checker plays** — every legal move (or top-K, K≈15, plus the played/second
move guaranteed) with:

- move in standard notation (e.g., `24/18 13/11`),
- equity (money) and/or match-equity (MWC / normalized equity),
- error size vs. best in **millipoints** (0 for the best move),
- rollout std error / confidence, and the rank.

For **cube decisions** — the three canonical equities:

- No double / no take (cubeless-in-context "no action") equity,
- Double / Take equity,
- Double / Pass equity,
- derived best action ∈ {No double, Double/Take, Double/Pass, Too good/…},
- error size in millipoints for each wrong choice,
- doubling window / cube efficiency where relevant.

Store `best_move`, `best_equity`, `second_best_move`, `equity_gap` (blunder
margin — used for scoring and for filtering near-free decisions; tiers are
human-difficulty-based, §3), and `phase` tag (§3).

### 1.4 Decision types and play mode

- **Checker plays** and **cube decisions** (take/pass and double/no-double) are
  both first-class. Split target ≈ **70% checker / 30% cube** (cube decisions
  are where humans and models both bleed equity, so they are worth
  over-weighting relative to their game frequency).
- **Match play vs. money play:** ship **both**, tagged. Money play is simpler to
  score (equity is linear). Match play requires match-equity-table context
  (score matters), which is exactly where models can be fooled — valuable
  signal. Default split ≈ 60% money / 40% match, with match positions spread
  across representative scores (incl. Crawford, 2-away/2-away, gammon-go/gammon-
  save situations).

---

## 2. Contamination avoidance (critical requirement)

The benchmark is worthless if models have seen the answers. Positions must be
**in-distribution** (arise from real, strong play) yet **out-of-corpus** (not in
pretraining text/images).

### 2.1 Generate, don't harvest

**Default: generate fresh games via strong-bot self-play**, then sample
positions from them. Use **GNU BG self-play** (open, scriptable) and/or XG
self-play to produce brand-new match/money games on the dataset creation date.
Do **not** harvest published matches, forums, book positions, or quiz sites.

Because the games never existed before we made them, their positions cannot be
in any pretraining corpus (modulo coincidental collisions, handled by dedup).

### 2.2 Avoid famous / canonical positions

Explicitly exclude:

- Textbook and quiz positions (Robertie, Magriel *Backgammon*, Woolsey/Trice,
  GammonVillage/BGonline problems, "position of the week" archives).
- Reference opening plays (e.g., the memorized 31/opening-roll best plays) —
  filter out ply-1/ply-2 opening positions and any position within N plies of
  the standard start where the answer is common knowledge.
- Any position whose XGID/GNU BG ID is **findable on the web** (§2.4).

### 2.3 Deduplicate against public collections

Build a **blocklist** of hashes from known public position sources we can
legally enumerate (GNU BG example databases, public match archives we scrape
*only to exclude*, opening books). Reject any generated position whose canonical
XGID (normalized) matches the blocklist. Keep the blocklist versioned.

Also dedup **within** the dataset (canonical XGID + color/symmetry
normalization) so near-duplicate positions don't inflate tiers.

### 2.4 Verify unfindability

For a sampled candidate before it enters the **public** split, spot-check
(scriptable, rate-limited web search of the exact XGID and GNU BG ID strings). A
zero-hit result is necessary-not-sufficient evidence of unfindability. Record
the check date. This is a sampling QA step, not per-position for the whole set.

### 2.5 Held-out split + canaries + provenance

- **Private held-out split:** the real leaderboard set is **never published in
  full**. We publish only (a) SHA-256 hashes of each held-out position record
  and (b) aggregate stats. Models are evaluated by us (or via a trusted runner)
  against the private set. This is the single strongest anti-contamination
  control — you cannot memorize what was never released.
- **Public dev split:** a small, openly published set (for prompt engineering,
  harness testing, and reproducibility demos) — assume it *will* eventually be
  ingested and treat its scores as non-authoritative over time.
- **Canary GUIDs:** embed a unique canary string (a fixed
  `BENCH-CANARY-<uuid>` token) in every published file and in a public "please
  don't train on this" manifest, so future corpora and future models can be
  audited for leakage.
- **Date-stamping:** every dataset artifact records a creation date and a
  content hash. A model with a training cutoff before the creation date
  provably could not have trained on it — the date stamp turns "trust me" into a
  checkable claim.

### 2.6 Residual risks (documented, accepted)

- **Format familiarity:** XGID/GNU BG notation is in pretraining. Accepted — we
  test mapping position→best play, not notation trivia.
- **Distributional leakage:** models may have learned general backgammon
  heuristics (good!). We measure whether those heuristics + reasoning reach
  rollout-optimal play; that is the point.
- **Public-dev decay:** public positions rot over time. Mitigated by the private
  held-out set being authoritative and by periodic dataset refreshes (new
  creation date, new canary).

---

## 3. Difficulty tiers

Tiers measure **how hard the decision is for top humans** — not how costly a
mistake is. The equity gap between best and second-best move is deliberately
**not** the tier axis: a huge-gap decision can be obvious (easy), a tiny-gap
decision can be famously hard, and the most valuable expert content is often
*large-gap decisions that strong humans still get wrong*. The gap keeps two
supporting roles only: it feeds scoring (equity loss), and it filters out
near-free decisions (gap < ~0.002) that add noise, not signal.

**Phase tags** (from board topology / rollout metadata): `race`,
`holding-game`, `blitz`, `priming`, `backgame`, `bearoff`, `opening-ish`,
`cube-action`.

### 3.1 How human difficulty is estimated

Our positions are fresh by design (§2), so they have no human play history.
Difficulty is therefore *estimated*, with three sources in increasing order of
authority:

1. **Known-hard taxonomy (prior).** Categories where expert errors are known to
   concentrate: backgame timing, prime-vs-prime, blitz cube windows,
   too-good/cash decisions, deep-anchor holding games, wastage-critical
   bearoffs. Used to seed tier assignments and as a QA cross-check.
2. **Human-error model (primary at scale).** Fit a model that predicts expert
   error rate / expected loss from position features, trained on large public
   corpora of *analyzed human matches* (e.g., online match archives run through
   GNU BG/XG, filtered to strong players by PR band). Training on public games
   does not contaminate the benchmark — our positions stay fresh; only
   difficulty *statistics* transfer. Apply the model to every candidate
   position.
3. **Expert panel calibration (ground truth).** A small panel of strong players
   (PR ≤ 5, ideally 2–4) plays the pilot set and a stratified sample of the
   dev/held-out sets blind. Their empirical miss rate and equity loss calibrate
   and validate the model, and hand-tier the marquee positions. Bonus: the same
   panel data doubles as a **measured human north star on our own positions**
   (§4.5), rather than only citing published PRs.

**Operational metric:** for each position we store `expected_expert_loss` (EEL,
predicted mean equity loss in millipoints for a reference PR-3 human) and
`expert_miss_rate` (predicted probability that player picks a non-best move).

### 3.2 Tier definitions

| Tier | Name | Human-difficulty criteria (provisional thresholds) | Intuition |
|------|------|-----------------------------------------------------|-----------|
| **T1** | Easy | experts essentially never err: miss rate < 2%, EEL < 1 mpt | "any decent club player finds it" |
| **T2** | Medium | club players err, experts rarely: miss rate 2–10% | solid-intermediate separators |
| **T3** | Hard | experts err at a real rate: miss rate 10–30% | strong-player separators |
| **T4** | Expert | even world-class players err often: miss rate > 30% or EEL > 10 mpt; historically debated categories | reference-grade; where PR-2 humans and top bots separate |

Notes:

- Thresholds are **provisional until panel calibration** (Phase 1 gate); the
  tier boundaries are re-fit so the panel's observed error rates match the tier
  intent.
- Bootstrap ordering: pilot (50 positions) is tiered from the taxonomy prior +
  author judgment, then panel-calibrated; the human-error model takes over for
  the full set in Phase 2.
- Cube decisions get their own within-tier weighting; near-doubling-window
  positions naturally land in T3/T4.
- Near-free decisions (gap < ~0.002) are down-sampled regardless of estimated
  difficulty — if the choice barely matters, it measures nothing — unless
  there's a large-blunder third option that humans actually pick.

### 3.3 Dataset size and splits

**Target total: ~1,500 positions** (v1 authoritative set), reached in stages:

| Split | Size | Purpose |
|-------|------|---------|
| Pilot (Phase 1) | 50 | wiring + calibration |
| Public dev | 150 | open, non-authoritative |
| Private held-out | ~1,300 | authoritative leaderboard set |

Per-tier target (of the ~1,300 held-out): **T1 25% / T2 30% / T3 30% /
T4 15%.** Skew away from too many T4 (rollout cost is high, variance eats
signal, and genuinely world-class-hard positions are the scarcest to source). Within each tier keep the ~70/30 checker/cube and ~60/40 money/match
splits from §1.4.

---

## 4. Evaluation protocol

### 4.1 Tracks

1. **Text-only** — prompt carries `ascii` + `board_json` + a natural-language
   statement of the decision. Primary track.
2. **Image** — prompt carries `image_png` only (plus minimal text: "You are on
   roll; what is the best play?"). Tests board reading + play.
3. **Text + image** (optional) — both, to measure whether images help or hurt.

### 4.2 Prompting

- A fixed **system prompt** stating the rules of notation and the exact required
  answer format. Versioned (`prompt_version`).
- The position payload for the track.
- A required **answer block** the model must emit, e.g.:

```
FINAL ANSWER: 24/18 13/11
```

  For cube: `FINAL ANSWER: Double, Take` / `No double` / `Double, Pass`.
- Chain-of-thought is **allowed** and separated from the final line; only the
  `FINAL ANSWER` line is parsed. (For the budget track, reasoning tokens count
  against the budget — §5.3.)

### 4.3 Answer format & equivalence

- Moves in standard slash notation; order-independent; `bar/` and `/off`
  supported; combined-die notation (`13/7` for a 6) accepted and normalized.
- **Normalization**: parse the move to the resulting board and compare
  *positions reached*, not strings — so `13/11 24/18` == `24/18 13/11` and any
  legal die-ordering that yields the same end position is treated as identical.
- Tolerance for "equivalent best moves": if the rollout has multiple moves
  within rollout noise (overlapping std-error) of the best, **any** of them
  scores as best.

### 4.4 Scoring

**Primary metric: equity loss (average error in millipoints).** For each
decision, error = `best_equity − chosen_equity` (≥ 0), read from stored
rollout data. Average over all positions (optionally per-tier and per-track).
This mirrors how humans are rated.

**Secondary metric: best-move accuracy** (% of positions where the model picks a
rollout-best move).

**PR-like score.** eXtreme Gammon's **Performance Rating (PR)** = average equity
loss per decision × 500, expressed in "millipoints per decision × constant" so
that lower is better and the scale matches human intuition:

- **PR ≈ 0** = flawless (rollout-perfect).
- **PR 2–4** = world-class human.
- **PR 5–8** = strong expert / open-tournament.
- **PR 10–15** = intermediate. **PR 20+** = beginner.

We compute **BenchPR** the same way: `BenchPR = 500 × mean(equity_loss)` over
the scored decisions (using the same money-equity units XG uses). Because it is
computed identically to XG PR, **models and humans sit on the same axis** — the
leaderboard can draw a "PR 2 top-human" line and a "PR 5 expert" line directly.

> Caveat to document: XG's exact PR constant and its treatment of cube vs.
> checker decisions must be matched precisely for the numbers to be literally
> comparable; we pin the formula in `harness/scoring.py` and validate against a
> few known XG-analyzed games. Until validated, we label it "BenchPR
> (PR-calibrated)".

### 4.5 Human north star

Anchor the leaderboard to top-human performance: draw horizontal reference lines
at **PR 2 (world-class), PR 4 (elite), PR 8 (strong club)**. The headline
question the benchmark answers: *"Does model X play at the level of a PR-2
human?"* Optionally include a couple of real reference points (e.g., published
tournament PRs) as annotated dots.

The expert panel used for tier calibration (§3.1) plays a sample of the actual
benchmark positions, so their measured BenchPR is a **direct human baseline on
this dataset** — plotted alongside the published-PR reference lines.

### 4.6 Sampling policy

- **Default run:** temperature 0 (or provider min), single attempt, greedy —
  the reproducible baseline.
- **Retries on unparseable output:** up to **2** re-asks with a
  format-reminder; if still unparseable, score as **worst legal move**
  (equity loss = gap to worst) and flag `parse_failed`. This punishes models
  that won't follow format without letting one bad parse nuke a run.
- **Self-consistency variant (reported separately):** N samples at
  temperature ~0.7, majority-vote / best-by-model-stated-confidence — this is a
  capability, and its cost is charged in the budget track (§5.3).

---

## 5. Model access / harness

### 5.1 Gateway

**Default: OpenRouter.** One API key, one OpenAI-compatible endpoint, hundreds of
models (Anthropic, OpenAI, Google, Meta, DeepSeek, etc.), pay-as-you-go, and —
crucially — **per-request usage/cost reporting** which we need for the cost axis
and the budget track. Minimal provider-account setup for a solo author.

Alternatives (note, don't default): direct provider APIs (best price/latency,
but N accounts to manage); **LiteLLM proxy** (self-hosted unified gateway, good
if we outgrow OpenRouter or want local models via Ollama).

### 5.2 Harness sketch

**Python, async batch runner.** Components:

- `client.py` — thin async OpenRouter client (httpx), ret/backoff, per-request
  `usage` capture (prompt/completion/reasoning tokens + cost).
- `runner.py` — async task pool over (model × position × track); concurrency
  caps; resumable.
- `prompts.py` — versioned prompt templates per track.
- `parse.py` — extract `FINAL ANSWER`, normalize moves to end-positions.
- `scoring.py` — equity-loss lookup, BenchPR, accuracy, per-tier aggregation.
- `cache.py` — content-addressed cache keyed by
  `(model, prompt_version, position_id, sampling_params)`; never pay twice for
  the same call; makes runs idempotent/resumable.
- `cost.py` — per-run cost tracking + rollups to results JSON.
- `report.py` — emit `results/*.json` consumed by the site.

Config via a `runs/<run>.yaml` (model list, track, sampling, budget). Every run
writes a manifest (dataset hash, prompt_version, timestamp) for reproducibility.

### 5.3 Fixed-dollar-budget ranking

A second leaderboard where **each model gets a fixed budget (default $10)** and
may spend it however it plays best on the *same* held-out set: bigger reasoning
budgets, retries, self-consistency N, best-of-N with a self-critique pass, etc.
Rank by **best BenchPR achieved within budget**.

- **Cost source:** OpenRouter's reported per-request cost (authoritative).
  Accumulate until the budget is exhausted; positions not yet answered when the
  budget runs out are scored as parse/timeout worst-case (creates a real
  incentive to allocate budget wisely).
- **Normalization:** because models have different token prices, the budget
  equalizes *dollars*, not *tokens* — a cheap model can afford many samples, an
  expensive model must be frugal. That is the intended, realistic trade-off.
  Report both "BenchPR @ $10" and the cost-to-reach-PR-X curve.
- Publish the spend strategy per model so results are reproducible.

---

## 6. Results website

**Default: static site generated from `results/*.json`, hosted on GitHub
Pages.** No backend. A small build step (Python or a light JS bundler) reads the
results JSON and renders:

- **Leaderboard table:** model, BenchPR (text), BenchPR (image), best-move
  accuracy, $ / run, BenchPR @ $10.
- **Skill-vs-cost scatter:** BenchPR (y, lower better) vs. $ per full run (x,
  log), with **human north-star lines** (PR 2 / 4 / 8).
- **Per-tier bars:** BenchPR by T1–T4 (shows where models fall apart — expect
  T4/cube to be brutal).
- **Text vs. image track** comparison per model.
- **Position explorer** (public dev set only): board image + top rollout moves +
  what each model answered (great for demos and error analysis).

Keep v1 dead simple (one page + a couple of charts). Charts: follow the repo's
dataviz conventions; ship light/dark, theme-aware, self-contained. Data flows
one way: results JSON → build → static HTML. No database.

Alternative: a notebook-generated report or an Observable/Streamlit dashboard —
rejected for v1 (hosting friction, not needed).

---

## 7. Roadmap / milestones

### Phase 0 — Scaffolding

- Repo tree (§8), config, licenses, canary manifest, CI stub.
- Position JSON schema + validator; ASCII renderer; SVG→PNG renderer.
- XGID and GNU BG ID parse/normalize round-trip.
- **Acceptance:** given a hand-entered XGID, produce a valid record with
  `board_json`, `ascii`, and `image_png`; schema validates; round-trip XGID↔JSON
  is lossless.

### Phase 1 — Pilot dataset (~50 positions)

- GNU BG self-play generator → sampler → dedup/blocklist → tiering (taxonomy
  prior + author judgment at this stage, per §3.1).
- **Ground truth via GNU BG rollouts** (the authoritative engine, §1.3).
- Manual QA of tier assignments; contamination spot-checks.
- **Expert panel calibration:** ≥2 strong players (PR ≤ 5) play the pilot blind;
  observed miss rates re-fit the tier thresholds and provide the first measured
  human-north-star data points.
- **Acceptance:** 50 positions across all tiers + both decision types, each with
  full rollout data; panel results recorded and tier thresholds re-fit; one
  end-to-end model run (one cheap model) produces a BenchPR and best-move
  accuracy; scoring reproduces by re-running from cache.

### Phase 2 — Full dataset + harness

- Scale to ~1,500 positions; finalize public/private split; publish hashes +
  canaries + creation date.
- Fit the **human-error model** (§3.1) on analyzed public match corpora,
  validate it against the panel data, and use it to tier the full set.
- Roll out the full authoritative set with **GNU BG rollouts** (the permanent
  ground-truth engine, §1.3) at the standard trial counts.
- Full async harness with caching, cost tracking, retries; run a slate of
  models on text + image tracks.
- **Acceptance:** full set rolled out; ≥6 models scored on both tracks; results
  JSON emitted; BenchPR validated against ≥3 XG-analyzed reference games to
  within a documented tolerance (a *metric-scale* calibration check, §4.4 — the
  truth engine remains GNU BG).

### Phase 3 — Website + budget ranking

- Static site on GitHub Pages; all charts; north-star lines.
- Fixed-$10 budget track implemented and run.
- **Acceptance:** public leaderboard live; skill-vs-cost and per-tier charts
  render; BenchPR @ $10 column populated; a reader can reproduce a run from the
  published manifest + public dev set.

### Milestone summary

| Phase | Deliverable | Gate |
|-------|-------------|------|
| 0 | Scaffolding + renderers | lossless XGID round-trip, valid record |
| 1 | 50-pos pilot + 1 model run | end-to-end BenchPR, reproducible |
| 2 | ~1,500 pos + full harness | GNU BG truth, ≥6 models, PR validated |
| 3 | Site + budget track | live leaderboard, BenchPR @ $10 |

---

## 8. Repository structure

```
backgammon-llm-benchmark/
├── PLAN.md                     # this document
├── README.md                   # quickstart + what the benchmark is
├── LICENSE                     # code license (e.g., MIT)
├── DATA_LICENSE                # dataset terms + "do not train" canary notice
├── CANARY.md                   # canary GUIDs + provenance manifest
├── pyproject.toml              # package + deps (httpx, pydantic, cairosvg, ...)
├── .gitignore
│
├── schema/
│   ├── position.schema.json    # JSON Schema for a position record
│   ├── rollout.schema.json     # JSON Schema for rollout ground-truth blob
│   └── results.schema.json     # JSON Schema for a model-run result file
│
├── positions/                  # canonical position records (source of truth)
│   ├── pilot/                  # Phase 1 ~50 positions (public)
│   ├── dev/                    # public dev split ~150 (non-authoritative)
│   └── heldout/                # PRIVATE authoritative set (gitignored)
│       └── hashes.json         # published SHA-256 hashes of held-out records
│
├── rollouts/                   # ground-truth rollout data, keyed by position id
│   ├── gnubg/                  # authoritative GNU BG rollouts
│   └── xg/                     # reserved: optional future XG cross-check only
│
├── data/                       # generated/intermediate artifacts
│   ├── selfplay/               # raw bot self-play games (source of positions)
│   ├── blocklist/              # dedup hashes of known public positions
│   └── manifests/              # dataset manifests (hash + creation date)
│
├── bgcore/                     # core domain layer (stdlib-only): board model, move rules, notation
│
├── render/                     # position -> text/image rendering
│   ├── ascii.py                # deterministic ASCII board render (versioned)
│   ├── svg.py                  # board -> SVG
│   ├── raster.py               # SVG -> PNG
│   └── images/                 # generated PNG/SVG (dev/pilot only; heldout ignored)
│
├── ids/                        # position ID parsing/normalization
│   ├── xgid.py                 # XGID <-> board_json, normalize/dedup key
│   └── gnubg_id.py             # GNU BG Position/Match ID <-> board_json
│
├── generate/                   # dataset construction pipeline
│   ├── selfplay.py             # drive GNU BG (and later XG) self-play
│   ├── sample.py               # sample candidate positions from games
│   ├── dedup.py                # blocklist + intra-set dedup
│   ├── tiering.py              # assign T1-T4 from estimated human difficulty (§3)
│   └── contamination.py        # web-findability spot-checks, canary injection
│
├── harness/                    # LLM evaluation harness
│   ├── client.py               # async OpenRouter client (usage/cost capture)
│   ├── runner.py               # async batch runner (model x position x track)
│   ├── prompts.py              # versioned prompt templates per track
│   ├── parse.py                # FINAL ANSWER extraction + move normalization
│   ├── scoring.py              # equity loss, BenchPR, accuracy, aggregation
│   ├── cache.py                # content-addressed response cache
│   ├── cost.py                 # per-run cost tracking + budget accounting
│   └── report.py               # emit results/*.json
│
├── runs/                       # run configs + manifests
│   └── example.yaml            # model list, track, sampling, budget
│
├── results/                    # model-run outputs consumed by the site
│   └── *.json
│
├── site/                       # static leaderboard (GitHub Pages)
│   ├── build.py                # results/*.json -> static HTML
│   ├── templates/              # page + chart templates
│   └── public/                 # built site output
│
├── scripts/                    # one-off / operational scripts
│   ├── build_dataset.py        # end-to-end: generate -> rollout -> tier -> emit
│   ├── run_benchmark.py        # execute a run config
│   ├── validate_pr.py          # calibrate BenchPR vs known XG-analyzed games
│   └── verify_position.py      # validate a single record against schema
│
├── tests/                      # unit/integration tests
│   ├── test_ids.py             # XGID / GNU BG round-trip
│   ├── test_render.py          # ASCII/SVG determinism
│   ├── test_parse.py           # move parsing/equivalence
│   └── test_scoring.py         # equity loss + BenchPR math
│
└── docs/
    ├── SCORING.md              # BenchPR definition + XG PR mapping
    ├── DATASET.md              # schema, tiers, splits, provenance
    ├── CONTAMINATION.md        # anti-contamination policy + audit steps
    └── HARNESS.md              # how to run models, add a model, budget track
```

Pragmatic notes: `positions/heldout/` and generated `render/images/` for heldout
are gitignored; only `hashes.json` and canaries are committed. `rollouts/` is
split by engine: `gnubg/` holds the authoritative rollouts, and `xg/` is
reserved for an optional future XG cross-check (unused in v1).

---

## 9. Open questions (author decisions needed)

1. **~~XG rollout automation.~~ RESOLVED 2026-07-15 — ground truth is GNU BG,
   permanently.** GNU BG rollouts are the authoritative source of truth in all
   phases (Decisions log; §1.3, §7). GNU BG is fully open and scriptable, which
   removes the automation problem entirely. The XG options once considered here
   — (b) XG batch analysis of exported sessions, (c) manual XG rollout import for
   marquee positions, (d) GUI automation (AutoHotkey/pywinauto) — are **moot**;
   at most XG serves as an optional future cross-check on a handful of positions,
   never as the authoritative engine. BenchPR calibration is unaffected: it is a
   metric-scale choice (item 2), not an engine choice.
2. **BenchPR constant.** Confirm the exact XG PR formula/constant and its
   cube-vs-checker handling so BenchPR is literally comparable to published human
   PRs (else label it "PR-calibrated"). *Still open — this is about the metric
   formula (the human-comparability axis, §4.4), and is independent of the
   ground-truth engine (§9.1).*
3. **~~Match-equity source.~~ RESOLVED 2026-07-15 — GNU BG's default MET.**
   Match-play normalized equities use GNU BG's built-in default match-equity
   table, because the MET must match the truth engine (§9.1). Recorded per
   position in `rollout_meta.met`.
4. **Dataset size vs. rollout budget.** ~1,500 at 1296+ trials is a lot of
   rollout compute — confirm the trial counts per tier and total time budget.
5. **Model slate + spend cap.** Which models for v1, and what total $ are we
   willing to spend across text/image/budget runs?
6. **Image style ablation.** Ship one theme in v1, or invest early in a
   style-robustness ablation (2+ themes)?
7. **Public vs. private ratio.** Is 150 public / 1,300 private the right split,
   and who runs models against the private set (author-only vs. trusted runner)?
8. **Licensing.** Code license (MIT?) and dataset license/terms (custom
   "no-train" + canary) — confirm before first public push.
9. **Expert panel + error-model corpus.** Who are the 2+ PR≤5 panelists, how are
   they compensated, and which analyzed-match corpus (source, size, PR filter)
   trains the human-error model? Panel time is the scarcest resource in the
   tiering pipeline (§3.1).

---

## 10. Guiding principles (tl;dr)

- **Rollouts are truth; equity loss is the score; BenchPR puts models on the
  human axis.**
- **Never publish the answers** — private held-out set + hashes + canaries +
  date stamps.
- **Generate fresh, in-distribution positions; exclude anything famous or
  findable.**
- **One position, many representations; two tracks, one scoring pipeline.**
- **Cost is a first-class axis** — skill-vs-dollars and a fixed-$10 budget rank.
- **Start tiny (50), prove the loop, then scale.**
