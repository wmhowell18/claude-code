"""Self-play game generation for training.

Generates training data by playing games between agents (warmstart pip count,
then neural network self-play). Supports recording network value estimates
for TD(lambda) target computation.

Includes a high-performance batched game simulator (`play_games_batched`)
that plays multiple games simultaneously with JIT-compiled batched inference,
providing ~50-100x speedup on TPU compared to sequential play.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass, field
from flax.training import train_state

from backgammon.core.board import Board, apply_move, generate_legal_moves, is_game_over, winner
from backgammon.core.types import Player, Move, GameOutcome
from backgammon.core.dice import roll_dice
from backgammon.evaluation.agents import Agent, pip_count_agent, random_agent


class GameStep(NamedTuple):
    """Single step in a game (state, action, outcome).

    Attributes:
        board: Board state before this move.
        player: Player who moved.
        legal_moves: All legal moves available.
        move_taken: The move that was actually played.
        dice: Dice roll for this turn.
    """
    board: Board
    player: Player
    legal_moves: List[Move]
    move_taken: Move
    dice: Tuple[int, int]


@dataclass
class GameResult:
    """Result of a completed game.

    Attributes:
        steps: Complete game trajectory.
        outcome: Final game outcome (None for draws).
        num_moves: Total number of moves played.
        starting_position: Initial board state.
        value_estimates: Optional per-step value estimates from the network
            during self-play. Shape: list of 5-element arrays (equity dist)
            from the perspective of the player to move at each step.
            Used for TD(lambda) target computation.
    """
    steps: List[GameStep]
    outcome: GameOutcome
    num_moves: int
    starting_position: Board
    value_estimates: Optional[List[np.ndarray]] = None


def play_game(
    white_agent: Agent,
    black_agent: Agent,
    starting_position: Board,
    max_moves: int = 200,
    rng: Optional[np.random.Generator] = None,
    record_value_estimates: bool = False,
) -> GameResult:
    """Play a single game between two agents.

    Args:
        white_agent: Agent playing white
        black_agent: Agent playing black
        starting_position: Initial board position
        max_moves: Maximum moves before declaring draw
        rng: Random number generator
        record_value_estimates: If True, capture network equity estimates
            at each step for TD(lambda) target computation. Only works
            when agents have a get_equity_estimate method (neural agents).

    Returns:
        GameResult with complete game trajectory
    """
    if rng is None:
        rng = np.random.default_rng()

    board = starting_position
    steps = []
    value_estimates = [] if record_value_estimates else None

    for move_num in range(max_moves):
        # Check if game is over
        if is_game_over(board):
            outcome = winner(board)
            return GameResult(
                steps=steps,
                outcome=outcome,
                num_moves=move_num,
                starting_position=starting_position,
                value_estimates=value_estimates,
            )

        # Current player
        current_player = board.player_to_move
        current_agent = white_agent if current_player == Player.WHITE else black_agent

        # Capture value estimate before move (for TD(lambda))
        if record_value_estimates:
            equity_est = _get_agent_equity(current_agent, board)
            value_estimates.append(equity_est)

        # Roll dice
        dice = roll_dice(rng)

        # Generate legal moves
        legal_moves = generate_legal_moves(board, current_player, dice)

        # Agent selects move
        move = current_agent.select_move(board, current_player, dice, legal_moves)

        # Record step
        steps.append(GameStep(
            board=board.copy(),
            player=current_player,
            legal_moves=legal_moves,
            move_taken=move,
            dice=dice,
        ))

        # Apply move
        board = apply_move(board, current_player, move)

    # Max moves reached - declare draw
    return GameResult(
        steps=steps,
        outcome=None,  # Draw
        num_moves=max_moves,
        starting_position=starting_position,
        value_estimates=value_estimates,
    )


def _get_agent_equity(agent: Agent, board: Board) -> Optional[np.ndarray]:
    """Try to get the equity estimate from an agent.

    Works with neural agents that have get_equity_estimate. Returns
    None for agents without this capability (pip count, random).

    Args:
        agent: Agent to query.
        board: Board state to evaluate.

    Returns:
        5-dim equity array or None if agent doesn't support it.
    """
    # The Agent wrapper stores the underlying function; check if the
    # agent's underlying object has get_equity_estimate.
    fn = agent.select_move_fn
    if hasattr(fn, '__self__') and hasattr(fn.__self__, 'get_equity_estimate'):
        return fn.__self__.get_equity_estimate(board)
    return None


def generate_training_batch(
    num_games: int,
    get_variant_fn,
    white_agent: Agent,
    black_agent: Agent,
    rng: Optional[np.random.Generator] = None,
    record_value_estimates: bool = False,
    max_moves: int = 200,
) -> List[GameResult]:
    """Generate a batch of training games.

    Args:
        num_games: Number of games to play
        get_variant_fn: Function returning position variants (e.g., get_early_training_variants)
        white_agent: Agent for white
        black_agent: Agent for black
        rng: Random number generator
        record_value_estimates: If True, record equity estimates for TD(lambda).
        max_moves: Maximum moves per game before declaring draw.

    Returns:
        List of completed game results
    """
    if rng is None:
        rng = np.random.default_rng()

    games = []
    variants = get_variant_fn()

    for _ in range(num_games):
        # Select random variant
        variant = variants[rng.integers(0, len(variants))]

        # Play game
        result = play_game(
            white_agent, black_agent, variant, rng=rng,
            record_value_estimates=record_value_estimates,
            max_moves=max_moves,
        )
        games.append(result)

    return games


def compute_game_statistics(games: List[GameResult]) -> dict:
    """Compute statistics from a batch of games.

    Args:
        games: List of game results

    Returns:
        Dictionary of statistics
    """
    total_games = len(games)
    white_wins = sum(1 for g in games if g.outcome and g.outcome.winner == Player.WHITE)
    black_wins = sum(1 for g in games if g.outcome and g.outcome.winner == Player.BLACK)
    draws = sum(1 for g in games if g.outcome is None)

    avg_moves = np.mean([g.num_moves for g in games])

    # Outcome types (normal, gammon, backgammon)
    gammons = sum(1 for g in games if g.outcome and g.outcome.points == 2)
    backgammons = sum(1 for g in games if g.outcome and g.outcome.points == 3)

    return {
        'total_games': total_games,
        'white_wins': white_wins,
        'black_wins': black_wins,
        'draws': draws,
        'white_win_rate': white_wins / total_games if total_games > 0 else 0.0,
        'avg_moves': avg_moves,
        'gammons': gammons,
        'backgammons': backgammons,
    }


# ==============================================================================
# BATCHED GAME SIMULATION (TPU/GPU OPTIMIZED)
# ==============================================================================
#
# The key performance insight: instead of playing 32 games sequentially
# (each making individual network calls), we step ALL games simultaneously
# and batch ALL network queries into a single JIT-compiled forward pass
# per game step. This reduces ~7,800 TPU dispatches to ~120, with each
# dispatch processing a much larger batch that TPUs handle efficiently.
#
# Additional optimizations:
# - JIT-compiled forward pass (avoids per-call TPU dispatch overhead)
# - Vectorized numpy encoding (eliminates per-point Python loops)
# - Combined equity+evaluation in single forward pass
# - Proper temperature-based exploration over values
# ==============================================================================


# JIT compilation cache (module-level to persist across calls)
_jit_inference_cache = {}


def _get_jit_inference(apply_fn):
    """Get or create a JIT-compiled inference function.

    Caches the JIT-compiled function by apply_fn identity so it's
    only compiled once per model. The compiled function takes (params, x)
    and returns the raw tuple from the model's forward pass.

    Args:
        apply_fn: The model's apply function (e.g., state.apply_fn).

    Returns:
        JIT-compiled function: (params, x) -> (equity, policy, cube, attn).
    """
    fn_id = id(apply_fn)
    if fn_id not in _jit_inference_cache:
        @jax.jit
        def _jit_infer(params, x):
            return apply_fn({'params': params}, x, training=False)
        _jit_inference_cache[fn_id] = _jit_infer
    return _jit_inference_cache[fn_id]


# Batch sizes to pad to, minimizing JIT recompilations on TPU.
# Each unique batch size triggers a new compilation; padding to these
# common sizes keeps the number of compilations small.
_BATCH_PAD_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def _pad_batch_size(n: int) -> int:
    """Find smallest padded batch size >= n."""
    for s in _BATCH_PAD_SIZES:
        if n <= s:
            return s
    return ((n + 255) // 256) * 256


def _encode_boards_fast(boards: List[Board]) -> np.ndarray:
    """Fast vectorized board encoding for raw encoding (feature_dim=2).

    Directly slices numpy arrays instead of calling per-point Python
    functions 26 times per board. ~10x faster than the generic path.

    Args:
        boards: List of Board objects.

    Returns:
        Array of shape (len(boards), 26, 2) with normalized checker counts.
    """
    n = len(boards)
    features = np.zeros((n, 26, 2), dtype=np.float32)
    inv15 = np.float32(1.0 / 15.0)
    for i, board in enumerate(boards):
        if board.player_to_move == Player.WHITE:
            features[i, :, 0] = board.white_checkers * inv15
            features[i, :, 1] = board.black_checkers * inv15
        else:
            features[i, :, 0] = board.black_checkers * inv15
            features[i, :, 1] = board.white_checkers * inv15
    return features


def _batched_inference(
    jit_fn,
    params,
    boards: List[Board],
) -> np.ndarray:
    """Run JIT-compiled batched inference on a list of boards.

    Handles encoding, batch padding (to minimize JIT recompilations),
    and returns raw 5-dim equity arrays.

    Args:
        jit_fn: JIT-compiled inference function.
        params: Model parameters.
        boards: List of Board objects to evaluate.

    Returns:
        Array of shape (len(boards), 5) with raw equity distributions.
    """
    n = len(boards)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float32)

    encoded = _encode_boards_fast(boards)

    # Pad batch to minimize JIT recompilations
    padded_n = _pad_batch_size(n)
    if padded_n > n:
        padding = np.zeros((padded_n - n, 26, 2), dtype=np.float32)
        encoded = np.concatenate([encoded, padding], axis=0)

    encoded_jax = jnp.array(encoded)
    equity, _, _, _ = jit_fn(params, encoded_jax)

    # Trim padding and convert to numpy
    return np.array(equity[:n], dtype=np.float32)


def _equity_to_value_np(equity: np.ndarray) -> np.ndarray:
    """Convert 5-dim equity distributions to scalar expected values.

    equity: [..., 5] = [win_normal, win_gammon, win_bg, lose_gammon, lose_bg]
    P(lose_normal) = 1 - sum(equity)
    Value = win_points - lose_points

    Args:
        equity: Array of shape (..., 5).

    Returns:
        Array of shape (...,) with scalar values in roughly [-3, +3].
    """
    win_value = (
        equity[..., 0] * 1.0
        + equity[..., 1] * 2.0
        + equity[..., 2] * 3.0
    )
    lose_normal = 1.0 - np.sum(equity, axis=-1)
    lose_value = (
        lose_normal * 1.0
        + equity[..., 3] * 2.0
        + equity[..., 4] * 3.0
    )
    return win_value - lose_value


def _terminal_value_for_player(board: Board, player: Player) -> float:
    """Get the value of a terminal position from a player's perspective.

    Args:
        board: Terminal board state.
        player: Player's perspective.

    Returns:
        Value in {-3, -2, -1, +1, +2, +3}.
    """
    outcome = winner(board)
    if outcome is None:
        return 0.0
    if outcome.winner == player:
        return float(outcome.points)
    else:
        return -float(outcome.points)


@dataclass
class _ActiveGame:
    """Internal state for a game in the batched simulator."""
    board: Board
    steps: list = field(default_factory=list)
    value_estimates: Optional[list] = None
    starting_position: Board = None
    done: bool = False
    outcome: Optional[GameOutcome] = None
    num_moves: int = 0
    # Per-step transient state (set during each step, not persisted)
    current_dice: Optional[Tuple[int, int]] = None
    current_moves: Optional[list] = None
    selected_move: Optional[Move] = None


def play_games_batched(
    num_games: int,
    state: train_state.TrainState,
    variants: List[Board],
    temperature: float = 0.3,
    max_moves: int = 200,
    rng: Optional[np.random.Generator] = None,
    record_value_estimates: bool = False,
) -> List[GameResult]:
    """Play multiple games simultaneously with batched neural network inference.

    Dramatically faster than sequential play because:
    1. All network queries across all active games are combined into single
       JIT-compiled batched forward passes (1 per game step instead of
       ~2 per move per game).
    2. Board encoding uses vectorized numpy operations (~10x faster).
    3. Batch padding minimizes JIT recompilations on TPU.
    4. Proper temperature-based exploration over value estimates.

    For 32 games at ~120 moves each:
    - Sequential: ~7,800 separate un-JITted network calls
    - Batched: ~120 JIT-compiled batched calls
    - Expected speedup: ~50-100x on TPU

    Args:
        num_games: Number of games to play simultaneously.
        state: Flax training state (model + params).
        variants: List of starting position variants to sample from.
        temperature: Exploration temperature (0=greedy, >0=softmax over values).
        max_moves: Maximum moves per game before declaring draw.
        rng: NumPy random generator.
        record_value_estimates: If True, record per-step equity estimates
            for TD(lambda) target computation.

    Returns:
        List of GameResult objects (same interface as generate_training_batch).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get JIT-compiled inference function (cached across calls)
    jit_fn = _get_jit_inference(state.apply_fn)

    # Initialize all games
    games = []
    for _ in range(num_games):
        variant = variants[rng.integers(len(variants))]
        games.append(_ActiveGame(
            board=variant.copy(),
            value_estimates=[] if record_value_estimates else None,
            starting_position=variant,
        ))

    for step_num in range(max_moves):
        # Identify active (non-finished) games
        active_indices = [i for i, g in enumerate(games) if not g.done]
        if not active_indices:
            break

        # --- Phase 1: Check for game-over, roll dice, generate legal moves ---
        still_active = []
        for i in active_indices:
            g = games[i]
            if is_game_over(g.board):
                g.done = True
                g.outcome = winner(g.board)
                continue
            g.current_dice = roll_dice(rng)
            g.current_moves = generate_legal_moves(
                g.board, g.board.player_to_move, g.current_dice
            )
            still_active.append(i)

        if not still_active:
            break

        # --- Phase 2: Collect boards for batched inference ---
        # (a) Current positions for equity estimates (TD(lambda))
        equity_boards = []
        equity_game_indices = []

        # (b) Result positions for move evaluation
        move_eval_boards = []
        # Track which result boards belong to which game/move
        games_needing_selection = []

        for i in still_active:
            g = games[i]
            player = g.board.player_to_move

            # Collect current position for equity estimate
            if record_value_estimates:
                equity_boards.append(g.board)
                equity_game_indices.append(i)

            # Single legal move → auto-select, no evaluation needed
            if len(g.current_moves) <= 1:
                g.selected_move = g.current_moves[0] if g.current_moves else ()
                continue

            # Multiple moves → collect result boards for evaluation
            move_info = []  # ('terminal', value) or ('network', idx)
            for move in g.current_moves:
                new_board = apply_move(g.board, player, move)
                if is_game_over(new_board):
                    move_info.append(('terminal',
                                      _terminal_value_for_player(new_board, player)))
                else:
                    move_info.append(('network', len(move_eval_boards)))
                    move_eval_boards.append(new_board)

            games_needing_selection.append((i, move_info))

        # --- Phase 3: Single batched inference for ALL boards ---
        # Combine equity boards and move-eval boards into one batch
        # to minimize TPU dispatches (one forward pass per game step).
        all_boards = equity_boards + move_eval_boards
        n_equity = len(equity_boards)
        equity_estimates = None
        move_eval_values = np.array([], dtype=np.float32)

        if all_boards:
            all_equity_raw = _batched_inference(jit_fn, state.params, all_boards)

            # Split results
            if n_equity > 0:
                equity_estimates = all_equity_raw[:n_equity]
            if move_eval_boards:
                move_eval_equity = all_equity_raw[n_equity:]
                move_eval_values = _equity_to_value_np(move_eval_equity)

        # --- Phase 4: Record equity estimates for TD(lambda) ---
        if record_value_estimates and equity_estimates is not None:
            for idx, game_i in enumerate(equity_game_indices):
                games[game_i].value_estimates.append(
                    equity_estimates[idx].copy()
                )

        # --- Phase 5: Select moves using temperature-based exploration ---
        for game_i, move_info in games_needing_selection:
            g = games[game_i]
            n_moves = len(move_info)
            values = np.empty(n_moves, dtype=np.float32)

            for mi, (mtype, mval) in enumerate(move_info):
                if mtype == 'terminal':
                    values[mi] = mval
                else:
                    # Network value is from board's player_to_move (=opponent
                    # after our move). Negate for our perspective.
                    values[mi] = -float(move_eval_values[mval])

            # Temperature-based move selection
            if temperature <= 0.0:
                best_idx = int(np.argmax(values))
            else:
                # Softmax with temperature over value estimates
                adjusted = values / temperature
                adjusted -= adjusted.max()  # numerical stability
                probs = np.exp(adjusted)
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs /= prob_sum
                else:
                    probs = np.ones(n_moves, dtype=np.float32) / n_moves
                best_idx = rng.choice(n_moves, p=probs)

            g.selected_move = g.current_moves[best_idx]

        # --- Phase 6: Apply moves and record game steps ---
        for i in still_active:
            g = games[i]
            if g.done:
                continue

            # Record step (preserve board state before move application)
            g.steps.append(GameStep(
                board=g.board.copy(),
                player=g.board.player_to_move,
                legal_moves=g.current_moves,
                move_taken=g.selected_move,
                dice=g.current_dice,
            ))

            # Apply the selected move
            g.board = apply_move(g.board, g.board.player_to_move, g.selected_move)
            g.num_moves += 1

            # Check if game ended after this move
            if is_game_over(g.board):
                g.done = True
                g.outcome = winner(g.board)

    # --- Convert to GameResult objects ---
    results = []
    for g in games:
        results.append(GameResult(
            steps=g.steps,
            outcome=g.outcome,
            num_moves=g.num_moves,
            starting_position=g.starting_position,
            value_estimates=g.value_estimates,
        ))

    return results
