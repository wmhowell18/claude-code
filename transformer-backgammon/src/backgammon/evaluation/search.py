"""Search module for backgammon move evaluation with lookahead.

Implements n-ply search with dice averaging for stronger move selection.
The key insight: instead of evaluating a position directly (0-ply),
evaluate what happens after the opponent responds to each possible
dice roll (1-ply), giving a much more accurate position assessment.

Terminology:
- 0-ply: Evaluate resulting position directly with the network.
- 1-ply: For each candidate move, apply it, then average over all 21
  opponent dice rolls (opponent plays best response via 0-ply).
- 2-ply: Like 1-ply, but opponent uses 1-ply for their evaluation.
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional
from flax.training import train_state

from backgammon.core.board import (
    Board,
    apply_move,
    generate_legal_moves,
    is_game_over,
    winner,
    pip_count,
)
from backgammon.core.types import Player, Move, Dice, LegalMoves, GameOutcome
from backgammon.core.dice import ALL_DICE_ROLLS, DICE_PROBABILITIES
from backgammon.encoding.encoder import raw_encoding_config, EncodingConfig


def _encode_boards_batch(
    boards: List[Board],
    encoding_config: EncodingConfig,
) -> np.ndarray:
    """Encode multiple boards into a batch array for network evaluation.

    Args:
        boards: List of board states to encode.
        encoding_config: Encoding configuration.

    Returns:
        Array of shape (len(boards), 26, feature_dim).
    """
    from backgammon.encoding.encoder import extract_position_features

    batch_size = len(boards)
    features = np.zeros(
        (batch_size, 26, encoding_config.feature_dim), dtype=np.float32
    )
    for i, board in enumerate(boards):
        for point in range(26):
            features[i, point] = extract_position_features(
                encoding_config, board, point
            )
    return features


def _equity_to_value(equity: jnp.ndarray) -> jnp.ndarray:
    """Convert 5-dim equity distribution to scalar expected value.

    Equity: [win_normal, win_gammon, win_bg, lose_gammon, lose_bg]
    P(lose_normal) = 1 - sum(equity)
    Value = win_probs*points - lose_probs*points

    Args:
        equity: Shape (..., 5) equity probabilities.

    Returns:
        Shape (...,) scalar expected value in roughly [-3, +3].
    """
    win_value = (
        equity[..., 0] * 1.0
        + equity[..., 1] * 2.0
        + equity[..., 2] * 3.0
    )
    lose_normal = 1.0 - jnp.sum(equity, axis=-1)
    lose_value = (
        lose_normal * 1.0
        + equity[..., 3] * 2.0
        + equity[..., 4] * 3.0
    )
    return win_value - lose_value


def _batch_evaluate(
    state: train_state.TrainState,
    boards: List[Board],
    encoding_config: EncodingConfig,
) -> np.ndarray:
    """Evaluate a batch of boards, returning value from each board's
    player_to_move perspective.

    Args:
        state: Network training state (model + params).
        boards: List of board states.
        encoding_config: Encoding configuration.

    Returns:
        Array of shape (len(boards),) with value estimates.
        Each value is from the perspective of that board's player_to_move.
    """
    if not boards:
        return np.array([], dtype=np.float32)

    encoded = _encode_boards_batch(boards, encoding_config)
    encoded_jax = jnp.array(encoded)

    equity, _, _, _ = state.apply_fn(
        {'params': state.params},
        encoded_jax,
        training=False,
    )
    values = _equity_to_value(equity)
    return np.array(values, dtype=np.float32)


def _terminal_value(board: Board, perspective: Player) -> float:
    """Get exact value of a terminal (game-over) position.

    Args:
        board: Terminal board state.
        perspective: Which player's perspective to return value for.

    Returns:
        Value in {-3, -2, -1, +1, +2, +3}.
    """
    outcome = winner(board)
    if outcome is None:
        return 0.0
    if outcome.winner == perspective:
        return float(outcome.points)
    else:
        return -float(outcome.points)


def evaluate_move_0ply(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    move: Move,
    encoding_config: Optional[EncodingConfig] = None,
) -> float:
    """Evaluate a single move at 0-ply (direct network evaluation).

    Apply the move, evaluate the resulting position with the network.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player making the move.
        move: Move to evaluate.
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Value estimate from player's perspective (higher = better).
    """
    if encoding_config is None:
        encoding_config = raw_encoding_config()

    new_board = apply_move(board, player, move)

    if is_game_over(new_board):
        return _terminal_value(new_board, player)

    # Network evaluates from new_board.player_to_move's perspective
    # (which is the opponent after apply_move). Negate for our perspective.
    values = _batch_evaluate(state, [new_board], encoding_config)
    return -float(values[0])


def select_move_0ply(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    legal_moves: LegalMoves,
    encoding_config: Optional[EncodingConfig] = None,
) -> Tuple[Move, float]:
    """Select best move using 0-ply evaluation (batch).

    Evaluates all legal moves in a single network forward pass.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player to move.
        legal_moves: List of legal moves.
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Tuple of (best_move, best_value).
    """
    if encoding_config is None:
        encoding_config = raw_encoding_config()

    if not legal_moves:
        return (), 0.0
    if len(legal_moves) == 1:
        val = evaluate_move_0ply(
            state, board, player, legal_moves[0], encoding_config
        )
        return legal_moves[0], val

    # Apply all moves and collect resulting boards
    result_boards = []
    terminal_indices = []
    terminal_values = []
    network_indices = []

    for i, move in enumerate(legal_moves):
        new_board = apply_move(board, player, move)
        if is_game_over(new_board):
            terminal_indices.append(i)
            terminal_values.append(_terminal_value(new_board, player))
        else:
            result_boards.append(new_board)
            network_indices.append(i)

    # Batch evaluate all non-terminal positions
    if result_boards:
        network_values = _batch_evaluate(state, result_boards, encoding_config)
    else:
        network_values = np.array([])

    # Combine values (negate network values since they're from opponent's POV)
    move_values = np.full(len(legal_moves), -np.inf, dtype=np.float32)
    for idx, val in zip(terminal_indices, terminal_values):
        move_values[idx] = val
    for i, idx in enumerate(network_indices):
        move_values[idx] = -network_values[i]

    best_idx = int(np.argmax(move_values))
    return legal_moves[best_idx], float(move_values[best_idx])


def evaluate_move_1ply(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    move: Move,
    encoding_config: Optional[EncodingConfig] = None,
) -> float:
    """Evaluate a single move at 1-ply (dice-averaged lookahead).

    Apply the move, then for each of 21 opponent dice rolls, find the
    opponent's best response (at 0-ply) and average the results weighted
    by dice probability.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player making the move.
        move: Move to evaluate.
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Value estimate from player's perspective (higher = better).
    """
    if encoding_config is None:
        encoding_config = raw_encoding_config()

    new_board = apply_move(board, player, move)

    if is_game_over(new_board):
        return _terminal_value(new_board, player)

    opponent = player.opponent()

    # Collect ALL positions across all 21 dice rolls for batch evaluation.
    # Structure: for each dice roll, store the resulting boards from each
    # of the opponent's legal moves.
    all_boards = []
    dice_move_map = []  # (dice_idx, num_moves, has_terminal, terminal_val)

    for dice_idx, dice_roll in enumerate(ALL_DICE_ROLLS):
        opp_moves = generate_legal_moves(new_board, opponent, dice_roll)

        if not opp_moves:
            # Opponent has no legal moves — position stays the same.
            # Evaluate from our perspective (it's our turn again).
            all_boards.append(new_board)
            dice_move_map.append((dice_idx, 1, False, 0.0))
            continue

        terminal_found = False
        terminal_val = 0.0
        n_network = 0

        for opp_move in opp_moves:
            after_opp = apply_move(new_board, opponent, opp_move)
            if is_game_over(after_opp):
                # Terminal — check if this is best for opponent
                v = _terminal_value(after_opp, opponent)
                if not terminal_found or v > terminal_val:
                    terminal_val = v
                    terminal_found = True
            else:
                all_boards.append(after_opp)
                n_network += 1

        dice_move_map.append(
            (dice_idx, n_network, terminal_found, terminal_val)
        )

    # Single batch evaluation of all collected boards
    if all_boards:
        all_values = _batch_evaluate(state, all_boards, encoding_config)
    else:
        all_values = np.array([])

    # Now compute the dice-averaged value.
    # For each dice roll, opponent picks the move that maximizes THEIR value
    # (which minimizes ours).
    board_cursor = 0
    weighted_value = 0.0

    for dice_idx, n_network, has_terminal, terminal_val in dice_move_map:
        dice_roll = ALL_DICE_ROLLS[dice_idx]
        prob = DICE_PROBABILITIES[dice_roll]

        # Collect opponent's values for this dice roll
        opp_best = -np.inf

        # Check terminal moves
        if has_terminal:
            opp_best = max(opp_best, terminal_val)

        # Check network-evaluated moves
        # The boards are from opponent's perspective (opponent just moved,
        # so player_to_move is us). Values are from our perspective.
        # Opponent wants to MINIMIZE our value = MAXIMIZE their own.
        for _ in range(n_network):
            # all_values[board_cursor] is from the board's player_to_move
            # perspective. After opponent moves, player_to_move = us.
            # So this value is from OUR perspective.
            our_val = float(all_values[board_cursor])
            # Opponent's value = negative of our value
            opp_val = -our_val
            opp_best = max(opp_best, opp_val)
            board_cursor += 1

        if opp_best == -np.inf:
            # No moves for opponent (shouldn't happen if we handled no-moves above)
            opp_best = 0.0

        # Convert back to our perspective
        our_value_for_this_dice = -opp_best
        weighted_value += prob * our_value_for_this_dice

    return weighted_value


def select_move_1ply(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    legal_moves: LegalMoves,
    encoding_config: Optional[EncodingConfig] = None,
) -> Tuple[Move, float]:
    """Select best move using 1-ply evaluation (dice-averaged lookahead).

    For each legal move, computes 1-ply value and picks the best.
    This batches the network evaluations across all moves and all
    opponent dice responses for efficiency.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player to move.
        legal_moves: List of legal moves.
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Tuple of (best_move, best_value).
    """
    if encoding_config is None:
        encoding_config = raw_encoding_config()

    if not legal_moves:
        return (), 0.0
    if len(legal_moves) == 1:
        val = evaluate_move_1ply(
            state, board, player, legal_moves[0], encoding_config
        )
        return legal_moves[0], val

    # For maximum efficiency, we collect ALL boards across ALL moves and
    # ALL opponent dice rolls into one giant batch for a single forward pass.
    all_boards = []

    # Track structure: for each of our moves, for each dice roll,
    # how many opponent-response boards were added
    # Structure: move_map[move_idx] = list of (n_network, has_terminal, terminal_val)
    move_map = []
    # Which of our moves produce immediate terminal positions
    terminal_moves = {}  # move_idx -> terminal_value

    for move_idx, move in enumerate(legal_moves):
        new_board = apply_move(board, player, move)

        if is_game_over(new_board):
            terminal_moves[move_idx] = _terminal_value(new_board, player)
            move_map.append(None)  # Sentinel: terminal
            continue

        opponent = player.opponent()
        dice_info = []

        for dice_roll in ALL_DICE_ROLLS:
            opp_moves = generate_legal_moves(new_board, opponent, dice_roll)

            if not opp_moves:
                # Opponent can't move — board unchanged, our turn again
                all_boards.append(new_board)
                dice_info.append((1, False, 0.0))
                continue

            terminal_found = False
            terminal_val = 0.0
            n_network = 0

            for opp_move in opp_moves:
                after_opp = apply_move(new_board, opponent, opp_move)
                if is_game_over(after_opp):
                    v = _terminal_value(after_opp, opponent)
                    if not terminal_found or v > terminal_val:
                        terminal_val = v
                        terminal_found = True
                else:
                    all_boards.append(after_opp)
                    n_network += 1

            dice_info.append((n_network, terminal_found, terminal_val))

        move_map.append(dice_info)

    # Single giant batch evaluation
    if all_boards:
        all_values = _batch_evaluate(state, all_boards, encoding_config)
    else:
        all_values = np.array([])

    # Compute value for each of our moves
    board_cursor = 0
    move_values = np.full(len(legal_moves), -np.inf, dtype=np.float32)

    for move_idx in range(len(legal_moves)):
        if move_idx in terminal_moves:
            move_values[move_idx] = terminal_moves[move_idx]
            continue

        dice_info = move_map[move_idx]
        weighted_value = 0.0

        for dice_idx, (n_network, has_terminal, terminal_val) in enumerate(
            dice_info
        ):
            dice_roll = ALL_DICE_ROLLS[dice_idx]
            prob = DICE_PROBABILITIES[dice_roll]

            opp_best = -np.inf

            if has_terminal:
                opp_best = max(opp_best, terminal_val)

            for _ in range(n_network):
                our_val = float(all_values[board_cursor])
                opp_val = -our_val
                opp_best = max(opp_best, opp_val)
                board_cursor += 1

            if opp_best == -np.inf:
                opp_best = 0.0

            weighted_value += prob * (-opp_best)

        move_values[move_idx] = weighted_value

    best_idx = int(np.argmax(move_values))
    return legal_moves[best_idx], float(move_values[best_idx])


def evaluate_move_2ply(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    move: Move,
    encoding_config: Optional[EncodingConfig] = None,
) -> float:
    """Evaluate a single move at 2-ply (opponent uses 1-ply response).

    Apply the move, then for each of 21 opponent dice rolls, the opponent
    selects their best move using 1-ply evaluation (dice-averaged lookahead
    from their perspective). This gives a much more accurate assessment
    than 1-ply at the cost of significantly more computation.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player making the move.
        move: Move to evaluate.
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Value estimate from player's perspective (higher = better).
    """
    if encoding_config is None:
        encoding_config = raw_encoding_config()

    new_board = apply_move(board, player, move)

    if is_game_over(new_board):
        return _terminal_value(new_board, player)

    opponent = player.opponent()

    weighted_value = 0.0

    for dice_roll in ALL_DICE_ROLLS:
        prob = DICE_PROBABILITIES[dice_roll]
        opp_moves = generate_legal_moves(new_board, opponent, dice_roll)

        if not opp_moves:
            # Opponent has no legal moves — position stays the same.
            # Evaluate from our perspective with 0-ply (no further search).
            values = _batch_evaluate(state, [new_board], encoding_config)
            our_val = float(values[0])
            weighted_value += prob * our_val
            continue

        # Opponent picks the move that maximizes THEIR value.
        # Each opponent move is evaluated at 1-ply from the opponent's
        # perspective (they average over OUR dice responses).
        opp_best = -np.inf

        for opp_move in opp_moves:
            after_opp = apply_move(new_board, opponent, opp_move)

            if is_game_over(after_opp):
                opp_val = _terminal_value(after_opp, opponent)
            else:
                # Evaluate at 1-ply from opponent's perspective.
                # This means: for each of OUR dice rolls, we pick our best
                # response (at 0-ply), and opponent averages the results.
                opp_val = _evaluate_position_1ply(
                    state, after_opp, opponent, encoding_config
                )

            opp_best = max(opp_best, opp_val)

        # Convert back to our perspective
        weighted_value += prob * (-opp_best)

    return weighted_value


def _evaluate_position_1ply(
    state: train_state.TrainState,
    board: Board,
    perspective: Player,
    encoding_config: EncodingConfig,
) -> float:
    """Evaluate a position at 1-ply from a given player's perspective.

    For each of 21 dice rolls for the OTHER player, that player picks
    their best 0-ply move, and we average the resulting values.

    This is the "inner loop" of 2-ply search: it tells the opponent
    how good a position is for them after they've moved.

    Args:
        state: Network training state.
        board: Board state to evaluate (it's perspective's "turn").
        perspective: Player whose perspective we evaluate from.
        encoding_config: Encoding configuration.

    Returns:
        Value from perspective's point of view.
    """
    other = perspective.opponent()

    # Collect all boards we need to evaluate in one batch
    all_boards = []
    dice_info = []  # (n_network, has_terminal, terminal_val)

    for dice_roll in ALL_DICE_ROLLS:
        moves = generate_legal_moves(board, perspective, dice_roll)

        if not moves:
            # No legal moves — use current board value
            all_boards.append(board)
            dice_info.append((1, False, 0.0))
            continue

        terminal_found = False
        terminal_val = -np.inf
        n_network = 0

        for m in moves:
            after = apply_move(board, perspective, m)
            if is_game_over(after):
                v = _terminal_value(after, perspective)
                if v > terminal_val:
                    terminal_val = v
                    terminal_found = True
            else:
                all_boards.append(after)
                n_network += 1

        dice_info.append((n_network, terminal_found, terminal_val))

    # Batch evaluate
    if all_boards:
        all_values = _batch_evaluate(state, all_boards, encoding_config)
    else:
        all_values = np.array([])

    # Compute dice-averaged value
    board_cursor = 0
    weighted_value = 0.0

    for dice_idx, (n_network, has_terminal, terminal_val) in enumerate(dice_info):
        prob = DICE_PROBABILITIES[ALL_DICE_ROLLS[dice_idx]]

        # Perspective picks the best move for this dice roll
        best_val = -np.inf

        if has_terminal:
            best_val = max(best_val, terminal_val)

        for _ in range(n_network):
            # Value is from the board's next player_to_move perspective,
            # which is the OTHER player after perspective moves.
            # Negate to get perspective's value.
            other_val = float(all_values[board_cursor])
            our_val = -other_val
            best_val = max(best_val, our_val)
            board_cursor += 1

        if best_val == -np.inf:
            best_val = 0.0

        weighted_value += prob * best_val

    return weighted_value


def select_move_2ply(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    legal_moves: LegalMoves,
    encoding_config: Optional[EncodingConfig] = None,
) -> Tuple[Move, float]:
    """Select best move using 2-ply evaluation.

    For each legal move, computes 2-ply value (opponent uses 1-ply
    responses) and picks the best. This is significantly more expensive
    than 1-ply but provides stronger play.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player to move.
        legal_moves: List of legal moves.
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Tuple of (best_move, best_value).
    """
    if encoding_config is None:
        encoding_config = raw_encoding_config()

    if not legal_moves:
        return (), 0.0
    if len(legal_moves) == 1:
        val = evaluate_move_2ply(
            state, board, player, legal_moves[0], encoding_config
        )
        return legal_moves[0], val

    best_move = legal_moves[0]
    best_value = -np.inf

    for move in legal_moves:
        val = evaluate_move_2ply(
            state, board, player, move, encoding_config
        )
        if val > best_value:
            best_value = val
            best_move = move

    return best_move, best_value


def select_move(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    dice: Dice,
    legal_moves: LegalMoves,
    ply: int = 0,
    encoding_config: Optional[EncodingConfig] = None,
) -> Tuple[Move, float]:
    """Select best move at a given search depth.

    Convenience function that dispatches to the appropriate ply-level.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player to move.
        dice: Current dice roll (unused for evaluation, but part of interface).
        legal_moves: List of legal moves.
        ply: Search depth (0, 1, or 2).
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Tuple of (best_move, best_value).
    """
    if ply == 0:
        return select_move_0ply(
            state, board, player, legal_moves, encoding_config
        )
    elif ply == 1:
        return select_move_1ply(
            state, board, player, legal_moves, encoding_config
        )
    elif ply == 2:
        return select_move_2ply(
            state, board, player, legal_moves, encoding_config
        )
    else:
        raise ValueError(f"Unsupported ply depth: {ply}. Use 0, 1, or 2.")


# ==============================================================================
# MOVE ORDERING HEURISTICS
# ==============================================================================


def _score_move_heuristic(board: Board, player: Player, move: Move) -> float:
    """Score a move with a fast heuristic for move ordering.

    Higher scores indicate more promising moves (evaluated first).
    This is used to order moves before expensive evaluation so that
    pruning (top-K) is more effective.

    Heuristic factors:
    - Pip count improvement (more = better)
    - Hitting opponent blots (big bonus)
    - Making new points (bonus)
    - Leaving blots (penalty)

    Args:
        board: Current board state
        player: Player making the move
        move: Move to score

    Returns:
        Heuristic score (higher = more promising)
    """
    score = 0.0

    # Pip count improvement
    new_board = apply_move(board, player, move)
    old_pips = pip_count(board, player)
    new_pips = pip_count(new_board, player)
    pip_improvement = old_pips - new_pips
    score += pip_improvement

    # Check each step for hits and blots
    for step in move:
        if step.hits_opponent:
            score += 20.0  # Big bonus for hitting

    # Check resulting position for blots and points made
    opponent = player.opponent()
    for point in range(1, 25):
        our_count = new_board.get_checkers(player, point)
        if our_count == 1:
            score -= 5.0  # Penalty for leaving a blot
        elif our_count >= 2:
            # Check if this is a new point (wasn't made before)
            old_count = board.get_checkers(player, point)
            if old_count < 2:
                score += 3.0  # Bonus for making a new point

    return score


def order_moves(
    board: Board,
    player: Player,
    legal_moves: LegalMoves,
) -> LegalMoves:
    """Order moves from most to least promising using heuristics.

    Args:
        board: Current board state
        player: Player making the move
        legal_moves: List of legal moves to order

    Returns:
        Moves sorted by heuristic score (best first)
    """
    if len(legal_moves) <= 1:
        return legal_moves

    scored = [
        (_score_move_heuristic(board, player, move), i, move)
        for i, move in enumerate(legal_moves)
    ]
    scored.sort(reverse=True)  # Best first

    return [move for _, _, move in scored]


def select_move_2ply_pruned(
    state: train_state.TrainState,
    board: Board,
    player: Player,
    legal_moves: LegalMoves,
    top_k: int = 10,
    encoding_config: Optional[EncodingConfig] = None,
) -> Tuple[Move, float]:
    """Select best move using 2-ply evaluation with move ordering and pruning.

    First orders moves by heuristic, then only evaluates the top-K most
    promising moves at full 2-ply depth. This dramatically reduces computation
    for positions with many legal moves.

    Args:
        state: Network training state.
        board: Current board state.
        player: Player to move.
        legal_moves: List of legal moves.
        top_k: Maximum number of moves to evaluate at 2-ply.
        encoding_config: Encoding config (defaults to raw).

    Returns:
        Tuple of (best_move, best_value).
    """
    if encoding_config is None:
        encoding_config = raw_encoding_config()

    if not legal_moves:
        return (), 0.0
    if len(legal_moves) == 1:
        val = evaluate_move_2ply(
            state, board, player, legal_moves[0], encoding_config
        )
        return legal_moves[0], val

    # Order moves by heuristic
    ordered_moves = order_moves(board, player, legal_moves)

    # Only evaluate top-K at 2-ply
    candidates = ordered_moves[:top_k]

    best_move = candidates[0]
    best_value = -np.inf

    for move in candidates:
        val = evaluate_move_2ply(
            state, board, player, move, encoding_config
        )
        if val > best_value:
            best_value = val
            best_move = move

    return best_move, best_value
