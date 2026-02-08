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

    equity, _, _ = state.apply_fn(
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
        ply: Search depth (0 or 1).
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
    else:
        raise ValueError(f"Unsupported ply depth: {ply}. Use 0 or 1.")
