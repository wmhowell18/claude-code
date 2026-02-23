"""TD(lambda) target computation for improved training signal.

Instead of using the raw game outcome as the training target for every
position in a game (Monte Carlo return, equivalent to lambda=1.0),
TD(lambda) blends the network's own predictions with the actual outcome.
This gives positions earlier in the game a softer, more informative
training signal.

Key insight from TD-Gammon: positions near the end of a game have a clear
outcome and should train toward it. Positions at the start of a game are
far from the outcome and benefit from bootstrapping off the network's
predictions at intermediate states.

The parameter lambda controls the blend:
- lambda=1.0: Pure Monte Carlo (use final game outcome for all positions)
- lambda=0.0: Pure one-step TD (bootstrap entirely from V(s_{t+1}))
- lambda=0.7: Good default for backgammon (empirically validated)

Implementation note: Following TD-Gammon (Tesauro 1995), all TD calculations
are performed from a fixed perspective (White). The network's per-player
predictions are converted to White's perspective, TD targets computed there,
then converted back to each step's player perspective for training.
This ensures deltas telescope correctly for lambda=1.0.
"""

import numpy as np
from typing import List, Optional

from backgammon.core.types import Player, GameOutcome
from backgammon.encoding.encoder import outcome_to_equity
from backgammon.training.self_play import GameResult


def _to_6dim(equity5: np.ndarray) -> np.ndarray:
    """Convert 5-dim equity to 6-dim (make lose_normal explicit).

    Input:  [win_normal, win_gammon, win_bg, lose_gammon, lose_bg]
    Output: [win_normal, win_gammon, win_bg, lose_normal, lose_gammon, lose_bg]
    """
    lose_normal = max(0.0, 1.0 - np.sum(equity5))
    return np.array([
        equity5[0], equity5[1], equity5[2],
        lose_normal, equity5[3], equity5[4],
    ], dtype=np.float32)


def _to_5dim(equity6: np.ndarray) -> np.ndarray:
    """Convert 6-dim equity to 5-dim (drop lose_normal, it's implicit).

    Input:  [win_normal, win_gammon, win_bg, lose_normal, lose_gammon, lose_bg]
    Output: [win_normal, win_gammon, win_bg, lose_gammon, lose_bg]
    """
    return np.array([
        equity6[0], equity6[1], equity6[2],
        equity6[4], equity6[5],
    ], dtype=np.float32)


def _flip_6dim(equity6: np.ndarray) -> np.ndarray:
    """Flip 6-dim equity from one player's perspective to the other's.

    This is a clean permutation: wins become losses and vice versa.
    [win_n, win_g, win_bg, lose_n, lose_g, lose_bg] ->
    [lose_n, lose_g, lose_bg, win_n, win_g, win_bg]
    """
    return np.array([
        equity6[3], equity6[4], equity6[5],
        equity6[0], equity6[1], equity6[2],
    ], dtype=np.float32)


def compute_td_lambda_targets(
    game: GameResult,
    lambda_param: float = 0.7,
) -> List[np.ndarray]:
    """Compute TD(lambda) targets for all positions in a game.

    Following TD-Gammon (Tesauro 1995), all computations are done from
    White's perspective so TD deltas telescope correctly for lambda=1.0.

    Requires that game.value_estimates is populated (network equity estimates
    were recorded during self-play).

    Falls back to pure Monte Carlo targets (lambda=1.0 behavior) when
    value estimates are not available.

    Args:
        game: Completed game with optional value_estimates.
        lambda_param: TD(lambda) parameter in [0, 1].
            0.0 = pure bootstrapping from network predictions.
            1.0 = pure Monte Carlo (game outcome only).
            0.7 = recommended default for backgammon.

    Returns:
        List of 5-dim equity target arrays, one per game step,
        from the perspective of the player to move at that step.
    """
    if game.outcome is None:
        return []  # Skip draws

    T = len(game.steps)
    if T == 0:
        return []

    # If no value estimates available, fall back to Monte Carlo
    if game.value_estimates is None or len(game.value_estimates) != T:
        return _monte_carlo_targets(game)

    # Check if any value estimates are None (non-neural agent steps)
    has_all_estimates = all(v is not None for v in game.value_estimates)
    if not has_all_estimates:
        return _monte_carlo_targets(game)

    # Convert all value estimates to 6-dim FROM WHITE's PERSPECTIVE.
    # V[t] is originally from game.steps[t].player's perspective.
    # We flip Black's estimates to White's perspective.
    V_white = []
    for t in range(T):
        v6 = _to_6dim(np.array(game.value_estimates[t], dtype=np.float32))
        if game.steps[t].player == Player.BLACK:
            v6 = _flip_6dim(v6)
        V_white.append(v6)

    # Compute final game outcome from White's perspective in 6-dim
    outcome_white = _to_6dim(outcome_to_equity(game.outcome, Player.WHITE).to_array())

    # Compute TD errors (delta_t) in White's perspective.
    # delta_t = V_white(s_{t+1}) - V_white(s_t)  for t < T-1
    # delta_{T-1} = outcome_white - V_white(s_{T-1})
    #
    # Because all values are from the same (White) perspective, deltas
    # telescope correctly: sum(deltas) = outcome_white - V_white(s_0).
    deltas = []
    for t in range(T - 1):
        delta = V_white[t + 1] - V_white[t]
        deltas.append(delta)

    # Last step
    delta_last = outcome_white - V_white[T - 1]
    deltas.append(delta_last)

    # Compute TD(lambda) targets using backward accumulation:
    # target_t = V_white(s_t) + sum_{k=t}^{T-1} lambda^{k-t} * delta_k
    targets_white = [None] * T
    eligibility = np.zeros(6, dtype=np.float32)

    for t in range(T - 1, -1, -1):
        eligibility = deltas[t] + lambda_param * eligibility
        target = V_white[t] + eligibility
        # Clamp to valid probability range
        target = np.clip(target, 0.0, 1.0)
        # Renormalize to a valid distribution (must sum to 1)
        target_sum = target.sum()
        if target_sum > 1e-8:
            target = target / target_sum
        targets_white[t] = target

    # Convert targets from White's perspective back to each step's player
    targets = []
    for t in range(T):
        target6 = targets_white[t]
        if game.steps[t].player == Player.BLACK:
            target6 = _flip_6dim(target6)
        targets.append(_to_5dim(target6))

    return targets


def _monte_carlo_targets(game: GameResult) -> List[np.ndarray]:
    """Compute pure Monte Carlo targets (lambda=1.0 equivalent).

    Each position gets the final game outcome as its target, which is
    the current behavior without TD(lambda).

    Args:
        game: Completed game.

    Returns:
        List of 5-dim equity target arrays.
    """
    targets = []
    for step in game.steps:
        equity = outcome_to_equity(game.outcome, step.player)
        targets.append(equity.to_array())
    return targets


def _flip_equity_if_needed(
    equity: np.ndarray,
    from_player: Player,
    to_player: Player,
) -> np.ndarray:
    """Flip 5-dim equity distribution if the perspective player changed.

    When converting equity from one player's perspective to the other's,
    wins become losses and vice versa.

    Args:
        equity: 5-dim equity from from_player's perspective.
        from_player: Player whose perspective the equity is from.
        to_player: Player whose perspective we want.

    Returns:
        Equity from to_player's perspective (5-dim).
    """
    if from_player == to_player:
        return equity.copy()

    # Convert to 6-dim, flip, convert back
    eq6 = _to_6dim(equity)
    flipped6 = _flip_6dim(eq6)
    return _to_5dim(flipped6)
