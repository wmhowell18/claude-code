"""Doubling cube for match and money play.

This module implements the doubling cube mechanics following the specification
in cube.mli, including:
- Cube state management (value, ownership)
- Cube decision rules (can_double, legal actions, apply actions)
- Match play (Crawford rule, score tracking)
- Match equity tables (Kazaross/Rockwell)
- Cubeful equity calculation (should_double, should_take)
"""

from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

from backgammon.core.types import (
    Player,
    Equity,
    CubeOwner,
    CubeAction,
    CubeState,
    MatchState,
    CubeEquity,
    CubeDecisionQuality,
)


# ==============================================================================
# CUBE STATE
# ==============================================================================

MAX_CUBE_VALUE = 64


def initial_cube() -> CubeState:
    """Create initial cube state (centered, value 1).

    Returns:
        CubeState with value=1, owner=CENTERED
    """
    return CubeState(value=1, owner=CubeOwner.CENTERED)


def owner_to_player(owner: CubeOwner) -> Optional[Player]:
    """Convert CubeOwner to Player (None if centered).

    Args:
        owner: Cube owner

    Returns:
        Player who owns the cube, or None if centered
    """
    if owner == CubeOwner.WHITE:
        return Player.WHITE
    elif owner == CubeOwner.BLACK:
        return Player.BLACK
    return None


def player_to_owner(player: Player) -> CubeOwner:
    """Convert Player to CubeOwner.

    Args:
        player: Player

    Returns:
        Corresponding CubeOwner
    """
    if player == Player.WHITE:
        return CubeOwner.WHITE
    return CubeOwner.BLACK


# ==============================================================================
# CUBE RULES
# ==============================================================================


def can_double(cube: CubeState, player: Player) -> bool:
    """Check if a player can offer a double.

    A player can double if:
    - The cube is centered (either player can double), OR
    - They own the cube
    - AND the cube value hasn't reached the maximum (64)

    Args:
        cube: Current cube state
        player: Player wanting to double

    Returns:
        True if the player can legally double
    """
    if cube.value >= MAX_CUBE_VALUE:
        return False

    if cube.owner == CubeOwner.CENTERED:
        return True

    # Player can double only if they own the cube
    return cube.owner == player_to_owner(player)


def legal_cube_actions(cube: CubeState, player: Player) -> List[CubeAction]:
    """Get legal cube actions for a player at the start of their turn.

    Before rolling dice, a player may choose to double (if allowed) or not.

    Args:
        cube: Current cube state
        player: Player whose turn it is

    Returns:
        List of legal cube actions
    """
    actions = [CubeAction.NO_DOUBLE]

    if can_double(cube, player):
        actions.append(CubeAction.DOUBLE)

    return actions


def legal_response_actions(cube: CubeState) -> List[CubeAction]:
    """Get legal response actions after a double is offered.

    Args:
        cube: Current cube state (before the double is applied)

    Returns:
        List of legal response actions (TAKE or PASS)
    """
    return [CubeAction.TAKE, CubeAction.PASS]


def apply_cube_action(
    cube: CubeState,
    player: Player,
    action: CubeAction,
) -> Tuple[Optional[CubeState], Optional[Tuple[Player, int]]]:
    """Apply a cube action and return the result.

    Args:
        cube: Current cube state
        player: Player performing the action
        action: The cube action to apply

    Returns:
        Tuple of (new_cube_state, game_over_result)
        - If game continues: (new_cube_state, None)
        - If game ends (pass): (None, (winner, points))

    Raises:
        ValueError: If the action is illegal
    """
    if action == CubeAction.NO_DOUBLE:
        return cube, None

    if action == CubeAction.DOUBLE:
        if not can_double(cube, player):
            raise ValueError(
                f"{player} cannot double: cube value={cube.value}, owner={cube.owner}"
            )
        # Double is offered but not yet resolved - the opponent must respond.
        # The cube value doubles and ownership transfers to the opponent.
        new_cube = CubeState(
            value=cube.value * 2,
            owner=player_to_owner(player.opponent()),
        )
        return new_cube, None

    if action == CubeAction.TAKE:
        # Opponent accepts the double. Cube state already updated by DOUBLE action.
        return cube, None

    if action == CubeAction.PASS:
        # Opponent declines the double. Current player wins at current cube value.
        # Note: the "current cube value" is the value BEFORE doubling.
        # When you pass a double, you lose the current stake.
        winner = player.opponent() if cube.owner != CubeOwner.CENTERED else player
        # The player who offered the double wins. In the flow:
        # Player A doubles -> Player B passes -> Player A wins at the pre-double value.
        # But since we track who is performing the action:
        # If `player` is passing, the opponent (who doubled) wins.
        points = cube.value  # Points at stake before the double
        return None, (player.opponent(), points)

    if action == CubeAction.BEAVER:
        if not can_double(cube, player):
            raise ValueError(f"{player} cannot beaver")
        # Beaver: accept the double and immediately redouble
        # Cube value quadruples, ownership returns to the original doubler
        new_cube = CubeState(
            value=cube.value * 4,
            owner=player_to_owner(player),
        )
        return new_cube, None

    raise ValueError(f"Unknown cube action: {action}")


# ==============================================================================
# MATCH PLAY
# ==============================================================================


def new_match(target_points: int) -> MatchState:
    """Create a new match.

    Args:
        target_points: Number of points to play to

    Returns:
        New MatchState
    """
    return MatchState(target_points=target_points)


def is_match_over(match: MatchState) -> bool:
    """Check if the match is over.

    Args:
        match: Current match state

    Returns:
        True if either player has reached the target score
    """
    return (
        match.white_score >= match.target_points
        or match.black_score >= match.target_points
    )


def match_winner(match: MatchState) -> Optional[Player]:
    """Get the match winner.

    Args:
        match: Current match state

    Returns:
        Winner player, or None if match is not over
    """
    if match.white_score >= match.target_points:
        return Player.WHITE
    if match.black_score >= match.target_points:
        return Player.BLACK
    return None


def update_match_score(
    match: MatchState,
    game_winner: Player,
    points: int,
) -> MatchState:
    """Update match score after a game.

    Also handles Crawford rule transitions:
    - If a player reaches match point - 1, the next game is Crawford
    - After the Crawford game, we enter post-Crawford play

    Args:
        match: Current match state
        game_winner: Who won the game
        points: Points won (cube_value * game_multiplier)

    Returns:
        Updated MatchState
    """
    new_white = match.white_score
    new_black = match.black_score

    if game_winner == Player.WHITE:
        new_white += points
    else:
        new_black += points

    # Determine Crawford state
    new_crawford = False
    new_post_crawford = match.post_crawford

    if match.crawford:
        # Crawford game just happened, now we're post-Crawford
        new_post_crawford = True
    elif not match.post_crawford:
        # Check if someone just reached match point - 1
        white_at_match_point = (
            new_white == match.target_points - 1
            and match.white_score < match.target_points - 1
        )
        black_at_match_point = (
            new_black == match.target_points - 1
            and match.black_score < match.target_points - 1
        )
        if white_at_match_point or black_at_match_point:
            new_crawford = True

    return MatchState(
        target_points=match.target_points,
        white_score=new_white,
        black_score=new_black,
        crawford=new_crawford,
        post_crawford=new_post_crawford,
    )


def is_crawford_game(match: MatchState) -> bool:
    """Check if the current game is a Crawford game.

    In the Crawford game, the doubling cube is disabled.

    Args:
        match: Current match state

    Returns:
        True if this is the Crawford game
    """
    return match.crawford


def can_double_in_match(
    cube: CubeState,
    player: Player,
    match: Optional[MatchState],
) -> bool:
    """Check if doubling is allowed considering match context.

    Doubling is disabled in the Crawford game.
    In money games (match=None), standard cube rules apply.

    Args:
        cube: Current cube state
        player: Player wanting to double
        match: Match state (None for money games)

    Returns:
        True if player can double
    """
    if match is not None and is_crawford_game(match):
        return False

    return can_double(cube, player)


def game_points(
    game_result_points: int,
    cube: CubeState,
) -> int:
    """Calculate total points won in a game considering the cube.

    Args:
        game_result_points: Base game points (1=normal, 2=gammon, 3=backgammon)
        cube: Current cube state

    Returns:
        Total points = base_points * cube_value
    """
    return game_result_points * cube.value


# ==============================================================================
# MATCH EQUITY TABLES
# ==============================================================================

# Kazaross/Rockwell Match Equity Table (MET)
# This is a well-known precomputed table giving the probability of winning
# a match from any score. Values are for the player who needs the number of
# points indicated by the row to win, against an opponent who needs the number
# of points indicated by the column.
#
# Table entries: MET[i][j] = probability of winning when you need i+1 points
# and opponent needs j+1 points.
#
# Source: Kit Woolsey's adaptation of the Kazaross/Rockwell table.

_MATCH_EQUITY_TABLE = np.array([
    # Opponent needs: 1     2      3      4      5      6      7      8      9     10     11     12     13     14     15
    [0.5000, 0.3006, 0.2495, 0.1812, 0.1515, 0.1083, 0.0914, 0.0650, 0.0553, 0.0393, 0.0337, 0.0240, 0.0207, 0.0148, 0.0128],  # You need 1
    [0.6994, 0.5000, 0.4020, 0.3179, 0.2646, 0.2070, 0.1736, 0.1349, 0.1140, 0.0882, 0.0751, 0.0580, 0.0498, 0.0385, 0.0333],  # You need 2
    [0.7505, 0.5980, 0.5000, 0.4110, 0.3505, 0.2851, 0.2433, 0.1958, 0.1684, 0.1344, 0.1163, 0.0923, 0.0805, 0.0637, 0.0558],  # You need 3
    [0.8188, 0.6821, 0.5890, 0.5000, 0.4354, 0.3653, 0.3176, 0.2635, 0.2297, 0.1887, 0.1654, 0.1348, 0.1190, 0.0963, 0.0855],  # You need 4
    [0.8485, 0.7354, 0.6495, 0.5646, 0.5000, 0.4295, 0.3788, 0.3215, 0.2839, 0.2381, 0.2113, 0.1754, 0.1567, 0.1291, 0.1159],  # You need 5
    [0.8917, 0.7930, 0.7149, 0.6347, 0.5705, 0.5000, 0.4467, 0.3851, 0.3440, 0.2936, 0.2632, 0.2222, 0.2004, 0.1677, 0.1523],  # You need 6
    [0.9086, 0.8264, 0.7567, 0.6824, 0.6212, 0.5533, 0.5000, 0.4382, 0.3953, 0.3425, 0.3097, 0.2652, 0.2415, 0.2048, 0.1878],  # You need 7
    [0.9350, 0.8651, 0.8042, 0.7365, 0.6785, 0.6149, 0.5618, 0.5000, 0.4548, 0.3998, 0.3650, 0.3172, 0.2914, 0.2505, 0.2314],  # You need 8
    [0.9447, 0.8860, 0.8316, 0.7703, 0.7161, 0.6560, 0.6047, 0.5452, 0.5000, 0.4441, 0.4083, 0.3587, 0.3322, 0.2889, 0.2688],  # You need 9
    [0.9607, 0.9118, 0.8656, 0.8113, 0.7619, 0.7064, 0.6575, 0.6002, 0.5559, 0.5000, 0.4630, 0.4113, 0.3836, 0.3374, 0.3160],  # You need 10
    [0.9663, 0.9249, 0.8837, 0.8346, 0.7887, 0.7368, 0.6903, 0.6350, 0.5917, 0.5370, 0.5000, 0.4476, 0.4194, 0.3722, 0.3502],  # You need 11
    [0.9760, 0.9420, 0.9077, 0.8652, 0.8246, 0.7778, 0.7348, 0.6828, 0.6413, 0.5887, 0.5524, 0.5000, 0.4710, 0.4228, 0.4000],  # You need 12
    [0.9793, 0.9502, 0.9195, 0.8810, 0.8433, 0.7996, 0.7585, 0.7086, 0.6678, 0.6164, 0.5806, 0.5290, 0.5000, 0.4513, 0.4282],  # You need 13
    [0.9852, 0.9615, 0.9363, 0.9037, 0.8709, 0.8323, 0.7952, 0.7495, 0.7111, 0.6626, 0.6278, 0.5772, 0.5487, 0.5000, 0.4765],  # You need 14
    [0.9872, 0.9667, 0.9442, 0.9145, 0.8841, 0.8477, 0.8122, 0.7686, 0.7312, 0.6840, 0.6498, 0.6000, 0.5718, 0.5235, 0.5000],  # You need 15
], dtype=np.float64)


def match_equity(player_away: int, opponent_away: int) -> float:
    """Look up match equity from the Kazaross/Rockwell table.

    Returns the probability that the player wins the match.

    Args:
        player_away: Points the player needs to win (1-15)
        opponent_away: Points the opponent needs to win (1-15)

    Returns:
        Probability of winning the match (0.0 to 1.0)
    """
    if player_away <= 0:
        return 1.0
    if opponent_away <= 0:
        return 0.0

    # Clamp to table bounds
    p = min(player_away, 15) - 1
    o = min(opponent_away, 15) - 1

    return float(_MATCH_EQUITY_TABLE[p][o])


def match_equity_from_state(
    match: MatchState,
    player: Player,
) -> float:
    """Calculate match equity for a player given the current match state.

    Args:
        match: Current match state
        player: Player to calculate equity for

    Returns:
        Probability of winning the match (0.0 to 1.0)
    """
    if player == Player.WHITE:
        player_away = match.target_points - match.white_score
        opponent_away = match.target_points - match.black_score
    else:
        player_away = match.target_points - match.black_score
        opponent_away = match.target_points - match.white_score

    return match_equity(player_away, opponent_away)


# ==============================================================================
# CUBEFUL EQUITY CALCULATION
# ==============================================================================


def calculate_cube_equity(
    equity: Equity,
    cube: CubeState,
    player: Player,
    match: Optional[MatchState] = None,
) -> CubeEquity:
    """Calculate cubeful equity for a position.

    For money games, uses the standard cubeful equity formula.
    For match play, uses match equity table lookups.

    Args:
        equity: Raw (cubeless) equity estimate from network
        cube: Current cube state
        player: Player to move
        match: Match state (None for money game)

    Returns:
        CubeEquity with all equity components
    """
    if match is not None:
        return _calculate_match_cube_equity(equity, cube, player, match)
    return _calculate_money_cube_equity(equity, cube, player)


def _calculate_money_cube_equity(
    equity: Equity,
    cube: CubeState,
    player: Player,
) -> CubeEquity:
    """Calculate cubeful equity for money games.

    In money games, the expected value scales linearly with cube value.

    The key insight is:
    - No-double equity: current expected value at current cube
    - Double/Take equity: expected value at doubled cube (opponent owns)
    - Double/Pass equity: player wins at current cube value

    Args:
        equity: Raw equity estimate
        cube: Current cube state
        player: Player considering the double

    Returns:
        CubeEquity for money game
    """
    ev = equity.expected_value()

    # No-double: equity stays at current cube value
    no_double_eq = ev * cube.value

    # If we double and opponent takes: equity at 2x cube value
    # but opponent now owns the cube (recube vig works against us slightly)
    doubled_value = cube.value * 2
    double_take_eq = ev * doubled_value

    # If we double and opponent passes: we win at current cube value
    double_pass_eq = cube.value

    # The opponent takes if take_equity > pass_equity
    # From opponent's perspective:
    #   take_equity = -ev * doubled_value (negative because opponent's perspective)
    #   pass_equity = -cube.value (they lose current stake)
    # Opponent takes if -ev * doubled_value > -cube.value
    # i.e., ev * doubled_value < cube.value
    # i.e., ev < 0.5 (for centered cube)

    take_equity_for_opponent = -ev * doubled_value
    pass_equity_for_opponent = -float(cube.value)

    # Double equity: depends on whether opponent takes or passes
    if take_equity_for_opponent >= pass_equity_for_opponent:
        # Opponent will take
        double_eq = double_take_eq
    else:
        # Opponent will pass
        double_eq = double_pass_eq

    # Cubeful equity: max of double and no-double from our perspective
    cubeful_eq = max(no_double_eq, double_eq)

    return CubeEquity(
        raw_equity=equity,
        cubeful_equity=cubeful_eq,
        double_equity=double_eq,
        no_double_equity=no_double_eq,
        take_equity=take_equity_for_opponent,
        pass_equity=pass_equity_for_opponent,
    )


def _calculate_match_cube_equity(
    equity: Equity,
    cube: CubeState,
    player: Player,
    match: MatchState,
) -> CubeEquity:
    """Calculate cubeful equity for match play.

    In match play, the value of points is nonlinear (depends on score).
    Uses the match equity table to convert point outcomes to match
    winning chances.

    Args:
        equity: Raw equity estimate
        cube: Current cube state
        player: Player considering the double
        match: Match state

    Returns:
        CubeEquity for match play
    """
    if player == Player.WHITE:
        player_away = match.target_points - match.white_score
        opp_away = match.target_points - match.black_score
    else:
        player_away = match.target_points - match.black_score
        opp_away = match.target_points - match.white_score

    # Calculate no-double match equity
    no_double_meq = _expected_match_equity(
        equity, player_away, opp_away, cube.value
    )

    # Calculate double/take match equity
    doubled_value = cube.value * 2
    double_take_meq = _expected_match_equity(
        equity, player_away, opp_away, doubled_value
    )

    # Double/pass: player wins cube.value points
    double_pass_meq = match_equity(
        player_away - cube.value, opp_away
    )

    # Opponent's take equity (from opponent's perspective)
    take_meq_for_opp = 1.0 - double_take_meq
    pass_meq_for_opp = 1.0 - double_pass_meq

    # Double equity: opponent's optimal response
    if take_meq_for_opp >= pass_meq_for_opp:
        double_meq = double_take_meq
    else:
        double_meq = double_pass_meq

    cubeful_meq = max(no_double_meq, double_meq)

    return CubeEquity(
        raw_equity=equity,
        cubeful_equity=cubeful_meq,
        double_equity=double_meq,
        no_double_equity=no_double_meq,
        take_equity=take_meq_for_opp,
        pass_equity=pass_meq_for_opp,
    )


def _expected_match_equity(
    equity: Equity,
    player_away: int,
    opp_away: int,
    stake: int,
) -> float:
    """Calculate expected match equity given an equity distribution and stake.

    Converts the 5-dimensional equity distribution into a single match
    equity value by weighting each outcome by its probability and looking
    up the resulting match equity.

    Args:
        equity: Equity distribution
        player_away: Points player needs
        opp_away: Points opponent needs
        stake: Current stake (cube value)

    Returns:
        Expected match equity (probability of winning match)
    """
    meq = 0.0

    # Win normal (1 * stake points)
    meq += equity.win_normal * match_equity(
        player_away - 1 * stake, opp_away
    )

    # Win gammon (2 * stake points)
    meq += equity.win_gammon * match_equity(
        player_away - 2 * stake, opp_away
    )

    # Win backgammon (3 * stake points)
    meq += equity.win_backgammon * match_equity(
        player_away - 3 * stake, opp_away
    )

    # Lose normal (opponent wins 1 * stake)
    meq += equity.lose_normal * match_equity(
        player_away, opp_away - 1 * stake
    )

    # Lose gammon (opponent wins 2 * stake)
    meq += equity.lose_gammon * match_equity(
        player_away, opp_away - 2 * stake
    )

    # Lose backgammon (opponent wins 3 * stake)
    meq += equity.lose_backgammon * match_equity(
        player_away, opp_away - 3 * stake
    )

    return meq


# ==============================================================================
# CUBE DECISIONS
# ==============================================================================


def should_double(cube_equity: CubeEquity) -> bool:
    """Determine if the player should double.

    The player should double if doubling gives higher equity than not doubling.

    Args:
        cube_equity: Calculated cube equity

    Returns:
        True if player should double
    """
    return cube_equity.double_equity > cube_equity.no_double_equity


def should_take(cube_equity: CubeEquity) -> bool:
    """Determine if the opponent should take (accept) a double.

    The opponent should take if taking gives higher equity than passing.

    Args:
        cube_equity: Calculated cube equity

    Returns:
        True if opponent should take
    """
    return cube_equity.take_equity >= cube_equity.pass_equity


def evaluate_cube_decision(
    action: CubeAction,
    cube_equity: CubeEquity,
) -> CubeDecisionQuality:
    """Evaluate the quality of a cube decision.

    Compares the action taken to the optimal action and computes
    the equity error.

    Args:
        action: The cube action that was taken
        cube_equity: The calculated cube equity for the position

    Returns:
        CubeDecisionQuality metrics
    """
    optimal_double = should_double(cube_equity)
    optimal_take = should_take(cube_equity)

    # Calculate equity of the action taken vs optimal
    if action == CubeAction.NO_DOUBLE:
        actual_eq = cube_equity.no_double_equity
        optimal_eq = (
            cube_equity.double_equity if optimal_double
            else cube_equity.no_double_equity
        )
    elif action == CubeAction.DOUBLE:
        actual_eq = cube_equity.double_equity
        optimal_eq = (
            cube_equity.double_equity if optimal_double
            else cube_equity.no_double_equity
        )
    elif action == CubeAction.TAKE:
        actual_eq = cube_equity.take_equity
        optimal_eq = (
            cube_equity.take_equity if optimal_take
            else cube_equity.pass_equity
        )
    elif action == CubeAction.PASS:
        actual_eq = cube_equity.pass_equity
        optimal_eq = (
            cube_equity.take_equity if optimal_take
            else cube_equity.pass_equity
        )
    else:
        actual_eq = 0.0
        optimal_eq = 0.0

    equity_error = abs(actual_eq - optimal_eq)

    return CubeDecisionQuality(
        equity_error=equity_error,
        was_blunder=equity_error > 0.1,
        was_correct=equity_error <= 0.02,
    )


# ==============================================================================
# REFERENCE CUBE DECISIONS (heuristic baseline)
# ==============================================================================


def reference_cube_decision(
    equity: Equity,
    cube: CubeState,
    player: Player,
    match: Optional[MatchState] = None,
) -> CubeAction:
    """Make a reference cube decision based on equity thresholds.

    This provides a simple heuristic baseline for cube decisions
    using standard money-game thresholds:
    - Double if equity > 0.0 (we're winning)
    - Take if equity > -0.5 (we're not too far behind)

    These are simplified rules. Real decisions depend on gammon rates,
    position volatility, and match score.

    Args:
        equity: Position equity estimate
        cube: Current cube state
        player: Player to move
        match: Match state (None for money game)

    Returns:
        Recommended cube action
    """
    if not can_double_in_match(cube, player, match):
        return CubeAction.NO_DOUBLE

    ev = equity.expected_value()

    # Simple thresholds for money game
    # In real play, these depend heavily on gammon rates and volatility
    if ev > 0.0:
        return CubeAction.DOUBLE
    return CubeAction.NO_DOUBLE


def reference_take_decision(
    equity: Equity,
    cube: CubeState,
) -> CubeAction:
    """Make a reference take/pass decision.

    Standard take point in money game is -0.5 expected value
    (25% winning chances assuming no gammons).

    Args:
        equity: Position equity from the perspective of the player being doubled
        cube: Current cube state (after the double)

    Returns:
        TAKE or PASS
    """
    ev = equity.expected_value()

    # Standard money game take point
    # With gammons, the actual take point is more nuanced
    if ev >= -0.5:
        return CubeAction.TAKE
    return CubeAction.PASS


# ==============================================================================
# CUBE ENCODING (for neural network input)
# ==============================================================================


def encode_cube_state(
    cube: CubeState,
    player: Player,
) -> NDArray[np.float32]:
    """Encode cube state as features for neural network input.

    Returns a 4-element feature vector:
    [cube_value_normalized, is_centered, we_own_cube, opp_owns_cube]

    Args:
        cube: Current cube state
        player: Player whose perspective to encode from

    Returns:
        Array of shape (4,) with cube features
    """
    # Normalize cube value: log2(value) / log2(64) maps 1->0, 64->1
    cube_value_norm = np.log2(cube.value) / np.log2(MAX_CUBE_VALUE)

    is_centered = 1.0 if cube.owner == CubeOwner.CENTERED else 0.0

    player_owner = player_to_owner(player)
    opp_owner = player_to_owner(player.opponent())

    we_own = 1.0 if cube.owner == player_owner else 0.0
    opp_owns = 1.0 if cube.owner == opp_owner else 0.0

    return np.array(
        [cube_value_norm, is_centered, we_own, opp_owns],
        dtype=np.float32,
    )


def encode_match_state(
    match: Optional[MatchState],
    player: Player,
) -> NDArray[np.float32]:
    """Encode match state as features for neural network input.

    Returns a 5-element feature vector:
    [is_money_game, our_score_norm, opp_score_norm, is_crawford, is_post_crawford]

    For money games (match=None), returns [1, 0, 0, 0, 0].

    Args:
        match: Match state (None for money game)
        player: Player whose perspective to encode from

    Returns:
        Array of shape (5,) with match features
    """
    if match is None:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if player == Player.WHITE:
        our_score = match.white_score
        opp_score = match.black_score
    else:
        our_score = match.black_score
        opp_score = match.white_score

    # Normalize scores by target
    our_score_norm = our_score / match.target_points
    opp_score_norm = opp_score / match.target_points

    return np.array([
        0.0,  # not money game
        our_score_norm,
        opp_score_norm,
        1.0 if match.crawford else 0.0,
        1.0 if match.post_crawford else 0.0,
    ], dtype=np.float32)


# Cube decision output dimension (for network head)
CUBE_DECISION_OUTPUT_DIM = 4  # no_double, double, take, pass
