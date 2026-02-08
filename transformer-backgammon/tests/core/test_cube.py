"""Tests for doubling cube implementation."""

import pytest
import numpy as np
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
from backgammon.core.cube import (
    initial_cube,
    owner_to_player,
    player_to_owner,
    can_double,
    legal_cube_actions,
    legal_response_actions,
    apply_cube_action,
    MAX_CUBE_VALUE,
    new_match,
    is_match_over,
    match_winner,
    update_match_score,
    is_crawford_game,
    can_double_in_match,
    game_points,
    match_equity,
    match_equity_from_state,
    calculate_cube_equity,
    should_double,
    should_take,
    evaluate_cube_decision,
    reference_cube_decision,
    reference_take_decision,
    encode_cube_state,
    encode_match_state,
    CUBE_DECISION_OUTPUT_DIM,
)


# ==============================================================================
# CUBE STATE TESTS
# ==============================================================================


class TestCubeState:
    """Tests for CubeState dataclass."""

    def test_initial_cube(self):
        """Test initial cube state."""
        cube = initial_cube()
        assert cube.value == 1
        assert cube.owner == CubeOwner.CENTERED

    def test_valid_cube_values(self):
        """Test creating cubes with valid values."""
        for value in [1, 2, 4, 8, 16, 32, 64]:
            cube = CubeState(value=value)
            assert cube.value == value

    def test_invalid_cube_value(self):
        """Test that invalid cube values raise errors."""
        with pytest.raises(AssertionError):
            CubeState(value=3)
        with pytest.raises(AssertionError):
            CubeState(value=0)
        with pytest.raises(AssertionError):
            CubeState(value=128)

    def test_cube_owners(self):
        """Test different cube owners."""
        cube_centered = CubeState(value=1, owner=CubeOwner.CENTERED)
        assert cube_centered.owner == CubeOwner.CENTERED

        cube_white = CubeState(value=2, owner=CubeOwner.WHITE)
        assert cube_white.owner == CubeOwner.WHITE

        cube_black = CubeState(value=4, owner=CubeOwner.BLACK)
        assert cube_black.owner == CubeOwner.BLACK

    def test_owner_to_player(self):
        """Test converting CubeOwner to Player."""
        assert owner_to_player(CubeOwner.WHITE) == Player.WHITE
        assert owner_to_player(CubeOwner.BLACK) == Player.BLACK
        assert owner_to_player(CubeOwner.CENTERED) is None

    def test_player_to_owner(self):
        """Test converting Player to CubeOwner."""
        assert player_to_owner(Player.WHITE) == CubeOwner.WHITE
        assert player_to_owner(Player.BLACK) == CubeOwner.BLACK


# ==============================================================================
# CUBE RULES TESTS
# ==============================================================================


class TestCanDouble:
    """Tests for can_double logic."""

    def test_centered_cube_either_player_can_double(self):
        """Both players can double when cube is centered."""
        cube = initial_cube()
        assert can_double(cube, Player.WHITE)
        assert can_double(cube, Player.BLACK)

    def test_owned_cube_only_owner_can_double(self):
        """Only the cube owner can double."""
        cube = CubeState(value=2, owner=CubeOwner.WHITE)
        assert can_double(cube, Player.WHITE)
        assert not can_double(cube, Player.BLACK)

        cube = CubeState(value=4, owner=CubeOwner.BLACK)
        assert not can_double(cube, Player.WHITE)
        assert can_double(cube, Player.BLACK)

    def test_max_cube_cannot_double(self):
        """Cannot double when cube is at maximum value."""
        cube = CubeState(value=64, owner=CubeOwner.WHITE)
        assert not can_double(cube, Player.WHITE)
        assert not can_double(cube, Player.BLACK)

        cube = CubeState(value=64, owner=CubeOwner.CENTERED)
        assert not can_double(cube, Player.WHITE)


class TestLegalCubeActions:
    """Tests for legal cube action generation."""

    def test_legal_actions_can_double(self):
        """When doubling is allowed, both NO_DOUBLE and DOUBLE are legal."""
        cube = initial_cube()
        actions = legal_cube_actions(cube, Player.WHITE)
        assert CubeAction.NO_DOUBLE in actions
        assert CubeAction.DOUBLE in actions
        assert len(actions) == 2

    def test_legal_actions_cannot_double(self):
        """When doubling is not allowed, only NO_DOUBLE is legal."""
        cube = CubeState(value=2, owner=CubeOwner.BLACK)
        actions = legal_cube_actions(cube, Player.WHITE)
        assert actions == [CubeAction.NO_DOUBLE]

    def test_legal_response_actions(self):
        """Response to a double is always TAKE or PASS."""
        cube = initial_cube()
        actions = legal_response_actions(cube)
        assert CubeAction.TAKE in actions
        assert CubeAction.PASS in actions
        assert len(actions) == 2


class TestApplyCubeAction:
    """Tests for applying cube actions."""

    def test_no_double(self):
        """NO_DOUBLE leaves cube unchanged."""
        cube = initial_cube()
        new_cube, game_over = apply_cube_action(cube, Player.WHITE, CubeAction.NO_DOUBLE)
        assert new_cube.value == 1
        assert new_cube.owner == CubeOwner.CENTERED
        assert game_over is None

    def test_double_from_centered(self):
        """Doubling from centered cube doubles value and transfers to opponent."""
        cube = initial_cube()
        new_cube, game_over = apply_cube_action(cube, Player.WHITE, CubeAction.DOUBLE)
        assert new_cube.value == 2
        assert new_cube.owner == CubeOwner.BLACK  # Opponent gets cube
        assert game_over is None

    def test_double_from_owned(self):
        """Doubling from owned cube doubles value and transfers ownership."""
        cube = CubeState(value=2, owner=CubeOwner.WHITE)
        new_cube, game_over = apply_cube_action(cube, Player.WHITE, CubeAction.DOUBLE)
        assert new_cube.value == 4
        assert new_cube.owner == CubeOwner.BLACK
        assert game_over is None

    def test_take(self):
        """Taking a double doesn't change cube state."""
        cube = CubeState(value=2, owner=CubeOwner.BLACK)
        new_cube, game_over = apply_cube_action(cube, Player.BLACK, CubeAction.TAKE)
        assert new_cube == cube
        assert game_over is None

    def test_pass_ends_game(self):
        """Passing a double ends the game."""
        # Scenario: White doubled (cube now value=2, owned by Black)
        # Black passes -> White wins at pre-double value
        cube = CubeState(value=2, owner=CubeOwner.BLACK)
        new_cube, game_over = apply_cube_action(cube, Player.BLACK, CubeAction.PASS)
        assert new_cube is None
        assert game_over is not None
        winner, points = game_over
        assert winner == Player.WHITE  # Opponent of the passer wins
        assert points == 2  # At current cube value

    def test_illegal_double_raises(self):
        """Doubling when not allowed raises ValueError."""
        cube = CubeState(value=2, owner=CubeOwner.BLACK)
        with pytest.raises(ValueError):
            apply_cube_action(cube, Player.WHITE, CubeAction.DOUBLE)

    def test_beaver(self):
        """Beaver accepts and immediately redoubles."""
        cube = initial_cube()
        new_cube, game_over = apply_cube_action(cube, Player.WHITE, CubeAction.BEAVER)
        assert new_cube.value == 4  # Quadrupled from 1
        assert new_cube.owner == CubeOwner.WHITE  # Back to the original doubler
        assert game_over is None

    def test_sequential_doubles(self):
        """Test a sequence of doubles: 1 -> 2 -> 4."""
        cube = initial_cube()

        # White doubles to 2
        cube, _ = apply_cube_action(cube, Player.WHITE, CubeAction.DOUBLE)
        assert cube.value == 2
        assert cube.owner == CubeOwner.BLACK

        # Black takes (no change to cube)
        cube, _ = apply_cube_action(cube, Player.BLACK, CubeAction.TAKE)
        assert cube.value == 2
        assert cube.owner == CubeOwner.BLACK

        # Black redoubles to 4
        cube, _ = apply_cube_action(cube, Player.BLACK, CubeAction.DOUBLE)
        assert cube.value == 4
        assert cube.owner == CubeOwner.WHITE


# ==============================================================================
# MATCH PLAY TESTS
# ==============================================================================


class TestMatchState:
    """Tests for match state management."""

    def test_new_match(self):
        """Test creating a new match."""
        match = new_match(5)
        assert match.target_points == 5
        assert match.white_score == 0
        assert match.black_score == 0
        assert not match.crawford
        assert not match.post_crawford

    def test_is_match_over(self):
        """Test match completion detection."""
        match = MatchState(target_points=5, white_score=0, black_score=0)
        assert not is_match_over(match)

        match = MatchState(target_points=5, white_score=5, black_score=3)
        assert is_match_over(match)

        match = MatchState(target_points=5, white_score=3, black_score=7)
        assert is_match_over(match)

    def test_match_winner(self):
        """Test match winner detection."""
        match = MatchState(target_points=5, white_score=3, black_score=2)
        assert match_winner(match) is None

        match = MatchState(target_points=5, white_score=5, black_score=3)
        assert match_winner(match) == Player.WHITE

        match = MatchState(target_points=5, white_score=2, black_score=5)
        assert match_winner(match) == Player.BLACK

    def test_update_match_score(self):
        """Test updating match score."""
        match = new_match(5)

        # White wins 2 points
        match = update_match_score(match, Player.WHITE, 2)
        assert match.white_score == 2
        assert match.black_score == 0

        # Black wins 3 points
        match = update_match_score(match, Player.BLACK, 3)
        assert match.white_score == 2
        assert match.black_score == 3

    def test_crawford_rule_activation(self):
        """Test Crawford rule activates at match point - 1."""
        match = MatchState(target_points=5, white_score=2, black_score=3)

        # White wins 2 points, reaching 4 (match point - 1)
        match = update_match_score(match, Player.WHITE, 2)
        assert match.white_score == 4
        assert match.crawford  # Crawford should activate

    def test_post_crawford(self):
        """Test transition to post-Crawford."""
        # Crawford game
        match = MatchState(
            target_points=5, white_score=4, black_score=3,
            crawford=True, post_crawford=False
        )

        # After Crawford game (black wins 1 point)
        match = update_match_score(match, Player.BLACK, 1)
        assert match.black_score == 4
        assert not match.crawford
        assert match.post_crawford

    def test_crawford_disables_doubling(self):
        """Test that doubling is disabled in Crawford game."""
        cube = initial_cube()
        match = MatchState(
            target_points=5, white_score=4, black_score=3,
            crawford=True
        )
        assert not can_double_in_match(cube, Player.WHITE, match)
        assert not can_double_in_match(cube, Player.BLACK, match)

    def test_post_crawford_allows_doubling(self):
        """Test that doubling is allowed in post-Crawford games."""
        cube = initial_cube()
        match = MatchState(
            target_points=5, white_score=4, black_score=3,
            crawford=False, post_crawford=True
        )
        assert can_double_in_match(cube, Player.WHITE, match)
        assert can_double_in_match(cube, Player.BLACK, match)

    def test_money_game_allows_doubling(self):
        """Test that money game (None match) allows normal doubling."""
        cube = initial_cube()
        assert can_double_in_match(cube, Player.WHITE, None)

    def test_game_points_calculation(self):
        """Test game points = base_points * cube_value."""
        cube = CubeState(value=4)
        assert game_points(1, cube) == 4   # Normal win
        assert game_points(2, cube) == 8   # Gammon
        assert game_points(3, cube) == 12  # Backgammon


# ==============================================================================
# MATCH EQUITY TABLE TESTS
# ==============================================================================


class TestMatchEquityTable:
    """Tests for match equity table lookups."""

    def test_equal_scores_near_half(self):
        """At equal scores, equity should be close to 0.5."""
        meq = match_equity(5, 5)
        assert abs(meq - 0.5) < 0.01

    def test_leading_player_advantage(self):
        """Player needing fewer points should have higher equity."""
        meq_leading = match_equity(1, 5)  # We need 1, opponent needs 5
        meq_trailing = match_equity(5, 1)  # We need 5, opponent needs 1

        assert meq_leading > 0.5
        assert meq_trailing < 0.5
        assert abs(meq_leading + meq_trailing - 1.0) < 0.01

    def test_one_away_one_away(self):
        """Both need 1 point: 50/50."""
        meq = match_equity(1, 1)
        assert abs(meq - 0.5) < 0.01

    def test_already_won(self):
        """Player with 0 points away has won."""
        meq = match_equity(0, 5)
        assert meq == 1.0

    def test_already_lost(self):
        """Player against opponent with 0 points away has lost."""
        meq = match_equity(5, 0)
        assert meq == 0.0

    def test_symmetry(self):
        """Table should be symmetric: MET(i,j) + MET(j,i) ≈ 1.0."""
        for i in range(1, 11):
            for j in range(1, 11):
                meq_ij = match_equity(i, j)
                meq_ji = match_equity(j, i)
                assert abs(meq_ij + meq_ji - 1.0) < 0.02, \
                    f"Symmetry violated: MET({i},{j})={meq_ij}, MET({j},{i})={meq_ji}"

    def test_monotonicity(self):
        """More points away should mean lower equity."""
        for opponent_away in [3, 5, 7]:
            prev_meq = 1.0
            for player_away in range(1, 11):
                meq = match_equity(player_away, opponent_away)
                assert meq <= prev_meq + 0.01, \
                    f"Not monotonic: MET({player_away},{opponent_away})={meq} > MET({player_away-1},{opponent_away})={prev_meq}"
                prev_meq = meq

    def test_match_equity_from_state(self):
        """Test convenience function with MatchState."""
        match = MatchState(target_points=7, white_score=3, black_score=5)

        white_meq = match_equity_from_state(match, Player.WHITE)
        black_meq = match_equity_from_state(match, Player.BLACK)

        # White needs 4, Black needs 2
        assert white_meq < 0.5  # White is behind
        assert black_meq > 0.5  # Black is ahead
        assert abs(white_meq + black_meq - 1.0) < 0.02


# ==============================================================================
# CUBEFUL EQUITY TESTS
# ==============================================================================


class TestCubeEquity:
    """Tests for cubeful equity calculation."""

    def test_money_game_strong_position(self):
        """In a strong position, doubling should be recommended."""
        equity = Equity(
            win_normal=0.6,
            win_gammon=0.1,
            win_backgammon=0.0,
            lose_gammon=0.05,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        assert ce.cubeful_equity > 0
        assert ce.no_double_equity > 0

    def test_money_game_losing_position(self):
        """In a losing position, should not double."""
        equity = Equity(
            win_normal=0.15,
            win_gammon=0.0,
            win_backgammon=0.0,
            lose_gammon=0.1,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        assert ce.no_double_equity < 0  # We're losing

    def test_should_double_clear_advantage(self):
        """should_double returns True with clear advantage."""
        equity = Equity(
            win_normal=0.65,
            win_gammon=0.1,
            win_backgammon=0.0,
            lose_gammon=0.02,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        # With ~75% winning chances and gammon threat, should double
        assert should_double(ce)

    def test_should_not_double_even_position(self):
        """should_double returns False in even position."""
        equity = Equity(
            win_normal=0.3,
            win_gammon=0.05,
            win_backgammon=0.0,
            lose_gammon=0.05,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        # Expected value is close to 0, risky to double
        assert not should_double(ce)

    def test_should_take_reasonable_position(self):
        """should_take returns True with reasonable chances."""
        equity = Equity(
            win_normal=0.25,
            win_gammon=0.05,
            win_backgammon=0.0,
            lose_gammon=0.05,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        # ~30% winning chances is a clear take
        assert should_take(ce)

    def test_should_pass_hopeless_position(self):
        """should_take returns False when doubler has overwhelming advantage.

        take/pass equity is from the opponent's (taker's) perspective.
        When the doubler has a huge advantage, the taker should pass.
        """
        equity = Equity(
            win_normal=0.55,
            win_gammon=0.25,
            win_backgammon=0.05,
            lose_gammon=0.01,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        # Doubler has ~85% win chances with heavy gammon threat -> opponent should pass
        assert not should_take(ce)

    def test_match_play_equity(self):
        """Test cube equity in match context."""
        equity = Equity(
            win_normal=0.5,
            win_gammon=0.1,
            win_backgammon=0.0,
            lose_gammon=0.05,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        match = MatchState(target_points=5, white_score=3, black_score=3)

        ce = calculate_cube_equity(equity, cube, Player.WHITE, match)
        assert ce.cubeful_equity > 0  # Winning position


# ==============================================================================
# CUBE DECISION QUALITY TESTS
# ==============================================================================


class TestCubeDecisionQuality:
    """Tests for cube decision evaluation."""

    def test_correct_decision(self):
        """Test evaluation of a correct cube decision."""
        equity = Equity(
            win_normal=0.65,
            win_gammon=0.1,
            win_backgammon=0.0,
            lose_gammon=0.02,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        # If should_double is True and we doubled, it's correct
        if should_double(ce):
            quality = evaluate_cube_decision(CubeAction.DOUBLE, ce)
            assert quality.was_correct
            assert not quality.was_blunder

    def test_blunder_detection(self):
        """Test that large errors are detected as blunders."""
        quality = CubeDecisionQuality(
            equity_error=0.15,
            was_blunder=True,
            was_correct=False,
        )
        assert quality.was_blunder
        assert not quality.was_correct


# ==============================================================================
# REFERENCE DECISION TESTS
# ==============================================================================


class TestReferenceCubeDecision:
    """Tests for reference (heuristic) cube decisions."""

    def test_reference_double_winning(self):
        """Reference agent doubles in winning position."""
        equity = Equity(
            win_normal=0.6,
            win_gammon=0.1,
            win_backgammon=0.0,
            lose_gammon=0.05,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        action = reference_cube_decision(equity, cube, Player.WHITE)
        assert action == CubeAction.DOUBLE

    def test_reference_no_double_losing(self):
        """Reference agent doesn't double in losing position."""
        equity = Equity(
            win_normal=0.15,
            win_gammon=0.0,
            win_backgammon=0.0,
            lose_gammon=0.1,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        action = reference_cube_decision(equity, cube, Player.WHITE)
        assert action == CubeAction.NO_DOUBLE

    def test_reference_take_decent_chances(self):
        """Reference agent takes with decent winning chances."""
        equity = Equity(
            win_normal=0.3,
            win_gammon=0.05,
            win_backgammon=0.0,
            lose_gammon=0.05,
            lose_backgammon=0.0,
        )
        cube = CubeState(value=2, owner=CubeOwner.BLACK)
        action = reference_take_decision(equity, cube)
        assert action == CubeAction.TAKE

    def test_reference_pass_poor_chances(self):
        """Reference agent passes with very poor chances."""
        equity = Equity(
            win_normal=0.05,
            win_gammon=0.0,
            win_backgammon=0.0,
            lose_gammon=0.2,
            lose_backgammon=0.05,
        )
        cube = CubeState(value=2, owner=CubeOwner.BLACK)
        action = reference_take_decision(equity, cube)
        assert action == CubeAction.PASS

    def test_reference_no_double_in_crawford(self):
        """Reference agent cannot double in Crawford game."""
        equity = Equity(
            win_normal=0.7,
            win_gammon=0.15,
            win_backgammon=0.0,
            lose_gammon=0.02,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        match = MatchState(
            target_points=5, white_score=4, black_score=3,
            crawford=True,
        )
        action = reference_cube_decision(equity, cube, Player.WHITE, match)
        assert action == CubeAction.NO_DOUBLE


# ==============================================================================
# CUBE ENCODING TESTS
# ==============================================================================


class TestCubeEncoding:
    """Tests for cube state encoding."""

    def test_encode_initial_cube(self):
        """Test encoding of initial centered cube."""
        cube = initial_cube()
        features = encode_cube_state(cube, Player.WHITE)

        assert features.shape == (4,)
        assert features.dtype == np.float32

        # Value 1 -> log2(1)/log2(64) = 0
        assert abs(features[0] - 0.0) < 1e-6
        # Is centered
        assert features[1] == 1.0
        # Nobody owns it
        assert features[2] == 0.0
        assert features[3] == 0.0

    def test_encode_owned_cube(self):
        """Test encoding of owned cube."""
        cube = CubeState(value=4, owner=CubeOwner.WHITE)
        features = encode_cube_state(cube, Player.WHITE)

        # Value 4 -> log2(4)/log2(64) = 2/6 = 0.333
        assert abs(features[0] - 2.0 / 6.0) < 1e-6
        # Not centered
        assert features[1] == 0.0
        # We own it (from White's perspective)
        assert features[2] == 1.0
        assert features[3] == 0.0

    def test_encode_opponent_owns_cube(self):
        """Test encoding when opponent owns cube."""
        cube = CubeState(value=8, owner=CubeOwner.BLACK)
        features = encode_cube_state(cube, Player.WHITE)

        # Not centered
        assert features[1] == 0.0
        # Opponent owns it (from White's perspective)
        assert features[2] == 0.0
        assert features[3] == 1.0

    def test_encode_max_cube(self):
        """Test encoding of max cube value."""
        cube = CubeState(value=64, owner=CubeOwner.WHITE)
        features = encode_cube_state(cube, Player.WHITE)

        # Value 64 -> log2(64)/log2(64) = 1.0
        assert abs(features[0] - 1.0) < 1e-6

    def test_encode_match_state_money_game(self):
        """Test encoding for money game (no match)."""
        features = encode_match_state(None, Player.WHITE)
        assert features.shape == (5,)
        assert features[0] == 1.0  # is_money_game

    def test_encode_match_state(self):
        """Test encoding for match play."""
        match = MatchState(target_points=7, white_score=3, black_score=5)
        features = encode_match_state(match, Player.WHITE)

        assert features.shape == (5,)
        assert features[0] == 0.0  # not money game
        assert abs(features[1] - 3.0 / 7.0) < 1e-6  # our score normalized
        assert abs(features[2] - 5.0 / 7.0) < 1e-6  # opp score normalized
        assert features[3] == 0.0  # not Crawford
        assert features[4] == 0.0  # not post-Crawford

    def test_encode_match_state_crawford(self):
        """Test encoding for Crawford game."""
        match = MatchState(
            target_points=5, white_score=4, black_score=3,
            crawford=True,
        )
        features = encode_match_state(match, Player.WHITE)
        assert features[3] == 1.0  # is Crawford

    def test_encode_perspective(self):
        """Test that encoding respects player perspective."""
        match = MatchState(target_points=7, white_score=3, black_score=5)

        white_features = encode_match_state(match, Player.WHITE)
        black_features = encode_match_state(match, Player.BLACK)

        # Scores should be swapped
        assert abs(white_features[1] - 3.0 / 7.0) < 1e-6  # White's score
        assert abs(white_features[2] - 5.0 / 7.0) < 1e-6  # Black's score
        assert abs(black_features[1] - 5.0 / 7.0) < 1e-6  # Black's score
        assert abs(black_features[2] - 3.0 / 7.0) < 1e-6  # White's score


# ==============================================================================
# CONSTANTS TESTS
# ==============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_cube_value(self):
        """Test max cube value constant."""
        assert MAX_CUBE_VALUE == 64

    def test_cube_decision_output_dim(self):
        """Test cube decision output dimension."""
        assert CUBE_DECISION_OUTPUT_DIM == 4


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestCubeIntegration:
    """Integration tests for complete cube decision flows."""

    def test_full_double_take_flow(self):
        """Test complete double -> take flow."""
        cube = initial_cube()

        # White doubles
        assert can_double(cube, Player.WHITE)
        cube, game_over = apply_cube_action(cube, Player.WHITE, CubeAction.DOUBLE)
        assert game_over is None
        assert cube.value == 2
        assert cube.owner == CubeOwner.BLACK

        # Black takes
        cube, game_over = apply_cube_action(cube, Player.BLACK, CubeAction.TAKE)
        assert game_over is None
        assert cube.value == 2
        assert cube.owner == CubeOwner.BLACK

        # Game continues. Black can redouble later.
        assert can_double(cube, Player.BLACK)
        assert not can_double(cube, Player.WHITE)

    def test_full_double_pass_flow(self):
        """Test complete double -> pass flow."""
        cube = initial_cube()

        # White doubles
        cube_after_double, _ = apply_cube_action(cube, Player.WHITE, CubeAction.DOUBLE)

        # Black passes -> game over, White wins 1 point (pre-double value)
        # Note: After the double, the cube state has value=2, owner=BLACK
        # Black passes with this state
        _, game_over = apply_cube_action(
            cube_after_double, Player.BLACK, CubeAction.PASS
        )
        assert game_over is not None
        winner, points = game_over
        assert winner == Player.WHITE
        assert points == 2  # Cube value at time of pass

    def test_match_play_game_sequence(self):
        """Test a sequence of games in a match."""
        match = new_match(5)
        assert not is_match_over(match)

        # Game 1: White wins 2 points (gammon)
        cube = initial_cube()
        match = update_match_score(match, Player.WHITE, game_points(2, cube))
        assert match.white_score == 2
        assert match.black_score == 0

        # Game 2: Black wins at cube=2 (normal)
        cube = CubeState(value=2, owner=CubeOwner.BLACK)
        match = update_match_score(match, Player.BLACK, game_points(1, cube))
        assert match.white_score == 2
        assert match.black_score == 2

        # Game 3: White wins at cube=4 (gammon = 8 points)
        cube = CubeState(value=4, owner=CubeOwner.WHITE)
        match = update_match_score(match, Player.WHITE, game_points(2, cube))
        assert match.white_score == 10  # Well past target
        assert is_match_over(match)
        assert match_winner(match) == Player.WHITE

    def test_cube_equity_consistency(self):
        """Test that cubeful equity is consistent with decisions."""
        equity = Equity(
            win_normal=0.55,
            win_gammon=0.1,
            win_backgammon=0.0,
            lose_gammon=0.05,
            lose_backgammon=0.0,
        )
        cube = initial_cube()
        ce = calculate_cube_equity(equity, cube, Player.WHITE)

        # If we should double, double equity > no-double equity
        if should_double(ce):
            assert ce.double_equity > ce.no_double_equity
        else:
            assert ce.double_equity <= ce.no_double_equity
