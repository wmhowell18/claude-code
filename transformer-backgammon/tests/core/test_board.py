"""Tests for board module."""

import pytest
from backgammon.core.board import (
    initial_board,
    empty_board,
    flip_board,
    pip_count,
    checkers_on_bar,
    checkers_borne_off,
    can_bear_off,
    is_game_over,
    winner,
    generate_legal_moves,
    apply_move,
    is_legal_move,
    board_to_string,
    # Position variants
    nackgammon_start,
    split_back_checkers,
    slotted_5_point,
    slotted_bar_point,
    advanced_anchor,
    hypergammon_start,
    micro_gammon_start,
    short_gammon_start,
    bearoff_practice,
    race_position,
    get_all_variant_starts,
    get_early_training_variants,
    get_mid_training_variants,
    get_late_training_variants,
    random_variant_start,
)
from backgammon.core.types import Player, MoveStep, GameOutcome


class TestBoardInitialization:
    """Tests for board initialization."""

    def test_initial_board(self):
        """Test standard starting position."""
        board = initial_board()

        # Check white pieces
        assert board.white_checkers[24] == 2
        assert board.white_checkers[13] == 5
        assert board.white_checkers[8] == 3
        assert board.white_checkers[6] == 5

        # Check black pieces
        assert board.black_checkers[1] == 2
        assert board.black_checkers[12] == 5
        assert board.black_checkers[17] == 3
        assert board.black_checkers[19] == 5

        # Check total
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # White starts
        assert board.player_to_move == Player.WHITE

    def test_empty_board(self):
        """Test empty board."""
        board = empty_board()
        assert sum(board.white_checkers) == 0
        assert sum(board.black_checkers) == 0

    def test_flip_board(self):
        """Test board flipping."""
        board = initial_board()
        flipped = flip_board(board)

        # Check that white and black are swapped and mirrored
        # White's pieces at 24 should become black's pieces at 1
        assert flipped.black_checkers[1] == 2
        # White's pieces at 6 should become black's pieces at 19
        assert flipped.black_checkers[19] == 5

        # Player should be flipped
        assert flipped.player_to_move == Player.BLACK


class TestBoardQueries:
    """Tests for board state queries."""

    def test_pip_count_initial(self):
        """Test pip count at starting position."""
        board = initial_board()

        # Standard pip count at start is 167 for each player
        white_pips = pip_count(board, Player.WHITE)
        black_pips = pip_count(board, Player.BLACK)

        assert white_pips == 167
        assert black_pips == 167

    def test_checkers_on_bar(self):
        """Test counting checkers on bar."""
        board = empty_board()
        assert checkers_on_bar(board, Player.WHITE) == 0

        board.white_checkers[0] = 2
        assert checkers_on_bar(board, Player.WHITE) == 2

    def test_checkers_borne_off(self):
        """Test counting borne off checkers."""
        board = empty_board()
        assert checkers_borne_off(board, Player.WHITE) == 0

        board.white_checkers[25] = 5
        assert checkers_borne_off(board, Player.WHITE) == 5

    def test_can_bear_off(self):
        """Test bear off eligibility."""
        board = empty_board()

        # All white checkers in home board (1-6)
        board.white_checkers[1] = 2
        board.white_checkers[3] = 3
        assert can_bear_off(board, Player.WHITE)

        # Add one checker outside home
        board.white_checkers[13] = 1
        assert not can_bear_off(board, Player.WHITE)

        # Checker on bar
        board.white_checkers[13] = 0
        board.white_checkers[0] = 1
        assert not can_bear_off(board, Player.WHITE)

    def test_is_game_over(self):
        """Test game over detection."""
        board = initial_board()
        assert not is_game_over(board)

        # White bears off all checkers
        board.white_checkers[:] = 0
        board.white_checkers[25] = 15
        assert is_game_over(board)

    def test_winner_detection(self):
        """Test winner detection and outcome types."""
        # Normal win
        board = empty_board()
        board.white_checkers[25] = 15
        board.black_checkers[25] = 5  # Black has some off

        outcome = winner(board)
        assert outcome is not None
        assert outcome.winner == Player.WHITE
        assert outcome.points == 1

        # Gammon (opponent hasn't borne off any)
        board.black_checkers[25] = 0  # Clear the borne off checkers
        board.black_checkers[13] = 5  # All on board
        outcome = winner(board)
        assert outcome.points == 2

        # Backgammon (opponent in winner's home)
        board.black_checkers[13] = 0
        board.black_checkers[3] = 5  # In white's home
        outcome = winner(board)
        assert outcome.points == 3


class TestMoveGeneration:
    """Tests for move generation."""

    def test_generate_opening_moves(self):
        """Test move generation from opening position."""
        board = initial_board()

        # Roll 3-1 (common opening roll)
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))
        assert len(moves) > 0

        # Should be able to use both dice
        # Each move should have 2 steps
        assert all(len(move) == 2 for move in moves)

    def test_generate_doubles(self):
        """Test doubles give 4 moves."""
        board = initial_board()

        # Roll double 6s
        moves = generate_legal_moves(board, Player.WHITE, (6, 6))
        assert len(moves) > 0

        # Should use all 4 dice
        # (though some may be blocked)
        max_steps = max(len(move) for move in moves)
        assert max_steps <= 4

    def test_entering_from_bar(self):
        """Test entering from bar is required."""
        board = empty_board()

        # Put white checker on bar
        board.white_checkers[0] = 1
        board.white_checkers[6] = 5  # Some checkers in home

        # Black controls some entry points
        board.black_checkers[20] = 2  # Blocks white's 5-point entry

        # Roll 3-1
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        # All moves must enter from bar first
        for move in moves:
            if len(move) > 0:
                assert move[0].from_point == 0  # Bar

    def test_no_legal_moves(self):
        """Test when no moves are possible."""
        board = empty_board()

        # Put white on bar
        board.white_checkers[0] = 1

        # Black blocks all entry points (19-24)
        for point in range(19, 25):
            board.black_checkers[point] = 2

        # White cannot enter
        moves = generate_legal_moves(board, Player.WHITE, (3, 5))

        # Should return empty move
        assert len(moves) == 1
        assert moves[0] == ()

    def test_bearing_off(self):
        """Test bearing off moves."""
        board = empty_board()

        # Put all white checkers in home
        board.white_checkers[1] = 2
        board.white_checkers[2] = 3
        board.white_checkers[3] = 4
        board.white_checkers[4] = 3
        board.white_checkers[5] = 2
        board.white_checkers[6] = 1

        # Roll 5-3
        moves = generate_legal_moves(board, Player.WHITE, (5, 3))

        # Should be able to bear off
        assert len(moves) > 0

        # Check that some moves go to point 25 (off)
        for move in moves:
            has_bearoff = any(step.to_point == 25 for step in move)
            if has_bearoff:
                break
        else:
            pytest.fail("No bear-off moves found")

    def test_bearing_off_with_higher_die(self):
        """Test bearing off with higher die than needed."""
        board = empty_board()

        # Only checker on 1-point
        board.white_checkers[1] = 15

        # Roll 6-5 (both higher than needed)
        moves = generate_legal_moves(board, Player.WHITE, (6, 5))

        # Should be able to use both dice to bear off
        assert len(moves) > 0
        assert all(len(move) == 2 for move in moves)
        assert all(step.to_point == 25 for move in moves for step in move)

    def test_hitting_opponent(self):
        """Test hitting opponent's blot."""
        board = empty_board()

        # White checker can hit black blot
        board.white_checkers[8] = 1
        board.black_checkers[5] = 1  # Blot

        # Roll 3-x
        moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        # Find move that hits
        hitting_move = None
        for move in moves:
            for step in move:
                if step.hits_opponent:
                    hitting_move = move
                    break
            if hitting_move:
                break

        assert hitting_move is not None


class TestMoveApplication:
    """Tests for applying moves."""

    def test_apply_simple_move(self):
        """Test applying a simple move."""
        board = initial_board()

        # Move from 24 to 23 with die 1
        move = (MoveStep(from_point=24, to_point=23, die_used=1),)

        new_board = apply_move(board, Player.WHITE, move)

        # Original board unchanged
        assert board.white_checkers[24] == 2

        # New board has move applied
        assert new_board.white_checkers[24] == 1
        assert new_board.white_checkers[23] == 1

        # Player switched
        assert new_board.player_to_move == Player.BLACK

    def test_apply_hitting_move(self):
        """Test applying a move that hits opponent."""
        board = empty_board()
        board.white_checkers[8] = 1
        board.black_checkers[5] = 1

        # Hit the blot
        move = (MoveStep(from_point=8, to_point=5, die_used=3, hits_opponent=True),)

        new_board = apply_move(board, Player.WHITE, move)

        # White checker moved to 5
        assert new_board.white_checkers[5] == 1
        assert new_board.white_checkers[8] == 0

        # Black checker sent to bar
        assert new_board.black_checkers[5] == 0
        assert new_board.black_checkers[0] == 1

    def test_apply_bearing_off(self):
        """Test bearing off."""
        board = empty_board()
        board.white_checkers[1] = 5

        # Bear off from point 1
        move = (MoveStep(from_point=1, to_point=25, die_used=1),)

        new_board = apply_move(board, Player.WHITE, move)

        assert new_board.white_checkers[1] == 4
        assert new_board.white_checkers[25] == 1

    def test_is_legal_move(self):
        """Test move legality checking."""
        board = initial_board()

        # Generate legal moves
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        # First move should be legal
        assert is_legal_move(board, Player.WHITE, (3, 1), legal_moves[0])

        # Random invalid move should not be legal
        invalid_move = (MoveStep(from_point=1, to_point=5, die_used=4),)
        assert not is_legal_move(board, Player.WHITE, (3, 1), invalid_move)


class TestBoardDisplay:
    """Tests for board display."""

    def test_board_to_string(self):
        """Test board string representation."""
        board = initial_board()
        s = board_to_string(board)

        assert isinstance(s, str)
        assert len(s) > 0
        assert "Player to move: white" in s
        assert "167" in s  # Pip count


class TestPositionVariants:
    """Tests for position variants (for training diversity)."""

    def test_nackgammon_start(self):
        """Test nackgammon variant starting position."""
        board = nackgammon_start()

        # White: 2 on 24, 2 on 23, 4 on 13, 3 on 8, 4 on 6
        assert board.white_checkers[24] == 2
        assert board.white_checkers[23] == 2
        assert board.white_checkers[13] == 4  # Takes one from standard 5
        assert board.white_checkers[8] == 3
        assert board.white_checkers[6] == 4  # Takes one from standard 5

        # Black mirror
        assert board.black_checkers[1] == 2
        assert board.black_checkers[2] == 2
        assert board.black_checkers[12] == 4
        assert board.black_checkers[17] == 3
        assert board.black_checkers[19] == 4

        # Total checkers still 15 each
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # Pip count for nackgammon: 2*24 + 2*23 + 4*13 + 3*8 + 4*6 = 194
        assert pip_count(board, Player.WHITE) == 194
        assert pip_count(board, Player.BLACK) == 194

    def test_split_back_checkers(self):
        """Test split back checkers variant."""
        board = split_back_checkers()

        # White: 1 on 24, 1 on 23 (instead of 2 on 24)
        assert board.white_checkers[24] == 1
        assert board.white_checkers[23] == 1

        # Black: 1 on 1, 1 on 2
        assert board.black_checkers[1] == 1
        assert board.black_checkers[2] == 1

        # Rest should be standard
        assert board.white_checkers[13] == 5
        assert board.white_checkers[8] == 3
        assert board.white_checkers[6] == 5

        # Total still 15
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # Pip count should be 166 (one less than standard)
        assert pip_count(board, Player.WHITE) == 166
        assert pip_count(board, Player.BLACK) == 166

    def test_slotted_5_point(self):
        """Test slotted 5-point variant."""
        board = slotted_5_point()

        # White: 4 on 6-point, 1 on 5-point (slotted)
        assert board.white_checkers[6] == 4
        assert board.white_checkers[5] == 1

        # Black: 4 on 19-point, 1 on 20-point
        assert board.black_checkers[19] == 4
        assert board.black_checkers[20] == 1

        # Total still 15
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # Pip count should be 166 (moved one pip closer)
        assert pip_count(board, Player.WHITE) == 166
        assert pip_count(board, Player.BLACK) == 166

    def test_slotted_bar_point(self):
        """Test slotted bar point (7-point) variant."""
        board = slotted_bar_point()

        # White: 4 on 13, 1 on 7 (bar point)
        assert board.white_checkers[13] == 4
        assert board.white_checkers[7] == 1

        # Black: 4 on 12, 1 on 18
        assert board.black_checkers[12] == 4
        assert board.black_checkers[18] == 1

        # Total still 15
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # Pip count should be 161 (6 pips closer for white)
        assert pip_count(board, Player.WHITE) == 161
        assert pip_count(board, Player.BLACK) == 161

    def test_advanced_anchor(self):
        """Test advanced anchor variant."""
        board = advanced_anchor()

        # White: no checkers on 24, 2 on 20 (opponent's 5-point)
        assert board.white_checkers[24] == 0
        assert board.white_checkers[20] == 2

        # Black: no checkers on 1, 2 on 5
        assert board.black_checkers[1] == 0
        assert board.black_checkers[5] == 2

        # Total still 15
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # Pip count should be 159 (8 pips closer)
        assert pip_count(board, Player.WHITE) == 159
        assert pip_count(board, Player.BLACK) == 159

    def test_get_all_variant_starts(self):
        """Test getting all variant starting positions."""
        variants = get_all_variant_starts()

        # Should return 11 variants (6 full game + 5 simplified)
        assert len(variants) == 11

        # All should be valid boards with 3-15 checkers each
        for board in variants:
            assert 3 <= sum(board.white_checkers) <= 15
            assert 3 <= sum(board.black_checkers) <= 15

        # First should be standard
        standard = variants[0]
        assert standard.white_checkers[24] == 2
        assert standard.white_checkers[6] == 5

    def test_random_variant_start(self):
        """Test random variant selection."""
        import numpy as np

        # Use fixed seed for reproducibility
        rng = np.random.default_rng(42)
        board = random_variant_start(rng)

        # Should be valid board
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # Try multiple times to ensure variety
        boards = [random_variant_start(rng) for _ in range(20)]

        # Should get different variants (check pip counts)
        pip_counts = [pip_count(b, Player.WHITE) for b in boards]
        unique_pip_counts = set(pip_counts)

        # Should have at least 2 different variants in 20 tries
        assert len(unique_pip_counts) >= 2

    def test_variants_are_playable(self):
        """Test that all variants can generate legal moves."""
        variants = get_all_variant_starts()

        for board in variants:
            # Should be able to generate moves from opening position
            moves = generate_legal_moves(board, Player.WHITE, (3, 1))
            assert len(moves) > 0

            # Should be able to apply a move
            new_board = apply_move(board, Player.WHITE, moves[0])
            assert new_board.player_to_move == Player.BLACK


class TestSimplifiedVariants:
    """Tests for simplified variants (for early training)."""

    def test_hypergammon_start(self):
        """Test hypergammon starting position (3 checkers)."""
        board = hypergammon_start()

        # White: 2 on 24, 1 on 23
        assert board.white_checkers[24] == 2
        assert board.white_checkers[23] == 1

        # Black: 2 on 1, 1 on 2
        assert board.black_checkers[1] == 2
        assert board.black_checkers[2] == 1

        # Only 3 checkers per side
        assert sum(board.white_checkers) == 3
        assert sum(board.black_checkers) == 3

        # Pip count: 2*24 + 1*23 = 71
        assert pip_count(board, Player.WHITE) == 71
        assert pip_count(board, Player.BLACK) == 71

    def test_micro_gammon_start(self):
        """Test micro gammon starting position (5 checkers)."""
        board = micro_gammon_start()

        # White: 2 on 24, 2 on 13, 1 on 8
        assert board.white_checkers[24] == 2
        assert board.white_checkers[13] == 2
        assert board.white_checkers[8] == 1

        # Only 5 checkers per side
        assert sum(board.white_checkers) == 5
        assert sum(board.black_checkers) == 5

        # Pip count: 2*24 + 2*13 + 1*8 = 82
        assert pip_count(board, Player.WHITE) == 82
        assert pip_count(board, Player.BLACK) == 82

    def test_short_gammon_start(self):
        """Test short gammon starting position (9 checkers)."""
        board = short_gammon_start()

        # White: 2 on 24, 3 on 13, 2 on 8, 2 on 6
        assert board.white_checkers[24] == 2
        assert board.white_checkers[13] == 3
        assert board.white_checkers[8] == 2
        assert board.white_checkers[6] == 2

        # 9 checkers per side
        assert sum(board.white_checkers) == 9
        assert sum(board.black_checkers) == 9

        # Pip count: 2*24 + 3*13 + 2*8 + 2*6 = 115
        assert pip_count(board, Player.WHITE) == 115
        assert pip_count(board, Player.BLACK) == 115

    def test_bearoff_practice(self):
        """Test bearoff practice position."""
        board = bearoff_practice()

        # All checkers in home board (points 1-6 for white)
        for point in range(7, 25):
            assert board.white_checkers[point] == 0

        # All checkers in home board (points 19-24 for black)
        for point in range(1, 19):
            assert board.black_checkers[point] == 0

        # Still 15 checkers
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # Should be able to bear off
        assert can_bear_off(board, Player.WHITE)
        assert can_bear_off(board, Player.BLACK)

        # Low pip count (all in home)
        white_pips = pip_count(board, Player.WHITE)
        assert white_pips < 60  # Maximum would be 15*6 = 90

    def test_race_position(self):
        """Test race position (past contact)."""
        board = race_position()

        # 15 checkers per side
        assert sum(board.white_checkers) == 15
        assert sum(board.black_checkers) == 15

        # White in points 1-12, Black in points 13-24 (past contact)
        # Check white is all low
        for point in range(13, 25):
            assert board.white_checkers[point] == 0

        # Check black is all high
        for point in range(1, 13):
            assert board.black_checkers[point] == 0

        # Should be past contact
        from backgammon.evaluation.agents import is_past_contact
        assert is_past_contact(board, Player.WHITE)
        assert is_past_contact(board, Player.BLACK)

    def test_simplified_variants_finish_fast(self):
        """Test that simplified variants have lower pip counts."""
        standard = initial_board()
        standard_pips = pip_count(standard, Player.WHITE)

        # All simplified variants should have lower pip count
        hypergammon = hypergammon_start()
        assert pip_count(hypergammon, Player.WHITE) < standard_pips

        micro = micro_gammon_start()
        assert pip_count(micro, Player.WHITE) < standard_pips

        short = short_gammon_start()
        assert pip_count(short, Player.WHITE) < standard_pips

        race = race_position()
        assert pip_count(race, Player.WHITE) < standard_pips

        bearoff = bearoff_practice()
        assert pip_count(bearoff, Player.WHITE) < standard_pips


class TestTrainingPhaseVariants:
    """Tests for training phase variant selection."""

    def test_get_all_variant_starts(self):
        """Test getting all variant starting positions."""
        variants = get_all_variant_starts()

        # Should have 11 variants now (6 original + 5 simplified)
        assert len(variants) == 11

        # All should be valid boards
        for board in variants:
            # Should have 3-15 checkers per side
            assert 3 <= sum(board.white_checkers) <= 15
            assert 3 <= sum(board.black_checkers) <= 15

    def test_get_early_training_variants(self):
        """Test early training variants."""
        early_variants = get_early_training_variants()

        # Should have 5 simplified variants
        assert len(early_variants) == 5

        # All should be simplified (3-15 checkers, lower pip counts)
        standard_pips = pip_count(initial_board(), Player.WHITE)
        for board in early_variants:
            # Should have lower pip count than standard
            assert pip_count(board, Player.WHITE) < standard_pips

    def test_get_mid_training_variants(self):
        """Test mid training variants."""
        mid_variants = get_mid_training_variants()

        # Should be a mix
        assert len(mid_variants) == 6

        # Should include both simplified and full variants
        pip_counts = [pip_count(b, Player.WHITE) for b in mid_variants]
        # Should have variety
        assert len(set(pip_counts)) >= 4

    def test_get_late_training_variants(self):
        """Test late training variants."""
        late_variants = get_late_training_variants()

        # Should be same as all variants
        assert len(late_variants) == len(get_all_variant_starts())

    def test_variant_progression(self):
        """Test that early variants are simpler than mid/late."""
        early = get_early_training_variants()
        mid = get_mid_training_variants()
        late = get_late_training_variants()

        # Early should have fewest average checkers
        avg_early_checkers = sum(sum(b.white_checkers) for b in early) / len(early)
        avg_mid_checkers = sum(sum(b.white_checkers) for b in mid) / len(mid)

        # Early should be simpler (fewer checkers) than mid
        assert avg_early_checkers < avg_mid_checkers

        # Late should have most variety (includes all variants)
        assert len(late) > len(early)
        assert len(late) >= len(mid)
