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
    get_all_variant_starts,
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

        # Should return 6 variants (standard + 5 variants)
        assert len(variants) == 6

        # All should be valid boards with 15 checkers each
        for board in variants:
            assert sum(board.white_checkers) == 15
            assert sum(board.black_checkers) == 15

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
