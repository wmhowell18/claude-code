"""Tests for agent module."""

import pytest
import numpy as np

from backgammon.core.types import Player
from backgammon.core.board import (
    initial_board,
    empty_board,
    generate_legal_moves,
    apply_move,
    pip_count,
    nackgammon_start,
)
from backgammon.evaluation.agents import (
    Agent,
    PipCountConfig,
    random_agent,
    pip_count_agent,
    greedy_pip_count_agent,
    count_blots,
    has_anchor,
    is_past_contact,
)


class TestAgentInterface:
    """Tests for agent interface."""

    def test_agent_creation(self):
        """Test creating a basic agent."""
        def dummy_select(board, player, dice, legal_moves):
            return legal_moves[0] if legal_moves else ()

        agent = Agent(name="DummyAgent", select_move_fn=dummy_select)
        assert agent.name == "DummyAgent"
        assert agent.select_move_fn is dummy_select

    def test_agent_select_move(self):
        """Test agent select_move method."""
        def dummy_select(board, player, dice, legal_moves):
            return legal_moves[0] if legal_moves else ()

        agent = Agent(name="DummyAgent", select_move_fn=dummy_select)
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        move = agent.select_move(board, Player.WHITE, (3, 1), legal_moves)
        assert move == legal_moves[0]


class TestRandomAgent:
    """Tests for random agent."""

    def test_random_agent_creation(self):
        """Test creating random agent."""
        agent = random_agent(seed=42)
        assert agent.name == "Random"

    def test_random_agent_selects_legal_move(self):
        """Test random agent selects a legal move."""
        agent = random_agent(seed=42)
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        move = agent.select_move(board, Player.WHITE, (3, 1), legal_moves)
        assert move in legal_moves

    def test_random_agent_empty_moves(self):
        """Test random agent with no legal moves."""
        agent = random_agent(seed=42)
        board = initial_board()

        move = agent.select_move(board, Player.WHITE, (3, 1), [])
        assert move == ()

    def test_random_agent_variety(self):
        """Test random agent produces variety."""
        agent = random_agent(seed=42)
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (6, 5))

        # Generate multiple moves
        moves_selected = []
        for _ in range(20):
            move = agent.select_move(board, Player.WHITE, (6, 5), legal_moves)
            moves_selected.append(move)

        # Should have some variety (not all the same)
        unique_moves = set(moves_selected)
        if len(legal_moves) > 1:
            assert len(unique_moves) > 1


class TestPipCountAgent:
    """Tests for pip count agent."""

    def test_pip_count_agent_creation(self):
        """Test creating pip count agent."""
        agent = pip_count_agent()
        assert agent.name == "PipCount"

    def test_pip_count_agent_with_config(self):
        """Test creating pip count agent with custom config."""
        config = PipCountConfig(
            blot_penalty=20.0,
            hit_bonus=30.0,
        )
        agent = pip_count_agent(config)
        assert agent.name == "PipCount"

    def test_pip_count_agent_selects_legal_move(self):
        """Test pip count agent selects a legal move."""
        agent = pip_count_agent()
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        move = agent.select_move(board, Player.WHITE, (3, 1), legal_moves)
        assert move in legal_moves

    def test_pip_count_agent_minimizes_pips(self):
        """Test pip count agent prefers moves that minimize pip count."""
        agent = pip_count_agent()
        board = initial_board()

        # Roll 6-5 (should prefer moving from back)
        legal_moves = generate_legal_moves(board, Player.WHITE, (6, 5))
        move = agent.select_move(board, Player.WHITE, (6, 5), legal_moves)

        # Apply move and check pip count decreased
        new_board = apply_move(board, Player.WHITE, move)
        old_pips = pip_count(board, Player.WHITE)
        new_pips = pip_count(new_board, Player.WHITE)

        assert new_pips < old_pips

    def test_pip_count_agent_empty_moves(self):
        """Test pip count agent with no legal moves."""
        agent = pip_count_agent()
        board = initial_board()

        move = agent.select_move(board, Player.WHITE, (3, 1), [])
        assert move == ()

    def test_pip_count_agent_deterministic(self):
        """Test pip count agent is deterministic."""
        agent = pip_count_agent()
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        move1 = agent.select_move(board, Player.WHITE, (3, 1), legal_moves)
        move2 = agent.select_move(board, Player.WHITE, (3, 1), legal_moves)

        # Same board, same moves, should select same move
        assert move1 == move2


class TestGreedyPipCountAgent:
    """Tests for greedy pip count agent."""

    def test_greedy_agent_creation(self):
        """Test creating greedy agent."""
        agent = greedy_pip_count_agent()
        assert agent.name == "GreedyPipCount"

    def test_greedy_agent_selects_legal_move(self):
        """Test greedy agent selects a legal move."""
        agent = greedy_pip_count_agent()
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        move = agent.select_move(board, Player.WHITE, (3, 1), legal_moves)
        assert move in legal_moves

    def test_greedy_agent_minimizes_pips(self):
        """Test greedy agent minimizes pip count."""
        agent = greedy_pip_count_agent()
        board = initial_board()

        # Roll 6-5
        legal_moves = generate_legal_moves(board, Player.WHITE, (6, 5))
        move = agent.select_move(board, Player.WHITE, (6, 5), legal_moves)

        # Check pip count decreased
        new_board = apply_move(board, Player.WHITE, move)
        assert pip_count(new_board, Player.WHITE) < pip_count(board, Player.WHITE)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_count_blots_initial(self):
        """Test counting blots at initial position."""
        board = initial_board()

        # Standard position has no blots
        assert count_blots(board, Player.WHITE) == 0
        assert count_blots(board, Player.BLACK) == 0

    def test_count_blots_with_blots(self):
        """Test counting blots with exposed checkers."""
        board = empty_board()

        # Add some blots
        board.white_checkers[8] = 1
        board.white_checkers[13] = 1
        board.white_checkers[20] = 1

        assert count_blots(board, Player.WHITE) == 3

        # Add made points
        board.white_checkers[6] = 2
        board.white_checkers[5] = 3

        # Still 3 blots
        assert count_blots(board, Player.WHITE) == 3

    def test_has_anchor_initial(self):
        """Test anchor detection at initial position."""
        board = initial_board()

        # At initial position, back checkers (2 on 24 for White, 2 on 1 for Black)
        # ARE anchors in opponent's home board
        assert has_anchor(board, Player.WHITE)  # 2 on 24 (in Black's home 19-24)
        assert has_anchor(board, Player.BLACK)  # 2 on 1 (in White's home 1-6)

    def test_has_anchor_with_anchor(self):
        """Test anchor detection with anchor."""
        board = empty_board()

        # White has anchor in Black's home (points 19-24)
        board.white_checkers[20] = 2  # Black's 5-point from White's perspective
        assert has_anchor(board, Player.WHITE)

        # Black has anchor in White's home (points 1-6)
        board.black_checkers[5] = 3  # White's 5-point from Black's perspective
        assert has_anchor(board, Player.BLACK)

    def test_has_anchor_blot_not_anchor(self):
        """Test that single checker is not an anchor."""
        board = empty_board()

        # Single checker is not an anchor
        board.white_checkers[5] = 1
        assert not has_anchor(board, Player.WHITE)

    def test_is_past_contact_initial(self):
        """Test past contact detection at initial position."""
        board = initial_board()

        # At start, not past contact
        assert not is_past_contact(board, Player.WHITE)
        assert not is_past_contact(board, Player.BLACK)

    def test_is_past_contact_race(self):
        """Test past contact detection in race."""
        board = empty_board()

        # White all ahead (points 1-6)
        board.white_checkers[1] = 3
        board.white_checkers[2] = 3
        board.white_checkers[3] = 3
        board.white_checkers[4] = 3
        board.white_checkers[5] = 3

        # Black all behind (points 19-24)
        board.black_checkers[19] = 3
        board.black_checkers[20] = 3
        board.black_checkers[21] = 3
        board.black_checkers[22] = 3
        board.black_checkers[23] = 3

        # Both past contact
        assert is_past_contact(board, Player.WHITE)
        assert is_past_contact(board, Player.BLACK)

    def test_is_past_contact_still_in_contact(self):
        """Test past contact when still in contact."""
        board = empty_board()

        # White at points 1-10
        board.white_checkers[1] = 5
        board.white_checkers[10] = 5

        # Black at points 8-15 (overlapping)
        board.black_checkers[8] = 5
        board.black_checkers[15] = 5

        # Still in contact
        assert not is_past_contact(board, Player.WHITE)
        assert not is_past_contact(board, Player.BLACK)


class TestAgentComparison:
    """Tests comparing different agents."""

    def test_pip_count_beats_random(self):
        """Test that pip count agent makes better moves than random."""
        pip_agent = pip_count_agent()
        random_agent_inst = random_agent(seed=42)

        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (6, 5))

        # Pip count agent should consistently pick good moves
        pip_move = pip_agent.select_move(board, Player.WHITE, (6, 5), legal_moves)
        pip_board = apply_move(board, Player.WHITE, pip_move)

        # Random might pick any move
        random_move = random_agent_inst.select_move(board, Player.WHITE, (6, 5), legal_moves)
        random_board = apply_move(board, Player.WHITE, random_move)

        # Pip count agent's move should be at least as good as random
        # (in terms of pip count)
        pip_pips = pip_count(pip_board, Player.WHITE)
        random_pips = pip_count(random_board, Player.WHITE)

        # Pip agent should never be worse
        assert pip_pips <= random_pips + 1  # Allow small margin

    def test_agents_work_with_variants(self):
        """Test that agents work with position variants."""
        pip_agent = pip_count_agent()
        greedy_agent = greedy_pip_count_agent()

        # Try with nackgammon
        board = nackgammon_start()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 1))

        # Both should select legal moves
        pip_move = pip_agent.select_move(board, Player.WHITE, (3, 1), legal_moves)
        greedy_move = greedy_agent.select_move(board, Player.WHITE, (3, 1), legal_moves)

        assert pip_move in legal_moves
        assert greedy_move in legal_moves
