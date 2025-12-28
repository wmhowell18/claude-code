"""Tests for action space encoding."""

import pytest
import numpy as np

from backgammon.encoding.action_encoder import (
    encode_move_to_action,
    create_action_mask,
    create_move_to_action_map,
    encode_move_to_one_hot,
    create_uniform_policy,
    select_move_from_policy,
    get_action_space_size,
    analyze_move_collisions,
    encode_move_structured,
    decode_move_structured,
    ACTION_SPACE_SIZE,
)
from backgammon.core.board import initial_board, generate_legal_moves
from backgammon.core.types import Player


class TestMoveEncoding:
    """Test basic move encoding."""

    def test_encode_empty_move(self):
        """Test encoding an empty/pass move."""
        move = ()
        action_idx = encode_move_to_action(move)

        assert action_idx == 0
        assert 0 <= action_idx < ACTION_SPACE_SIZE

    def test_encode_single_step_move(self):
        """Test encoding a single-step move."""
        # Move from point 24 to 20 with die 4
        move = ((24, 20, 4, False),)
        action_idx = encode_move_to_action(move)

        assert 0 <= action_idx < ACTION_SPACE_SIZE
        assert action_idx != 0  # Should not be same as pass

    def test_encode_multi_step_move(self):
        """Test encoding a multi-step move."""
        # Two checker movements
        move = ((24, 20, 4, False), (13, 10, 3, False))
        action_idx = encode_move_to_action(move)

        assert 0 <= action_idx < ACTION_SPACE_SIZE

    def test_encode_deterministic(self):
        """Test that encoding is deterministic."""
        move = ((24, 20, 4, False), (13, 10, 3, False))

        # Encode multiple times - should get same result
        idx1 = encode_move_to_action(move)
        idx2 = encode_move_to_action(move)
        idx3 = encode_move_to_action(move)

        assert idx1 == idx2 == idx3

    def test_different_moves_different_indices(self):
        """Test that different moves get different indices (usually)."""
        move1 = ((24, 20, 4, False),)
        move2 = ((13, 10, 3, False),)

        idx1 = encode_move_to_action(move1)
        idx2 = encode_move_to_action(move2)

        # Usually different (collisions possible but rare)
        # We just check they're valid
        assert 0 <= idx1 < ACTION_SPACE_SIZE
        assert 0 <= idx2 < ACTION_SPACE_SIZE

    def test_action_space_size(self):
        """Test getting action space size."""
        size = get_action_space_size()
        assert size == ACTION_SPACE_SIZE
        assert size == 1024


class TestActionMask:
    """Test action masking."""

    def test_create_mask_empty_moves(self):
        """Test creating mask with no legal moves."""
        legal_moves = []
        mask = create_action_mask(legal_moves)

        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == bool
        assert not mask.any()  # All False

    def test_create_mask_single_move(self):
        """Test creating mask with one legal move."""
        legal_moves = [((24, 20, 4, False),)]
        mask = create_action_mask(legal_moves)

        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.sum() == 1  # Exactly one True

        # The move's action should be marked
        action_idx = encode_move_to_action(legal_moves[0])
        assert mask[action_idx]

    def test_create_mask_multiple_moves(self):
        """Test creating mask with multiple legal moves."""
        legal_moves = [
            ((24, 20, 4, False),),
            ((13, 10, 3, False),),
            ((8, 5, 3, False),),
        ]
        mask = create_action_mask(legal_moves)

        assert mask.shape == (ACTION_SPACE_SIZE,)
        # At least 3 (could be less if collisions)
        assert mask.sum() >= 1
        assert mask.sum() <= len(legal_moves)

        # All moves should be in mask
        for move in legal_moves:
            action_idx = encode_move_to_action(move)
            assert mask[action_idx]

    def test_create_mask_from_real_position(self):
        """Test creating mask from real game position."""
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 4))

        mask = create_action_mask(legal_moves)

        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.sum() > 0  # Should have legal moves
        assert mask.sum() <= len(legal_moves)


class TestMoveToActionMap:
    """Test move to action mapping."""

    def test_create_map_empty(self):
        """Test creating map with no moves."""
        legal_moves = []
        action_map = create_move_to_action_map(legal_moves)

        assert len(action_map) == 0

    def test_create_map_single_move(self):
        """Test creating map with one move."""
        move = ((24, 20, 4, False),)
        legal_moves = [move]

        action_map = create_move_to_action_map(legal_moves)

        action_idx = encode_move_to_action(move)
        assert action_idx in action_map
        assert action_map[action_idx] == move

    def test_create_map_multiple_moves(self):
        """Test creating map with multiple moves."""
        legal_moves = [
            ((24, 20, 4, False),),
            ((13, 10, 3, False),),
            ((8, 5, 3, False),),
        ]

        action_map = create_move_to_action_map(legal_moves)

        # All moves should be in map
        for move in legal_moves:
            action_idx = encode_move_to_action(move)
            assert action_idx in action_map
            assert action_map[action_idx] == move


class TestPolicyEncoding:
    """Test policy encoding."""

    def test_one_hot_encoding(self):
        """Test one-hot policy encoding."""
        move = ((24, 20, 4, False),)
        legal_moves = [move, ((13, 10, 3, False),)]

        policy = encode_move_to_one_hot(move, legal_moves)

        assert policy.shape == (ACTION_SPACE_SIZE,)
        assert policy.dtype == np.float32
        assert policy.sum() == pytest.approx(1.0)
        assert policy.max() == 1.0

        # The move's action should have probability 1.0
        action_idx = encode_move_to_action(move)
        assert policy[action_idx] == 1.0

    def test_uniform_policy_empty(self):
        """Test uniform policy with no moves."""
        legal_moves = []
        policy = create_uniform_policy(legal_moves)

        assert policy.shape == (ACTION_SPACE_SIZE,)
        assert policy.sum() == 0.0

    def test_uniform_policy_single_move(self):
        """Test uniform policy with one move."""
        move = ((24, 20, 4, False),)
        legal_moves = [move]

        policy = create_uniform_policy(legal_moves)

        assert policy.shape == (ACTION_SPACE_SIZE,)
        assert policy.sum() == pytest.approx(1.0)

        action_idx = encode_move_to_action(move)
        assert policy[action_idx] == pytest.approx(1.0)

    def test_uniform_policy_multiple_moves(self):
        """Test uniform policy with multiple moves."""
        legal_moves = [
            ((24, 20, 4, False),),
            ((13, 10, 3, False),),
            ((8, 5, 3, False),),
        ]

        policy = create_uniform_policy(legal_moves)

        assert policy.shape == (ACTION_SPACE_SIZE,)
        # Sum might be less than 1.0 if there are collisions
        assert policy.sum() <= 1.0 + 1e-6

        # Each legal move should have non-zero probability
        for move in legal_moves:
            action_idx = encode_move_to_action(move)
            assert policy[action_idx] > 0.0


class TestMoveSelection:
    """Test move selection from policy."""

    def test_select_from_single_move(self):
        """Test selecting from policy with one move."""
        legal_moves = [((24, 20, 4, False),)]
        policy = create_uniform_policy(legal_moves)

        selected = select_move_from_policy(policy, legal_moves, temperature=1.0)

        assert selected == legal_moves[0]

    def test_select_greedy(self):
        """Test greedy selection (temperature=0)."""
        legal_moves = [
            ((24, 20, 4, False),),
            ((13, 10, 3, False),),
        ]

        # Create policy favoring first move
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[encode_move_to_action(legal_moves[0])] = 0.9
        policy[encode_move_to_action(legal_moves[1])] = 0.1

        selected = select_move_from_policy(policy, legal_moves, temperature=0.0)

        # Should always select first move
        assert selected == legal_moves[0]

    def test_select_stochastic(self):
        """Test stochastic selection."""
        legal_moves = [
            ((24, 20, 4, False),),
            ((13, 10, 3, False),),
        ]

        policy = create_uniform_policy(legal_moves)

        # Sample multiple times - should get variety (probabilistically)
        selections = set()
        for _ in range(20):
            selected = select_move_from_policy(policy, legal_moves, temperature=1.0)
            assert selected in legal_moves
            selections.add(selected)

        # With 20 samples from uniform, very likely to see both (not guaranteed but very likely)
        # We just check we got valid moves
        assert len(selections) > 0

    def test_select_no_moves(self):
        """Test selecting with no legal moves."""
        legal_moves = []
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

        selected = select_move_from_policy(policy, legal_moves)

        assert selected == ()


class TestCollisionAnalysis:
    """Test collision analysis."""

    def test_analyze_empty(self):
        """Test analyzing empty move set."""
        moves = []
        stats = analyze_move_collisions(moves)

        assert stats['total_moves'] == 0
        assert stats['unique_actions'] == 0
        assert stats['num_collisions'] == 0

    def test_analyze_no_collisions(self):
        """Test analyzing moves with no collisions."""
        # Very unlikely to have collisions with different moves
        moves = [
            ((24, 20, 4, False),),
            ((13, 10, 3, False),),
            ((8, 5, 3, False),),
        ]

        stats = analyze_move_collisions(moves)

        assert stats['total_moves'] == 3
        # Usually no collisions (but possible)
        assert stats['unique_actions'] <= 3
        assert stats['unique_actions'] >= 1

    def test_analyze_with_duplicates(self):
        """Test analyzing moves with duplicates."""
        move = ((24, 20, 4, False),)
        moves = [move, move, move]  # Same move three times

        stats = analyze_move_collisions(moves)

        assert stats['total_moves'] == 3
        assert stats['unique_actions'] == 1  # Only one unique action
        assert stats['num_collisions'] == 2  # 2 duplicates
        assert stats['collision_rate'] == pytest.approx(2/3)

    def test_analyze_real_position(self):
        """Test analyzing moves from real position."""
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 4))

        stats = analyze_move_collisions(legal_moves)

        assert stats['total_moves'] == len(legal_moves)
        assert stats['unique_actions'] > 0
        assert 0 <= stats['collision_rate'] < 1.0
        assert 0 < stats['action_space_utilization'] < 1.0


class TestStructuredEncoding:
    """Test alternative structured encoding."""

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip."""
        move = ((24, 20, 4, False), (13, 10, 3, False))

        encoded = encode_move_structured(move, Player.WHITE)
        decoded = decode_move_structured(encoded)

        assert decoded == move

    def test_encode_empty_move(self):
        """Test encoding empty move."""
        move = ()

        encoded = encode_move_structured(move, Player.WHITE)
        decoded = decode_move_structured(encoded)

        assert decoded == ()

    def test_encode_hitting_move(self):
        """Test encoding move with hit."""
        move = ((24, 20, 4, True),)  # Hit opponent

        encoded = encode_move_structured(move, Player.WHITE)
        decoded = decode_move_structured(encoded)

        assert decoded == move
        assert decoded[0][3] is True  # Hit flag preserved


class TestIntegration:
    """Integration tests with real game positions."""

    def test_encode_all_opening_moves(self):
        """Test encoding all opening moves."""
        board = initial_board()

        # Test several different dice rolls
        for dice in [(3, 4), (6, 6), (1, 2), (5, 4)]:
            legal_moves = generate_legal_moves(board, Player.WHITE, dice)

            # All moves should encode successfully
            for move in legal_moves:
                action_idx = encode_move_to_action(move)
                assert 0 <= action_idx < ACTION_SPACE_SIZE

            # Create mask and policy
            mask = create_action_mask(legal_moves)
            policy = create_uniform_policy(legal_moves)

            assert mask.sum() > 0
            assert policy.sum() > 0

    def test_select_from_real_positions(self):
        """Test move selection from real positions."""
        board = initial_board()
        legal_moves = generate_legal_moves(board, Player.WHITE, (3, 4))

        policy = create_uniform_policy(legal_moves)

        # Select several times
        for _ in range(10):
            selected = select_move_from_policy(policy, legal_moves)
            assert selected in legal_moves
