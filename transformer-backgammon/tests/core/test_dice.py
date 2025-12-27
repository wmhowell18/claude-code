"""Tests for dice utilities."""

import pytest
import numpy as np
from backgammon.core.dice import (
    all_dice_rolls,
    is_doubles,
    dice_values,
    roll_dice,
    canonicalize_dice,
    dice_to_string,
    ALL_DICE_ROLLS,
    DICE_PROBABILITIES,
)


class TestDiceUtilities:
    """Tests for dice utility functions."""

    def test_all_dice_rolls(self):
        """Test that we get all 21 unique dice rolls."""
        rolls = all_dice_rolls()
        assert len(rolls) == 21

        # Check all doubles are present
        for i in range(1, 7):
            assert (i, i) in rolls

        # Check no duplicates (e.g., both (2,3) and (3,2))
        seen = set()
        for roll in rolls:
            canonical = tuple(sorted(roll))
            assert canonical not in seen
            seen.add(canonical)

    def test_is_doubles(self):
        """Test doubles detection."""
        assert is_doubles((1, 1))
        assert is_doubles((6, 6))
        assert not is_doubles((1, 2))
        assert not is_doubles((5, 6))

    def test_dice_values(self):
        """Test getting dice values for moves."""
        # Non-doubles give 2 values
        assert dice_values((3, 5)) == [3, 5]
        assert dice_values((1, 6)) == [1, 6]

        # Doubles give 4 values
        assert dice_values((4, 4)) == [4, 4, 4, 4]
        assert dice_values((2, 2)) == [2, 2, 2, 2]

    def test_roll_dice(self):
        """Test dice rolling."""
        rng = np.random.default_rng(42)

        # Roll multiple times
        for _ in range(100):
            dice = roll_dice(rng)
            assert len(dice) == 2
            assert 1 <= dice[0] <= 6
            assert 1 <= dice[1] <= 6

    def test_canonicalize_dice(self):
        """Test dice canonicalization."""
        assert canonicalize_dice((5, 3)) == (3, 5)
        assert canonicalize_dice((1, 6)) == (1, 6)
        assert canonicalize_dice((4, 4)) == (4, 4)
        assert canonicalize_dice((2, 1)) == (1, 2)

    def test_dice_to_string(self):
        """Test dice string conversion."""
        assert dice_to_string((3, 5)) == "3-5"
        assert dice_to_string((4, 4)) == "Double 4s"
        assert dice_to_string((1, 1)) == "Double 1s"

    def test_all_dice_rolls_constant(self):
        """Test that ALL_DICE_ROLLS constant is correct."""
        assert len(ALL_DICE_ROLLS) == 21
        assert ALL_DICE_ROLLS == all_dice_rolls()

    def test_dice_probabilities(self):
        """Test that dice probabilities sum to 1."""
        total_prob = sum(DICE_PROBABILITIES.values())
        assert abs(total_prob - 1.0) < 1e-6

        # Check individual probabilities
        assert abs(DICE_PROBABILITIES[(1, 1)] - 1/36) < 1e-6  # Doubles
        assert abs(DICE_PROBABILITIES[(1, 2)] - 1/18) < 1e-6  # Non-doubles
