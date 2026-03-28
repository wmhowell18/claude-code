"""Tests for dice utilities."""

import pytest
import numpy as np
from collections import Counter
from backgammon.core.dice import (
    all_dice_rolls,
    is_doubles,
    dice_values,
    roll_dice,
    canonicalize_dice,
    dice_to_string,
    ALL_DICE_ROLLS,
    DICE_PROBABILITIES,
    StratifiedDiceSampler,
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


class TestStratifiedDiceSampler:
    """Tests for stratified dice sampling."""

    def test_deck_has_36_outcomes(self):
        """One full deck should yield exactly 36 rolls."""
        rng = np.random.default_rng(42)
        sampler = StratifiedDiceSampler(rng)
        rolls = [sampler.roll() for _ in range(36)]
        assert len(rolls) == 36

    def test_exact_frequencies_per_epoch(self):
        """Each epoch should have correct frequencies: doubles 1x, non-doubles 2x."""
        rng = np.random.default_rng(42)
        sampler = StratifiedDiceSampler(rng)
        rolls = [sampler.roll() for _ in range(36)]
        counts = Counter(rolls)

        for dice_roll in ALL_DICE_ROLLS:
            if is_doubles(dice_roll):
                assert counts[dice_roll] == 1, f"{dice_roll} should appear 1x, got {counts[dice_roll]}"
            else:
                assert counts[dice_roll] == 2, f"{dice_roll} should appear 2x, got {counts[dice_roll]}"

    def test_auto_refill(self):
        """Sampler should auto-refill after 36 rolls."""
        rng = np.random.default_rng(42)
        sampler = StratifiedDiceSampler(rng)
        # Draw 72 rolls (2 full decks)
        rolls = [sampler.roll() for _ in range(72)]
        assert len(rolls) == 72

        # Each half should independently have correct frequencies
        for epoch_rolls in [rolls[:36], rolls[36:]]:
            counts = Counter(epoch_rolls)
            for dice_roll in ALL_DICE_ROLLS:
                expected = 1 if is_doubles(dice_roll) else 2
                assert counts[dice_roll] == expected

    def test_rolls_are_canonical(self):
        """All rolls should be in canonical form (smaller die first)."""
        rng = np.random.default_rng(42)
        sampler = StratifiedDiceSampler(rng)
        for _ in range(72):
            roll = sampler.roll()
            assert roll[0] <= roll[1], f"Roll {roll} not canonical"

    def test_shuffled_order(self):
        """Different seeds should produce different orderings."""
        sampler1 = StratifiedDiceSampler(np.random.default_rng(1))
        sampler2 = StratifiedDiceSampler(np.random.default_rng(2))
        rolls1 = [sampler1.roll() for _ in range(36)]
        rolls2 = [sampler2.roll() for _ in range(36)]
        # Same set, different order (extremely unlikely to be identical)
        assert Counter(rolls1) == Counter(rolls2)
        assert rolls1 != rolls2
