"""Pytest configuration and shared fixtures."""

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture(scope="session")
def rng_key():
    """Create a JAX random key for testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_board():
    """Create a sample board state for testing."""
    from backgammon.core.board import initial_board
    return initial_board()
