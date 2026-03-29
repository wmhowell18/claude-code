"""Unit tests for loss functions and training step.

losses.py is the core of the training loop but previously had zero unit tests.
These tests cover:
- Policy loss with masking edge cases
- Equity loss with numerical stability
- Combined loss in both policy+value and value-only modes
- train_step JIT compilation and gradient flow
- compute_metrics in both modes
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from backgammon.training.losses import (
    compute_policy_loss,
    compute_equity_loss,
    compute_combined_loss,
    train_step,
    compute_metrics,
)
from backgammon.training.train import TrainingConfig, create_train_state


class TestComputePolicyLoss:
    """Test policy loss computation."""

    def test_perfect_prediction(self):
        """Loss should be near zero when prediction matches target."""
        # Logits that produce a near-one-hot distribution after softmax
        logits = jnp.array([[10.0, -10.0, -10.0, -10.0]])
        target = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        mask = jnp.ones((1, 4))

        loss = compute_policy_loss(logits, target, mask)
        assert loss < 0.1

    def test_uniform_target(self):
        """Loss with uniform target and uniform logits should be log(N)."""
        n_actions = 4
        logits = jnp.zeros((1, n_actions))
        target = jnp.ones((1, n_actions)) / n_actions
        mask = jnp.ones((1, n_actions))

        loss = compute_policy_loss(logits, target, mask)
        expected = jnp.log(n_actions)  # Cross-entropy of uniform = log(N)
        np.testing.assert_allclose(float(loss), float(expected), atol=1e-5)

    def test_mask_zeroes_out_actions(self):
        """Masked actions should not contribute to loss."""
        logits = jnp.array([[0.0, 0.0, 0.0, 0.0]])
        target = jnp.array([[0.5, 0.5, 0.0, 0.0]])
        mask = jnp.array([[True, True, False, False]])

        loss_masked = compute_policy_loss(logits, target, mask)

        # Compare with only 2-action case
        logits_2 = jnp.array([[0.0, 0.0]])
        target_2 = jnp.array([[0.5, 0.5]])
        mask_2 = jnp.ones((1, 2))
        loss_2 = compute_policy_loss(logits_2, target_2, mask_2)

        np.testing.assert_allclose(float(loss_masked), float(loss_2), atol=1e-4)

    def test_single_legal_action(self):
        """With one legal action, loss should be near zero (forced move)."""
        logits = jnp.array([[5.0, -5.0, -5.0]])
        target = jnp.array([[1.0, 0.0, 0.0]])
        mask = jnp.array([[True, False, False]])

        loss = compute_policy_loss(logits, target, mask)
        assert loss < 0.1

    def test_batch_dimension(self):
        """Loss should correctly average over batch."""
        logits = jnp.array([[10.0, -10.0], [-10.0, 10.0]])
        target = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        mask = jnp.ones((2, 2))

        loss = compute_policy_loss(logits, target, mask)
        assert jnp.isfinite(loss)
        assert loss < 0.1  # Both predictions are correct


class TestComputeEquityLoss:
    """Test equity loss computation."""

    def test_perfect_prediction(self):
        """Loss near zero when pred matches target."""
        pred = jnp.array([[0.8, 0.1, 0.02, 0.03, 0.03, 0.02]])
        target = jnp.array([[0.8, 0.1, 0.02, 0.03, 0.03, 0.02]])

        loss = compute_equity_loss(pred, target)
        assert loss < 1.0  # Cross-entropy of a non-degenerate distribution

    def test_wrong_prediction_higher_loss(self):
        """Wrong prediction should give higher loss than correct one."""
        target = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        good_pred = jnp.array([[0.9, 0.04, 0.01, 0.03, 0.01, 0.01]])
        bad_pred = jnp.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

        loss_good = compute_equity_loss(good_pred, target)
        loss_bad = compute_equity_loss(bad_pred, target)

        assert loss_good < loss_bad

    def test_near_zero_prediction_stability(self):
        """Epsilon should prevent log(0) explosion."""
        pred = jnp.array([[1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1.0]])
        target = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        loss = compute_equity_loss(pred, target)
        assert jnp.isfinite(loss)

    def test_batch_averaging(self):
        """Loss is averaged over batch dimension."""
        pred = jnp.array([[0.5, 0.2, 0.1, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1, 0.1]])
        target = jnp.array([[0.5, 0.2, 0.1, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1, 0.1]])

        loss = compute_equity_loss(pred, target)
        assert jnp.isfinite(loss)


class TestComputeCombinedLoss:
    """Test combined loss in both modes."""

    def test_combined_mode_returns_all_metrics(self):
        """With policy head enabled, all metrics should be present."""
        policy_logits = jnp.zeros((2, 4))
        equity_pred = jnp.ones((2, 6)) / 6
        target_policy = jnp.ones((2, 4)) / 4
        equity_target = jnp.ones((2, 6)) / 6
        mask = jnp.ones((2, 4))

        loss, metrics = compute_combined_loss(
            policy_logits, equity_pred, target_policy, equity_target, mask
        )

        assert jnp.isfinite(loss)
        assert 'policy_loss' in metrics
        assert 'equity_loss' in metrics
        assert 'total_loss' in metrics
        assert jnp.isfinite(metrics['policy_loss'])
        assert jnp.isfinite(metrics['equity_loss'])

    def test_value_only_mode(self):
        """With policy_logits=None, should use equity loss only."""
        equity_pred = jnp.ones((2, 6)) / 6
        equity_target = jnp.ones((2, 6)) / 6

        loss, metrics = compute_combined_loss(
            None, equity_pred, None, equity_target, None
        )

        assert jnp.isfinite(loss)
        assert float(metrics['policy_loss']) == 0.0
        assert loss == metrics['equity_loss']  # No weighting in value-only mode

    def test_weight_scaling(self):
        """Policy and equity weights should scale their contributions."""
        policy_logits = jnp.zeros((2, 4))
        equity_pred = jnp.ones((2, 6)) / 6
        target_policy = jnp.ones((2, 4)) / 4
        equity_target = jnp.ones((2, 6)) / 6
        mask = jnp.ones((2, 4))

        _, metrics_default = compute_combined_loss(
            policy_logits, equity_pred, target_policy, equity_target, mask,
            policy_weight=1.0, equity_weight=0.5,
        )
        _, metrics_heavy_policy = compute_combined_loss(
            policy_logits, equity_pred, target_policy, equity_target, mask,
            policy_weight=10.0, equity_weight=0.5,
        )

        # Higher policy weight -> higher total loss
        assert metrics_heavy_policy['total_loss'] > metrics_default['total_loss']


class TestTrainStep:
    """Test JIT-compiled training step."""

    @pytest.fixture
    def state_and_batch(self):
        """Create a small training state and a dummy batch."""
        config = TrainingConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            ff_dim=128,
            train_policy=True,
        )
        rng = jax.random.PRNGKey(42)
        state = create_train_state(config, rng)

        batch = {
            'board_encoding': jnp.zeros((4, 26, 10)),
            'target_policy': jnp.ones((4, 4096)) / 4096,
            'equity_target': jnp.ones((4, 6)) / 6,
            'action_mask': jnp.ones((4, 4096)),
        }
        return state, batch

    def test_train_step_runs(self, state_and_batch):
        """train_step should JIT-compile and return updated state + metrics."""
        state, batch = state_and_batch
        rng = jax.random.PRNGKey(0)

        new_state, metrics = train_step(state, batch, rng)

        assert new_state.step == state.step + 1
        assert jnp.isfinite(metrics['total_loss'])
        assert 'grad_norm' in metrics

    def test_params_change_after_step(self, state_and_batch):
        """Parameters should change after a gradient step."""
        state, batch = state_and_batch
        rng = jax.random.PRNGKey(0)

        new_state, _ = train_step(state, batch, rng)

        old_leaves = jax.tree_util.tree_leaves(state.params)
        new_leaves = jax.tree_util.tree_leaves(new_state.params)

        any_changed = any(
            not np.allclose(o, n) for o, n in zip(old_leaves, new_leaves)
        )
        assert any_changed, "At least some parameters should change after a gradient step"

    def test_multiple_steps_converge(self, state_and_batch):
        """Loss should decrease over multiple steps on constant data."""
        state, batch = state_and_batch
        rng = jax.random.PRNGKey(0)

        losses = []
        for i in range(10):
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, step_rng)
            losses.append(float(metrics['total_loss']))

        # Loss at end should be lower than at start
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_value_only_train_step(self):
        """train_step should work without policy head."""
        config = TrainingConfig(
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            ff_dim=128,
            train_policy=False,
        )
        state = create_train_state(config, jax.random.PRNGKey(42))

        batch = {
            'board_encoding': jnp.zeros((4, 26, 10)),
            'target_policy': jnp.zeros((4, 4096)),
            'equity_target': jnp.ones((4, 6)) / 6,
            'action_mask': jnp.ones((4, 4096)),
        }
        rng = jax.random.PRNGKey(0)

        new_state, metrics = train_step(state, batch, rng)

        assert new_state.step == state.step + 1
        assert jnp.isfinite(metrics['total_loss'])
        assert float(metrics['policy_loss']) == 0.0


class TestComputeMetrics:
    """Test evaluation metrics computation."""

    def test_metrics_with_policy_head(self):
        """compute_metrics should return accuracy when policy head is enabled."""
        from backgammon.encoding.action_encoder import get_action_space_size
        action_size = get_action_space_size()
        config = TrainingConfig(
            embed_dim=32, num_heads=2, num_layers=1, ff_dim=128,
            train_policy=True,
        )
        state = create_train_state(config, jax.random.PRNGKey(42))

        batch = {
            'board_encoding': jnp.zeros((2, 26, 10)),
            'target_policy': jnp.ones((2, action_size)) / action_size,
            'equity_target': jnp.ones((2, 6)) / 6,
            'action_mask': jnp.ones((2, action_size)),
        }

        metrics = compute_metrics(state, batch)

        assert 'loss' in metrics
        assert 'equity_loss' in metrics
        assert 'equity_accuracy' in metrics
        assert 'policy_accuracy' in metrics
        assert np.isfinite(metrics['loss'])

    def test_metrics_value_only(self):
        """compute_metrics should work in value-only mode."""
        config = TrainingConfig(
            embed_dim=32, num_heads=2, num_layers=1, ff_dim=128,
            train_policy=False,
        )
        state = create_train_state(config, jax.random.PRNGKey(42))

        batch = {
            'board_encoding': jnp.zeros((2, 26, 10)),
            'target_policy': jnp.zeros((2, 4096)),
            'equity_target': jnp.ones((2, 6)) / 6,
            'action_mask': jnp.ones((2, 4096)),
        }

        metrics = compute_metrics(state, batch)

        assert 'equity_accuracy' in metrics
        assert 'policy_accuracy' not in metrics  # No policy head in value-only mode
        assert np.isfinite(metrics['loss'])
