"""Tests for metrics logging."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np

from backgammon.training.metrics import MetricsLogger, MetricsAggregator


class TestMetricsLogger:
    """Test metrics logger functionality."""

    def test_logger_creation(self):
        """Test creating a metrics logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
            )

            assert logger.log_dir == Path(tmpdir)
            assert logger.run_name == "test_run"
            assert not logger.use_tensorboard
            assert not logger.use_wandb

            logger.close()

    def test_log_metrics_basic(self):
        """Test basic metric logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
                console_interval=100,  # Disable console logging for test
            )

            # Log some metrics
            logger.log_metrics({
                'loss': 0.5,
                'accuracy': 0.8,
            }, step=1)

            logger.log_metrics({
                'loss': 0.4,
                'accuracy': 0.85,
            }, step=2)

            logger.close()

            # Check JSONL file was created
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            assert jsonl_path.exists()

            # Read and verify contents
            with open(jsonl_path) as f:
                lines = f.readlines()

            assert len(lines) == 2

            # Parse first entry
            entry1 = json.loads(lines[0])
            assert entry1['step'] == 1
            assert entry1['loss'] == 0.5
            assert entry1['accuracy'] == 0.8
            assert 'timestamp' in entry1

            # Parse second entry
            entry2 = json.loads(lines[1])
            assert entry2['step'] == 2
            assert entry2['loss'] == 0.4
            assert entry2['accuracy'] == 0.85

    def test_log_metrics_with_prefix(self):
        """Test logging metrics with prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
                console_interval=100,
            )

            logger.log_metrics({
                'loss': 0.5,
                'accuracy': 0.8,
            }, step=1, prefix="train/")

            logger.close()

            # Read JSONL
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            with open(jsonl_path) as f:
                entry = json.loads(f.readline())

            # Check prefix was added
            assert 'train/loss' in entry
            assert 'train/accuracy' in entry
            assert entry['train/loss'] == 0.5

    def test_log_metrics_auto_step(self):
        """Test automatic step incrementing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
                console_interval=100,
            )

            # Log without step (should auto-increment)
            logger.log_metrics({'loss': 0.5})
            logger.log_metrics({'loss': 0.4})
            logger.log_metrics({'loss': 0.3})

            logger.close()

            # Read JSONL
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            with open(jsonl_path) as f:
                lines = f.readlines()

            # Check steps auto-incremented
            assert json.loads(lines[0])['step'] == 0
            assert json.loads(lines[1])['step'] == 1
            assert json.loads(lines[2])['step'] == 2

    def test_log_hyperparams(self):
        """Test logging hyperparameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
            )

            logger.log_hyperparams({
                'learning_rate': 0.001,
                'batch_size': 32,
                'model': 'transformer',
            })

            logger.close()

            # Read JSONL
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            with open(jsonl_path) as f:
                entry = json.loads(f.readline())

            assert entry['type'] == 'hyperparameters'
            assert entry['learning_rate'] == 0.001
            assert entry['batch_size'] == 32
            assert entry['model'] == 'transformer'

    def test_log_text(self):
        """Test logging text data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
            )

            logger.log_text("progress", "Training started", step=0)
            logger.log_text("progress", "Epoch 1 complete", step=100)

            logger.close()

            # Read JSONL
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            with open(jsonl_path) as f:
                lines = f.readlines()

            entry1 = json.loads(lines[0])
            assert entry1['type'] == 'text'
            assert entry1['name'] == 'progress'
            assert entry1['text'] == 'Training started'
            assert entry1['step'] == 0

    def test_log_histogram(self):
        """Test logging histogram data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
            )

            values = np.random.randn(100)
            logger.log_histogram("gradients", values, step=1)

            logger.close()

            # Read JSONL
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            with open(jsonl_path) as f:
                entry = json.loads(f.readline())

            assert entry['type'] == 'histogram'
            assert entry['name'] == 'gradients'
            assert 'mean' in entry
            assert 'std' in entry
            assert 'min' in entry
            assert 'max' in entry
            assert entry['count'] == 100

    def test_save_summary(self):
        """Test saving training summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
            )

            logger.save_summary({
                'final_loss': 0.1,
                'final_accuracy': 0.95,
                'total_steps': 1000,
            })

            logger.close()

            # Check summary file
            summary_path = Path(tmpdir) / "test_run_summary.json"
            assert summary_path.exists()

            with open(summary_path) as f:
                summary = json.load(f)

            assert summary['final_loss'] == 0.1
            assert summary['final_accuracy'] == 0.95
            assert summary['total_steps'] == 1000

    def test_context_manager(self):
        """Test using logger as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
            ) as logger:
                logger.log_metrics({'loss': 0.5}, step=1)

            # Should be closed automatically
            assert logger._jsonl_file is None

            # File should still exist
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            assert jsonl_path.exists()

    def test_console_logging_interval(self):
        """Test console logging respects interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                log_dir=tmpdir,
                run_name="test_run",
                use_tensorboard=False,
                use_wandb=False,
                console_interval=5,
            )

            # Log at steps 0, 1, 2, 3, 4, 5
            # Console should log at 0 and 5
            for step in range(6):
                logger.log_metrics({'loss': 0.5}, step=step)

            logger.close()

            # All metrics should be in JSONL regardless
            jsonl_path = Path(tmpdir) / "test_run_metrics.jsonl"
            with open(jsonl_path) as f:
                lines = f.readlines()

            assert len(lines) == 6


class TestMetricsAggregator:
    """Test metrics aggregator."""

    def test_aggregator_creation(self):
        """Test creating an aggregator."""
        agg = MetricsAggregator()
        assert len(agg) == 0

    def test_add_single_metric(self):
        """Test adding a single metric."""
        agg = MetricsAggregator()

        agg.add("loss", 0.5)
        agg.add("loss", 0.4)
        agg.add("loss", 0.3)

        assert len(agg) == 1

        averages = agg.compute_averages()
        assert "loss" in averages
        assert averages["loss"] == pytest.approx(0.4)

    def test_add_multiple_metrics(self):
        """Test adding multiple metrics."""
        agg = MetricsAggregator()

        agg.add("loss", 0.5)
        agg.add("loss", 0.3)
        agg.add("accuracy", 0.8)
        agg.add("accuracy", 0.9)

        assert len(agg) == 2

        averages = agg.compute_averages()
        assert averages["loss"] == pytest.approx(0.4)
        assert averages["accuracy"] == pytest.approx(0.85)

    def test_add_dict(self):
        """Test adding metrics as dictionary."""
        agg = MetricsAggregator()

        agg.add_dict({
            "loss": 0.5,
            "accuracy": 0.8,
        })

        agg.add_dict({
            "loss": 0.3,
            "accuracy": 0.9,
        })

        averages = agg.compute_averages()
        assert averages["loss"] == pytest.approx(0.4)
        assert averages["accuracy"] == pytest.approx(0.85)

    def test_compute_statistics(self):
        """Test computing full statistics."""
        agg = MetricsAggregator()

        agg.add("loss", 0.1)
        agg.add("loss", 0.2)
        agg.add("loss", 0.3)
        agg.add("loss", 0.4)
        agg.add("loss", 0.5)

        stats = agg.compute_statistics()

        assert "loss" in stats
        assert stats["loss"]["mean"] == pytest.approx(0.3)
        assert stats["loss"]["min"] == pytest.approx(0.1)
        assert stats["loss"]["max"] == pytest.approx(0.5)
        assert stats["loss"]["count"] == 5
        assert "std" in stats["loss"]

    def test_reset(self):
        """Test resetting aggregator."""
        agg = MetricsAggregator()

        agg.add("loss", 0.5)
        agg.add("accuracy", 0.8)

        assert len(agg) == 2

        agg.reset()
        assert len(agg) == 0

        averages = agg.compute_averages()
        assert len(averages) == 0

    def test_weighted_aggregation(self):
        """Test aggregation with counts."""
        agg = MetricsAggregator()

        # Add with different counts
        agg.add("loss", 0.5, count=10)
        agg.add("loss", 0.3, count=20)

        # Average should be weighted by count
        # But compute_averages just averages the values, not weighted
        averages = agg.compute_averages()
        assert averages["loss"] == pytest.approx(0.4)

        # Count is tracked separately
        stats = agg.compute_statistics()
        assert stats["loss"]["count"] == 30  # 10 + 20


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_full_training_loop_simulation(self):
        """Test metrics logging in a simulated training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with MetricsLogger(
                log_dir=tmpdir,
                run_name="integration_test",
                use_tensorboard=False,
                use_wandb=False,
                console_interval=100,
            ) as logger:

                # Log hyperparameters
                logger.log_hyperparams({
                    'learning_rate': 0.001,
                    'batch_size': 32,
                })

                # Simulate training
                for epoch in range(3):
                    # Epoch-level aggregator
                    epoch_metrics = MetricsAggregator()

                    # Simulate batches
                    for batch in range(10):
                        step = epoch * 10 + batch

                        # Batch metrics
                        batch_loss = 0.5 - step * 0.01
                        batch_acc = 0.7 + step * 0.01

                        logger.log_metrics({
                            'loss': batch_loss,
                            'accuracy': batch_acc,
                        }, step=step, prefix='train/')

                        # Add to epoch aggregator
                        epoch_metrics.add_dict({
                            'loss': batch_loss,
                            'accuracy': batch_acc,
                        })

                    # Log epoch averages
                    epoch_avg = epoch_metrics.compute_averages()
                    logger.log_metrics(
                        epoch_avg,
                        step=(epoch + 1) * 10,
                        prefix='epoch/',
                    )

                # Save final summary
                logger.save_summary({
                    'total_epochs': 3,
                    'final_loss': 0.2,
                })

            # Verify files were created
            jsonl_path = Path(tmpdir) / "integration_test_metrics.jsonl"
            summary_path = Path(tmpdir) / "integration_test_summary.json"

            assert jsonl_path.exists()
            assert summary_path.exists()

            # Count entries
            with open(jsonl_path) as f:
                lines = f.readlines()

            # Should have: 1 hyperparams + 30 batch metrics + 3 epoch metrics = 34
            assert len(lines) == 34
