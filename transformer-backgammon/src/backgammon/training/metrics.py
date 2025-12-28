"""Metrics logging for training with TensorBoard and Weights & Biases support.

Provides unified interface for logging training metrics to various backends.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import numpy as np

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# Optional Weights & Biases support
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class MetricsLogger:
    """Unified metrics logger supporting multiple backends.

    Supports:
    - Console logging (always enabled)
    - JSONL file logging (always enabled)
    - TensorBoard (optional, if installed)
    - Weights & Biases (optional, if installed and configured)

    Args:
        log_dir: Directory for logs
        run_name: Name of this training run
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_config: Configuration dict for W&B
        console_interval: Log to console every N steps
    """

    log_dir: Path
    run_name: str = "backgammon_training"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_config: Optional[Dict[str, Any]] = None
    console_interval: int = 10

    # Internal state
    _tensorboard_writer: Optional[Any] = field(default=None, init=False, repr=False)
    _wandb_initialized: bool = field(default=False, init=False, repr=False)
    _jsonl_file: Optional[Any] = field(default=None, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _start_time: float = field(default_factory=time.time, init=False, repr=False)

    def __post_init__(self):
        """Initialize logging backends."""
        # Create log directory
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize JSONL logging
        jsonl_path = self.log_dir / f"{self.run_name}_metrics.jsonl"
        self._jsonl_file = open(jsonl_path, 'a')

        # Initialize TensorBoard if requested
        if self.use_tensorboard:
            if not HAS_TENSORBOARD:
                print("Warning: TensorBoard requested but not installed. Install with: pip install tensorboard")
                self.use_tensorboard = False
            else:
                tb_dir = self.log_dir / "tensorboard" / self.run_name
                self._tensorboard_writer = SummaryWriter(str(tb_dir))
                print(f"TensorBoard logging to: {tb_dir}")

        # Initialize Weights & Biases if requested
        if self.use_wandb:
            if not HAS_WANDB:
                print("Warning: W&B requested but not installed. Install with: pip install wandb")
                self.use_wandb = False
            else:
                if self.wandb_project is None:
                    self.wandb_project = "backgammon-transformer"

                wandb.init(
                    project=self.wandb_project,
                    name=self.run_name,
                    config=self.wandb_config or {},
                )
                self._wandb_initialized = True
                print(f"W&B logging to project: {self.wandb_project}")

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step number (auto-incremented if None)
            prefix: Prefix to add to all metric names (e.g., "train/", "eval/")
        """
        if step is None:
            step = self._step_count
            self._step_count += 1

        # Add prefix to metric names if provided
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Add timestamp and step
        log_entry = {
            "step": step,
            "timestamp": time.time() - self._start_time,
            **metrics,
        }

        # Log to JSONL file
        self._jsonl_file.write(json.dumps(log_entry) + '\n')
        self._jsonl_file.flush()

        # Log to TensorBoard
        if self.use_tensorboard and self._tensorboard_writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self._tensorboard_writer.add_scalar(name, value, step)
                elif isinstance(value, np.ndarray):
                    self._tensorboard_writer.add_histogram(name, value, step)

        # Log to Weights & Biases
        if self.use_wandb and self._wandb_initialized:
            wandb.log(metrics, step=step)

        # Log to console
        if step % self.console_interval == 0:
            self._log_console(step, metrics)

    def _log_console(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log metrics to console in readable format."""
        elapsed = time.time() - self._start_time
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        print(f"[Step {step:6d}] [{elapsed:8.1f}s] {metrics_str}")

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameter name -> value
        """
        # Log to JSONL
        log_entry = {
            "type": "hyperparameters",
            "timestamp": time.time() - self._start_time,
            **params,
        }
        self._jsonl_file.write(json.dumps(log_entry) + '\n')
        self._jsonl_file.flush()

        # TensorBoard hparams
        if self.use_tensorboard and self._tensorboard_writer is not None:
            # Convert values to scalars for TensorBoard
            scalar_params = {
                k: v for k, v in params.items()
                if isinstance(v, (int, float, str, bool))
            }
            if scalar_params:
                self._tensorboard_writer.add_hparams(
                    scalar_params,
                    {},  # Empty metrics dict (will be filled later)
                )

        # W&B automatically logs config on init, but we can update it
        if self.use_wandb and self._wandb_initialized:
            wandb.config.update(params)

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """Log text data.

        Args:
            name: Name of the text entry
            text: Text content
            step: Training step
        """
        if step is None:
            step = self._step_count

        # Log to JSONL
        log_entry = {
            "type": "text",
            "name": name,
            "text": text,
            "step": step,
            "timestamp": time.time() - self._start_time,
        }
        self._jsonl_file.write(json.dumps(log_entry) + '\n')
        self._jsonl_file.flush()

        # Log to TensorBoard
        if self.use_tensorboard and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_text(name, text, step)

        # W&B doesn't have direct text logging in the same way

    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """Log histogram data.

        Args:
            name: Name of the histogram
            values: Array of values
            step: Training step
        """
        if step is None:
            step = self._step_count

        # Log summary statistics to JSONL
        log_entry = {
            "type": "histogram",
            "name": name,
            "step": step,
            "timestamp": time.time() - self._start_time,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
        self._jsonl_file.write(json.dumps(log_entry) + '\n')
        self._jsonl_file.flush()

        # Log to TensorBoard
        if self.use_tensorboard and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_histogram(name, values, step)

        # Log to W&B
        if self.use_wandb and self._wandb_initialized:
            wandb.log({name: wandb.Histogram(values)}, step=step)

    def save_summary(self, metrics: Dict[str, Any]) -> None:
        """Save final training summary.

        Args:
            metrics: Final metrics to save
        """
        summary_path = self.log_dir / f"{self.run_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Training summary saved to: {summary_path}")

        # Log to W&B summary
        if self.use_wandb and self._wandb_initialized:
            for k, v in metrics.items():
                wandb.run.summary[k] = v

    def close(self) -> None:
        """Close all logging backends."""
        # Close JSONL file
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None

        # Close TensorBoard
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
            self._tensorboard_writer = None

        # Finish W&B run
        if self._wandb_initialized:
            wandb.finish()
            self._wandb_initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MetricsAggregator:
    """Aggregates metrics over multiple steps for averaging.

    Useful for computing epoch-level or phase-level averages.
    """

    def __init__(self):
        """Initialize empty aggregator."""
        self._values: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}

    def add(self, name: str, value: float, count: int = 1) -> None:
        """Add a metric value.

        Args:
            name: Metric name
            value: Metric value
            count: Number of samples this value represents
        """
        if name not in self._values:
            self._values[name] = []
            self._counts[name] = 0

        self._values[name].append(value)
        self._counts[name] += count

    def add_dict(self, metrics: Dict[str, float], count: int = 1) -> None:
        """Add multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            count: Number of samples these values represent
        """
        for name, value in metrics.items():
            self.add(name, value, count)

    def compute_averages(self) -> Dict[str, float]:
        """Compute average value for each metric.

        Returns:
            Dictionary of metric name -> average value
        """
        averages = {}
        for name, values in self._values.items():
            averages[name] = np.mean(values)

        return averages

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics (mean, std, min, max) for each metric.

        Returns:
            Dictionary of metric name -> statistics dict
        """
        stats = {}
        for name, values in self._values.items():
            values_array = np.array(values)
            stats[name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'count': self._counts[name],
            }

        return stats

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self._values.clear()
        self._counts.clear()

    def __len__(self) -> int:
        """Return number of metrics being tracked."""
        return len(self._values)
