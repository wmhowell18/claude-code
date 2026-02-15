# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Or install with development tools
pip install -e ".[dev]"

# Or install everything (dev + training tools)
pip install -e ".[all]"
```

### 2. Verify Installation

```bash
## Run tests
pytest

# Check imports work
python -c "import backgammon; print(backgammon.__version__)"
```

### 3. Development Workflow

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=backgammon --cov-report=html

# Run tests in parallel
pytest -n auto

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Lint
flake8 src/ tests/
```

## Project Structure

```
transformer-backgammon/
├── src/backgammon/          # Main package
│   ├── core/                # Game logic
│   │   ├── types.py         # Data structures
│   │   ├── board.py         # Board representation
│   │   └── dice.py          # Dice utilities
│   ├── encoding/            # Board encoding
│   ├── network/             # Transformer architecture
│   ├── training/            # Training loop
│   ├── evaluation/          # Position evaluation
│   └── utils/               # Utilities
│
├── tests/                   # Test suite
│   ├── core/                # Core tests
│   ├── encoding/            # Encoding tests
│   ├── network/             # Network tests
│   ├── training/            # Training tests
│   └── evaluation/          # Evaluation tests
│
├── scripts/                 # Utility scripts
│
├── pyproject.toml           # Package configuration
├── requirements.txt         # Core dependencies
└── requirements-dev.txt     # Development dependencies
```

## Next Steps

1. **Run the smoke test** (`python scripts/smoke_test.py`) to validate training + search quickly
2. **Try the example trainer** (`python scripts/train_example.py`) for a short end-to-end run
3. **Run focused tests first** (`pytest tests/core -q`) before full-suite execution
4. **Scale up** to longer training runs once local checks pass

## Dependencies

### Core (required)
- **JAX**: GPU-accelerated numerical computing
- **Flax**: Neural network library for JAX
- **Optax**: Optimization library for JAX
- **NumPy**: Numerical arrays

### Development (optional)
- **pytest**: Testing framework
- **black**: Code formatter
- **mypy**: Type checker
- **wandb**: Experiment tracking (for training)

## Troubleshooting

### JAX Installation Issues

If you have issues with JAX/CUDA:

```bash
# CPU-only (for development)
pip install jax jaxlib

# GPU support (CUDA 11)
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# GPU support (CUDA 12)
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Import Errors

Make sure you installed in editable mode:
```bash
pip install -e .
```

### Test Failures

If tests fail, ensure dependencies are installed and run a focused subset first (e.g. `pytest tests/core -q`).

## Development Tips

1. **Start small**: Implement one function at a time
2. **Test as you go**: Write tests alongside implementation
3. **Use type hints**: Makes code more maintainable
4. **Follow the .mli specs**: Interface files define what to implement

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Project README](README.md)
- [Architecture Specification](types.mli, board.mli, etc.)
