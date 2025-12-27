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
# Run tests (will fail initially - we haven't implemented yet)
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
├── configs/                 # Configuration files
├── benchmarks/              # Benchmark position suites
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Utility scripts
│
├── pyproject.toml           # Package configuration
├── requirements.txt         # Core dependencies
└── requirements-dev.txt     # Development dependencies
```

## Next Steps

1. **Implement core types** (`src/backgammon/core/types.py`)
2. **Implement board logic** (`src/backgammon/core/board.py`)
3. **Write tests** (`tests/core/test_board.py`)
4. **Verify everything works** (`pytest`)

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

Initially, tests will fail because we haven't implemented the modules yet. This is expected!

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
