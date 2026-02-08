"""Position evaluation and move selection."""

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

from backgammon.evaluation.network_agent import (
    NeuralNetworkAgent,
    create_neural_agent,
    create_mcts_agent,
)

from backgammon.evaluation.evaluator import (
    TrainingEvaluator,
    EvalConfig,
    evaluate_vs_opponent,
)

__all__ = [
    # Agents
    "Agent",
    "PipCountConfig",
    "random_agent",
    "pip_count_agent",
    "greedy_pip_count_agent",
    "count_blots",
    "has_anchor",
    "is_past_contact",
    # Neural network agents
    "NeuralNetworkAgent",
    "create_neural_agent",
    "create_mcts_agent",
    # Evaluation
    "TrainingEvaluator",
    "EvalConfig",
    "evaluate_vs_opponent",
]
