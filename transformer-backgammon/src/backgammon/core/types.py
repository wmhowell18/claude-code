"""Core type definitions for transformer backgammon.

This module defines all the core data structures used throughout the project,
following the specification in types.mli.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray


# ==============================================================================
# BOARD REPRESENTATION
# ==============================================================================

# Type aliases
Point = int  # 0-25: 0=bar, 1-24=points, 25=off
CheckerCount = int  # 0-15


class Player(Enum):
    """Player colors."""
    WHITE = "white"
    BLACK = "black"

    def opponent(self) -> "Player":
        """Return the opponent player."""
        return Player.BLACK if self == Player.WHITE else Player.WHITE

    def __str__(self) -> str:
        return self.value


# Dice type
Dice = Tuple[int, int]  # (die1, die2) where 1 <= die1, die2 <= 6


@dataclass(frozen=True)
class MoveStep:
    """A single checker movement.

    Attributes:
        from_point: Starting position (0=bar, 1-24=points, 25=off)
        to_point: Ending position (0=bar, 1-24=points, 25=off)
        die_used: Which die value was used (1-6)
        hits_opponent: Whether this move hits an opponent blot
    """
    from_point: Point
    to_point: Point
    die_used: int
    hits_opponent: bool = False

    def __post_init__(self):
        """Validate move step."""
        assert 0 <= self.from_point <= 25, f"Invalid from_point: {self.from_point}"
        assert 0 <= self.to_point <= 25, f"Invalid to_point: {self.to_point}"
        assert 1 <= self.die_used <= 6, f"Invalid die: {self.die_used}"


# A complete move (may use 1-4 dice for doubles)
Move = Tuple[MoveStep, ...]  # Use tuple for immutability


@dataclass
class Board:
    """Board state representation.

    The board has 26 positions:
    - Position 0: Bar (where hit checkers go)
    - Positions 1-24: Regular points
    - Position 25: Off (where borne-off checkers go)

    For White:
    - Home board: points 1-6
    - Moves from high to low (24 → 1 → off)

    For Black:
    - Home board: points 19-24
    - Moves from low to high (1 → 24 → off)

    Attributes:
        white_checkers: Array of checker counts for white (length 26)
        black_checkers: Array of checker counts for black (length 26)
        player_to_move: Which player moves next
    """
    white_checkers: NDArray[np.int32] = field(default_factory=lambda: np.zeros(26, dtype=np.int32))
    black_checkers: NDArray[np.int32] = field(default_factory=lambda: np.zeros(26, dtype=np.int32))
    player_to_move: Player = Player.WHITE

    def __post_init__(self):
        """Validate board state."""
        assert len(self.white_checkers) == 26, "white_checkers must have length 26"
        assert len(self.black_checkers) == 26, "black_checkers must have length 26"
        assert all(0 <= c <= 15 for c in self.white_checkers), "Invalid white checker count"
        assert all(0 <= c <= 15 for c in self.black_checkers), "Invalid black checker count"

    def copy(self) -> "Board":
        """Create a deep copy of the board."""
        return Board(
            white_checkers=self.white_checkers.copy(),
            black_checkers=self.black_checkers.copy(),
            player_to_move=self.player_to_move,
        )

    def get_checkers(self, player: Player, point: Point) -> CheckerCount:
        """Get number of checkers at a point for a player."""
        if player == Player.WHITE:
            return int(self.white_checkers[point])
        else:
            return int(self.black_checkers[point])

    def set_checkers(self, player: Player, point: Point, count: CheckerCount) -> None:
        """Set number of checkers at a point for a player (mutates board)."""
        assert 0 <= count <= 15, f"Invalid checker count: {count}"
        if player == Player.WHITE:
            self.white_checkers[point] = count
        else:
            self.black_checkers[point] = count


# Legal moves for a given board + dice
LegalMoves = List[Move]


# ==============================================================================
# NEURAL NETWORK REPRESENTATIONS
# ==============================================================================

@dataclass
class Equity:
    """Network output: equity estimation.

    Represents the probability distribution over game outcomes.
    Note: P(lose normal) = 1 - sum(all other probabilities)

    Attributes:
        win_normal: P(win without gammon)
        win_gammon: P(win with gammon, no backgammon)
        win_backgammon: P(win with backgammon)
        lose_gammon: P(lose with gammon, no backgammon)
        lose_backgammon: P(lose with backgammon)
    """
    win_normal: float
    win_gammon: float
    win_backgammon: float
    lose_gammon: float
    lose_backgammon: float

    def __post_init__(self):
        """Validate probabilities."""
        total = (
            self.win_normal
            + self.win_gammon
            + self.win_backgammon
            + self.lose_gammon
            + self.lose_backgammon
        )
        assert 0.0 <= total <= 1.01, f"Probabilities sum to {total}, should be <= 1.0"

    @property
    def lose_normal(self) -> float:
        """P(lose without gammon)."""
        return 1.0 - (
            self.win_normal
            + self.win_gammon
            + self.win_backgammon
            + self.lose_gammon
            + self.lose_backgammon
        )

    def expected_value(self, cube_value: int = 1) -> float:
        """Calculate expected game value (in points).

        Args:
            cube_value: Current doubling cube value (1, 2, 4, 8, 16, 32, 64)

        Returns:
            Expected value: positive = good for current player, negative = bad
        """
        win_value = (
            self.win_normal * 1.0
            + self.win_gammon * 2.0
            + self.win_backgammon * 3.0
        )
        lose_value = (
            self.lose_normal * -1.0
            + self.lose_gammon * -2.0
            + self.lose_backgammon * -3.0
        )
        return (win_value + lose_value) * cube_value

    def to_array(self) -> NDArray[np.float32]:
        """Convert to numpy array [win_normal, win_gammon, win_bg, lose_gammon, lose_bg]."""
        return np.array(
            [
                self.win_normal,
                self.win_gammon,
                self.win_backgammon,
                self.lose_gammon,
                self.lose_backgammon,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_array(arr: NDArray[np.float32]) -> "Equity":
        """Create from numpy array."""
        assert len(arr) == 5, "Array must have exactly 5 elements"
        return Equity(
            win_normal=float(arr[0]),
            win_gammon=float(arr[1]),
            win_backgammon=float(arr[2]),
            lose_gammon=float(arr[3]),
            lose_backgammon=float(arr[4]),
        )


@dataclass
class EncodedBoard:
    """Board encoded as features for neural network input.

    Attributes:
        position_features: [batch, seq_len, feature_dim] - features for each position
        dice_features: Optional dice encoding
        batch_size: Number of boards in batch
        sequence_length: Length of position sequence (always 26)
        feature_dim: Dimension of features per position
    """
    position_features: NDArray[np.float32]  # [batch, 26, feature_dim]
    dice_features: Optional[NDArray[np.float32]] = None  # [batch, dice_feature_dim]
    batch_size: int = 1
    sequence_length: int = 26
    feature_dim: int = 0

    def __post_init__(self):
        """Validate shapes."""
        assert len(self.position_features.shape) == 3, "position_features must be 3D"
        assert self.position_features.shape[1] == 26, "sequence_length must be 26"
        self.batch_size = self.position_features.shape[0]
        self.sequence_length = self.position_features.shape[1]
        self.feature_dim = self.position_features.shape[2]


@dataclass
class EncodingConfig:
    """Configuration for board encoding.

    Attributes:
        use_one_hot_counts: Encode checker counts as one-hot vectors (vs continuous)
        include_geometric_features: Add geometric features (distance to home, etc.)
        include_strategic_features: Add strategic features (blots, anchors, etc.)
        include_dice_encoding: Include dice features in encoding
        include_global_features: Add global board features broadcast to each position.
            Adds 8 features: is_contact, our_pip_norm, opp_pip_norm,
            our_home_points, opp_home_points, our_prime_len, opp_prime_len,
            our_bearoff_progress.
        feature_dim: Resulting feature dimension per position
    """
    use_one_hot_counts: bool = False
    include_geometric_features: bool = False
    include_strategic_features: bool = False
    include_dice_encoding: bool = False
    include_global_features: bool = False
    feature_dim: int = 2  # Will be computed based on flags


@dataclass
class TransformerConfig:
    """Configuration for transformer architecture.

    Attributes:
        num_layers: Number of transformer blocks
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        dropout_rate: Dropout probability
        layer_norm_epsilon: Epsilon for layer normalization
        input_feature_dim: Dimension of input features per position
        use_learned_positional_encoding: Whether to use learned position embeddings
        use_policy_head: Whether to include policy prediction head
        num_actions: Number of actions in policy head output
        return_attention_weights: Whether to return attention weights for visualization
    """
    num_layers: int = 4
    embed_dim: int = 128
    num_heads: int = 8
    ff_dim: int = 512
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    input_feature_dim: int = 2
    use_learned_positional_encoding: bool = True
    use_policy_head: bool = False
    num_actions: int = 1024  # Default action space size
    return_attention_weights: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert 0 <= self.dropout_rate < 1, "dropout_rate must be in [0, 1)"


@dataclass
class TrainingConfig:
    """Configuration for training loop.

    Attributes:
        learning_rate: Initial learning rate
        optimizer: Optimizer name ("adam", "sgd", etc.)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        minibatches_per_epoch: Number of minibatches per epoch
        replay_buffer_size: Experience replay buffer capacity
        min_replay_size: Minimum buffer size before training starts
        games_per_iteration: Number of self-play games per iteration
        eval_every_n_games: Evaluate after this many games
        eval_num_games: Number of games for evaluation
        checkpoint_every_n_games: Save checkpoint after this many games
        checkpoint_dir: Directory for saving checkpoints
    """
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    batch_size: int = 256
    num_epochs: int = 100
    minibatches_per_epoch: int = 100
    replay_buffer_size: int = 100000
    min_replay_size: int = 10000
    games_per_iteration: int = 100
    eval_every_n_games: int = 1000
    eval_num_games: int = 100
    checkpoint_every_n_games: int = 5000
    checkpoint_dir: str = "./checkpoints"


@dataclass
class Config:
    """Complete configuration for the backgammon AI.

    Attributes:
        transformer: Transformer architecture config
        encoding: Board encoding config
        training: Training loop config
        seed: Random seed
        device: Device to use ("cpu", "gpu", "tpu")
        use_jit: Whether to use JAX JIT compilation
        log_level: Logging level
        log_dir: Directory for logs
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
    """
    transformer: TransformerConfig
    encoding: EncodingConfig
    training: TrainingConfig
    seed: int = 42
    device: str = "cpu"
    use_jit: bool = True
    log_level: str = "info"
    log_dir: str = "./logs"
    use_wandb: bool = False
    wandb_project: str = "transformer-backgammon"


# ==============================================================================
# GAME SIMULATION
# ==============================================================================

@dataclass
class GameOutcome:
    """Game outcome with points won/lost.

    Attributes:
        winner: Which player won
        points: Points won (1=normal, 2=gammon, 3=backgammon)
    """
    winner: Player
    points: int  # 1, 2, or 3

    def __post_init__(self):
        """Validate outcome."""
        assert self.points in [1, 2, 3], f"Points must be 1, 2, or 3, got {self.points}"

    def is_gammon(self) -> bool:
        """Check if outcome is a gammon (includes backgammon)."""
        return self.points >= 2

    def is_backgammon(self) -> bool:
        """Check if outcome is a backgammon."""
        return self.points == 3


@dataclass
class GameRecord:
    """Complete game record.

    Attributes:
        game_id: Unique game identifier
        positions: List of (board, dice, move) tuples
        outcome: Final game outcome
        num_plies: Number of plies played
        white_player: White player name/type
        black_player: Black player name/type
        timestamp: Unix timestamp of game start
    """
    game_id: int
    positions: List[Tuple[Board, Dice, Move]]
    outcome: GameOutcome
    num_plies: int
    white_player: str
    black_player: str
    timestamp: float


# ==============================================================================
# TRAINING DATA STRUCTURES
# ==============================================================================

@dataclass
class TrainingExample:
    """A single training example.

    Attributes:
        board_state: Board position
        player: Player to move
        target_equity: Target equity from game outcome
        target_policy: Optional policy target from MCTS
        game_id: Which game this came from
        ply_number: Which ply in the game
    """
    board_state: Board
    player: Player
    target_equity: Equity
    target_policy: Optional[NDArray[np.float32]] = None
    game_id: int = 0
    ply_number: int = 0


@dataclass
class Batch:
    """Training batch.

    Attributes:
        boards: Encoded boards [batch_size, 26, feature_dim]
        target_equities: Target equity values [batch_size, 5]
        target_policies: Optional policy targets [batch_size, num_moves]
    """
    boards: EncodedBoard
    target_equities: NDArray[np.float32]  # [batch_size, 5]
    target_policies: Optional[NDArray[np.float32]] = None  # [batch_size, num_moves]


@dataclass
class TrainingMetrics:
    """Training metrics for logging.

    Attributes:
        epoch: Current epoch number
        total_games: Total games played
        total_positions: Total positions trained on
        equity_loss: Current equity loss
        policy_loss: Current policy loss (if using policy head)
        total_loss: Combined loss
        win_rate_vs_random: Win rate against random baseline
        win_rate_vs_baseline: Win rate against other baseline
        positions_per_second: Training throughput
        games_per_minute: Game generation rate
    """
    epoch: int
    total_games: int
    total_positions: int
    equity_loss: float
    policy_loss: Optional[float] = None
    total_loss: float = 0.0
    win_rate_vs_random: Optional[float] = None
    win_rate_vs_baseline: Optional[float] = None
    positions_per_second: float = 0.0
    games_per_minute: float = 0.0


# ==============================================================================
# EVALUATION AND SEARCH
# ==============================================================================

@dataclass
class SearchConfig:
    """Configuration for move selection search.

    Attributes:
        ply_depth: Search depth (0=greedy, 1=1-ply, 2=2-ply, etc.)
        average_dice: Whether to average over all 21 dice outcomes
        num_dice_samples: If not averaging, how many dice to sample
        prune_to_top_k: Only consider top-k moves at each ply
    """
    ply_depth: int = 1
    average_dice: bool = True
    num_dice_samples: Optional[int] = None
    prune_to_top_k: Optional[int] = None


@dataclass
class MoveEvaluation:
    """Evaluation result for a single move.

    Attributes:
        move: The move being evaluated
        equity: Equity after this move
        expected_equity: Single expected value score
        evaluations_count: How many positions were evaluated
    """
    move: Move
    equity: Equity
    expected_equity: float
    evaluations_count: int = 1


@dataclass
class SearchResult:
    """Result of searching for best move.

    Attributes:
        best_move: The best move found
        best_equity: Expected equity of best move
        all_moves: All evaluated moves, sorted by equity
        positions_evaluated: Total positions evaluated
        time_ms: Search time in milliseconds
    """
    best_move: Move
    best_equity: float
    all_moves: List[MoveEvaluation]
    positions_evaluated: int
    time_ms: float


# ==============================================================================
# UTILITY TYPES
# ==============================================================================

# Result type for error handling (simplified - could use typing.Result in Python 3.12+)
Result = Union[Tuple[bool, any], Tuple[bool, str]]  # (success, value_or_error)
