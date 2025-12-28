"""Action space encoding for backgammon moves.

Maps moves to/from fixed-size action space for neural network training.
Uses a deterministic encoding scheme based on checker movements.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from backgammon.core.types import Move, MoveStep, Player

# Action space configuration
ACTION_SPACE_SIZE = 1024  # Fixed size for neural network output

# Maximum move sequence length in backgammon (usually 4 for doubles)
MAX_MOVE_STEPS = 4


def encode_move_to_action(move: Move) -> int:
    """Encode a move to a fixed action index.

    Uses a deterministic encoding based on the sequence of checker movements.
    The encoding is:
    - Empty move (pass) -> 0
    - Each move step contributes to the hash based on (from_point, to_point)
    - Use a stable hash to ensure consistency

    Args:
        move: Tuple of MoveSteps representing the move

    Returns:
        Action index in range [0, ACTION_SPACE_SIZE)
    """
    if not move:
        return 0  # Pass/no move

    # Create a deterministic hash from the move sequence
    # We'll use the sequence of (from, to) pairs
    move_signature = []
    for step in move:
        # MoveStep is a NamedTuple with attributes: from_point, to_point, die_used, hits_opponent
        # Handle both NamedTuple and regular tuple formats
        if hasattr(step, 'from_point'):
            from_point, to_point = step.from_point, step.to_point
        else:
            from_point, to_point = step[0], step[1]
        move_signature.append((from_point, to_point))

    # Sort to ensure canonical ordering for equivalent moves
    # (though in practice, move generation should already be ordered)
    move_signature = tuple(sorted(move_signature))

    # Compute stable hash
    # Use a custom hash that's deterministic across runs
    hash_value = _stable_hash(move_signature)

    # Map to action space
    return hash_value % ACTION_SPACE_SIZE


def create_action_mask(legal_moves: List[Move]) -> np.ndarray:
    """Create a boolean mask indicating legal actions.

    Args:
        legal_moves: List of legal moves in current position

    Returns:
        Boolean array of shape (ACTION_SPACE_SIZE,) where True indicates
        a legal action
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

    for move in legal_moves:
        action_idx = encode_move_to_action(move)
        mask[action_idx] = True

    return mask


def create_move_to_action_map(legal_moves: List[Move]) -> Dict[int, Move]:
    """Create mapping from action indices to moves.

    Args:
        legal_moves: List of legal moves

    Returns:
        Dictionary mapping action_idx -> move
    """
    action_map = {}
    for move in legal_moves:
        action_idx = encode_move_to_action(move)
        action_map[action_idx] = move

    return action_map


def encode_move_to_one_hot(
    move: Move,
    legal_moves: List[Move],
) -> np.ndarray:
    """Encode a move as a one-hot policy over legal moves.

    Args:
        move: The move that was played
        legal_moves: All legal moves in the position

    Returns:
        One-hot array of shape (ACTION_SPACE_SIZE,) with 1.0 at the
        action index and 0.0 elsewhere
    """
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

    # Get action index for the played move
    action_idx = encode_move_to_action(move)
    policy[action_idx] = 1.0

    return policy


def create_uniform_policy(legal_moves: List[Move]) -> np.ndarray:
    """Create uniform policy over legal moves.

    Useful for random play or initialization.

    Args:
        legal_moves: List of legal moves

    Returns:
        Uniform probability distribution over legal actions
    """
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

    if not legal_moves:
        return policy

    # Compute uniform probability
    prob = 1.0 / len(legal_moves)

    for move in legal_moves:
        action_idx = encode_move_to_action(move)
        policy[action_idx] = prob

    return policy


def select_move_from_policy(
    policy_probs: np.ndarray,
    legal_moves: List[Move],
    temperature: float = 1.0,
) -> Move:
    """Select a move by sampling from policy distribution.

    Args:
        policy_probs: Policy probabilities (ACTION_SPACE_SIZE,)
        legal_moves: List of legal moves
        temperature: Sampling temperature (0 = greedy, higher = more random)

    Returns:
        Selected move from legal_moves
    """
    if not legal_moves:
        return ()

    if len(legal_moves) == 1:
        return legal_moves[0]

    # Get probabilities for each legal move
    move_probs = []
    for move in legal_moves:
        action_idx = encode_move_to_action(move)
        move_probs.append(policy_probs[action_idx])

    move_probs = np.array(move_probs)

    # Apply temperature
    if temperature == 0.0:
        # Greedy selection
        best_idx = np.argmax(move_probs)
        return legal_moves[best_idx]
    else:
        # Temperature scaling
        move_probs = move_probs ** (1.0 / temperature)
        move_probs = move_probs / move_probs.sum()

        # Sample
        idx = np.random.choice(len(legal_moves), p=move_probs)
        return legal_moves[idx]


def _stable_hash(obj: Tuple) -> int:
    """Compute a stable hash that's consistent across Python runs.

    Args:
        obj: Object to hash (must be hashable)

    Returns:
        Stable integer hash value
    """
    # Python's hash() is not stable across runs (uses random seed)
    # We need a deterministic hash for reproducibility

    # Convert to string and hash that
    s = str(obj)

    # Simple polynomial rolling hash
    hash_value = 0
    prime = 31
    mod = 2**31 - 1  # Large prime

    for char in s:
        hash_value = (hash_value * prime + ord(char)) % mod

    return hash_value


def get_action_space_size() -> int:
    """Get the size of the action space.

    Returns:
        Size of the action space (fixed)
    """
    return ACTION_SPACE_SIZE


def analyze_move_collisions(moves: List[Move]) -> Dict[str, any]:
    """Analyze hash collisions in a set of moves.

    Useful for debugging and validation.

    Args:
        moves: List of moves to analyze

    Returns:
        Dictionary with collision statistics
    """
    action_indices = [encode_move_to_action(move) for move in moves]

    unique_indices = set(action_indices)
    num_collisions = len(action_indices) - len(unique_indices)

    # Find colliding moves
    index_to_moves = {}
    for move, idx in zip(moves, action_indices):
        if idx not in index_to_moves:
            index_to_moves[idx] = []
        index_to_moves[idx].append(move)

    colliding_groups = {
        idx: moves_list
        for idx, moves_list in index_to_moves.items()
        if len(moves_list) > 1
    }

    return {
        'total_moves': len(moves),
        'unique_actions': len(unique_indices),
        'num_collisions': num_collisions,
        'collision_rate': num_collisions / len(moves) if moves else 0.0,
        'colliding_groups': colliding_groups,
        'action_space_utilization': len(unique_indices) / ACTION_SPACE_SIZE,
    }


# Additional encoding strategies for future improvements

def encode_move_structured(move: Move, player: Player) -> Tuple[int, ...]:
    """Encode move as structured tuple (alternative encoding).

    This is a more explicit encoding that could be used for
    more sophisticated action representations.

    Args:
        move: Move to encode
        player: Player making the move

    Returns:
        Tuple representing the move structure
    """
    if not move:
        return (0,)  # Pass

    # Encode each step
    encoding = []
    for step in move:
        # Handle both NamedTuple and regular tuple formats
        if hasattr(step, 'from_point'):
            from_point = step.from_point
            to_point = step.to_point
            die_value = step.die_used
            hits = step.hits_opponent
        else:
            from_point, to_point, die_value, hits = step

        # Encode step: pack into single integer
        # Format: from_point (6 bits) | to_point (6 bits) | die (3 bits) | hit (1 bit)
        step_code = (
            (from_point & 0x3F) << 10 |
            (to_point & 0x3F) << 4 |
            (die_value & 0x7) << 1 |
            (1 if hits else 0)
        )
        encoding.append(step_code)

    # Pad to fixed length
    while len(encoding) < MAX_MOVE_STEPS:
        encoding.append(0)

    return tuple(encoding[:MAX_MOVE_STEPS])


def decode_move_structured(encoding: Tuple[int, ...]) -> Move:
    """Decode structured encoding back to move.

    Args:
        encoding: Tuple encoding from encode_move_structured

    Returns:
        Decoded move (tuple of MoveSteps)
    """
    steps = []

    for step_code in encoding:
        if step_code == 0:
            continue  # Padding

        # Unpack step
        from_point = (step_code >> 10) & 0x3F
        to_point = (step_code >> 4) & 0x3F
        die_value = (step_code >> 1) & 0x7
        hits = bool(step_code & 1)

        steps.append((from_point, to_point, die_value, hits))

    return tuple(steps)
