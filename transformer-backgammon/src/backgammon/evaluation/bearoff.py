"""One-sided bearoff database (gnubg-style).

Precomputes, for every one-sided bearoff position (up to 15 checkers on
the 6 home-board points), the exact probability distribution over the
number of rolls needed to bear off all checkers under near-optimal play
(the move minimizing expected rolls is chosen for every roll — the same
convention as GNU Backgammon's one-sided database).

From two one-sided distributions the exact win probability of a pure
bearoff-vs-bearoff race follows by convolution: the player on roll wins
iff they need no more rolls than their opponent (X <= Y), since they
roll first.

Position space: distributions of 0..15 checkers over 6 points =
C(21, 6) = 54,264 positions. The full database builds in a couple of
minutes in pure Python and is cached to disk (~7 MB) after the first
build.

Limitations (documented, deliberate for v1):
- One-sided: each side's roll distribution assumes play that minimizes
  its own expected rolls. In rare positions, maximizing win probability
  against a specific opponent distribution differs slightly.
- Gammons are ignored by bearoff_equity (both sides being in their home
  boards makes gammons rare but not impossible).
"""

import os
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np

from backgammon.core.types import Board, Player, Move, Dice, LegalMoves
from backgammon.core.board import apply_move, can_bear_off
from backgammon.core.dice import ALL_DICE_ROLLS, DICE_PROBABILITIES

# Home board has 6 points; distances 1-6 from bearing off.
N_POINTS = 6
MAX_CHECKERS = 15

# Upper bound on rolls to bear off. The worst position (15 checkers on
# the 6-point, 90 pips) needs well under 32 rolls even with minimal
# dice; the build asserts no probability mass is lost to truncation.
MAX_ROLLS = 32

# Number of positions with c checkers (sum <= s) on k points: C(s+k, k)
TOTAL_POSITIONS = comb(MAX_CHECKERS + N_POINTS, N_POINTS)  # 54,264

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "backgammon"
)


# ==============================================================================
# POSITION INDEXING (combinatorial ranking)
# ==============================================================================

# _N_LE[k][s] = number of k-tuples of non-negative ints with sum <= s
_N_LE = [
    [comb(s + k, k) for s in range(MAX_CHECKERS + 1)]
    for k in range(N_POINTS + 1)
]


def position_index(counts: Tuple[int, ...]) -> int:
    """Rank a bearoff position (checkers per point, distances 1-6).

    Positions are ranked lexicographically among all 6-tuples with
    sum <= 15. The empty position () -> 0 is "all checkers borne off".

    Args:
        counts: Tuple of 6 checker counts, counts[i] = checkers at
            distance i+1 from bearing off.

    Returns:
        Index in [0, TOTAL_POSITIONS).
    """
    rank = 0
    remaining = MAX_CHECKERS
    for i in range(N_POINTS):
        c = counts[i]
        k = N_POINTS - i - 1
        # Tuples that start with a smaller count at this position
        for v in range(c):
            rank += _N_LE[k][remaining - v]
        remaining -= c
    return rank


def enumerate_positions(max_checkers: int = MAX_CHECKERS) -> List[Tuple[int, ...]]:
    """Enumerate all bearoff positions with at most max_checkers checkers."""
    positions = []

    def rec(prefix: List[int], remaining: int) -> None:
        if len(prefix) == N_POINTS:
            positions.append(tuple(prefix))
            return
        for c in range(remaining + 1):
            rec(prefix + [c], remaining - c)

    rec([], max_checkers)
    return positions


# ==============================================================================
# ONE-SIDED MOVE GENERATION
# ==============================================================================


def _die_successors(counts: Tuple[int, ...], die: int) -> List[Tuple[int, ...]]:
    """All positions reachable by playing one die in pure bearoff.

    Rules (one-sided, all checkers in the home board):
    - Bear off from point `die` if occupied.
    - Move any checker from point p > die to point p - die.
    - If no checker on `die` or higher, bear off from the highest
      occupied point (the "overage" rule).

    In pure bearoff a die is always playable while checkers remain.

    Args:
        counts: Position (checkers per distance 1-6).
        die: Die value 1-6.

    Returns:
        List of successor positions (may contain duplicates).
    """
    succs = []

    # Bear off exactly
    if counts[die - 1] > 0:
        c = list(counts)
        c[die - 1] -= 1
        succs.append(tuple(c))

    # Move within the board: p -> p - die
    for p in range(die + 1, N_POINTS + 1):
        if counts[p - 1] > 0:
            c = list(counts)
            c[p - 1] -= 1
            c[p - 1 - die] += 1
            succs.append(tuple(c))

    if not succs:
        # No checker on `die` or higher: bear off from highest occupied
        for p in range(die - 1, 0, -1):
            if counts[p - 1] > 0:
                c = list(counts)
                c[p - 1] -= 1
                succs.append(tuple(c))
                break

    return succs


def _roll_successors(
    counts: Tuple[int, ...], roll: Dice
) -> List[Tuple[int, ...]]:
    """All positions reachable by playing a full dice roll in pure bearoff.

    Args:
        counts: Position before the roll.
        roll: Dice roll (a, b); doubles play four moves.

    Returns:
        List of unique successor positions.
    """
    a, b = roll
    orders = [(a, a, a, a)] if a == b else [(a, b), (b, a)]

    out = set()
    for order in orders:
        states = {counts}
        for die in order:
            nxt = set()
            for s in states:
                if sum(s) == 0:
                    nxt.add(s)  # Already borne off; extra pips unused
                else:
                    nxt.update(_die_successors(s, die))
            states = nxt
        out |= states
    return list(out)


# ==============================================================================
# DATABASE
# ==============================================================================


class BearoffDatabase:
    """Exact one-sided bearoff roll distributions.

    Attributes:
        distributions: (TOTAL_POSITIONS, MAX_ROLLS) float32 array;
            distributions[i, n] = P(position i bears off in exactly n
            rolls) under expected-rolls-minimizing play. Rows for
            positions above max_checkers are all zero.
        expected_rolls: (TOTAL_POSITIONS,) float32 array of expected
            rolls to bear off.
        max_checkers: Number of checkers the build covered.
    """

    def __init__(
        self,
        distributions: np.ndarray,
        expected_rolls: np.ndarray,
        max_checkers: int,
    ):
        self.distributions = distributions
        self.expected_rolls = expected_rolls
        self.max_checkers = max_checkers

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, max_checkers: int = MAX_CHECKERS) -> "BearoffDatabase":
        """Build the database by dynamic programming.

        Positions are processed in increasing pip order; every dice roll
        strictly reduces the pip count, so all successors are already
        solved when a position is reached.

        Args:
            max_checkers: Limit the build to positions with at most this
                many checkers (successors stay within the limit, so the
                subset is closed). 15 = full database.

        Returns:
            Built database.
        """
        dist = np.zeros((TOTAL_POSITIONS, MAX_ROLLS), dtype=np.float32)
        exp = np.zeros(TOTAL_POSITIONS, dtype=np.float32)

        positions = enumerate_positions(max_checkers)
        positions.sort(key=lambda c: sum((i + 1) * c[i] for i in range(N_POINTS)))

        rolls = [(roll, DICE_PROBABILITIES[roll]) for roll in ALL_DICE_ROLLS]

        for counts in positions:
            idx = position_index(counts)
            if sum(counts) == 0:
                dist[idx, 0] = 1.0
                exp[idx] = 0.0
                continue

            dist_acc = np.zeros(MAX_ROLLS, dtype=np.float64)
            exp_acc = 0.0

            for roll, prob in rolls:
                succs = _roll_successors(counts, roll)
                # Near-optimal play: minimize expected remaining rolls
                best = min(succs, key=lambda s: exp[position_index(s)])
                best_idx = position_index(best)
                exp_acc += prob * float(exp[best_idx])
                dist_acc[1:] += prob * dist[best_idx, :-1]

            exp[idx] = 1.0 + exp_acc
            dist[idx] = dist_acc.astype(np.float32)

            # Truncation check: all probability mass must fit in MAX_ROLLS
            total = float(dist_acc.sum())
            if abs(total - 1.0) > 1e-6:
                raise RuntimeError(
                    f"Bearoff distribution for {counts} sums to {total}; "
                    f"increase MAX_ROLLS"
                )

        return cls(dist, exp, max_checkers)

    def save(self, path: str) -> None:
        """Save the database to a compressed .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            distributions=self.distributions,
            expected_rolls=self.expected_rolls,
            max_checkers=np.array([self.max_checkers]),
        )

    @classmethod
    def load(cls, path: str) -> "BearoffDatabase":
        """Load a database saved by save()."""
        data = np.load(path)
        return cls(
            distributions=data["distributions"],
            expected_rolls=data["expected_rolls"],
            max_checkers=int(data["max_checkers"][0]),
        )

    @classmethod
    def load_or_build(
        cls,
        path: Optional[str] = None,
        max_checkers: int = MAX_CHECKERS,
    ) -> "BearoffDatabase":
        """Load the cached database, building and caching it if missing.

        Args:
            path: Cache file path. Defaults to
                ~/.cache/backgammon/bearoff_<max_checkers>.npz.
            max_checkers: Build limit if the cache is missing. A cached
                file built with fewer checkers than requested is rebuilt.

        Returns:
            Loaded or freshly built database.
        """
        if path is None:
            path = os.path.join(
                _DEFAULT_CACHE_DIR, f"bearoff_{max_checkers}.npz"
            )

        if os.path.exists(path):
            db = cls.load(path)
            if db.max_checkers >= max_checkers:
                return db

        db = cls.build(max_checkers)
        db.save(path)
        return db

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def roll_distribution(self, counts: Tuple[int, ...]) -> np.ndarray:
        """P(exactly n rolls to bear off) for n in [0, MAX_ROLLS)."""
        self._check(counts)
        return self.distributions[position_index(counts)]

    def rolls_to_bear_off(self, counts: Tuple[int, ...]) -> float:
        """Expected number of rolls to bear off all checkers."""
        self._check(counts)
        return float(self.expected_rolls[position_index(counts)])

    def win_probability(
        self,
        counts_on_roll: Tuple[int, ...],
        counts_opponent: Tuple[int, ...],
    ) -> float:
        """Exact P(player on roll wins) in a mutual bearoff race.

        The player on roll needs X rolls, the opponent Y rolls; rolling
        first, the player on roll wins iff X <= Y.

        Args:
            counts_on_roll: Home-board counts of the player on roll.
            counts_opponent: Home-board counts of the opponent.

        Returns:
            Win probability in [0, 1].
        """
        x = self.roll_distribution(counts_on_roll)
        y = self.roll_distribution(counts_opponent)
        # P(Y >= n) for each n
        y_ge = np.concatenate([np.cumsum(y[::-1])[::-1], [0.0]])
        return float(np.dot(x, y_ge[: len(x)]))

    def _check(self, counts: Tuple[int, ...]) -> None:
        if len(counts) != N_POINTS:
            raise ValueError(f"Expected {N_POINTS} counts, got {len(counts)}")
        if sum(counts) > self.max_checkers:
            raise ValueError(
                f"Position has {sum(counts)} checkers but database was "
                f"built with max_checkers={self.max_checkers}"
            )


# ==============================================================================
# ENGINE INTEGRATION
# ==============================================================================


def home_board_counts(board: Board, player: Player) -> Optional[Tuple[int, ...]]:
    """Extract a player's home board as bearoff counts (distances 1-6).

    Args:
        board: Board state.
        player: Player to extract.

    Returns:
        6-tuple of counts indexed by distance from bearing off, or None
        if the player has checkers outside their home board (or on the
        bar) and therefore cannot bear off yet.
    """
    if not can_bear_off(board, player):
        return None

    if player == Player.WHITE:
        # White's home is points 1-6; distance = point number
        return tuple(int(board.white_checkers[p]) for p in range(1, 7))
    # Black's home is points 19-24; distance = 25 - point
    return tuple(int(board.black_checkers[25 - d]) for d in range(1, 7))


def bearoff_win_probability(
    db: BearoffDatabase, board: Board
) -> Optional[float]:
    """Exact win probability for board.player_to_move in a mutual bearoff.

    Args:
        db: Bearoff database.
        board: Board state; both players must be able to bear off (which
            in this engine implies a pure race).

    Returns:
        P(player_to_move wins), or None if either player still has
        checkers outside their home board.
    """
    mover = board.player_to_move
    ours = home_board_counts(board, mover)
    theirs = home_board_counts(board, mover.opponent())
    if ours is None or theirs is None:
        return None
    return db.win_probability(ours, theirs)


def bearoff_equity(db: BearoffDatabase, board: Board) -> Optional[float]:
    """Cubeless equity (2p - 1) for board.player_to_move in mutual bearoff.

    Gammons are ignored (rare once both sides bear off; see module
    docstring).

    Returns:
        Equity in [-1, 1], or None if not a mutual bearoff position.
    """
    p = bearoff_win_probability(db, board)
    if p is None:
        return None
    return 2.0 * p - 1.0


def select_bearoff_move(
    db: BearoffDatabase,
    board: Board,
    player: Player,
    legal_moves: LegalMoves,
) -> Optional[Move]:
    """Pick the database-optimal move in a bearoff position.

    If both players are bearing off, maximizes the exact win probability
    (after our move the opponent is on roll: we win iff X' < Y, so we
    maximize P(X' < Y), tie-broken by expected rolls). If only we are
    bearing off, minimizes expected rolls.

    Args:
        db: Bearoff database.
        board: Current board state.
        player: Player to move.
        legal_moves: Engine-generated legal moves.

    Returns:
        Best move, or None if the player cannot bear off yet (caller
        should fall back to another agent).
    """
    if not legal_moves:
        return None
    if home_board_counts(board, player) is None:
        return None

    opp_counts = home_board_counts(board, player.opponent())
    opp_y_gt: Optional[np.ndarray] = None
    if opp_counts is not None:
        y = db.roll_distribution(opp_counts)
        # P(Y > n) for each n
        tail = np.cumsum(y[::-1])[::-1]
        opp_y_gt = np.concatenate([tail[1:], [0.0]])

    best_move = None
    best_key = None

    for move in legal_moves:
        after = apply_move(board, player, move)
        counts = home_board_counts(after, player)
        # Bearing off keeps all checkers in the home board
        assert counts is not None

        exp = db.rolls_to_bear_off(counts)
        if opp_y_gt is not None:
            x = db.roll_distribution(counts)
            win_p = float(np.dot(x, opp_y_gt[: len(x)]))
            key = (-win_p, exp)
        else:
            key = (0.0, exp)

        if best_key is None or key < best_key:
            best_key = key
            best_move = move

    return best_move


# ==============================================================================
# SHARED DATABASE FOR EXACT ENDGAME EVALUATION
# ==============================================================================
#
# Search (evaluation/search.py), batched self-play (training/self_play.py),
# and the neural agent's equity estimates consult a process-wide shared
# database to replace network evaluation with exact values in mutual-bearoff
# positions — the same trick gnubg uses. Exact endgame values sharpen both
# play and the TD training targets for everything upstream of the bearoff.
#
# Opt-in: nothing is loaded or built until enable_exact_bearoff() is called
# (train() does this when TrainingConfig.use_exact_bearoff is set), so unit
# tests and network-only experiments pay no cost.

_shared_db: Optional[BearoffDatabase] = None


def enable_exact_bearoff(
    db: Optional[BearoffDatabase] = None,
    path: Optional[str] = None,
    max_checkers: int = MAX_CHECKERS,
) -> BearoffDatabase:
    """Enable exact bearoff evaluation in search and self-play.

    Args:
        db: Database to share. If None, load_or_build() is used (first
            call on a machine builds the full database in ~1 minute and
            caches it; later calls load the cache in milliseconds).
        path: Cache file path forwarded to load_or_build() when db is None.
        max_checkers: Build limit forwarded to load_or_build() when db is
            None. Positions with more checkers per side fall back to the
            network.

    Returns:
        The now-shared database.
    """
    global _shared_db
    if db is None:
        db = BearoffDatabase.load_or_build(path=path, max_checkers=max_checkers)
    _shared_db = db
    return db


def disable_exact_bearoff() -> None:
    """Disable exact bearoff evaluation (revert to pure network)."""
    global _shared_db
    _shared_db = None


def get_exact_bearoff_db() -> Optional[BearoffDatabase]:
    """The shared bearoff database, or None when exact evaluation is off."""
    return _shared_db


def _exact_bearoff_counts(
    db: BearoffDatabase, board: Board
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Home-board counts (mover's, opponent's) when the position is
    EXACTLY evaluable from the one-sided database, else None.

    Exactness requires:
    - Both players bearing off (in this engine the home boards are
      disjoint, so mutual bearoff implies a pure race).
    - Both players have borne off at least one checker, so gammons and
      backgammons are impossible and the win probability fully
      determines the equity (the database ignores gammons).
    - Both sides within db.max_checkers (only relevant for partial
      builds; the full database covers every mutual-bearoff position).
    """
    # Cheap pre-filter: rejects everything outside the late endgame
    # with two scalar reads before the per-point can_bear_off scans.
    if board.white_checkers[25] < 1 or board.black_checkers[25] < 1:
        return None

    mover = board.player_to_move
    ours = home_board_counts(board, mover)
    if ours is None or sum(ours) > db.max_checkers:
        return None
    theirs = home_board_counts(board, mover.opponent())
    if theirs is None or sum(theirs) > db.max_checkers:
        return None
    return ours, theirs


def exact_bearoff_value(db: BearoffDatabase, board: Board) -> Optional[float]:
    """Exact scalar value for board.player_to_move, or None.

    Returns 2p - 1 in [-1, 1] (expected points: gammons are impossible
    under the exactness gate, so win/loss probability is the whole
    story), or None when the position is not exactly evaluable and the
    caller should fall back to the network.
    """
    counts = _exact_bearoff_counts(db, board)
    if counts is None:
        return None
    return 2.0 * db.win_probability(*counts) - 1.0


def exact_bearoff_equity6(
    db: BearoffDatabase, board: Board
) -> Optional[np.ndarray]:
    """Exact 6-dim equity distribution for board.player_to_move, or None.

    Returns [win_n, win_g, win_bg, lose_n, lose_g, lose_bg] with the
    gammon/backgammon slots exactly zero (impossible under the exactness
    gate), or None when the caller should fall back to the network.
    """
    counts = _exact_bearoff_counts(db, board)
    if counts is None:
        return None
    p = db.win_probability(*counts)
    equity = np.zeros(6, dtype=np.float32)
    equity[0] = p
    equity[3] = 1.0 - p
    return equity


def bearoff_agent(
    db: BearoffDatabase,
    fallback=None,
    name: str = "Bearoff",
):
    """Create an agent playing database-perfect bearoffs.

    Uses the bearoff database whenever the player to move can bear off;
    otherwise delegates to the fallback agent.

    Args:
        db: Bearoff database.
        fallback: Agent used outside bearoff (defaults to
            pip_count_agent()).
        name: Agent name.

    Returns:
        Agent instance.
    """
    from backgammon.evaluation.agents import Agent, pip_count_agent

    if fallback is None:
        fallback = pip_count_agent()

    def select(
        board: Board, player: Player, dice: Dice, legal_moves: LegalMoves
    ) -> Move:
        if not legal_moves:
            return ()
        move = select_bearoff_move(db, board, player, legal_moves)
        if move is not None:
            return move
        return fallback.select_move(board, player, dice, legal_moves)

    return Agent(name=name, select_move_fn=select)
