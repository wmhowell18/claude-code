"""Checker-play rules: legal move generation, application, equivalence (PLAN.md §4.3).

All logic is expressed in the mover-relative board frame documented in
:mod:`bgcore.board`: the mover advances from low indices to high indices, enters
from the bar on index ``d`` (die value), and bears off *beyond* index 24 from the
home board (indices 19..24).

Rules implemented
-----------------
* Bar entry has priority: while the mover has checkers on the bar it must enter
  them (on an open entry point) before any other checker moves.
* Hitting: landing on a point holding exactly one opponent checker sends it to
  the opponent's bar.
* Bear-off, including the highest-point / overflow rule: a die larger than the
  highest occupied home point may bear a checker off that highest point; an exact
  die always bears off its point; otherwise the die must be played within the board.
* Must play both dice when possible; when only one die can be played, the larger
  die must be used if that is legal; doubles play up to four checkers.

Move *equivalence* is by resulting position: two move strings are equivalent iff
they leave the checkers in the same place (PLAN.md §4.3). This is realised by
matching a parsed move against the enumerated legal moves (which carry correct
intermediate hits) and comparing the resulting layout.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bgcore.board import HOME_END, HOME_START, Board


class IllegalMove(ValueError):
    """Raised when a move string is not legal in the given position."""


# A hop is a single die application:
#   {"frm": int|"bar", "to": int|"off", "hit": bool, "die": int}
Hop = dict[str, Any]


@dataclass
class LegalMove:
    """One legal full move (all dice consumed as required)."""

    notation: str
    hops: list[Hop]
    points: list[int]
    bar: dict[str, int]
    off: dict[str, int]
    endpoint_multiset: Counter = field(default_factory=Counter)
    hit_points: frozenset = frozenset()

    def signature(self) -> tuple:
        """Resulting-layout signature used for equivalence comparisons."""
        return (tuple(self.points), self.bar["x"], self.bar["o"], self.off["x"])


# -- low-level state helpers ---------------------------------------------


def _all_home(points: list[int], bar_x: int) -> bool:
    if bar_x > 0:
        return False
    for i in range(1, HOME_START):
        if points[i] > 0:
            return False
    return True


def _legal_hops(points: list[int], bar_x: int, d: int) -> list[Hop]:
    hops: list[Hop] = []
    if bar_x > 0:
        j = d  # bar entry lands on index == die
        if points[j] >= -1:
            hops.append({"frm": "bar", "to": j, "hit": points[j] == -1, "die": d})
        return hops  # must enter from the bar first
    home = _all_home(points, bar_x)
    min_home = None
    if home:
        for i in range(HOME_START, HOME_END + 1):
            if points[i] > 0:
                min_home = i
                break
    for p in range(1, 25):
        if points[p] <= 0:
            continue
        dest = p + d
        if dest <= 24:
            if points[dest] >= -1:
                hops.append({"frm": p, "to": dest, "hit": points[dest] == -1, "die": d})
        elif home:
            if dest == 25:  # exact bear-off
                hops.append({"frm": p, "to": "off", "hit": False, "die": d})
            elif min_home is not None and p == min_home:  # overflow off highest point
                hops.append({"frm": p, "to": "off", "hit": False, "die": d})
    return hops


def _apply_hop(state: tuple, hop: Hop) -> tuple:
    points, bar_x, bar_o, off_x = state
    np = list(points)
    if hop["frm"] == "bar":
        bar_x -= 1
    else:
        np[hop["frm"]] -= 1
    if hop["to"] == "off":
        off_x += 1
    else:
        j = hop["to"]
        if np[j] < 0:  # opponent blot -> bar
            np[j] = 0
            bar_o += 1
        np[j] += 1
    return (np, bar_x, bar_o, off_x)


# -- notation chain building (mover-relative index -> notation point) -----


def _pt(x: Any) -> Any:
    if x == "bar":
        return "bar"
    if x == "off":
        return "off"
    return 25 - x  # board index -> notation point number


def _chains_from_hops(hops: list[Hop]) -> list[dict]:
    """Merge single-die hops of the same checker into slash chains (notation coords)."""
    chains: list[dict] = []
    for h in hops:
        a = _pt(h["frm"])
        b = _pt(h["to"])
        attached = False
        if a != "bar":
            for ch in chains:
                if ch["pts"][-1] == a:
                    ch["pts"].append(b)
                    ch["hits"].append(h["hit"])
                    attached = True
                    break
        if not attached:
            chains.append({"pts": [a, b], "hits": [h["hit"]]})
    return chains


def _render_chain(ch: dict) -> str:
    pts = ch["pts"]
    hits = ch["hits"]
    out = str(pts[0])
    for i, p in enumerate(pts[1:]):
        out += "/" + str(p) + ("*" if hits[i] else "")
    return out


def _tok_sort_key(tok: str) -> tuple:
    head = tok.split("/", 1)[0]
    start = 25 if head == "bar" else int(head)
    return (-start, tok)


def format_hops(hops: list[Hop]) -> str:
    """Render a list of hops as canonical XG-style move notation."""
    if not hops:
        return "Cannot Move"
    chains = _chains_from_hops(hops)
    toks = [_render_chain(ch) for ch in chains]
    cnt = Counter(toks)
    parts = []
    for tok in sorted(set(toks), key=_tok_sort_key):
        n = cnt[tok]
        parts.append(f"{tok}({n})" if n > 1 else tok)
    return " ".join(parts)


def _endpoint_multiset(hops: list[Hop]) -> Counter:
    chains = _chains_from_hops(hops)
    return Counter((ch["pts"][0], ch["pts"][-1]) for ch in chains)


def _hit_points(hops: list[Hop]) -> frozenset:
    pts = set()
    for h in hops:
        if h["hit"]:
            pts.add(_pt(h["to"]))
    return frozenset(pts)


# -- full-move generation -------------------------------------------------


def _dice_sequence(dice: list[int]) -> tuple[int, ...]:
    if len(dice) == 2 and dice[0] == dice[1]:
        return (dice[0],) * 4
    return tuple(dice)


def _recurse(state: tuple, remaining: tuple, hops: list[Hop], results: list) -> None:
    points, bar_x, _bar_o, _off_x = state
    tried: set[int] = set()
    for i, d in enumerate(remaining):
        if d in tried:
            continue
        tried.add(d)
        for hop in _legal_hops(points, bar_x, d):
            nstate = _apply_hop(state, hop)
            _recurse(nstate, remaining[:i] + remaining[i + 1 :], hops + [hop], results)
    results.append((len(hops), hops, state))


def generate_moves(board: Board) -> list[LegalMove]:
    """Return every legal full move for ``board.dice`` (deduped by end position).

    Result is sorted by notation for determinism. An empty list means the mover
    has no legal play ("Cannot Move").
    """
    if not board.dice:
        return []
    seq = _dice_sequence(board.dice)
    start_state = (list(board.points), int(board.bar["x"]), int(board.bar["o"]), int(board.off["x"]))
    results: list = []
    _recurse(start_state, seq, [], results)

    max_len = max(r[0] for r in results)
    if max_len == 0:
        return []
    keep = [r for r in results if r[0] == max_len]

    # Larger-die rule: if at most one die can ever be played, the larger die must
    # be used when doing so is legal.
    if max_len == 1 and len(board.dice) == 2 and board.dice[0] != board.dice[1]:
        larger = max(board.dice)
        if any(r[1][0]["die"] == larger for r in keep):
            keep = [r for r in keep if r[1][0]["die"] == larger]

    # Dedup by resulting layout; among equivalent hop-sequences pick the one with
    # the lexicographically smallest notation for determinism.
    best: dict[tuple, LegalMove] = {}
    for _n, hops, state in keep:
        points, bar_x, bar_o, off_x = state
        mv = LegalMove(
            notation=format_hops(hops),
            hops=hops,
            points=points,
            bar={"x": bar_x, "o": bar_o},
            off={"x": off_x, "o": int(board.off["o"])},
            endpoint_multiset=_endpoint_multiset(hops),
            hit_points=_hit_points(hops),
        )
        sig = mv.signature()
        if sig not in best or mv.notation < best[sig].notation:
            best[sig] = mv
    return sorted(best.values(), key=lambda m: m.notation)


def legal_moves(board: Board) -> list[str]:
    """Convenience: sorted notation strings for all legal moves (``[]`` if none)."""
    return [m.notation for m in generate_moves(board)]


# -- applying / matching a move string ------------------------------------


def _applied_signature(board: Board, tokens: list[dict]):
    """Resulting-layout signature of applying a parsed move's chains verbatim.

    Walks each token's point chain as consecutive single-die hops (converting
    notation points to board indices) and applies them, detecting intermediate
    hits exactly as :func:`_apply_hop` does. Because a full move is identified by
    the position it reaches (module docstring), this lets :func:`_match` recognise
    a spelling whose per-checker endpoint multiset differs from the representative
    :func:`generate_moves` kept (e.g. a single checker written ``10/3`` versus the
    equivalent two-checker ``10/4 4/3``). Returns ``None`` if a coordinate is out
    of range / underflows, in which case the caller falls back to "no match".
    """
    points = list(board.points)
    bar_x = int(board.bar["x"])
    bar_o = int(board.bar["o"])
    off_x = int(board.off["x"])
    for tok in tokens:
        chain = tok["chain"]
        for a, b in zip(chain, chain[1:]):
            if a == "bar":
                if bar_x <= 0:
                    return None
                bar_x -= 1
            else:
                ai = 25 - a
                if ai < 0 or ai > 25 or points[ai] <= 0:
                    return None
                points[ai] -= 1
            if b == "off":
                off_x += 1
            else:
                j = 25 - b
                if j < 0 or j > 25:
                    return None
                if points[j] < 0:
                    points[j] = 0
                    bar_o += 1
                points[j] += 1
    return (tuple(points), bar_x, bar_o, off_x)


def _match(board: Board, move_str: str):
    """Find the generated :class:`LegalMove` a notation string denotes, or ``None``.

    Matching is by the multiset of per-checker (start, end) endpoints, so any die
    ordering or combined-die spelling that names the same checkers matches; ``*``
    markers only disambiguate when several legal moves share endpoints. When no
    generated move shares the endpoint multiset (a single checker written through
    an intermediate resting point yields a different multiset than the equivalent
    two-checker representative :func:`generate_moves` kept), fall back to matching
    by the resulting layout — the move's true identity — so both spellings resolve
    to the same legal move.
    """
    from bgcore import notation as _notation

    tokens = _notation.parse_move(move_str)
    if tokens is None:  # "Cannot Move"
        return "cannot-move"
    want = Counter()
    want_hits = set()
    for tok in tokens:
        want[(tok["from"], tok["to"])] += 1
        for hp in tok["hits"]:
            want_hits.add(hp)

    legal = generate_moves(board)
    candidates = [m for m in legal if m.endpoint_multiset == want]
    if not candidates:
        # Resulting-position fallback: the same legal end layout spelled with a
        # different per-checker endpoint decomposition (e.g. "10/3" vs "10/4 4/3").
        sig = _applied_signature(board, tokens)
        if sig is not None:
            by_sig = [m for m in legal if m.signature() == sig]
            if by_sig:
                return by_sig[0]  # signatures are unique across generate_moves
        return None
    if len(candidates) == 1:
        return candidates[0]
    exact = [m for m in candidates if set(m.hit_points) == want_hits]
    if exact:
        return sorted(exact, key=lambda m: m.notation)[0]
    return sorted(candidates, key=lambda m: m.notation)[0]


def is_legal(board: Board, move_str: str) -> bool:
    """True if ``move_str`` is a legal play in ``board``."""
    m = _match(board, move_str)
    if m == "cannot-move":
        return len(generate_moves(board)) == 0
    return m is not None


def apply_move(board: Board, move_str: str) -> Board:
    """Apply ``move_str`` and return the resulting board in the mover's frame.

    The returned board has the mover's checkers moved, hits reflected on the
    opponent's bar, ``dice`` cleared, and ``turn``/perspective unchanged. Raises
    :class:`IllegalMove` if the move is not legal. Use :func:`bgcore.board.flip`
    to hand the turn to the opponent.
    """
    m = _match(board, move_str)
    if m == "cannot-move":
        if generate_moves(board):
            raise IllegalMove("moves are available; 'Cannot Move' is not legal")
        nb = board.copy()
        nb.dice = []
        nb.refresh_pip()
        return nb
    if m is None:
        raise IllegalMove(f"illegal move for this roll: {move_str!r}")
    nb = board.copy()
    nb.points = list(m.points)
    nb.bar = {"x": m.bar["x"], "o": m.bar["o"]}
    nb.off = {"x": m.off["x"], "o": m.off["o"]}
    nb.dice = []
    nb.refresh_pip()
    return nb


def moves_equivalent(board: Board, s1: str, s2: str) -> bool:
    """True iff two move strings reach the same resulting position (PLAN.md §4.3)."""
    m1 = _match(board, s1)
    m2 = _match(board, s2)
    if m1 == "cannot-move" or m2 == "cannot-move":
        no_moves = len(generate_moves(board)) == 0
        return no_moves and (m1 == "cannot-move") and (m2 == "cannot-move")
    if m1 is None or m2 is None:
        return False
    return m1.signature() == m2.signature()
