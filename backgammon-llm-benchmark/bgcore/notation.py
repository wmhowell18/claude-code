"""Parse / format XG-style move and cube notation (PLAN.md §4.2-4.3).

Checker moves use standard slash notation in the mover's own point numbers
(24 = back checkers, 1 = ace point), matching how humans and XG write them::

    24/18 13/11        two checkers
    24/18*             a hit
    6/off              bear off
    bar/22             bar entry
    13/11(2)           repetition
    Cannot Move        no legal play

Cube-decision answers are case-insensitive with common synonyms::

    double | no double | take | pass          (drop = pass; beaver optional)

Notation point numbers are the mover-relative complement of the board index used
in :mod:`bgcore.board`: ``point = 25 - index`` (so the ace point 1 is board index
24). Round-trip ``format_move(parse_move(s))`` is stable/canonical.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


class NotationError(ValueError):
    """Raised when a move / cube string cannot be parsed."""


_CANNOT_MOVE = {"cannot move", "cannotmove", "cant move", "no move", "no play", "dance"}

_CUBE_SYNONYMS = {
    "double": "double",
    "dbl": "double",
    "redouble": "double",
    "no double": "no double",
    "nodouble": "no double",
    "no-double": "no double",
    "roll": "no double",
    "no": "no double",
    "nd": "no double",
    "take": "take",
    "t": "take",
    "accept": "take",
    "pass": "pass",
    "drop": "pass",
    "fold": "pass",
    "decline": "pass",
    "p": "pass",
    "beaver": "beaver",
}

_POINT_RE = re.compile(r"^(bar|off|\d{1,2})$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"^(?P<chain>[^()]+?)(?:\((?P<count>\d+)\))?$")


def _parse_point(tok: str) -> Any:
    m = _POINT_RE.match(tok)
    if not m:
        raise NotationError(f"bad point {tok!r}")
    low = tok.lower()
    if low in ("bar", "off"):
        return low
    n = int(tok)
    if not (1 <= n <= 24):
        raise NotationError(f"point {n} out of range 1..24")
    return n


def parse_move(s: str) -> list[dict] | None:
    """Parse a checker move string.

    Returns ``None`` for a "Cannot Move" string, otherwise a list of expanded
    token dicts (one per checker played), each with keys ``from``, ``to``,
    ``hits`` (list of landing points that hit), ``chain`` (full slash path) and
    ``chain_hits`` (per-step hit flags). Raises :class:`NotationError` on garbage.
    """
    text = s.strip()
    if not text:
        raise NotationError("empty move string")
    norm = re.sub(r"[.\s]+", " ", text.lower()).strip()
    if norm in _CANNOT_MOVE:
        return None

    tokens: list[dict] = []
    for raw in text.split():
        m = _TOKEN_RE.match(raw.strip())
        if not m:
            raise NotationError(f"bad move token {raw!r}")
        chain_str = m.group("chain")
        count = int(m.group("count")) if m.group("count") else 1
        parts = chain_str.split("/")
        if len(parts) < 2:
            raise NotationError(f"move token needs a '/': {raw!r}")
        chain: list[Any] = []
        chain_hits: list[bool] = []
        for i, part in enumerate(parts):
            hit = part.endswith("*")
            clean = part[:-1] if hit else part
            pt = _parse_point(clean)
            chain.append(pt)
            if i == 0:
                if hit:
                    raise NotationError("'*' cannot mark the first point")
            else:
                chain_hits.append(hit)
        hits = [chain[i + 1] for i, h in enumerate(chain_hits) if h]
        token = {
            "from": chain[0],
            "to": chain[-1],
            "hits": hits,
            "chain": chain,
            "chain_hits": chain_hits,
        }
        for _ in range(count):
            tokens.append(dict(token, chain=list(chain), chain_hits=list(chain_hits), hits=list(hits)))
    return tokens


def _render_chain(chain: list[Any], chain_hits: list[bool]) -> str:
    out = str(chain[0])
    for i, p in enumerate(chain[1:]):
        out += "/" + str(p) + ("*" if chain_hits[i] else "")
    return out


def _tok_sort_key(tok: str) -> tuple:
    head = tok.split("/", 1)[0]
    start = 25 if head == "bar" else int(head)
    return (-start, tok)


def format_move(tokens: list[dict] | None) -> str:
    """Render parsed tokens back to canonical notation (stable round-trip)."""
    if tokens is None or len(tokens) == 0:
        return "Cannot Move"
    rendered = [_render_chain(t["chain"], t["chain_hits"]) for t in tokens]
    cnt = Counter(rendered)
    parts = []
    for tok in sorted(set(rendered), key=_tok_sort_key):
        n = cnt[tok]
        parts.append(f"{tok}({n})" if n > 1 else tok)
    return " ".join(parts)


# -- cube decisions -------------------------------------------------------


def parse_cube(s: str) -> str:
    """Canonicalize a cube-decision answer to one of
    ``double`` / ``no double`` / ``take`` / ``pass`` / ``beaver``.

    Accepts case-insensitive synonyms (e.g. ``drop`` -> ``pass``). A two-part
    answer such as ``"Double, Take"`` returns the *action* part (``double``);
    use :func:`parse_cube_answer` to recover both parts. Raises
    :class:`NotationError` if unrecognized.
    """
    norm = re.sub(r"[\s,]+", " ", s.strip().lower()).strip()
    if norm in _CUBE_SYNONYMS:
        return _CUBE_SYNONYMS[norm]
    first = norm.split(" ")[0]
    if first in _CUBE_SYNONYMS and first not in ("no",):
        return _CUBE_SYNONYMS[first]
    raise NotationError(f"unrecognized cube decision {s!r}")


def parse_cube_answer(s: str) -> tuple[str, str | None]:
    """Parse a possibly two-part cube answer like ``"Double, Take"``.

    Returns ``(action, response)`` where ``action`` is ``double`` / ``no double``
    and ``response`` is ``take`` / ``pass`` / ``beaver`` / ``None``.
    """
    norm = re.sub(r"[\s]+", " ", s.strip().lower())
    pieces = [p.strip() for p in re.split(r"[,/]| and ", norm) if p.strip()]
    if not pieces:
        raise NotationError(f"empty cube answer {s!r}")
    action = parse_cube(pieces[0])
    response = parse_cube(pieces[1]) if len(pieces) > 1 else None
    return action, response


def format_cube(action: str) -> str:
    """Canonical display form of a cube decision (identity on canonical input)."""
    return parse_cube(action)
