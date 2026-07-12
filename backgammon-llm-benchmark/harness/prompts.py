"""Versioned prompt templates per track (PLAN.md §4.1-4.2).

Holds the fixed system prompt (notation rules + required answer format) and the
per-track payload builders:

* **text** — ``ascii`` render + ``board_json`` + a natural-language statement of
  the decision.
* **image** — the rendered PNG plus a minimal text frame ("You are on roll…").
* **text+image** — both.

Each decision has a **checker** variant (final line ``MOVE: <move>``) and a
**cube** variant (final line ``ACTION: <double|no double|take|pass>``). Chain of
thought is allowed above the final line; only that last contract line is parsed
(:mod:`harness.parse`).

Everything here is deterministic given the inputs (no timestamps, no randomness)
and versioned by :data:`PROMPT_VERSION`, so a run is reproducible.
"""

from __future__ import annotations

import json
from typing import Any

from bgcore.board import Board, pip_counts
from harness.client import user_message

__all__ = [
    "PROMPT_VERSION",
    "Track",
    "system_prompt",
    "board_json_text",
    "decision_statement",
    "answer_contract",
    "user_text",
    "build_messages",
]

# Bump when the wording of any template below changes.
PROMPT_VERSION = "bench-1"

Track = str  # one of "text" | "image" | "text+image"
_TRACKS = ("text", "image", "text+image")

_SYSTEM = """\
You are a world-class backgammon engine. You are given a single position and must \
choose the best play. Reason as carefully as you like, then commit to one answer.

Board and notation conventions:
- The position is always shown from the perspective of the player on roll ("X").
  You are X. Your opponent is "O".
- Points are numbered 1-24 in your own numbering: 24 is your back checkers' start,
  1 is your ace point. You move checkers from high-numbered points toward 1 and
  bear off past the 1 point.
- Checker moves use standard slash notation, e.g. `24/18 13/11`. A hit is marked
  with `*` (e.g. `24/18*`). Bear-off is `6/off`; bar entry is `bar/22`. Repeated
  moves may be written `13/11(2)`. Move order does not matter; a play is judged by
  the position it reaches. If you have no legal move, answer `Cannot Move`.
- Cube decisions: if you are on roll deciding whether to double, answer `double`
  or `no double`. If you are facing a double, answer `take` or `pass`.

ANSWER FORMAT (required):
- End your reply with a single final line and nothing after it.
- For a checker play, the final line must be exactly:
    MOVE: <your move>
- For a cube decision, the final line must be exactly:
    ACTION: <double | no double | take | pass>
Only that final line is scored. Do not add commentary after it."""


def system_prompt() -> str:
    """The fixed, versioned system prompt (notation rules + answer contract)."""
    return _SYSTEM


# -- payload pieces -------------------------------------------------------


def board_json_text(board: Board) -> str:
    """Deterministic pretty-printed ``board_json`` for the text payload."""
    return json.dumps(board.to_json(), indent=2, sort_keys=True)


def _score_line(board: Board) -> str:
    length = int(board.score.get("length", 0) or 0)
    if length:
        crawf = " (Crawford)" if board.score.get("crawford") else ""
        return f"Match to {length}{crawf}. Score X {board.score['x']}, O {board.score['o']}."
    return "Money game (unlimited)."


def decision_statement(board: Board) -> str:
    """A one-paragraph natural-language statement of the decision to make."""
    px, po = pip_counts(board)
    owner = {"center": "in the center", "x": "on your side (X)", "o": "on O's side"}[
        board.cube["owner"]
    ]
    lines = [
        _score_line(board),
        f"Cube value {board.cube['value']}, {owner}.",
        f"Pip count: X {px}, O {po}.",
    ]
    if board.decision_type == "cube":
        lines.append(
            "This is a CUBE decision. Decide the correct cube action for the player "
            "on roll (X)."
        )
    else:
        dice = board.dice
        roll = f"{dice[0]}-{dice[1]}" if len(dice) == 2 else "?"
        lines.append(f"You are on roll with dice {roll}. Choose the best checker play.")
    return "\n".join(lines)


def answer_contract(board: Board) -> str:
    """The strict final-line reminder for this decision type."""
    if board.decision_type == "cube":
        return "Finish with exactly:\nACTION: <double | no double | take | pass>"
    return "Finish with exactly:\nMOVE: <your move in slash notation>"


def user_text(board: Board, *, ascii_text: str | None = None, include_board_json: bool = True) -> str:
    """Assemble the text payload: ASCII board + board_json + decision + contract."""
    blocks: list[str] = []
    if ascii_text:
        blocks.append("Position (ASCII):\n" + ascii_text)
    if include_board_json:
        blocks.append("Position (board_json):\n" + board_json_text(board))
    blocks.append(decision_statement(board))
    blocks.append(answer_contract(board))
    return "\n\n".join(blocks)


def _image_frame_text(board: Board) -> str:
    """Minimal text for the image track (PLAN §4.1: image carries png + minimal text)."""
    if board.decision_type == "cube":
        head = (
            "The image shows a backgammon position from the perspective of the player "
            "on roll (X). Decide the correct cube action."
        )
    else:
        head = (
            "The image shows a backgammon position from the perspective of the player "
            "on roll (X). Choose the best checker play for the dice shown."
        )
    return head + "\n\n" + answer_contract(board)


# -- top-level builder ----------------------------------------------------


def build_messages(
    board: Board,
    *,
    track: Track = "text",
    ascii_text: str | None = None,
    image_png: bytes | None = None,
) -> list[dict[str, Any]]:
    """Build the full ``messages`` list (system + user) for a track.

    ``ascii_text`` should be the deterministic ASCII render (text tracks);
    ``image_png`` the rendered PNG bytes (image tracks). Rendering is the runner's
    job so this module has no image dependency and stays deterministic/testable.
    """
    if track not in _TRACKS:
        raise ValueError(f"unknown track {track!r}; expected one of {_TRACKS}")
    if track in ("image", "text+image") and image_png is None:
        raise ValueError(f"track {track!r} requires image_png bytes")

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt()}]

    if track == "text":
        messages.append(user_message(user_text(board, ascii_text=ascii_text)))
    elif track == "image":
        messages.append(user_message(_image_frame_text(board), images=[image_png]))
    else:  # text+image
        messages.append(user_message(user_text(board, ascii_text=ascii_text), images=[image_png]))
    return messages
