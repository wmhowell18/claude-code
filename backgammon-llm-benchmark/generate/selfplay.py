"""Drive strong-bot self-play and extract candidate decisions (PLAN.md §2.1).

Generates brand-new games via GNU BG self-play on the dataset creation date so
sampled positions cannot be in any pretraining corpus, then extracts a stream of
candidate decisions (board + phase context) for the sampler.

The gnubg subprocess is **injected** as a plain ``Callable[[list[str]], str]``
(taking a command list, returning the exported .mat text), so the whole
orchestrator is unit-testable with canned output and never spawns gnubg in CI.
The default runner wires :func:`generate.gnubg.run_gnubg`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from bgcore.board import Board
from generate import gnubg, tiering

# A runner takes the gnubg command list and returns the exported .mat text.
Runner = Callable[[list[str]], str]


@dataclass(frozen=True)
class RunConfig:
    """Self-play run configuration (PLAN.md §2.1)."""

    games: int = 10
    seed: int = 0
    plies: int = 2
    match_length: int | None = None  # None = money session
    export_path: str = "data/selfplay/session.mat"

    @property
    def play_mode(self) -> str:
        return "match" if self.match_length else "money"


@dataclass
class Candidate:
    """A candidate decision with the light context the sampler needs."""

    board: Board
    decision_type: str  # "checker" | "cube"
    play_mode: str  # "money" | "match"
    phase: str
    game_index: int
    move_number: int
    seed: int
    source: str = "gnubg-selfplay"

    def key(self) -> str:
        """Canonical dedup key (mover-relative canonical form)."""
        from bgcore.board import canonical_key

        return canonical_key(self.board)


def build_commands(config: RunConfig) -> list[str]:
    """Build the gnubg command script for a run (delegates to :mod:`generate.gnubg`)."""
    return gnubg.selfplay_commands(
        config.games,
        seed=config.seed,
        plies=config.plies,
        match_length=config.match_length,
        export_path=config.export_path,
    )


def extract_candidates(
    mat_text: str,
    *,
    config: RunConfig | None = None,
    validate_boards: bool = True,
) -> list[Candidate]:
    """Parse exported .mat text into :class:`Candidate` decisions with phase tags.

    Board reconstruction is delegated to :func:`generate.gnubg.parse_match`
    (which replays via :mod:`bgcore`); the phase is attached with
    :func:`generate.tiering.classify_phase`.
    """
    seed = config.seed if config else 0
    out: list[Candidate] = []
    for d in gnubg.parse_match(mat_text, validate_boards=validate_boards):
        out.append(
            Candidate(
                board=d.board,
                decision_type=d.decision_type,
                play_mode=d.play_mode,
                phase=tiering.classify_phase(d.board),
                game_index=d.game_index,
                move_number=d.move_number,
                seed=seed,
                source=d.source,
            )
        )
    return out


def generate(config: RunConfig, runner: Runner | None = None) -> list[Candidate]:
    """Run one self-play session and return its candidate decisions.

    ``runner`` (defaulting to :func:`generate.gnubg.run_gnubg`) is the injected
    subprocess boundary: it takes the command list and returns .mat text. Passing
    a canned-output stand-in makes the whole path testable without gnubg.
    """
    run = runner or gnubg.run_gnubg
    commands = build_commands(config)
    mat_text = run(commands)
    return extract_candidates(mat_text, config=config)


def merge(streams: Iterable[Iterable[Candidate]]) -> list[Candidate]:
    """Flatten several candidate streams (e.g. one per run) into one list."""
    out: list[Candidate] = []
    for s in streams:
        out.extend(s)
    return out
