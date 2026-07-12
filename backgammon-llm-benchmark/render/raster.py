"""SVG -> PNG rasterization (PLAN.md §1.2).

Rasterizes the committed SVG to a PNG via ``cairosvg`` at a chosen width. Records
``IMAGE_RENDER_VERSION`` (from :mod:`render.svg`) so a rendering change is
traceable. ``cairosvg`` is an optional dependency (the ``render`` extra); a clear
:class:`ImportError` is raised if it is not installed.
"""

from __future__ import annotations

from bgcore.board import Board
from render.svg import IMAGE_RENDER_VERSION, render as render_svg

__all__ = ["IMAGE_RENDER_VERSION", "svg_to_png", "board_to_png"]

_MISSING = (
    "cairosvg is required to rasterize SVG -> PNG. Install the render extra: "
    "pip install 'backgammon-llm-benchmark[render]' (or pip install cairosvg)."
)


def svg_to_png(svg: str, *, width: int = 1024) -> bytes:
    """Rasterize an SVG string to PNG bytes at the given output ``width``."""
    try:
        import cairosvg  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without cairosvg
        raise ImportError(_MISSING) from exc
    return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width)


def board_to_png(board: Board, *, width: int = 1024) -> bytes:
    """Render a board straight to PNG bytes (SVG -> PNG)."""
    return svg_to_png(render_svg(board), width=width)
