"""SVG -> PNG rasterization (PLAN.md §1.2).

Rasterizes the committed SVG to a fixed-target PNG (e.g. 1024x768) via
cairosvg/resvg. Records ``image_render_version`` so a rendering change is traceable.
"""
