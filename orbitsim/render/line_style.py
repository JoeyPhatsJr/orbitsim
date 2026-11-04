"""Named line styles for every orbit/trajectory line on the map (pure, no Panda3D).

Widths are framebuffer **pixels** (GL line width), so they are constant on screen at any
zoom; multiply by ``px_scale(framebuffer height)`` so a Retina (2x) framebuffer doesn't turn
them into hairlines. Your own orbit, the planned path and encounters are bold; reference
orbits (Moon, planets) are quiet unless targeted.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

Color = Tuple[float, float, float, float]

REFERENCE_HEIGHT_PX = 900


@dataclass(frozen=True)
class LineStyle:
    """How one polyline is drawn.

    color : RGBA of the main stroke.
    width_px : main stroke width in framebuffer pixels (before px_scale).
    halo_px : extra width of the dark under-stroke; 0 draws no halo.
    fade_minimum : alpha at the far end of the path (1.0 = no fade).
    dash : (points drawn, points skipped), or None for a solid line.
    """

    color: Color
    width_px: float
    halo_px: float
    fade_minimum: float
    dash: Optional[Tuple[int, int]] = None


ORBIT_STYLE = LineStyle((0.22, 0.84, 1.0, 1.0), 3.5, 2.5, 0.65)
PLAN_STYLE = LineStyle((1.0, 0.62, 0.18, 1.0), 3.5, 2.5, 0.75, dash=(6, 4))
REFERENCE_STYLE = LineStyle((0.60, 0.64, 0.71, 0.55), 2.0, 0.0, 1.0)
TARGET_REFERENCE_STYLE = LineStyle((0.86, 0.88, 0.92, 0.9), 2.75, 1.5, 1.0)
ENCOUNTER_STYLE = LineStyle((1.0, 0.83, 0.30, 1.0), 4.5, 2.5, 0.8)


def reference_style(targeted: bool) -> LineStyle:
    """Style for a body's reference orbit: highlighted while that body is the target."""
    return TARGET_REFERENCE_STYLE if targeted else REFERENCE_STYLE


def px_scale(framebuffer_height_px: int) -> float:
    """Line-width multiplier for a framebuffer of this height (never below 1)."""
    return max(1.0, framebuffer_height_px / REFERENCE_HEIGHT_PX)
