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
# The planned path while the burn is still changing: an instant two-body estimate,
# fainter and finer-dashed so it reads as provisional until the exact path settles.
ESTIMATE_STYLE = LineStyle((1.0, 0.62, 0.18, 0.45), 3.0, 0.0, 0.75, dash=(3, 4))
TARGET_REFERENCE_STYLE = LineStyle((0.86, 0.88, 0.92, 0.9), 2.75, 1.5, 1.0)
ENCOUNTER_STYLE = LineStyle((1.0, 0.83, 0.30, 1.0), 4.5, 2.5, 0.8)


def reference_style(targeted: bool) -> LineStyle:
    """Style for a body's reference orbit: highlighted while that body is the target."""
    return TARGET_REFERENCE_STYLE if targeted else REFERENCE_STYLE


# Below this camera distance the planets' heliocentric orbits are just stray straight
# lines across the Earth-Moon view (Earth's own passes through the ship), so the sandbox
# hides them until you zoom out to planetary scale (~0.13 AU).
PLANET_ORBITS_MIN_DISTANCE_M = 2.0e10


def planet_orbits_visible(camera_distance_m: float) -> bool:
    """Whether the sandbox shows the planets' reference orbits at this zoom."""
    return camera_distance_m >= PLANET_ORBITS_MIN_DISTANCE_M


def px_scale(framebuffer_height_px: int, window_height: int) -> float:
    """Line-width multiplier (never below 1).

    Display density (framebuffer pixels per window unit: 2 on Retina) times a window
    factor that grows lines proportionally on windows taller than REFERENCE_HEIGHT_PX.
    """
    if framebuffer_height_px <= 0 or window_height <= 0:
        return 1.0
    density = framebuffer_height_px / window_height
    return max(1.0, density * max(1.0, window_height / REFERENCE_HEIGHT_PX))
