"""Pure responsive layout calculations for gameplay UI."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenLayout:
    width_px: int
    height_px: int
    ui_scale: float
    safe_x: float
    safe_y: float
    compact: bool
    orbit_width: float
    vessel_width: float
    planner_width: float


class ResponsiveLayout:
    """Map a viewport into stable aspect2d sizing from 720p through 4K."""

    @staticmethod
    def calculate(width_px: int, height_px: int) -> ScreenLayout:
        if width_px <= 0 or height_px <= 0:
            raise ValueError("viewport dimensions must be positive")
        ui_scale = max(0.84, min(1.35, height_px / 1080.0))
        aspect = width_px / height_px
        compact = height_px <= 800 or aspect < 1.55
        safe_x = 0.045 if width_px < 1600 else 0.035
        safe_y = 0.055 if height_px <= 800 else 0.045
        side = 0.72 if compact else min(0.88, 0.74 + max(0.0, aspect - 16 / 9) * 0.18)
        return ScreenLayout(
            width_px, height_px, ui_scale, safe_x, safe_y, compact,
            orbit_width=side,
            vessel_width=0.64 if compact else 0.72,
            planner_width=1.12 if compact else 1.28,
        )
