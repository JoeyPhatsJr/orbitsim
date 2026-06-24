"""Floating-origin transform: physics-space float64 -> render-space float32.

render_pos = (physics_pos_m - origin_m) / scale_m_per_unit

The subtraction happens in float64 BEFORE the float32 cast, which preserves
millimeter precision near the focus even at solar-system absolute coordinates.
"""
import numpy as np


class RenderTransform:
    """Maps physics-space SI float64 positions to render-space float32 positions.

    Parameters
    ----------
    origin_m : np.ndarray
        The physics point (float64, shape (3,)) currently mapped to render (0,0,0).
    scale_m_per_unit : float
        Meters per render unit (set from camera zoom).
    """

    def __init__(self, origin_m: np.ndarray, scale_m_per_unit: float) -> None:
        self.origin_m = np.asarray(origin_m, dtype=np.float64).copy()
        self.scale_m_per_unit = float(scale_m_per_unit)

    def set_origin(self, origin_m: np.ndarray) -> None:
        """Re-center the render space on a new physics point."""
        self.origin_m = np.asarray(origin_m, dtype=np.float64).copy()

    def to_render(self, physics_pos_m: np.ndarray) -> tuple[float, float, float]:
        """Convert a physics-space position to a render-space (x, y, z) tuple."""
        local = np.asarray(physics_pos_m, dtype=np.float64) - self.origin_m  # float64 first
        scaled = (local / self.scale_m_per_unit).astype(np.float32)          # then cast
        return (float(scaled[0]), float(scaled[1]), float(scaled[2]))
