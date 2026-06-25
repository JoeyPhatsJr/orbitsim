"""Orbit-style camera: focus/zoom across a huge dynamic range.

This module's pure functions (zoom_to_scale, clamp_distance) are unit-tested.
The CameraRig class that drives Panda3D is added in Task 8 and exercised by the
visual checkpoint, not unit tests.
"""

MIN_DISTANCE_M = 10.0
MAX_DISTANCE_M = 1.0e13  # ~67 AU camera pull-back, enough to frame the outer planets

# 1000 render units span the camera-to-focus distance, keeping the visible
# scene comfortably inside float32-friendly coordinates.
RENDER_UNITS_ACROSS_VIEW = 1000.0


def clamp_distance(distance_m: float) -> float:
    """Clamp a camera distance to the supported zoom range [10 m, 1e12 m]."""
    return max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance_m))


def zoom_to_scale(distance_m: float) -> float:
    """Meters per render unit for a given camera-to-focus distance [m]."""
    return distance_m / RENDER_UNITS_ACROSS_VIEW


import math


class CameraRig:
    """Orbit camera: azimuth/elevation around a focus, log zoom drives scale.

    Parameters
    ----------
    base : ShowBase
    transform : RenderTransform
        Its scale_m_per_unit is updated as the camera zooms.
    """

    def __init__(self, base, transform) -> None:
        self.base = base
        self.transform = transform
        self.distance_m = 2.0e7
        self.azimuth = 0.0
        self.elevation = 0.3
        self._apply_scale()

    def _apply_scale(self) -> None:
        self.transform.scale_m_per_unit = zoom_to_scale(self.distance_m)

    def set_distance(self, distance_m: float) -> None:
        self.distance_m = clamp_distance(distance_m)
        self._apply_scale()

    def zoom(self, factor: float) -> None:
        """Multiply camera distance (factor < 1 zooms in, > 1 zooms out)."""
        self.set_distance(self.distance_m * factor)

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth += d_azimuth
        self.elevation = max(-1.5, min(1.5, self.elevation + d_elevation))

    def apply(self) -> None:
        """Place the camera in render space (focus is always render origin)."""
        # In render units the focus sits at (0,0,0); camera distance is fixed
        # at RENDER_UNITS_ACROSS_VIEW because scale already encodes zoom.
        d = RENDER_UNITS_ACROSS_VIEW
        ce = math.cos(self.elevation)
        x = d * ce * math.cos(self.azimuth)
        y = d * ce * math.sin(self.azimuth)
        z = d * math.sin(self.elevation)
        self.base.camera.set_pos(x, y, z)
        self.base.camera.look_at(0, 0, 0)
        lens = self.base.camLens
        lens.set_near_far(0.01, 1.0e6)
