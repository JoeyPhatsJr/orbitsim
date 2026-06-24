"""Orbit-style camera: focus/zoom across a huge dynamic range.

This module's pure functions (zoom_to_scale, clamp_distance) are unit-tested.
The CameraRig class that drives Panda3D is added in Task 8 and exercised by the
visual checkpoint, not unit tests.
"""

MIN_DISTANCE_M = 10.0
MAX_DISTANCE_M = 1.0e12

# 1000 render units span the camera-to-focus distance, keeping the visible
# scene comfortably inside float32-friendly coordinates.
RENDER_UNITS_ACROSS_VIEW = 1000.0


def clamp_distance(distance_m: float) -> float:
    """Clamp a camera distance to the supported zoom range [10 m, 1e12 m]."""
    return max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance_m))


def zoom_to_scale(distance_m: float) -> float:
    """Meters per render unit for a given camera-to-focus distance [m]."""
    return distance_m / RENDER_UNITS_ACROSS_VIEW
