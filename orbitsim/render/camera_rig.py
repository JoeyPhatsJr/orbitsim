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


def smoothing_alpha(dt_s: float, response_s: float) -> float:
    """Frame-rate-independent exponential response in ``[0, 1]``."""
    if dt_s <= 0.0:
        return 0.0
    if response_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt_s / response_s)


def smooth_log_distance(current_m: float, target_m: float, alpha: float) -> float:
    """Interpolate zoom in log space so near and map scales feel consistent."""
    a = max(0.0, min(1.0, alpha))
    lo = math.log(max(MIN_DISTANCE_M, current_m))
    hi = math.log(max(MIN_DISTANCE_M, target_m))
    return math.exp(lo + (hi - lo) * a)


def smooth_angle(current_rad: float, target_rad: float, alpha: float) -> float:
    """Interpolate an angle along its shortest wrapped arc."""
    a = max(0.0, min(1.0, alpha))
    delta = (target_rad - current_rad + math.pi) % (2.0 * math.pi) - math.pi
    return current_rad + delta * a


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
        self.target_distance_m = self.distance_m
        self.target_azimuth = self.azimuth
        self.target_elevation = self.elevation
        self.zoom_response_s = 0.16
        self.orbit_response_s = 0.10
        self._apply_scale()

    def _apply_scale(self) -> None:
        self.transform.scale_m_per_unit = zoom_to_scale(self.distance_m)

    def set_distance(self, distance_m: float) -> None:
        """Set current and target distance immediately (initial framing/load)."""
        self.distance_m = clamp_distance(distance_m)
        self.target_distance_m = self.distance_m
        self._apply_scale()

    def move_to_distance(self, distance_m: float) -> None:
        """Set a smoothed destination distance (interactive framing)."""
        self.target_distance_m = clamp_distance(distance_m)

    def zoom(self, factor: float) -> None:
        """Multiply camera distance (factor < 1 zooms in, > 1 zooms out)."""
        self.move_to_distance(self.target_distance_m * factor)

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.target_azimuth += d_azimuth
        self.target_elevation = max(
            -1.5, min(1.5, self.target_elevation + d_elevation)
        )

    def update(self, dt_s: float) -> None:
        """Ease current camera state toward input targets."""
        zoom_a = smoothing_alpha(dt_s, self.zoom_response_s)
        orbit_a = smoothing_alpha(dt_s, self.orbit_response_s)
        self.distance_m = smooth_log_distance(
            self.distance_m, self.target_distance_m, zoom_a
        )
        self.azimuth = smooth_angle(self.azimuth, self.target_azimuth, orbit_a)
        self.elevation += (self.target_elevation - self.elevation) * orbit_a
        self._apply_scale()

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
        # Near/far must adapt to zoom. At ship view (distance_m=80, scale=0.08),
        # Earth center is ~8.6e7 render units away; at map zoom (2e7 m, scale=2e4)
        # it's only ~340 units. Scale far so world-space coverage stays constant
        # (~10 AU). Near is set as a fraction of far to maintain depth precision
        # while keeping nearby geometry (ship model at ~1000 units) visible.
        far = max(1.5e12 / max(self.distance_m, MIN_DISTANCE_M), 1.0e6)
        near = min(max(far * 1e-7, 0.01), d * 0.1)
        lens.set_near_far(near, far)
