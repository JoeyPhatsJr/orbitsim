"""Tests for camera zoom and smoothing math (pure, no Panda3D)."""
import math

import pytest

from orbitsim.render.camera_rig import (
    MAX_DISTANCE_M,
    MIN_DISTANCE_M,
    clamp_distance,
    smooth_angle,
    smooth_log_distance,
    smoothing_alpha,
    zoom_to_scale,
)


def test_scale_proportional_to_distance():
    assert zoom_to_scale(1.0e6) == 1.0e6 / 1000.0
    assert zoom_to_scale(1.0e9) == 1.0e9 / 1000.0


def test_scale_monotonic():
    assert zoom_to_scale(10.0) < zoom_to_scale(1.0e12)


def test_clamp_distance_bounds():
    assert clamp_distance(1.0) == MIN_DISTANCE_M
    assert clamp_distance(1.0e15) == MAX_DISTANCE_M
    assert clamp_distance(1.0e6) == 1.0e6


def test_smoothing_alpha_is_frame_rate_independent():
    whole = smoothing_alpha(1.0, 0.25)
    half = smoothing_alpha(0.5, 0.25)
    assert 1.0 - whole == pytest.approx((1.0 - half) ** 2)


def test_log_distance_smoothing_has_equal_zoom_feel_across_scales():
    alpha = 0.5
    assert smooth_log_distance(100.0, 10_000.0, alpha) == pytest.approx(1_000.0)
    assert smooth_log_distance(1.0e6, 1.0e8, alpha) == pytest.approx(1.0e7)


def test_angle_smoothing_takes_short_path_across_wrap():
    current = math.radians(179.0)
    target = math.radians(-179.0)
    result = smooth_angle(current, target, 0.5)
    assert math.degrees(result) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Opening camera: whole orbit in view, ship never behind Earth, tilted to the Sun
# ---------------------------------------------------------------------------
from orbitsim.render.camera_rig import (  # noqa: E402
    OPENING_DISTANCE_FACTOR,
    OPENING_SUN_TILT_RAD,
    opening_camera,
)


def _dir(az, el):
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def test_camera_looks_down_on_the_orbit_tilted_toward_the_sun():
    az, el, _ = opening_camera((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 8.8e6)
    assert az == pytest.approx(0.0, abs=1e-12)                 # leaning toward +x (the Sun)
    assert el == pytest.approx(math.pi / 2 - OPENING_SUN_TILT_RAD)


def test_camera_stays_on_the_sun_side_of_the_orbit_plane():
    # Orbit normal pointing away from the Sun gets flipped: we view the sunward face.
    az, el, _ = opening_camera((1.0, 0.0, 0.5), (0.0, 0.0, -1.0), 8.8e6)
    c = _dir(az, el)
    assert c[2] > 0.0 and c[0] > 0.0


def test_camera_is_at_least_70_deg_above_the_orbit_plane():
    # Keeps the ship from hiding behind Earth from LEO (sin 70 deg * r > R_earth).
    for sun in ((1.0, 0.0, 0.0), (0.3, -0.9, 0.4), (0.0, 1.0, -0.2)):
        for normal in ((0.0, 0.0, 1.0), (0.1, 0.2, 0.97)):
            az, el, _ = opening_camera(sun, normal, 8.8e6)
            c = _dir(az, el)
            n = normal / __import__("numpy").linalg.norm(normal)
            assert abs(float(__import__("numpy").dot(c, n))) >= math.sin(math.radians(70)) - 1e-9


def test_distance_frames_the_whole_orbit():
    _, _, d = opening_camera((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 8.8e6)
    assert d == pytest.approx(OPENING_DISTANCE_FACTOR * 8.8e6)


def test_rejects_degenerate_input():
    with pytest.raises(ValueError):
        opening_camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 8.8e6)
    with pytest.raises(ValueError):
        opening_camera((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 8.8e6)
    with pytest.raises(ValueError):
        opening_camera((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0)
