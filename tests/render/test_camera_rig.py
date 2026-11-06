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


# ---------------------------------------------------------------------------
# Film offset: the focus sits above the navball cluster, not at screen centre
# ---------------------------------------------------------------------------
_FOCUS_PROBE = r"""
import numpy as np
from panda3d.core import Point2, loadPrcFileData
loadPrcFileData("", "window-type offscreen\nwin-size 1600 900\naudio-library-name null")
from direct.showbase.ShowBase import ShowBase
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.ui.hud_layout import FOCUS_SCREEN_Y
base = ShowBase()
rig = CameraRig(base, RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=2.0e4))
out = []
for focus_y, el in ((FOCUS_SCREEN_Y, 0.3), (FOCUS_SCREEN_Y, -1.2), (0.0, 0.3)):
    rig.focus_screen_y = focus_y
    rig.elevation = el
    rig.apply()
    p = Point2()
    base.camLens.project(base.cam.get_relative_point(base.render, (0, 0, 0)), p)
    out.append((float(p.x), float(p.y)))
print("RESULT", repr([FOCUS_SCREEN_Y] + out))
base.destroy()
"""


def test_focus_projects_above_the_navball():
    import ast
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run([sys.executable, "-c", _FOCUS_PROBE], cwd=root,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    focus_y, *points = ast.literal_eval(line[len("RESULT "):])
    for x, y in points[:2]:
        assert abs(x) < 1e-3 and abs(y - focus_y) < 0.01, points
    assert abs(points[2][1]) < 1e-3                       # default rig: screen centre
