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
