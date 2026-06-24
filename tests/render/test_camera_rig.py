"""Tests for camera zoom->scale mapping (pure math)."""
from orbitsim.render.camera_rig import zoom_to_scale, clamp_distance, MIN_DISTANCE_M, MAX_DISTANCE_M


def test_scale_proportional_to_distance():
    assert zoom_to_scale(1.0e6) == 1.0e6 / 1000.0
    assert zoom_to_scale(1.0e9) == 1.0e9 / 1000.0


def test_scale_monotonic():
    assert zoom_to_scale(10.0) < zoom_to_scale(1.0e12)


def test_clamp_distance_bounds():
    assert clamp_distance(1.0) == MIN_DISTANCE_M
    assert clamp_distance(1.0e15) == MAX_DISTANCE_M
    assert clamp_distance(1.0e6) == 1.0e6
