"""Tests for the 2D map view pure helpers."""
import math
import numpy as np
import pytest

from orbitsim.render.map_view import ecliptic_project, circle_points, view_extents


class TestEclipticProject:
    def test_origin_subtraction(self):
        pos = np.array([1e8, 2e8, 3e8])
        origin = np.array([1e8, 0.0, 0.0])
        x, y = ecliptic_project(pos, origin)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(2e8)

    def test_ignores_z(self):
        pos = np.array([100.0, 200.0, 999.0])
        origin = np.zeros(3)
        x, y = ecliptic_project(pos, origin)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_zero_origin(self):
        pos = np.array([5e6, -3e6, 1e6])
        x, y = ecliptic_project(pos, np.zeros(3))
        assert x == pytest.approx(5e6)
        assert y == pytest.approx(-3e6)


class TestCirclePoints:
    def test_count(self):
        pts = circle_points(1.0, n=64)
        assert len(pts) == 64

    def test_radius(self):
        r = 1e9
        pts = circle_points(r, n=128)
        for x, y in pts:
            assert math.sqrt(x * x + y * y) == pytest.approx(r, rel=1e-10)

    def test_first_point_on_x_axis(self):
        r = 500.0
        pts = circle_points(r, n=32)
        assert pts[0][0] == pytest.approx(r)
        assert pts[0][1] == pytest.approx(0.0, abs=1e-10)


class TestViewExtents:
    def test_symmetric(self):
        left, right, bottom, top = view_extents(0.0, 0.0, 100.0)
        assert left == pytest.approx(-100.0)
        assert right == pytest.approx(100.0)
        assert bottom == pytest.approx(-100.0)
        assert top == pytest.approx(100.0)

    def test_offset_center(self):
        left, right, bottom, top = view_extents(50.0, -30.0, 10.0)
        assert left == pytest.approx(40.0)
        assert right == pytest.approx(60.0)
        assert bottom == pytest.approx(-40.0)
        assert top == pytest.approx(-20.0)
