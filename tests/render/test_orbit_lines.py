"""Tests for orbit-line sampling (pure math)."""
import numpy as np
from orbitsim.render.orbit_lines import sample_orbit_points
from orbitsim.core.elements import KeplerianElements
from orbitsim.core.constants import MU_EARTH


def test_ellipse_sample_shape_and_radius():
    elem = KeplerianElements(a=8.0e6, e=0.1, i=0.3, raan=1.0, argp=0.5, nu=0.0, mu=MU_EARTH)
    pts = sample_orbit_points(elem, n=128)
    assert pts.shape == (128, 3)
    radii = np.linalg.norm(pts, axis=1)
    rp = elem.a * (1 - elem.e)  # periapsis
    ra = elem.a * (1 + elem.e)  # apoapsis
    assert radii.min() >= rp - 1.0
    assert radii.max() <= ra + 1.0


def test_ellipse_is_closed():
    elem = KeplerianElements(a=8.0e6, e=0.2, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH)
    pts = sample_orbit_points(elem, n=64)
    # First and last sampled true anomalies are adjacent around the loop;
    # distance between them is one segment, far smaller than the orbit size.
    seg = np.linalg.norm(pts[0] - pts[-1])
    assert seg < 0.5 * elem.a


def test_hyperbola_sample_finite():
    a = -1.0e7  # negative for hyperbola
    elem = KeplerianElements(a=a, e=1.4, i=0.2, raan=0.5, argp=0.3, nu=0.0, mu=MU_EARTH)
    pts = sample_orbit_points(elem, n=100)
    assert pts.shape == (100, 3)
    assert np.isfinite(pts).all()
