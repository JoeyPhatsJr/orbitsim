"""Tests for orbit-line sampling (pure math)."""
import numpy as np
import pytest
from orbitsim.render.orbit_lines import (
    build_orbit_node,
    orbit_shape_changed,
    path_fade_alphas,
    sample_orbit_points,
)
from orbitsim.core.elements import KeplerianElements
from orbitsim.core.constants import MU_EARTH


def _elem(**over):
    base = dict(a=7.0e6, e=0.1, i=0.5, raan=0.3, argp=0.4, nu=0.0, mu=MU_EARTH)
    base.update(over)
    return KeplerianElements(**base)


def test_shape_unchanged_for_identical():
    assert orbit_shape_changed(_elem(), _elem()) is False


def test_shape_unchanged_ignores_true_anomaly():
    assert orbit_shape_changed(_elem(nu=0.0), _elem(nu=1.7)) is False


def test_shape_changed_on_semimajor_axis():
    # default tol 1e-6 rel -> ~7 m at a=7e6; a real burn shifts a far more.
    assert orbit_shape_changed(_elem(a=7.0e6), _elem(a=7.0e6 + 100.0)) is True


def test_shape_changed_on_angles():
    assert orbit_shape_changed(_elem(argp=0.4), _elem(argp=0.4 + 1e-3)) is True


def test_shape_angle_wrap_near_zero_is_unchanged():
    # raan flips between ~0 and ~2pi under recovery noise; that is NOT a real change.
    two_pi = 2.0 * np.pi
    assert orbit_shape_changed(_elem(raan=1e-12), _elem(raan=two_pi - 1e-12)) is False


def test_none_counts_as_changed():
    assert orbit_shape_changed(None, _elem()) is True
    assert orbit_shape_changed(_elem(), None) is True


def test_ellipse_sample_shape_and_radius():
    elem = KeplerianElements(a=8.0e6, e=0.1, i=0.3, raan=1.0, argp=0.5, nu=0.0, mu=MU_EARTH)
    pts = sample_orbit_points(elem, n=128)
    assert pts.shape == (128, 3)
    radii = np.linalg.norm(pts, axis=1)
    rp = elem.periapsis_radius
    ra = elem.apoapsis_radius
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


def test_rendered_line_has_halo_and_color_strokes():
    node = build_orbit_node([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert node.get_num_children() == 2
    assert node.get_child(0).name == "trajectory_halo"
    assert node.get_child(1).name == "trajectory_color"


def test_path_fade_tracks_distance_not_sample_count():
    points = [(0.0, 0.0, 0.0), (9.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    alpha = path_fade_alphas(points, minimum=0.25)
    assert alpha[0] == 1.0
    assert alpha[-1] == pytest.approx(0.25)
    assert alpha[1] < 0.5  # already 90% of the path despite being the middle sample
