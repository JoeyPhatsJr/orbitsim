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


# ---------------------------------------------------------------------------
# sample_relative_orbit_points (heliocentric reference lines in the sandbox)
# ---------------------------------------------------------------------------
from orbitsim.core.constants import MU_SUN
from orbitsim.core.planets import mars_state_at, sun_state_at, A_MARS
from orbitsim.core.state import StateVector
from orbitsim.render.orbit_lines import sample_relative_orbit_points


def test_relative_orbit_points_trace_heliocentric_circle():
    """Mars' geocentric state relative to the geocentric Sun is its
    heliocentric orbit: every sampled point sits at ~A_MARS from the center."""
    t = 1.0e6
    pts = sample_relative_orbit_points(mars_state_at(t), sun_state_at(t), MU_SUN, n=64)
    assert pts.shape == (64, 3)
    radii = np.linalg.norm(pts, axis=1)
    assert np.allclose(radii, A_MARS, rtol=1e-6)


def test_relative_orbit_points_passes_through_current_relative_position():
    t = 2.5e6
    mars = mars_state_at(t)
    sun = sun_state_at(t)
    rel_now = mars.r - sun.r
    pts = sample_relative_orbit_points(mars, sun, MU_SUN, n=512)
    nearest = np.min(np.linalg.norm(pts - rel_now, axis=1))
    # Orbit circumference / 512 samples bounds the gap to the nearest sample.
    assert nearest < 2.0 * np.pi * A_MARS / 512 * 1.5


def test_relative_orbit_points_degenerate_state_falls_back_to_circle():
    body = StateVector(r=np.array([1.5e11, 0.0, 0.0]), v=np.zeros(3), mu=0.0, epoch_s=0.0)
    center = StateVector(r=np.zeros(3), v=np.zeros(3), mu=MU_SUN, epoch_s=0.0)
    pts = sample_relative_orbit_points(body, center, MU_SUN, n=32)
    assert pts.shape == (32, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.5e11, rtol=1e-9)


# ---------------------------------------------------------------------------
# Vectorized sampling: same points as converting each true anomaly separately,
# without one elements_to_state call per point (it rebuilt 9 orbit lines x 256
# points every frame of a zoom: ~45 ms/frame).
# ---------------------------------------------------------------------------
import pytest

from orbitsim.core.elements import KeplerianElements, elements_to_state
from orbitsim.render import orbit_lines as _ol


def _reference_points(el, nus):
    out = []
    for nu in nus:
        k = KeplerianElements(a=el.a, e=el.e, i=el.i, raan=el.raan, argp=el.argp,
                              nu=float(nu), mu=el.mu, epoch_s=el.epoch_s)
        out.append(elements_to_state(k).r)
    return np.array(out)


_CASES = [
    KeplerianElements(a=7.0e6, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=3.986e14),
    KeplerianElements(a=2.4e7, e=0.73, i=0.9, raan=2.1, argp=4.0, nu=1.0, mu=3.986e14),
    KeplerianElements(a=2.279e11, e=0.0934, i=0.0323, raan=0.865, argp=5.0, nu=3.0, mu=1.327e20),
    KeplerianElements(a=-1.2e7, e=1.8, i=0.4, raan=5.5, argp=0.3, nu=0.2, mu=3.986e14),
]


@pytest.mark.parametrize("el", _CASES, ids=["circular", "eccentric-inclined", "mars-like", "hyperbola"])
def test_sample_orbit_points_matches_per_point_conversion(el):
    n = 64
    if el.e < 1.0:
        nus = np.linspace(0.0, 2.0 * np.pi, n)
    else:
        limit = 0.99 * np.arccos(-1.0 / el.e)
        nus = np.linspace(-limit, limit, n)
    got = _ol.sample_orbit_points(el, n=n)
    assert got.shape == (n, 3) and got.dtype == np.float64
    np.testing.assert_allclose(got, _reference_points(el, nus), rtol=1e-12, atol=1e-6)


def test_sample_orbit_points_does_not_convert_point_by_point(monkeypatch):
    calls = []
    real = _ol.elements_to_state

    def counting(el):
        calls.append(1)
        return real(el)

    monkeypatch.setattr(_ol, "elements_to_state", counting)
    _ol.sample_orbit_points(_CASES[1], n=256)
    assert len(calls) <= 1


# ---------------------------------------------------------------------------
# Styled lines (map readability plan 1, Task 2)
# ---------------------------------------------------------------------------
from panda3d.core import RenderModeAttrib  # noqa: E402

from orbitsim.render.line_style import ORBIT_STYLE, PLAN_STYLE, REFERENCE_STYLE  # noqa: E402


def _strokes(node):
    return list(node.get_children())


def _thickness(np_):
    return np_.node().get_geom_state(0).get_attrib(RenderModeAttrib).get_thickness()


def _strip_count(np_):
    return np_.node().get_geom(0).get_primitive(0).get_num_primitives()


STYLED_POINTS = [(float(i), 0.0, 0.0) for i in range(40)]


def test_width_scales_with_px_scale():
    node = build_orbit_node(STYLED_POINTS, ORBIT_STYLE, px_scale=2.0)
    color_stroke = _strokes(node)[-1]
    assert _thickness(color_stroke) == ORBIT_STYLE.width_px * 2.0


def test_halo_only_when_style_has_one():
    assert len(_strokes(build_orbit_node(STYLED_POINTS, ORBIT_STYLE))) == 2
    assert len(_strokes(build_orbit_node(STYLED_POINTS, REFERENCE_STYLE))) == 1


def test_dashed_plan_breaks_into_dash_strips():
    solid = _strokes(build_orbit_node(STYLED_POINTS, ORBIT_STYLE))[-1]
    dashed = _strokes(build_orbit_node(STYLED_POINTS, PLAN_STYLE))[-1]
    assert _strip_count(solid) == 1
    assert _strip_count(dashed) == 4        # 40 pts, (6 on, 4 off)


def test_dashed_two_points_still_draws():
    node = build_orbit_node([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], PLAN_STYLE)
    assert _strip_count(_strokes(node)[-1]) == 1


def test_lines_ignore_scene_lighting():
    # Lines have no normals; scene lights (ambient + sun) rendered them at ~10% of
    # their colour, which is why orbits were nearly invisible. They must be unlit.
    from panda3d.core import AmbientLight, NodePath
    scene = NodePath("render")
    amb = scene.attach_new_node(AmbientLight("amb"))
    scene.set_light(amb)
    node = build_orbit_node(STYLED_POINTS, ORBIT_STYLE)
    node.reparent_to(scene)
    from panda3d.core import LightAttrib
    for stroke in _strokes(node):
        lights = stroke.get_net_state().get_attrib(LightAttrib)
        assert lights is None or lights.get_num_on_lights() == 0
