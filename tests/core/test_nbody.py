import numpy as np
from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core import nbody as nb


def test_bodies_sit_on_the_barycenter_axis_at_t0():
    e = nb.EARTH.state_at(0.0)
    m = nb.MOON.state_at(0.0)
    # Moon on +x at (1-ratio)d, Earth on -x at -ratio*d.
    assert np.allclose(m.r, [nb.MOON_X, 0.0, 0.0])
    assert np.allclose(e.r, [nb.EARTH_X, 0.0, 0.0])
    # Mass-weighted positions cancel at the barycenter.
    bary = MU_EARTH * e.r + MU_MOON * m.r
    assert np.linalg.norm(bary) < 1e-3 * MU_TOTAL_FOR_TEST


def test_bodies_are_circular_and_rotate_with_omega():
    # Quarter period later, the Moon has rotated 90 degrees (+x -> +y).
    t = (np.pi / 2) / nb.OMEGA_EM
    m = nb.MOON.state_at(t)
    assert np.allclose(m.r, [0.0, nb.MOON_X, 0.0], atol=1.0)
    # Circular speed = omega * radius; velocity perpendicular to radius.
    assert abs(np.linalg.norm(m.r) - nb.MOON_X) < 1.0
    assert abs(np.linalg.norm(m.v) - nb.OMEGA_EM * nb.MOON_X) < 1e-6
    assert abs(np.dot(m.r, m.v)) < 1.0


def test_single_attractor_matches_point_mass_gravity():
    r = np.array([2.0e7, 0.0, 0.0])
    a = nb.gravity_accel(r, 0.0, attractors=[nb.EARTH])
    e = nb.EARTH.state_at(0.0).r
    d = r - e
    expected = -MU_EARTH * d / np.linalg.norm(d)**3
    assert np.allclose(a, expected, rtol=1e-12)


def test_two_attractors_sum():
    r = np.array([1.0e8, 5.0e7, 0.0])
    a_both = nb.gravity_accel(r, 0.0, attractors=nb.EARTH_MOON)
    a_e = nb.gravity_accel(r, 0.0, attractors=[nb.EARTH])
    a_m = nb.gravity_accel(r, 0.0, attractors=[nb.MOON])
    assert np.allclose(a_both, a_e + a_m, rtol=1e-12)


MU_TOTAL_FOR_TEST = MU_EARTH + MU_MOON
