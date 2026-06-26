import numpy as np
from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
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


def _leo_state():
    r = 7.0e6
    # State referenced to Earth's *barycentric* position so it's a clean Earth orbit.
    e = nb.EARTH.state_at(0.0)
    return StateVector(r=e.r + np.array([r, 0.0, 0.0]),
                       v=e.v + np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                       mu=MU_EARTH, epoch_s=0.0)


def test_reduces_to_two_body_with_only_earth():
    st = _leo_state()
    period = 2 * np.pi * np.sqrt(7.0e6**3 / MU_EARTH)
    # Geocentric two-body reference (subtract Earth's fixed barycentric offset).
    e = nb.EARTH.state_at(0.0)
    geo = StateVector(r=st.r - e.r, v=st.v - e.v, mu=MU_EARTH, epoch_s=0.0)
    # Quarter orbit: fine step so 2nd-order Verlet error is sub-metre.
    out = nb.propagate_nbody(st, period / 4, attractors=[nb.EARTH], max_step_s=0.5)
    ref = propagate_kepler(geo, period / 4).r + nb.EARTH.state_at(period / 4).r
    assert np.linalg.norm(out.r - ref) < 1.0
    # One period closes.
    out2 = nb.propagate_nbody(st, period, attractors=[nb.EARTH], max_step_s=0.5)
    ref2 = propagate_kepler(geo, period).r + nb.EARTH.state_at(period).r
    assert np.linalg.norm(out2.r - ref2) < 10.0


def test_propagation_is_reversible():
    st = _leo_state()
    T = 3600.0 * 3
    fwd = nb.propagate_nbody(st, T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    back = nb.propagate_nbody(fwd, -T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    assert np.linalg.norm(back.r - st.r) < 1.0


MU_TOTAL_FOR_TEST = MU_EARTH + MU_MOON
