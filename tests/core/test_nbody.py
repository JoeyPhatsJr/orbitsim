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


def test_reduces_to_two_body_with_stationary_earth():
    # A point mass fixed at the origin (radius 0 => never moves) makes the ship's
    # motion a clean Kepler orbit, isolating the Verlet integrator from the moving-
    # frame residual a barycentrically-orbiting Earth would add (Omega^2 * r_earth,
    # ~35 m/quarter-orbit — that residual is real physics, not integrator error).
    earth0 = nb._CircularBody(MU_EARTH, 0.0)
    r = 7.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)
    # Quarter orbit: fine step so 2nd-order Verlet error is sub-metre.
    out = nb.propagate_nbody(st, period / 4, attractors=[earth0], max_step_s=0.5)
    assert np.linalg.norm(out.r - propagate_kepler(st, period / 4).r) < 1.0
    # One period closes.
    out2 = nb.propagate_nbody(st, period, attractors=[earth0], max_step_s=0.5)
    assert np.linalg.norm(out2.r - propagate_kepler(st, period).r) < 10.0


def test_propagation_is_reversible():
    st = _leo_state()
    T = 3600.0 * 3
    fwd = nb.propagate_nbody(st, T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    back = nb.propagate_nbody(fwd, -T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    assert np.linalg.norm(back.r - st.r) < 1.0


def test_jacobi_constant_conserved_over_seven_days():
    # A ship out between Earth and Moon where both attractors matter.
    e = nb.EARTH.state_at(0.0)
    st = StateVector(r=np.array([1.2e8, 0.0, 0.0]),
                     v=np.array([0.0, 900.0, 50.0]), mu=MU_EARTH, epoch_s=0.0)
    c0 = nb.jacobi_constant(st, 0.0)
    # Velocity Verlet's Jacobi error is bounded O(h^2) (no secular drift); 200 s steps
    # give ~2.7e-7 here. (Do not relax the 1e-6 tolerance — tighten the step instead.)
    far = nb.propagate_nbody(st, 7 * 86400.0, attractors=nb.EARTH_MOON, max_step_s=200.0)
    c1 = nb.jacobi_constant(far, far.epoch_s)
    assert abs(c1 - c0) / abs(c0) < 1e-6


MU_TOTAL_FOR_TEST = MU_EARTH + MU_MOON
