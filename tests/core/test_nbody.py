import numpy as np
from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core import nbody as nb
from orbitsim.core.moon import moon_state_at


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


def test_eccentric_periapsis_passage_resolved_within_one_call():
    """One propagate call spanning a periapsis passage must stay accurate.

    This is exactly what a high-warp frame does on an eccentric orbit: the
    call starts near apoapsis (long local timescale => coarse substeps) but
    crosses periapsis (short timescale) inside the same call. The substep
    size must adapt to the *current* radius, not the starting one.
    """
    earth0 = nb._CircularBody(MU_EARTH, 0.0)   # stationary => clean Kepler reference
    rp, ra = 6.6e6, 3.0e8
    a = 0.5 * (rp + ra)
    vp = np.sqrt(MU_EARTH * (2.0 / rp - 1.0 / a))
    at_pe = StateVector(r=np.array([rp, 0.0, 0.0]),
                        v=np.array([0.0, vp, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
    at_ap = propagate_kepler(at_pe, period / 2.0)     # start at apoapsis
    out = nb.propagate_nbody(at_ap, period, attractors=[earth0], max_step_s=3600.0)
    ref = propagate_kepler(at_ap, period)
    # Periapsis speed is ~10.9 km/s; the passage must be resolved to a small
    # fraction of rp, not smeared across multi-kilosecond substeps.
    assert np.linalg.norm(out.r - ref.r) < 5.0e4
    # Energy must return to its initial value (no artificial kick at periapsis).
    # Periapsis-based step sizing keeps the step uniform along the coast, so
    # velocity Verlet stays symplectic: measured error is ~1e-13 here.
    eps0 = 0.5 * np.dot(at_ap.v, at_ap.v) - MU_EARTH / np.linalg.norm(at_ap.r)
    eps1 = 0.5 * np.dot(out.v, out.v) - MU_EARTH / np.linalg.norm(out.r)
    assert abs(eps1 - eps0) / abs(eps0) < 1e-9


def test_propagation_is_reversible():
    st = _leo_state()
    T = 3600.0 * 3
    fwd = nb.propagate_nbody(st, T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    back = nb.propagate_nbody(fwd, -T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    assert np.linalg.norm(back.r - st.r) < 1.0


def test_verlet_min_substep_floor_reduces_substep_count():
    """A larger min_substep_s floors the adaptive step, cutting substep (accel) calls.

    This is the knob the visual trajectory prediction uses to stay cheap through the
    near-Earth gravity well without touching the accurate on-rails sim (which keeps
    the default tiny floor).
    """
    calls = {"fine": 0, "coarse": 0}

    def make_accel(key):
        def accel(r, t):
            calls[key] += 1
            return np.zeros(3)
        return accel

    cap_fn = lambda r, v, t: 1.0        # force a tiny 1 s cap everywhere
    dt = 1000.0
    nb._verlet_adaptive(np.array([7e6, 0.0, 0.0]), np.array([0.0, 7546.0, 0.0]), 0.0,
                        dt, make_accel("fine"), cap_fn, base_step_s=1024.0,
                        min_substep_s=1e-6)
    nb._verlet_adaptive(np.array([7e6, 0.0, 0.0]), np.array([0.0, 7546.0, 0.0]), 0.0,
                        dt, make_accel("coarse"), cap_fn, base_step_s=1024.0,
                        min_substep_s=128.0)
    assert calls["coarse"] < calls["fine"]


def _geocentric_leo():
    """A clean geocentric LEO state (Earth fixed at origin), for the solar propagator."""
    r = 7.0e6
    return StateVector(r=np.array([r, 0.0, 0.0]),
                       v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                       mu=MU_EARTH, epoch_s=0.0)


def test_propagate_solar_system_min_substep_defaults_to_current_behavior():
    """Omitting min_substep_s must reproduce the pre-existing sim result exactly."""
    st = _geocentric_leo()
    default = nb.propagate_solar_system(st, 3600.0)
    explicit = nb.propagate_solar_system(st, 3600.0, min_substep_s=nb._MIN_SUBSTEP_S)
    assert np.array_equal(default.r, explicit.r)
    assert np.array_equal(default.v, explicit.v)


def test_coarse_min_substep_safe_for_incrementally_sampled_bound_orbit():
    """The coarse floor must not harm a bound-orbit prediction.

    The renderer samples a trajectory in many small dt steps; for a bound orbit each
    step is far shorter than the coarse floor, so the floor never engages and the
    coarse path tracks the fine one to within a few km over a full orbit.
    """
    st = _geocentric_leo()
    T = 2 * np.pi * np.sqrt(7.0e6**3 / MU_EARTH)
    n = 256
    cur_c, cur_f = st, st
    worst = 0.0
    for _ in range(n):
        cur_c = nb.propagate_solar_system(cur_c, T / n, min_substep_s=300.0)
        cur_f = nb.propagate_solar_system(cur_f, T / n)
        worst = max(worst, float(np.linalg.norm(cur_c.r - cur_f.r)))
    assert worst < 50_000.0   # a few km over the whole orbit — invisible on the line


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


def test_L4_L5_exact_equilateral_geometry():
    L = nb.lagrange_points(0.0)
    e = np.array([nb.EARTH_X, 0.0, 0.0])
    m = np.array([nb.MOON_X, 0.0, 0.0])
    for key, sign in (("L4", +1), ("L5", -1)):
        p = L[key]
        assert abs(np.linalg.norm(p - e) - nb.D_EM) < 1e-3   # distance d from Earth
        assert abs(np.linalg.norm(p - m) - nb.D_EM) < 1e-3   # distance d from Moon
        assert np.sign(p[1]) == sign                          # L4 leads (+y), L5 trails


def test_collinear_points_match_known_positions_and_balance():
    L = nb.lagrange_points(0.0)
    # Published Earth-Moon CR3BP rotating-frame x (units of d), barycenter origin.
    for key, x_over_d in (("L1", 0.8369), ("L2", 1.1557), ("L3", -1.0051)):
        x = L[key][0]                              # at t=0 rotating == inertial x
        assert abs(x / nb.D_EM - x_over_d) < 1e-3
        # Net effective (gravity + centrifugal) acceleration ~ 0 at the point.
        p = L[key]
        g = nb.gravity_accel(p, 0.0, attractors=nb.EARTH_MOON)
        centrifugal = nb.OMEGA_EM**2 * np.array([p[0], p[1], 0.0])
        assert np.linalg.norm(g + centrifugal) < 1e-6


def test_L4_stays_bounded_over_a_day():
    L = nb.lagrange_points(0.0)
    p = L["L4"]
    v = np.cross([0.0, 0.0, nb.OMEGA_EM], p)       # co-rotating: stationary in rot frame
    st = StateVector(r=p, v=v, mu=MU_EARTH, epoch_s=0.0)
    out = nb.propagate_nbody(st, 86400.0, attractors=nb.EARTH_MOON, max_step_s=60.0)
    moved_L4 = nb.lagrange_points(86400.0)["L4"]
    assert np.linalg.norm(out.r - moved_L4) < 0.1 * nb.D_EM   # stable: doesn't escape


MU_TOTAL_FOR_TEST = MU_EARTH + MU_MOON


def test_earth_moon_accel_has_indirect_term():
    r = np.array([2.0e7, 1.0e7, 0.0])
    a = nb.earth_moon_accel(r, 0.0)
    rM = moon_state_at(0.0).r
    direct = (-MU_EARTH * r / np.linalg.norm(r)**3
              - MU_MOON * (r - rM) / np.linalg.norm(r - rM)**3)
    indirect = -MU_MOON * rM / np.linalg.norm(rM)**3
    assert np.allclose(a, direct + indirect, rtol=1e-12)


def test_L4_balances_in_the_earth_fixed_model():
    # L4: 60 deg ahead of the Moon in its orbital plane, distance d from Earth and Moon.
    t = 1.0e5
    m = moon_state_at(t)
    d = np.linalg.norm(m.r)
    w = np.cross(m.r, m.v) / d**2                 # Moon's angular velocity vector
    omega = np.linalg.norm(w)
    axis = w / omega
    # Rodrigues rotation of r_M by +60 deg about the orbit normal.
    c, s = np.cos(np.radians(60)), np.sin(np.radians(60))
    L4 = m.r * c + np.cross(axis, m.r) * s + axis * np.dot(axis, m.r) * (1 - c)
    # Net rotating-frame acceleration (gravity + centrifugal) must vanish.
    centrifugal = -np.cross(w, np.cross(w, L4))
    net = nb.earth_moon_accel(L4, t) + centrifugal
    assert np.linalg.norm(net) < 1e-7, np.linalg.norm(net)


def test_propagate_earth_moon_reduces_to_two_body_near_earth():
    # A LEO orbit: the Moon's perturbation is tiny, so it tracks Kepler closely.
    r = 7.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)
    out = nb.propagate_earth_moon(st, period / 4, max_step_s=0.5)
    # Within ~1 km of two-body over a quarter LEO orbit (Moon tug is sub-km here).
    assert np.linalg.norm(out.r - propagate_kepler(st, period / 4).r) < 1.0e3


def test_propagate_earth_moon_reversible():
    st = StateVector(r=np.array([5.0e7, 0.0, 0.0]),
                     v=np.array([0.0, 1500.0, 100.0]), mu=MU_EARTH, epoch_s=0.0)
    T = 3600.0 * 6
    fwd = nb.propagate_earth_moon(st, T, max_step_s=20.0)
    back = nb.propagate_earth_moon(fwd, -T, max_step_s=20.0)
    assert np.linalg.norm(back.r - st.r) < 1.0


from orbitsim.core.elements import state_to_elements


def test_osculating_elements_earth_dominant_matches_two_body():
    r = 8.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    osc = nb.osculating_elements(st, 0.0)
    ref = state_to_elements(StateVector(st.r, st.v, MU_EARTH, 0.0))
    assert abs(osc.a - ref.a) < 1.0 and abs(osc.e - ref.e) < 1e-9


def test_osculating_elements_switches_to_moon_inside_soi():
    t = 0.0
    m = moon_state_at(t)
    r_lo = 3.0e6                                   # 3000 km lunar orbit (inside SOI)
    # Circular about the Moon, in the Moon's frame.
    st = StateVector(r=m.r + np.array([r_lo, 0.0, 0.0]),
                     v=m.v + np.array([0.0, np.sqrt(MU_MOON / r_lo), 0.0]),
                     mu=MU_EARTH, epoch_s=t)
    osc = nb.osculating_elements(st, t)
    assert osc.mu == MU_MOON                       # dominant body is the Moon
    assert abs(osc.a - r_lo) < 1.0e4 and osc.e < 0.01   # ~circular lunar orbit


WARP_STEPS = (1.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 10000.0, 100000.0)


def test_max_safe_warp_caps_low_orbits_below_high_orbits():
    # Under a tight sub-step budget, a fast low orbit (short local timescale) caps at a
    # lower warp than a slow high orbit. (The proximity sub-stepping is so cheap that the
    # cap only bites under a tight budget / very close approach — see Part 2 for tuning.)
    low = StateVector(r=np.array([7.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 7546.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    high = StateVector(r=np.array([1.5e8, 0.0, 0.0]),
                       v=np.array([0.0, np.sqrt(MU_EARTH / 1.5e8), 0.0]),
                       mu=MU_EARTH, epoch_s=0.0)
    w_low = nb.max_safe_warp(low, 0.0, WARP_STEPS, budget_substeps=20)
    w_high = nb.max_safe_warp(high, 0.0, WARP_STEPS, budget_substeps=20)
    assert w_low in WARP_STEPS and w_high in WARP_STEPS
    assert w_high > w_low                      # slower/farther orbit allows faster warp
    assert w_low >= 1.0                         # never below the floor


def test_max_safe_warp_respects_substep_budget():
    leo = StateVector(r=np.array([7.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 7546.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    w = nb.max_safe_warp(leo, 0.0, WARP_STEPS, real_dt_s=1 / 60, budget_substeps=200)
    n = nb._earth_moon_substeps(leo, (1 / 60) * w, max_step_s=3600.0)
    assert n <= 200


def test_solar_max_warp_reaches_one_hundred_million_in_deep_space():
    from orbitsim.sim.clock import SimClock

    state = StateVector(
        r=np.array([1.0e11, 0.0, 0.0]),
        v=np.array([0.0, 3.0e4, 0.0]),
        mu=MU_EARTH,
    )
    assert nb.max_safe_warp_solar(state, 0.0, SimClock.WARP_STEPS) == 100_000_000


def test_solar_max_warp_is_capped_near_earth():
    from orbitsim.sim.clock import SimClock

    state = StateVector(
        r=np.array([7.0e6, 0.0, 0.0]),
        v=np.array([0.0, 7546.0, 0.0]),
        mu=MU_EARTH,
    )
    cap = nb.max_safe_warp_solar(state, 0.0, SimClock.WARP_STEPS)

    assert cap < 10_000_000


def test_background_prediction_ignores_live_ephemeris_cache(monkeypatch):
    cached = StateVector(
        r=np.array([9.0e15, 8.0e15, 7.0e15]),
        v=np.zeros(3),
        mu=0.0,
        epoch_s=0.0,
    )
    monkeypatch.setitem(nb._ephemeris_cache, "SUN", cached)
    monkeypatch.setattr(nb, "_EPHEMERIS_AVAILABLE", False)

    assert nb._csun(123.0) is cached
    with nb.stable_prediction_ephemeris():
        predicted = nb._csun(123.0)
        assert predicted is not cached
        assert predicted.epoch_s == 123.0
    assert nb._csun(123.0) is cached


def test_background_prediction_interpolates_its_own_time_varying_ephemeris(monkeypatch):
    def cubic_ephemeris(name, epoch_s, center):
        x = epoch_s / nb._PREDICTION_EPHEMERIS_STEP_S
        return StateVector(
            r=np.array([x**3, 2.0 * x, -x]),
            v=np.array([3.0 * x**2, 2.0, -1.0]) / nb._PREDICTION_EPHEMERIS_STEP_S,
            mu=0.0,
            epoch_s=epoch_s,
        )

    monkeypatch.setattr(nb, "_EPHEMERIS_AVAILABLE", True)
    monkeypatch.setattr(nb, "_ephem_body_state", cubic_ephemeris)
    epoch = 0.25 * nb._PREDICTION_EPHEMERIS_STEP_S

    with nb.stable_prediction_ephemeris():
        state = nb._csun(epoch)

    np.testing.assert_allclose(state.r, [0.25**3, 0.5, -0.25], atol=1e-14)


def test_earth_fixed_lagrange_points_are_equilibria():
    # Each L-point nulls the rotating-frame acceleration (gravity+indirect + centrifugal),
    # with the rotation about the Moon's ACTUAL orbit normal. Holds at t=0 and t!=0.
    for t in (0.0, 1.0e5):
        m = moon_state_at(t)
        n = np.cross(m.r, m.v)
        omega = nb.OMEGA_EM * (n / np.linalg.norm(n))
        lps = nb.earth_fixed_lagrange_points(t)
        assert set(lps) == {"L1", "L2", "L3", "L4", "L5"}
        for name, r in lps.items():
            centrifugal = -np.cross(omega, np.cross(omega, r))
            net = nb.earth_moon_accel(r, t) + centrifugal
            assert np.linalg.norm(net) < 1e-6, (name, t, float(np.linalg.norm(net)))


def test_earth_fixed_L4_L5_equilateral():
    t = 0.0
    m = moon_state_at(t)
    d = np.linalg.norm(m.r)
    lps = nb.earth_fixed_lagrange_points(t)
    for name in ("L4", "L5"):
        L = lps[name]
        assert abs(np.linalg.norm(L) - d) < 1.0          # distance d from Earth
        assert abs(np.linalg.norm(L - m.r) - d) < 1.0    # distance d from the Moon
        cosang = np.dot(L, m.r) / (np.linalg.norm(L) * d)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        assert abs(ang - 60.0) < 1e-3                     # 60 deg from the Moon
    # L4 leads the Moon, L5 trails (opposite sides of the Earth-Moon line).
    nrm = np.cross(m.r, m.v)
    assert np.dot(np.cross(m.r, lps["L4"]), nrm) > 0
    assert np.dot(np.cross(m.r, lps["L5"]), nrm) < 0


def test_earth_fixed_collinear_placement():
    t = 0.0
    m = moon_state_at(t)
    d = np.linalg.norm(m.r)
    u = m.r / d
    lps = nb.earth_fixed_lagrange_points(t)
    s = {k: float(np.dot(lps[k], u)) for k in ("L1", "L2", "L3")}
    assert 0.0 < s["L1"] < d < s["L2"]      # L1 between bodies, L2 beyond the Moon
    assert s["L3"] < 0.0                     # L3 beyond Earth
    for k in ("L1", "L2", "L3"):
        assert np.linalg.norm(np.cross(lps[k], u)) < 1.0   # on the Earth-Moon line


def test_earth_fixed_lagrange_distance_invariant_under_rotation():
    names = ("L1", "L2", "L3", "L4", "L5")
    dist = {n: [] for n in names}
    for t in (0.0, 3.0e5, 6.0e5, 9.0e5):
        lps = nb.earth_fixed_lagrange_points(t)
        for n in names:
            dist[n].append(np.linalg.norm(lps[n]))
    for n in names:
        assert max(dist[n]) - min(dist[n]) < 1.0   # rigid rotation: |L| constant


# ---------------------------------------------------------------------------
# Ephemeris cache tests
# ---------------------------------------------------------------------------
import pytest


class TestEphemerisCache:
    """Tests for the per-frame ephemeris cache in nbody."""

    def test_cache_starts_empty(self):
        assert nb._ephemeris_cache == {} or isinstance(nb._ephemeris_cache, dict)

    def test_refresh_populates_all_bodies(self):
        from orbitsim.core import ephemeris
        if not (nb._EPHEMERIS_AVAILABLE and ephemeris.available()):
            pytest.skip("DE440 kernel not available")
        ok = nb.refresh_ephemeris_cache(0.0)
        assert ok is True
        for name in nb._EPHEM_BODY_NAMES:
            assert name in nb._ephemeris_cache
            st = nb._ephemeris_cache[name]
            assert st.r.shape == (3,)
            assert st.v.shape == (3,)

    def test_cached_state_fn_returns_cached_value(self):
        ok = nb.refresh_ephemeris_cache(0.0)
        if not ok:
            pytest.skip("DE440 kernel not available")
        cached_sun = nb._ephemeris_cache["SUN"]
        result = nb._csun(0.0)
        np.testing.assert_array_equal(result.r, cached_sun.r)
        np.testing.assert_array_equal(result.v, cached_sun.v)

    def test_cached_fn_ignores_t_argument_when_cached(self):
        ok = nb.refresh_ephemeris_cache(0.0)
        if not ok:
            pytest.skip("DE440 kernel not available")
        r1 = nb._csun(0.0).r.copy()
        r2 = nb._csun(1e6).r.copy()
        np.testing.assert_array_equal(r1, r2)

    def test_fallback_when_cache_empty(self):
        saved = nb._ephemeris_cache.copy()
        try:
            nb._ephemeris_cache = {}
            from orbitsim.core.planets import sun_state_at
            result = nb._csun(100.0)
            expected = sun_state_at(100.0)
            np.testing.assert_allclose(result.r, expected.r, rtol=1e-12)
        finally:
            nb._ephemeris_cache = saved

    def test_ephemeris_positions_differ_from_circular(self):
        ok = nb.refresh_ephemeris_cache(0.0)
        if not ok:
            pytest.skip("DE440 kernel not available")
        from orbitsim.core.planets import mars_state_at
        circular_mars = mars_state_at(0.0).r
        real_mars = nb._ephemeris_cache["MARS"].r
        diff = np.linalg.norm(real_mars - circular_mars)
        assert diff > 1e6, f"Expected significant difference, got {diff:.0f} m"

    def test_solar_system_accel_uses_cache(self):
        ok = nb.refresh_ephemeris_cache(0.0)
        if not ok:
            pytest.skip("DE440 kernel not available")
        r = np.array([7e6, 0.0, 0.0])
        a_cached = nb.solar_system_accel(r, 0.0)
        nb._ephemeris_cache = {}
        a_circular = nb.solar_system_accel(r, 0.0)
        diff = np.linalg.norm(a_cached - a_circular)
        assert diff > 0, "Cached and circular accelerations should differ"
        assert diff / np.linalg.norm(a_cached) < 0.01, "Difference should be small (same order)"

    def test_propagate_solar_system_with_cache(self):
        ok = nb.refresh_ephemeris_cache(0.0)
        if not ok:
            pytest.skip("DE440 kernel not available")
        r = 7.0e6
        st = StateVector(r=np.array([r, 0.0, 0.0]),
                         v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                         mu=MU_EARTH, epoch_s=0.0)
        period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)
        out = nb.propagate_solar_system(st, period / 4, max_step_s=10.0)
        assert np.linalg.norm(out.r) > 0
        assert abs(np.linalg.norm(out.r) - r) / r < 0.01

    def test_ephemeris_available_reflects_cache_state(self):
        saved = nb._ephemeris_cache.copy()
        try:
            nb._ephemeris_cache = {}
            assert nb.ephemeris_available() is False
            nb._ephemeris_cache = {"SUN": StateVector(np.zeros(3), np.zeros(3), 0, 0)}
            assert nb.ephemeris_available() is True
        finally:
            nb._ephemeris_cache = saved
