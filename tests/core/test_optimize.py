"""Tests for the porkchop grid and transfer optimizer."""
import numpy as np
from orbitsim.core.optimize import porkchop
from orbitsim.core.transfers import hohmann
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH


def _circular(r_m):
    v = np.sqrt(MU_EARTH / r_m)
    return StateVector(r=np.array([r_m, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)


def test_porkchop_minimum_near_hohmann():
    """The grid minimum for two coplanar circular orbits should match Hohmann.

    Crucial: the departure axis must span a full *synodic* period. A Hohmann-cost
    transfer only exists when the target happens to sit ~180 deg ahead at arrival,
    and that phasing recurs every synodic period T_syn = 2*pi / |w1 - w2|. Scanning a
    narrower window (e.g. one Hohmann time-of-flight) never catches the right phase,
    so the grid minimum there is ~2x Hohmann -- a property of the *geometry*, not the
    solver. Over a synodic period the true ~1.0x Hohmann basin appears.
    """
    r1, r2 = 7000e3, 14000e3
    dep = _circular(r1)
    arr_circular_v = np.sqrt(MU_EARTH / r2)
    arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, arr_circular_v, 0.0]), mu=MU_EARTH)

    h = hohmann(r1, r2, MU_EARTH)
    w1 = np.sqrt(MU_EARTH / r1**3)
    w2 = np.sqrt(MU_EARTH / r2**3)
    t_syn = 2.0 * np.pi / abs(w1 - w2)

    dep_times = np.linspace(0.0, t_syn, 20)
    tof_grid = np.linspace(0.5 * h.time_of_flight_s, 1.5 * h.time_of_flight_s, 30)

    dv, (i, j) = porkchop(dep, arr, dep_times, tof_grid, MU_EARTH)
    assert dv.shape == (len(dep_times), len(tof_grid))
    assert np.isfinite(dv[i, j])
    # The basin minimum should be within 10% of the ideal Hohmann total dv.
    assert dv[i, j] < 1.1 * h.dv_total_mps


def test_optimize_transfer_beats_or_matches_grid():
    from orbitsim.core.optimize import optimize_transfer
    r1, r2 = 7000e3, 14000e3
    dep = _circular(r1)
    arr_v = np.sqrt(MU_EARTH / r2)
    arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, arr_v, 0.0]), mu=MU_EARTH)
    h = hohmann(r1, r2, MU_EARTH)
    dep_times = np.linspace(0.0, h.time_of_flight_s, 8)
    tof_grid = np.linspace(0.5 * h.time_of_flight_s, 1.5 * h.time_of_flight_s, 20)

    dv_grid, (i, j) = porkchop(dep, arr, dep_times, tof_grid, MU_EARTH)
    sol = optimize_transfer(dep, arr, dep_times, tof_grid, MU_EARTH)
    assert sol.kind == "lambert"
    # Refined solution is no worse than the coarse grid minimum (+ small tolerance).
    assert sol.dv_total_mps <= dv_grid[i, j] * 1.01


def test_earth_to_mars_porkchop_has_window():
    import pytest
    pytest.importorskip("skyfield")
    from orbitsim.core.ephemeris import available
    if not available():
        pytest.skip("DE440 kernel unavailable (offline)")
    from orbitsim.core.optimize import interplanetary_porkchop
    # 2031 launch window scan, TOF 150–300 days.
    base = 31.0 * 365.25 * 86400.0
    dep_times = np.linspace(base, base + 2 * 365.25 * 86400.0, 24)  # 2 years
    tof_grid = np.linspace(150 * 86400.0, 300 * 86400.0, 16)
    dv, (i, j) = interplanetary_porkchop("EARTH", "MARS", dep_times, tof_grid)
    assert dv.shape == (24, 16)
    finite = dv[np.isfinite(dv)]
    assert finite.size > 0
    # Minimum heliocentric transfer dv (departure + arrival v-infinity) is in a
    # physically plausible band — a few km/s to low tens of km/s.
    assert 2.0e3 < dv[i, j] < 5.0e4


# ---------------------------------------------------------------------------
# intercept_node tests
# ---------------------------------------------------------------------------

import numpy as np
from orbitsim.core.state import StateVector
from orbitsim.core.bodies import EARTH
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.maneuvers import apply_maneuver
from orbitsim.core.optimize import intercept_node


def _ship_and_target():
    mu = EARTH.mu
    # Ship in a circular ~7,000 km LEO (coplanar, equatorial).
    r1 = 7.0e6
    ship = StateVector(r=np.array([r1, 0.0, 0.0]),
                       v=np.array([0.0, np.sqrt(mu / r1), 0.0]), mu=mu, epoch_s=0.0)
    # Target in a higher circular orbit, phased ~100° ahead (off 180° to avoid the
    # Lambert plane singularity).
    r2 = 4.0e7
    ang = np.radians(100.0)
    tv = np.sqrt(mu / r2)
    target = StateVector(
        r=np.array([r2 * np.cos(ang), r2 * np.sin(ang), 0.0]),
        v=np.array([-tv * np.sin(ang), tv * np.cos(ang), 0.0]), mu=mu, epoch_s=0.0)
    return ship, target, mu


def test_intercept_node_closes_the_loop():
    # The node must actually intercept: after applying the burn, the post-burn
    # trajectory passes close to the target. We verify this directly with
    # closest_approach (no need to recover the solver's internal time-of-flight),
    # so the test does not constrain the solver's tof discretization.
    from orbitsim.core.rendezvous import closest_approach
    ship, target, mu = _ship_and_target()
    dep = np.linspace(0.0, 3.0e3, 16)
    tof = np.linspace(1.0e3, 4.0e4, 40)
    node = intercept_node(ship, target, mu, dep, tof)
    t_dep = node.epoch_s - ship.epoch_s
    post = apply_maneuver(ship, node)                      # state at the node epoch
    target_at_dep = propagate_kepler(target, t_dep)        # target at the same epoch
    ca = closest_approach(post, target_at_dep, window_s=6.0e4, coarse_samples=2000)
    assert ca.separation_m < 1.0e4    # within 10 km of a moving target — a real intercept


def test_intercept_node_rtn_projection_is_lossless():
    # The node's RTN components must recompose to the inertial burn vector.
    ship, target, mu = _ship_and_target()
    dep = np.linspace(0.0, 3.0e3, 12)
    tof = np.linspace(1.0e3, 4.0e4, 30)
    node = intercept_node(ship, target, mu, dep, tof, refine=False)
    burn = propagate_kepler(ship, node.epoch_s - ship.epoch_s)
    v_hat = burn.v / np.linalg.norm(burn.v)
    h = np.cross(burn.r, burn.v); h_hat = h / np.linalg.norm(h)
    r_hat = np.cross(h_hat, v_hat)
    recomposed = (node.dv_prograde_mps * v_hat + node.dv_normal_mps * h_hat
                  + node.dv_radial_mps * r_hat)
    assert node.magnitude_mps > 0.0
    # Orthonormal RTN basis -> recomposed magnitude must equal the node magnitude
    # (lossless projection); a wrong/ non-orthonormal basis would break this.
    assert abs(np.linalg.norm(recomposed) - node.magnitude_mps) < 1e-9


def test_intercept_node_raises_when_infeasible():
    ship, target, mu = _ship_and_target()
    # All TOFs non-positive => no feasible Lambert cell.
    import pytest
    with pytest.raises(ValueError):
        intercept_node(ship, target, mu, np.array([0.0]), np.array([-1.0, 0.0]))


# ---------------------------------------------------------------------------
# Callable-based optimizer tests (interplanetary transfers)
# ---------------------------------------------------------------------------

from orbitsim.core.optimize import porkchop_callable, intercept_node_callable


def _keplerian_target_fn(target_state):
    """Wrap a Keplerian target as a callable for comparison testing."""
    def fn(t_abs):
        dt = t_abs - target_state.epoch_s
        return propagate_kepler(target_state, dt)
    return fn


def test_porkchop_callable_matches_keplerian():
    """porkchop_callable with a Keplerian callable should give the same result
    as the original porkchop function."""
    from orbitsim.core.optimize import porkchop
    ship, target, mu = _ship_and_target()
    dep_times = np.linspace(0.0, 3.0e3, 8)
    tof_grid = np.linspace(1.0e3, 4.0e4, 12)
    dv_orig, (i0, j0) = porkchop(ship, target, dep_times, tof_grid, mu)
    target_fn = _keplerian_target_fn(target)
    dv_call, (i1, j1) = porkchop_callable(ship, target_fn, mu, dep_times, tof_grid)
    np.testing.assert_allclose(dv_call, dv_orig, rtol=1e-10)
    assert (i0, j0) == (i1, j1)


def test_intercept_node_callable_matches_keplerian():
    """intercept_node_callable with a Keplerian callable produces the same node
    as the original intercept_node."""
    ship, target, mu = _ship_and_target()
    dep = np.linspace(0.0, 3.0e3, 12)
    tof = np.linspace(1.0e3, 4.0e4, 30)
    node_orig = intercept_node(ship, target, mu, dep, tof, refine=False)
    target_fn = _keplerian_target_fn(target)
    node_call = intercept_node_callable(ship, target_fn, mu, dep, tof, refine=False)
    assert abs(node_orig.epoch_s - node_call.epoch_s) < 1e-9
    assert abs(node_orig.dv_prograde_mps - node_call.dv_prograde_mps) < 1e-9
    assert abs(node_orig.dv_normal_mps - node_call.dv_normal_mps) < 1e-9
    assert abs(node_orig.dv_radial_mps - node_call.dv_radial_mps) < 1e-9


def test_intercept_node_callable_with_planet_target():
    """intercept_node_callable works with a planet state_at function (Mars)."""
    from orbitsim.core.planets import mars_state_at, A_EARTH, A_MARS
    from orbitsim.core.constants import MU_SUN
    mu = EARTH.mu
    r0 = 7.0e6
    ship = StateVector(r=np.array([r0, 0.0, 0.0]),
                       v=np.array([0.0, np.sqrt(mu / r0), 0.0]), mu=mu, epoch_s=0.0)
    hohmann_tof = np.pi * np.sqrt(((A_EARTH + A_MARS) / 2.0)**3 / MU_SUN)
    w_earth = np.sqrt(MU_SUN / A_EARTH**3)
    w_mars = np.sqrt(MU_SUN / A_MARS**3)
    synodic = 2.0 * np.pi / abs(w_earth - w_mars)
    dep = np.linspace(0.0, min(synodic, 2 * 365.25 * 86400.0), 20)
    tof = np.linspace(0.3 * hohmann_tof, 2.0 * hohmann_tof, 20)
    node = intercept_node_callable(ship, mars_state_at, mu, dep, tof, refine=False)
    assert node.magnitude_mps > 0.0
    assert node.magnitude_mps < 5.0e4


def test_intercept_node_callable_raises_when_infeasible():
    ship, _, mu = _ship_and_target()
    import pytest
    def dummy_target(t):
        return StateVector(r=np.array([1e12, 0, 0]), v=np.zeros(3), mu=mu, epoch_s=t)
    with pytest.raises(ValueError):
        intercept_node_callable(ship, dummy_target, mu, np.array([0.0]), np.array([-1.0, 0.0]))


# --- Interplanetary Earth-departure planning -------------------------------
import pytest
from orbitsim.core.optimize import hyperbolic_injection_velocity, interplanetary_departure_node


def _outgoing_v_infinity(r, v, mu):
    """Independently compute the outgoing hyperbolic-excess velocity vector of the
    orbit (r, v) via the eccentricity vector and its asymptote true anomaly.

    Deliberately a *different* formulation than the constructor under test (which
    works from the orbit equation / true anomaly at r), so agreement is a real
    round-trip, not a tautology.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    r_mag = np.linalg.norm(r)
    energy = np.dot(v, v) / 2.0 - mu / r_mag
    assert energy > 0.0, "orbit is not hyperbolic"
    e_vec = ((np.dot(v, v) - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)
    assert e > 1.0
    h_hat = np.cross(r, v)
    h_hat /= np.linalg.norm(h_hat)
    p_hat = e_vec / e                     # toward periapsis
    q_hat = np.cross(h_hat, p_hat)        # prograde, 90 deg past periapsis
    nu_inf = np.arccos(-1.0 / e)
    asymptote_hat = np.cos(nu_inf) * p_hat + np.sin(nu_inf) * q_hat
    return np.sqrt(2.0 * energy) * asymptote_hat


def test_hyperbolic_injection_reproduces_requested_v_infinity():
    r = np.array([7.0e6, 0.0, 0.0])
    v_inf = np.array([2000.0, 2500.0, 800.0])   # arbitrary 3D excess velocity
    v_inj = hyperbolic_injection_velocity(r, v_inf, MU_EARTH)
    v_inf_out = _outgoing_v_infinity(r, v_inj, MU_EARTH)
    assert np.allclose(v_inf_out, v_inf, rtol=1e-6, atol=1e-3)


def test_hyperbolic_injection_speed_matches_energy():
    r = np.array([0.0, 7.2e6, 0.0])
    v_inf = np.array([3100.0, 0.0, 0.0])
    v_inj = hyperbolic_injection_velocity(r, v_inf, MU_EARTH)
    r_mag = np.linalg.norm(r)
    expected_speed = np.sqrt(np.dot(v_inf, v_inf) + 2.0 * MU_EARTH / r_mag)
    assert abs(np.linalg.norm(v_inj) - expected_speed) < 1e-6


def test_hyperbolic_injection_is_coplanar_with_r_and_vinf():
    r = np.array([5.0e6, 4.0e6, 0.0])
    v_inf = np.array([-1500.0, 2200.0, 900.0])
    v_inj = hyperbolic_injection_velocity(r, v_inf, MU_EARTH)
    plane_normal = np.cross(r, v_inf)
    plane_normal /= np.linalg.norm(plane_normal)
    assert abs(np.dot(v_inj, plane_normal)) < 1e-3


def test_hyperbolic_injection_raises_when_vinf_parallel_to_r():
    r = np.array([7.0e6, 0.0, 0.0])
    v_inf = np.array([3000.0, 0.0, 0.0])   # radial: no unique orbit plane
    with pytest.raises(ValueError):
        hyperbolic_injection_velocity(r, v_inf, MU_EARTH)


def _leo_ship():
    r0 = 7.0e6
    return StateVector(r=np.array([r0, 0.0, 0.0]),
                       v=np.array([0.0, np.sqrt(MU_EARTH / r0), 0.0]),
                       mu=MU_EARTH, epoch_s=0.0)


def _mars_window():
    from orbitsim.core.planets import A_EARTH, A_MARS
    from orbitsim.core.constants import MU_SUN
    hohmann_tof = np.pi * np.sqrt(((A_EARTH + A_MARS) / 2.0) ** 3 / MU_SUN)
    w_earth = np.sqrt(MU_SUN / A_EARTH ** 3)
    w_mars = np.sqrt(MU_SUN / A_MARS ** 3)
    synodic = 2.0 * np.pi / abs(w_earth - w_mars)
    dep = np.linspace(0.0, min(synodic, 2 * 365.25 * 86400.0), 20)
    tof = np.linspace(0.5 * hohmann_tof, 1.5 * hohmann_tof, 20)
    return dep, tof


def test_interplanetary_departure_earth_to_mars_is_escape_within_window():
    from orbitsim.core.planets import mars_state_at, sun_state_at
    ship = _leo_ship()
    dep, tof = _mars_window()
    node = interplanetary_departure_node(ship, mars_state_at, sun_state_at, dep, tof)

    # Epoch lands inside the swept departure window.
    assert dep[0] <= node.epoch_s - ship.epoch_s <= dep[-1]

    # Applying the burn yields an Earth-escape hyperbola (specific energy > 0).
    post = apply_maneuver(ship, node)
    r_mag = np.linalg.norm(post.r)
    energy = np.dot(post.v, post.v) / 2.0 - MU_EARTH / r_mag
    assert energy > 0.0

    # And the injection cost is physically sane (well under an obviously-wrong value).
    assert 0.0 < node.magnitude_mps < 20_000.0


def test_interplanetary_departure_raises_when_infeasible():
    from orbitsim.core.planets import mars_state_at, sun_state_at
    ship = _leo_ship()
    with pytest.raises(ValueError):
        interplanetary_departure_node(
            ship, mars_state_at, sun_state_at, np.array([0.0]), np.array([-1.0, 0.0]))


# ---------------------------------------------------------------------------
# Multi-core fan-out: a passed executor must not change results
# ---------------------------------------------------------------------------
#
# The `executor` parameter fans departure-time rows across worker threads or
# processes. Correctness invariant: the pooled result is *identical* to the
# serial result, whatever the worker count. We validate the fan-out logic with
# a ThreadPoolExecutor (fast, in-process) and picklability with a real
# ProcessPoolExecutor on a small grid.
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from orbitsim.core.optimize import (
    intercept_node, interplanetary_departure_node_by_name, planning_planet_state,
)


def test_porkchop_executor_matches_serial_threads():
    r1, r2 = 7000e3, 14000e3
    dep = _circular(r1)
    arr_v = np.sqrt(MU_EARTH / r2)
    arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, arr_v, 0.0]), mu=MU_EARTH)
    dep_times = np.linspace(0.0, 3.0e4, 17)
    tof_grid = np.linspace(1.0e3, 4.0e4, 23)

    dv_serial, argmin_serial = porkchop(dep, arr, dep_times, tof_grid, MU_EARTH)
    with ThreadPoolExecutor(max_workers=4) as ex:
        dv_par, argmin_par = porkchop(dep, arr, dep_times, tof_grid, MU_EARTH, executor=ex)
    assert np.array_equal(dv_serial, dv_par)     # bit-identical, inf-aware
    assert argmin_serial == argmin_par


def test_porkchop_executor_matches_serial_processes():
    r1, r2 = 7000e3, 14000e3
    dep = _circular(r1)
    arr_v = np.sqrt(MU_EARTH / r2)
    arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, arr_v, 0.0]), mu=MU_EARTH)
    dep_times = np.linspace(0.0, 3.0e4, 8)       # small grid: spawn cost dominates
    tof_grid = np.linspace(1.0e3, 4.0e4, 10)

    dv_serial, argmin_serial = porkchop(dep, arr, dep_times, tof_grid, MU_EARTH)
    with ProcessPoolExecutor(max_workers=2) as ex:
        dv_par, argmin_par = porkchop(dep, arr, dep_times, tof_grid, MU_EARTH, executor=ex)
    assert np.array_equal(dv_serial, dv_par)
    assert argmin_serial == argmin_par


def test_intercept_node_executor_matches_serial():
    ship, target, mu = _ship_and_target()
    dep = np.linspace(0.0, 3.0e3, 13)
    tof = np.linspace(1.0e3, 4.0e4, 29)
    node_serial = intercept_node(ship, target, mu, dep, tof, refine=False)
    with ThreadPoolExecutor(max_workers=4) as ex:
        node_par = intercept_node(ship, target, mu, dep, tof, refine=False, executor=ex)
    assert node_serial.epoch_s == node_par.epoch_s
    assert node_serial.dv_prograde_mps == node_par.dv_prograde_mps
    assert node_serial.dv_normal_mps == node_par.dv_normal_mps
    assert node_serial.dv_radial_mps == node_par.dv_radial_mps


def test_planning_planet_state_returns_geocentric_offline_or_online():
    # Whatever the ephemeris availability, planning_planet_state must return a
    # finite geocentric StateVector (real DE440 or circular fallback).
    st = planning_planet_state("MARS", 0.0)
    assert st.r.shape == (3,) and np.all(np.isfinite(st.r))
    assert st.mu == MU_EARTH


def test_interplanetary_departure_by_name_escape_within_window():
    ship = _leo_ship()
    dep, tof = _mars_window()
    node = interplanetary_departure_node_by_name(ship, "MARS", "SUN", dep, tof)
    assert dep[0] <= node.epoch_s - ship.epoch_s <= dep[-1]
    post = apply_maneuver(ship, node)
    energy = np.dot(post.v, post.v) / 2.0 - MU_EARTH / np.linalg.norm(post.r)
    assert energy > 0.0
    assert 0.0 < node.magnitude_mps < 20_000.0


def test_interplanetary_departure_by_name_executor_matches_serial():
    ship = _leo_ship()
    dep, tof = _mars_window()
    node_serial = interplanetary_departure_node_by_name(ship, "MARS", "SUN", dep, tof)
    with ThreadPoolExecutor(max_workers=4) as ex:
        node_par = interplanetary_departure_node_by_name(
            ship, "MARS", "SUN", dep, tof, executor=ex)
    assert node_serial.epoch_s == node_par.epoch_s
    assert node_serial.dv_prograde_mps == node_par.dv_prograde_mps
    assert node_serial.dv_normal_mps == node_par.dv_normal_mps
    assert node_serial.dv_radial_mps == node_par.dv_radial_mps
