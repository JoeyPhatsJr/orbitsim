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
