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
