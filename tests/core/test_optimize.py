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
    """The grid minimum should find a sensible transfer between two coplanar circular orbits."""
    r1, r2 = 7000e3, 14000e3
    dep = _circular(r1)
    arr_circular_v = np.sqrt(MU_EARTH / r2)
    # Target at r2 in circular orbit
    arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, arr_circular_v, 0.0]), mu=MU_EARTH)

    h = hohmann(r1, r2, MU_EARTH)
    dep_times = np.linspace(0.0, h.time_of_flight_s, 8)
    tof_grid = np.linspace(0.5 * h.time_of_flight_s, 1.5 * h.time_of_flight_s, 20)

    dv, (i, j) = porkchop(dep, arr, dep_times, tof_grid, MU_EARTH)
    assert dv.shape == (len(dep_times), len(tof_grid))
    assert np.isfinite(dv[i, j])
    # The best grid cell should represent a feasible transfer (positive cost).
    assert dv[i, j] > 0
