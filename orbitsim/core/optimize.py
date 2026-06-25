"""delta-V optimizer: porkchop grids + local refinement."""
import numpy as np

from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.transfers import lambert


def porkchop(
    state_dep: StateVector,
    state_arr: StateVector,
    dep_times_s: np.ndarray,
    tof_grid_s: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Grid of Lambert solves: total delta-V over departure time x time-of-flight.

    Parameters
    ----------
    state_dep, state_arr : StateVector
        Departure and arrival bodies'/vessels' states at epoch 0 of the grid.
    dep_times_s : np.ndarray
        Departure times relative to the states' epoch [s], shape (m,).
    tof_grid_s : np.ndarray
        Times of flight to test [s], shape (n,).
    mu : float

    Returns
    -------
    (dv_total, argmin) : (np.ndarray, (int, int))
        dv_total[i, j] for dep_times_s[i] and tof_grid_s[j]; argmin index pair.
        Infeasible cells are np.inf.
    """
    m = len(dep_times_s)
    n = len(tof_grid_s)
    dv = np.full((m, n), np.inf, dtype=np.float64)

    for i, t_dep in enumerate(dep_times_s):
        dep_state = propagate_kepler(state_dep, float(t_dep))
        for j, tof in enumerate(tof_grid_s):
            if tof <= 0:
                continue
            arr_state = propagate_kepler(state_arr, float(t_dep + tof))
            try:
                v1, v2 = lambert(dep_state.r, arr_state.r, float(tof), mu)
            except Exception:
                continue
            dv_dep = np.linalg.norm(v1 - dep_state.v)
            dv_arr = np.linalg.norm(arr_state.v - v2)
            dv[i, j] = dv_dep + dv_arr

    flat = int(np.argmin(dv))
    argmin = (flat // n, flat % n)
    return dv, argmin
