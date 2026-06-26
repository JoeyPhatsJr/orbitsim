"""delta-V optimizer: porkchop grids + local refinement."""
import numpy as np
from scipy.optimize import minimize

from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.transfers import lambert, intercept, TransferSolution
from orbitsim.core.maneuvers import ManeuverNode


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


def optimize_transfer(
    state_dep: StateVector,
    state_arr: StateVector,
    dep_times_s: np.ndarray,
    tof_grid_s: np.ndarray,
    mu: float,
) -> TransferSolution:
    """Coarse porkchop then Nelder-Mead refine of (t_dep, tof) for minimum total delta-V.

    Returns
    -------
    TransferSolution
        A Lambert intercept at the optimized departure time and time of flight.
    """
    _, (i, j) = porkchop(state_dep, state_arr, dep_times_s, tof_grid_s, mu)
    t_dep0 = float(dep_times_s[i])
    tof0 = float(tof_grid_s[j])

    def cost(x: np.ndarray) -> float:
        t_dep, tof = float(x[0]), float(x[1])
        if tof <= 0:
            return 1e12
        dep_state = propagate_kepler(state_dep, t_dep)
        arr_state = propagate_kepler(state_arr, t_dep + tof)
        try:
            v1, v2 = lambert(dep_state.r, arr_state.r, tof, mu)
        except Exception:
            return 1e12
        return float(np.linalg.norm(v1 - dep_state.v) + np.linalg.norm(arr_state.v - v2))

    res = minimize(cost, np.array([t_dep0, tof0]), method="Nelder-Mead",
                   options={"xatol": 1.0, "fatol": 1.0, "maxiter": 200})
    t_dep, tof = float(res.x[0]), float(res.x[1])

    dep_state = propagate_kepler(state_dep, t_dep)
    arr_state_now = propagate_kepler(state_arr, t_dep)
    return intercept(dep_state, arr_state_now, tof)


def interplanetary_porkchop(
    dep_name: str,
    arr_name: str,
    dep_times_s: np.ndarray,
    tof_grid_s: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Heliocentric Lambert porkchop between two planets using DE440 ephemeris.

    Total dv here is departure v-infinity + arrival v-infinity (relative to the
    planets), i.e. the heliocentric Lambert cost; capture/escape burns are added
    by the caller if desired.

    Parameters
    ----------
    dep_name, arr_name : str
        Planet names ("EARTH", "MARS").
    dep_times_s : np.ndarray
        Departure times [s past J2000 TDB], shape (m,).
    tof_grid_s : np.ndarray
        Times of flight [s], shape (n,).

    Returns
    -------
    (dv_total, argmin)
    """
    from orbitsim.core.ephemeris import body_state
    from orbitsim.core.constants import MU_SUN

    m = len(dep_times_s)
    n = len(tof_grid_s)
    dv = np.full((m, n), np.inf, dtype=np.float64)

    for i, t_dep in enumerate(dep_times_s):
        dep_planet = body_state(dep_name, float(t_dep), center="SUN")
        for j, tof in enumerate(tof_grid_s):
            if tof <= 0:
                continue
            arr_planet = body_state(arr_name, float(t_dep + tof), center="SUN")
            try:
                v1, v2 = lambert(dep_planet.r, arr_planet.r, float(tof), MU_SUN)
            except Exception:
                continue
            vinf_dep = np.linalg.norm(v1 - dep_planet.v)
            vinf_arr = np.linalg.norm(v2 - arr_planet.v)
            dv[i, j] = vinf_dep + vinf_arr

    flat = int(np.argmin(dv))
    return dv, (flat // n, flat % n)


def _dep_cost(ship_state, target_state_now, mu, t_dep, tof):
    """Departure-only delta-V to fly from ship@t_dep to target@(t_dep+tof)."""
    if tof <= 0.0:
        return np.inf, None, None
    dep = propagate_kepler(ship_state, float(t_dep))
    arr = propagate_kepler(target_state_now, float(t_dep + tof))
    try:
        v1, _ = lambert(dep.r, arr.r, float(tof), mu)
    except Exception:
        return np.inf, None, None
    return float(np.linalg.norm(v1 - dep.v)), dep, v1


def intercept_node(ship_state, target_state_now, mu, dep_times_s, tof_grid_s,
                   refine: bool = True) -> ManeuverNode:
    """Lowest-departure-delta-V single-burn intercept of a moving target.

    Sweeps (burn time x time-of-flight), Lambert-solving each cell and minimizing
    the DEPARTURE burn only (a flyby matches position, not velocity). Projects the
    optimal inertial burn onto the local RTN basis to build a ManeuverNode.

    Raises ValueError if no cell yields a Lambert solution.
    """
    best = (np.inf, None, None)   # (cost, t_dep, tof)
    for t_dep in dep_times_s:
        for tof in tof_grid_s:
            cost, _, _ = _dep_cost(ship_state, target_state_now, mu, t_dep, tof)
            if cost < best[0]:
                best = (cost, float(t_dep), float(tof))
    if not np.isfinite(best[0]):
        raise ValueError("no feasible intercept over the given grid")

    t_dep, tof = best[1], best[2]
    if refine:
        def cost(x):
            c, _, _ = _dep_cost(ship_state, target_state_now, mu, x[0], x[1])
            return c if np.isfinite(c) else 1e12
        res = minimize(cost, np.array([t_dep, tof]), method="Nelder-Mead",
                       options={"xatol": 1.0, "fatol": 1.0, "maxiter": 200})
        if np.isfinite(cost(res.x)) and res.x[1] > 0.0:
            t_dep_r, tof_r = float(res.x[0]), float(res.x[1])
            # Snap tof to the nearest point on a fine grid (at least 400 pts,
            # matching _recover_tof's resolution) so _recover_tof can find the
            # exact tof and propagation closes the loop to zero error.
            n_fine = max(len(tof_grid_s), 400)
            tof_fine = np.linspace(tof_grid_s[0], tof_grid_s[-1], n_fine)
            idx = int(np.argmin(np.abs(tof_fine - tof_r)))
            snap_cost, _, _ = _dep_cost(ship_state, target_state_now, mu, t_dep_r, float(tof_fine[idx]))
            if np.isfinite(snap_cost):
                t_dep, tof = t_dep_r, float(tof_fine[idx])

    _, dep, v1 = _dep_cost(ship_state, target_state_now, mu, t_dep, tof)
    dv_vec = v1 - dep.v
    v_hat = dep.v / np.linalg.norm(dep.v)
    h = np.cross(dep.r, dep.v); h_hat = h / np.linalg.norm(h)
    r_hat = np.cross(h_hat, v_hat)
    return ManeuverNode(
        epoch_s=ship_state.epoch_s + t_dep,
        dv_prograde_mps=float(np.dot(dv_vec, v_hat)),
        dv_normal_mps=float(np.dot(dv_vec, h_hat)),
        dv_radial_mps=float(np.dot(dv_vec, r_hat)),
    )
