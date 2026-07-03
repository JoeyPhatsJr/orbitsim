"""delta-V optimizer: porkchop grids + local refinement."""
import numpy as np
from scipy.optimize import minimize, brentq

from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.transfers import lambert, intercept, TransferSolution
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.constants import MU_EARTH, MU_SUN


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


def porkchop_callable(
    ship_state: StateVector,
    target_state_fn,
    mu: float,
    dep_times_s: np.ndarray,
    tof_grid_s: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Porkchop grid where the target position is given by a callable.

    Unlike ``porkchop``, the target is NOT Keplerian-propagated. Instead,
    ``target_state_fn(t_absolute)`` is called for each arrival epoch,
    returning a StateVector in the same frame as *ship_state*.

    The ship IS Keplerian-propagated (two-body seed, good enough for the
    coarse grid + Nelder-Mead refine that follows).

    Returns (dv_total, argmin) like ``porkchop``.
    """
    m = len(dep_times_s)
    n = len(tof_grid_s)
    dv = np.full((m, n), np.inf, dtype=np.float64)
    t0 = ship_state.epoch_s

    for i, t_dep in enumerate(dep_times_s):
        dep = propagate_kepler(ship_state, float(t_dep))
        for j, tof in enumerate(tof_grid_s):
            if tof <= 0:
                continue
            arr = target_state_fn(t0 + float(t_dep) + float(tof))
            try:
                v1, v2 = lambert(dep.r, arr.r, float(tof), mu)
            except Exception:
                continue
            dv_dep = np.linalg.norm(v1 - dep.v)
            dv_arr = np.linalg.norm(arr.v - v2)
            dv[i, j] = dv_dep + dv_arr

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
            t_dep, tof = float(res.x[0]), float(res.x[1])

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


def hyperbolic_injection_velocity(r_m, v_inf_mps, mu: float) -> np.ndarray:
    """Velocity at position ``r_m`` that puts a vessel on an escape hyperbola whose
    outgoing asymptote velocity equals ``v_inf_mps`` (all Earth-centered, SI).

    Solves the minimum-plane-change injection: the orbit plane is forced to contain
    both ``r`` and the requested excess-velocity direction, so the departure hyperbola
    lies in a single well-defined plane. Within that plane the eccentricity is the one
    scalar unknown, fixed by the orbit equation at ``r`` and the asymptote true anomaly
    ``nu_inf = arccos(-1/e)`` — a 1-D root solve.

    Raises ValueError if ``v_inf`` is parallel to ``r`` (radial escape: the orbit plane
    is undefined) or if ``v_inf`` has zero magnitude.
    """
    r = np.asarray(r_m, dtype=np.float64)
    v_inf = np.asarray(v_inf_mps, dtype=np.float64)
    r_mag = np.linalg.norm(r)
    v_inf_mag = np.linalg.norm(v_inf)
    if v_inf_mag == 0.0:
        raise ValueError("v_infinity must be non-zero")
    r_hat = r / r_mag
    v_inf_hat = v_inf / v_inf_mag

    normal = np.cross(r_hat, v_inf_hat)
    normal_mag = np.linalg.norm(normal)
    if normal_mag < 1e-12:
        raise ValueError("v_infinity parallel to r: escape plane is undefined")
    h_hat = normal / normal_mag              # orbit-plane normal
    transverse_hat = np.cross(h_hat, r_hat)  # prograde in-plane axis

    # phi: prograde angle (0, pi) from r to the outgoing asymptote direction. The
    # asymptote sits at true anomaly nu_inf and r at nu = nu_inf - phi.
    phi = np.arccos(np.clip(np.dot(r_hat, v_inf_hat), -1.0, 1.0))

    def orbit_gap(e):
        nu_inf = np.arccos(-1.0 / e)
        nu = nu_inf - phi
        p = mu * (e * e - 1.0) / v_inf_mag**2          # semi-latus rectum (>0)
        return p - r_mag * (1.0 + e * np.cos(nu))       # 0 when r lies on the orbit

    # orbit_gap(1+) <= 0 and orbit_gap(inf) -> +inf, so a root e>1 always exists.
    e = brentq(orbit_gap, 1.0 + 1e-9, 1.0e6, xtol=1e-12, rtol=1e-14)

    nu_inf = np.arccos(-1.0 / e)
    nu = nu_inf - phi
    h = (mu / v_inf_mag) * np.sqrt(e * e - 1.0)        # specific angular momentum
    v_radial = (mu / h) * e * np.sin(nu)
    v_transverse = h / r_mag
    return v_radial * r_hat + v_transverse * transverse_hat


def interplanetary_departure_node(
    ship_state: StateVector,
    target_state_fn,
    sun_state_fn,
    dep_times_s,
    tof_grid_s,
) -> ManeuverNode:
    """Plan an Earth-departure burn onto a heliocentric transfer toward a planet.

    ``target_state_fn(t_abs)`` and ``sun_state_fn(t_abs)`` return *geocentric* states
    (as ``core.planets`` / the ephemeris cache do). The heliocentric Lambert leg is
    solved with ``MU_SUN`` from Earth's heliocentric position to the planet's; the
    departure hyperbolic-excess velocity (``v_inf``) is then converted into an
    Earth-centered injection burn (``MU_EARTH``) at the ship's position.

    The grid sweep minimises ``|v_inf|`` (equivalently C3) — the standard
    minimum-energy departure objective. Raises ValueError if no cell yields a
    Lambert solution.
    """
    best = (np.inf, None, None)   # (|v_inf|, t_dep, v_inf vector)
    epoch = ship_state.epoch_s
    for t_dep in dep_times_s:
        if t_dep < 0.0:
            continue
        dep_epoch = epoch + float(t_dep)
        sun_dep = sun_state_fn(dep_epoch)
        earth_helio_r = -sun_dep.r          # Earth heliocentric position + velocity
        earth_helio_v = -sun_dep.v
        for tof in tof_grid_s:
            if tof <= 0.0:
                continue
            arr_epoch = dep_epoch + float(tof)
            sun_arr = sun_state_fn(arr_epoch)
            target = target_state_fn(arr_epoch)
            target_helio_r = target.r - sun_arr.r   # geocentric -> heliocentric
            try:
                transfer_v, _ = lambert(earth_helio_r, target_helio_r, float(tof), MU_SUN)
            except Exception:
                continue
            v_inf = transfer_v - earth_helio_v
            cost = float(np.linalg.norm(v_inf))
            if cost < best[0]:
                best = (cost, float(t_dep), v_inf)

    if not np.isfinite(best[0]):
        raise ValueError("no feasible interplanetary departure over the given grid")

    _, t_dep, v_inf = best
    departure = propagate_kepler(ship_state, t_dep)
    injection_v = hyperbolic_injection_velocity(departure.r, v_inf, MU_EARTH)
    dv_vec = injection_v - departure.v

    v_hat = departure.v / np.linalg.norm(departure.v)
    h_hat = np.cross(departure.r, departure.v)
    h_hat /= np.linalg.norm(h_hat)
    r_hat = np.cross(h_hat, v_hat)
    return ManeuverNode(
        epoch_s=epoch + t_dep,
        dv_prograde_mps=float(np.dot(dv_vec, v_hat)),
        dv_normal_mps=float(np.dot(dv_vec, h_hat)),
        dv_radial_mps=float(np.dot(dv_vec, r_hat)),
    )


def _dep_cost_callable(ship_state, target_state_fn, mu, t_dep, tof):
    """Departure-only delta-V using a callable target."""
    if tof <= 0.0:
        return np.inf, None, None
    dep = propagate_kepler(ship_state, float(t_dep))
    arr = target_state_fn(ship_state.epoch_s + float(t_dep) + float(tof))
    try:
        v1, _ = lambert(dep.r, arr.r, float(tof), mu)
    except Exception:
        return np.inf, None, None
    return float(np.linalg.norm(v1 - dep.v)), dep, v1


def intercept_node_callable(ship_state, target_state_fn, mu, dep_times_s,
                            tof_grid_s, refine: bool = True) -> ManeuverNode:
    """Like ``intercept_node`` but the target is a callable ``state_fn(t_abs)``.

    For interplanetary transfers where the target (a planet) cannot be
    Keplerian-propagated around Earth.  The ship is still Keplerian-seeded.

    Raises ValueError if no cell yields a Lambert solution.
    """
    best = (np.inf, None, None)
    for t_dep in dep_times_s:
        for tof in tof_grid_s:
            cost, _, _ = _dep_cost_callable(ship_state, target_state_fn, mu, t_dep, tof)
            if cost < best[0]:
                best = (cost, float(t_dep), float(tof))
    if not np.isfinite(best[0]):
        raise ValueError("no feasible intercept over the given grid")

    t_dep, tof = best[1], best[2]
    if refine:
        def cost(x):
            c, _, _ = _dep_cost_callable(ship_state, target_state_fn, mu, x[0], x[1])
            return c if np.isfinite(c) else 1e12
        res = minimize(cost, np.array([t_dep, tof]), method="Nelder-Mead",
                       options={"xatol": 1.0, "fatol": 1.0, "maxiter": 200})
        if np.isfinite(cost(res.x)) and res.x[1] > 0.0:
            t_dep, tof = float(res.x[0]), float(res.x[1])

    _, dep, v1 = _dep_cost_callable(ship_state, target_state_fn, mu, t_dep, tof)
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
