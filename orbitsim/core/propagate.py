"""Two-body propagation (analytic + numeric)."""
from typing import Sequence
import numpy as np
from scipy.integrate import solve_ivp

from orbitsim.core.state import StateVector
from orbitsim.core.elements import (  # noqa: F401 (elements_to_state re-exported)
    state_to_elements, elements_to_state, elements_to_state_rotated, KeplerianElements,
)
from orbitsim.core.frames import perifocal_rotation
from orbitsim.core.kepler import (
    true_to_eccentric_anomaly,
    eccentric_to_mean_anomaly,
    mean_to_true_anomaly,
    true_to_hyperbolic_anomaly,
    hyperbolic_to_mean_anomaly,
    mean_to_true_anomaly_hyperbolic,
)
from orbitsim.core.constants import R_EARTH, J2_EARTH
from orbitsim.core.bodies import CelestialBody


def propagate_kepler(state: StateVector, dt: float) -> StateVector:
    """Analytic two-body propagation by dt seconds (on-rails).

    Parameters
    ----------
    state : StateVector
        Initial state vector.
    dt : float
        Time step [s].

    Returns
    -------
    StateVector
        Propagated state vector.

    Notes
    -----
    1. Convert state to elements.
    2. Compute mean anomaly at current true anomaly.
    3. Advance mean anomaly by n*dt.
    4. Solve for new true anomaly.
    5. Return new state.
    """
    return KeplerOrbit(state).state_at(dt)


def _solve_kepler_elliptic_vec(M: np.ndarray, e: float, tol: float = 1e-12,
                               max_iter: int = 50) -> np.ndarray:
    """Vectorized ``kepler.solve_kepler_elliptic``: same start, Newton step and per-element
    stopping rule (|f| < tol, checked after the update), applied to an array of M."""
    M_norm = np.asarray(M, dtype=np.float64) % (2.0 * np.pi)
    E = (M_norm + e * np.sin(M_norm)) if e < 0.8 else np.full_like(M_norm, np.pi)
    active = M_norm >= 1e-14
    E = np.where(active, E, 0.0)
    for _ in range(max_iter):
        if not active.any():
            return E
        idx = np.flatnonzero(active)
        Ea, Ma = E[idx], M_norm[idx]
        f = Ea - e * np.sin(Ea) - Ma
        fp = 1.0 - e * np.cos(Ea)
        E[idx] = Ea - f / fp
        active[idx[np.abs(f) < tol]] = False
    if active.any():
        raise RuntimeError(f"Kepler solve did not converge for {int(active.sum())} sample(s), e={e}")
    return E


class KeplerOrbit:
    """One state's two-body orbit, set up once for propagation to many epochs.

    ``state_at(dt)`` is bit-identical to ``propagate_kepler(state, dt)`` (which delegates
    here) but derives the elements, mean motion, initial mean anomaly and perifocal
    rotation only once. ``positions(dts)`` is a vectorized positions-only fast path for
    scans (agrees with ``state_at`` to ~1e-12 relative; elliptic orbits are vectorized,
    hyperbolic ones fall back to ``state_at`` per epoch).
    """

    def __init__(self, state: StateVector) -> None:
        elem = state_to_elements(state)
        self.elements = elem
        e, a = elem.e, elem.a
        self.elliptic = e < 1.0
        if self.elliptic:
            E0 = true_to_eccentric_anomaly(elem.nu, e)
            self._M0 = eccentric_to_mean_anomaly(E0, e)
        else:
            F0 = true_to_hyperbolic_anomaly(elem.nu, e)
            self._M0 = hyperbolic_to_mean_anomaly(F0, e)
        self._n = np.sqrt(elem.mu / abs(a) ** 3)
        self._q = perifocal_rotation(elem.raan, elem.i, elem.argp)

    def state_at(self, dt: float) -> StateVector:
        """State ``dt`` seconds after the initial state (exactly ``propagate_kepler``)."""
        elem = self.elements
        e = elem.e
        if self.elliptic:
            M = (self._M0 + self._n * dt) % (2.0 * np.pi)
            nu_new = mean_to_true_anomaly(M, e)
        else:
            M = self._M0 + self._n * dt
            nu_new = mean_to_true_anomaly_hyperbolic(M, e)
        new_elem = KeplerianElements(
            a=elem.a,
            e=elem.e,
            i=elem.i,
            raan=elem.raan,
            argp=elem.argp,
            nu=nu_new,
            mu=elem.mu,
            epoch_s=elem.epoch_s + dt,
        )
        return elements_to_state_rotated(new_elem, self._q)

    def positions(self, dts) -> np.ndarray:
        """Positions [m], shape (len(dts), 3), at each offset in ``dts`` [s]."""
        dts = np.asarray(dts, dtype=np.float64)
        if not self.elliptic:
            return np.array([self.state_at(float(t)).r for t in dts], dtype=np.float64).reshape(-1, 3)
        elem = self.elements
        e = elem.e
        M = (self._M0 + self._n * dts) % (2.0 * np.pi)
        E = _solve_kepler_elliptic_vec(M, e)
        nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                              np.sqrt(1.0 - e) * np.cos(E / 2.0)) % (2.0 * np.pi)
        radius = elem.semi_latus_rectum / (1.0 + e * np.cos(nu))
        r_pqw = np.stack([radius * np.cos(nu), radius * np.sin(nu), np.zeros_like(nu)], axis=1)
        return np.ascontiguousarray(r_pqw @ self._q.T, dtype=np.float64)


def propagate_numeric(
    state: StateVector,
    dt: float,
    *,
    j2: bool = False,
    third_bodies: Sequence[tuple[CelestialBody, np.ndarray]] = (),
) -> StateVector:
    """High-fidelity numeric propagation via scipy.integrate.solve_ivp (DOP853).

    Parameters
    ----------
    state : StateVector
        Initial state vector.
    dt : float
        Time step [s].
    j2 : bool
        Include J2 oblateness perturbation (Earth only for now).
    third_bodies : sequence of (CelestialBody, position)
        Third-body gravitational perturbations (not yet used).

    Returns
    -------
    StateVector
        Propagated state vector.
    """
    mu = state.mu
    y0 = np.concatenate([state.r, state.v])

    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        r_vec = y[:3]
        v_vec = y[3:]
        r_mag = np.linalg.norm(r_vec)
        r3 = r_mag**3

        a_grav = -mu * r_vec / r3

        a_total = a_grav

        if j2:
            r5 = r_mag**5
            z = r_vec[2]
            z2_r2 = (z / r_mag) ** 2
            factor = 1.5 * J2_EARTH * mu * R_EARTH**2 / r5
            a_j2 = factor * np.array(
                [
                    r_vec[0] * (5.0 * z2_r2 - 1.0),
                    r_vec[1] * (5.0 * z2_r2 - 1.0),
                    r_vec[2] * (5.0 * z2_r2 - 3.0),
                ]
            )
            a_total = a_total + a_j2

        return np.concatenate([v_vec, a_total])

    sol = solve_ivp(
        deriv,
        [0.0, dt],
        y0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-6,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"propagate_numeric failed: {sol.message}")

    r_final = sol.y[:3, -1]
    v_final = sol.y[3:, -1]

    return StateVector(
        r=r_final,
        v=v_final,
        mu=mu,
        epoch_s=state.epoch_s + dt,
    )
