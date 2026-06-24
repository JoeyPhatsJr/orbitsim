"""Two-body propagation (analytic + numeric)."""
from typing import Sequence
import numpy as np
from scipy.integrate import solve_ivp

from orbitsim.core.state import StateVector
from orbitsim.core.elements import state_to_elements, elements_to_state, KeplerianElements
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
    elem = state_to_elements(state)
    e = elem.e
    a = elem.a

    if e < 1.0:
        E0 = true_to_eccentric_anomaly(elem.nu, e)
        M0 = eccentric_to_mean_anomaly(E0, e)
        n = np.sqrt(elem.mu / abs(a) ** 3)
        M = (M0 + n * dt) % (2.0 * np.pi)
        nu_new = mean_to_true_anomaly(M, e)
    else:
        F0 = true_to_hyperbolic_anomaly(elem.nu, e)
        M0 = hyperbolic_to_mean_anomaly(F0, e)
        n = np.sqrt(elem.mu / abs(a) ** 3)
        M = M0 + n * dt
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
    return elements_to_state(new_elem)


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
