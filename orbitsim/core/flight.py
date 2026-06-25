"""Continuous-thrust flight physics (pure, float64, SI). Real rocket equation +
two-body powered RK4 integrator."""
import numpy as np

from orbitsim.core.state import StateVector


def tsiolkovsky_dv(ve_mps: float, m0_kg: float, mf_kg: float) -> float:
    """Ideal delta-V = ve * ln(m0/mf) [m/s]. Requires 0 < mf <= m0, ve > 0."""
    if ve_mps <= 0.0:
        raise ValueError(f"ve must be positive, got {ve_mps}")
    if mf_kg <= 0.0 or m0_kg < mf_kg:
        raise ValueError(f"need 0 < mf <= m0, got m0={m0_kg}, mf={mf_kg}")
    return float(ve_mps * np.log(m0_kg / mf_kg))


def fuel_burned_for_dv(ve_mps: float, m0_kg: float, dv_mps: float) -> float:
    """Propellant mass [kg] to produce dv from initial mass m0 — the rocket-equation
    inverse, m0 * (1 - exp(-dv/ve)). Requires ve > 0, m0 > 0, dv >= 0."""
    if ve_mps <= 0.0:
        raise ValueError(f"ve must be positive, got {ve_mps}")
    if m0_kg <= 0.0:
        raise ValueError(f"m0 must be positive, got {m0_kg}")
    if dv_mps < 0.0:
        raise ValueError(f"dv must be non-negative, got {dv_mps}")
    return float(m0_kg * (1.0 - np.exp(-dv_mps / ve_mps)))


def mass_flow_rate(throttle: float, max_thrust_n: float, ve_mps: float
                   ) -> float:
    """Propellant mass flow ṁ = throttle * thrust / ve [kg/s]."""
    if ve_mps <= 0.0:
        raise ValueError(f"ve must be positive, got {ve_mps}")
    return float(throttle * max_thrust_n / ve_mps)


def thrust_accel_mps2(throttle: float, max_thrust_n: float, mass_kg: float
                      ) -> float:
    """Thrust acceleration magnitude = throttle * thrust / mass [m/s^2]."""
    if mass_kg <= 0.0:
        raise ValueError(f"mass must be positive, got {mass_kg}")
    return float(throttle * max_thrust_n / mass_kg)


def _gravity_accel(r, mu):
    """Two-body gravitational acceleration [m/s^2] (zero when mu == 0)."""
    rn = np.linalg.norm(r)
    if mu != 0.0 and rn > 0.0:
        return -mu * r / rn**3
    return np.zeros(3)


def integrate_powered(
    state: StateVector,
    dry_mass_kg: float,
    fuel_kg: float,
    thrust_dir_unit: np.ndarray,
    throttle: float,
    max_thrust_n: float,
    ve_mps: float,
    dt_s: float,
    substeps: int = 50,
) -> tuple:
    """Integrate r, v, fuel over dt_s under two-body gravity + thrust.

    Operator splitting per substep: the thrust contributes an *exact*
    rocket-equation velocity impulse over the burn portion of the substep
    (so the total delta-V telescopes to ve*ln(m0/mf) and fuel reaches exactly
    zero, independent of how depletion aligns with the grid), then the state
    drifts under two-body gravity via RK4. Thrust direction is held constant
    over the interval (the sim layer slews attitude separately).

    Returns
    -------
    (StateVector, float)
        New state (same mu, epoch_s + dt_s) and remaining fuel [kg].
    """
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    thrust_dir_unit = np.asarray(thrust_dir_unit, dtype=np.float64)
    r = np.asarray(state.r, dtype=np.float64).copy()
    v = np.asarray(state.v, dtype=np.float64).copy()
    fuel = float(fuel_kg)
    mu = state.mu
    h = dt_s / substeps
    mdot = mass_flow_rate(throttle, max_thrust_n, ve_mps) if ve_mps > 0 else 0.0

    for _ in range(substeps):
        # Thrust: exact rocket-equation impulse over the burning portion of h.
        if throttle > 0.0 and fuel > 0.0 and mdot > 0.0:
            t_burn = min(h, fuel / mdot)
            m_start = dry_mass_kg + fuel
            m_end = m_start - mdot * t_burn
            v = v + ve_mps * np.log(m_start / m_end) * thrust_dir_unit
            fuel = max(0.0, fuel - mdot * t_burn)
        # Gravity drift: RK4 on the two-body field for the full substep.
        k1r = v
        k1v = _gravity_accel(r, mu)
        k2r = v + 0.5 * h * k1v
        k2v = _gravity_accel(r + 0.5 * h * k1r, mu)
        k3r = v + 0.5 * h * k2v
        k3v = _gravity_accel(r + 0.5 * h * k2r, mu)
        k4r = v + h * k3v
        k4v = _gravity_accel(r + h * k3r, mu)
        r = r + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    return (
        StateVector(r=r, v=v, mu=mu, epoch_s=state.epoch_s + dt_s),
        float(fuel),
    )
