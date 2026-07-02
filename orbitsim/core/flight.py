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

    Strang (symmetric) operator splitting per substep: half the substep's
    rocket-equation velocity impulse, a gravity drift over the full substep
    (RK4 on the two-body field), then the other half impulse. The impulses
    are *exact* rocket-equation increments, so the total delta-V telescopes
    to ve*ln(m0/mf) and fuel reaches exactly zero independent of how
    depletion aligns with the grid; the symmetric placement makes the split
    2nd-order accurate (impulse-first is only 1st order). Thrust direction
    is held constant over the interval (the sim layer slews attitude
    separately).

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
        # First half of the substep's exact rocket-equation impulse.
        dm_half = 0.0
        if throttle > 0.0 and fuel > 0.0 and mdot > 0.0:
            t_burn = min(h, fuel / mdot)
            dm_half = 0.5 * mdot * t_burn
            m_start = dry_mass_kg + fuel
            v = v + ve_mps * np.log(m_start / (m_start - dm_half)) * thrust_dir_unit
            fuel = max(0.0, fuel - dm_half)
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
        # Second half impulse (telescopes with the first: total is exact).
        if dm_half > 0.0:
            m_mid = dry_mass_kg + fuel
            v = v + ve_mps * np.log(m_mid / (m_mid - dm_half)) * thrust_dir_unit
            fuel = max(0.0, fuel - dm_half)

    return (
        StateVector(r=r, v=v, mu=mu, epoch_s=state.epoch_s + dt_s),
        float(fuel),
    )


def integrate_powered_nbody(
    state: StateVector,
    dry_mass_kg: float,
    fuel_kg: float,
    thrust_dir_unit: np.ndarray,
    throttle: float,
    max_thrust_n: float,
    ve_mps: float,
    dt_s: float,
) -> tuple:
    """Integrate r, v, fuel over dt_s under earth_moon_accel + thrust.

    Strang splitting per substep: half the exact rocket-equation velocity
    impulse, a velocity-Verlet drift under earth_moon_accel, then the other
    half impulse (2nd order; same fuel-reaches-zero and delta-V telescoping
    guarantees as integrate_powered). Substep count is proximity-aware.

    Returns
    -------
    (StateVector, float)
        New state (epoch_s + dt_s) and remaining fuel [kg].
    """
    from orbitsim.core.nbody import earth_moon_accel, _earth_moon_substeps

    thrust_dir_unit = np.asarray(thrust_dir_unit, dtype=np.float64)
    r = np.asarray(state.r, dtype=np.float64).copy()
    v = np.asarray(state.v, dtype=np.float64).copy()
    fuel = float(fuel_kg)
    t = state.epoch_s

    # Proximity sub-stepping is only meaningful when the bodies act (mu != 0).
    # In free space (mu == 0, e.g. the rocket-equation unit test) there are no
    # bodies, so use a fixed grid; the exact impulse telescopes regardless of n.
    n = _earth_moon_substeps(state, dt_s, max_step_s=3600.0) if state.mu != 0.0 else 50
    h = dt_s / n
    mdot = mass_flow_rate(throttle, max_thrust_n, ve_mps) if ve_mps > 0 else 0.0

    for _ in range(n):
        # First half of the substep's exact rocket-equation impulse.
        dm_half = 0.0
        if throttle > 0.0 and fuel > 0.0 and mdot > 0.0:
            t_burn = min(h, fuel / mdot)
            dm_half = 0.5 * mdot * t_burn
            m_start = dry_mass_kg + fuel
            v = v + ve_mps * np.log(m_start / (m_start - dm_half)) * thrust_dir_unit
            fuel = max(0.0, fuel - dm_half)
        # Gravity: velocity-Verlet under earth_moon_accel (zero if mu==0).
        if state.mu != 0.0:
            a0 = earth_moon_accel(r, t)
            v_half = v + 0.5 * a0 * h
            r = r + v_half * h
            t = t + h
            a1 = earth_moon_accel(r, t)
            v = v_half + 0.5 * a1 * h
        else:
            r = r + v * h
            t = t + h
        # Second half impulse (telescopes with the first: total is exact).
        if dm_half > 0.0:
            m_mid = dry_mass_kg + fuel
            v = v + ve_mps * np.log(m_mid / (m_mid - dm_half)) * thrust_dir_unit
            fuel = max(0.0, fuel - dm_half)

    return StateVector(r=r, v=v, mu=state.mu, epoch_s=state.epoch_s + dt_s), float(fuel)


def integrate_powered_solar(
    state: StateVector,
    dry_mass_kg: float,
    fuel_kg: float,
    thrust_dir_unit: np.ndarray,
    throttle: float,
    max_thrust_n: float,
    ve_mps: float,
    dt_s: float,
) -> tuple:
    """Integrate under solar_system_accel + thrust (same Strang splitting as nbody variant)."""
    from orbitsim.core.nbody import solar_system_accel, _solar_system_substeps

    thrust_dir_unit = np.asarray(thrust_dir_unit, dtype=np.float64)
    r = np.asarray(state.r, dtype=np.float64).copy()
    v = np.asarray(state.v, dtype=np.float64).copy()
    fuel = float(fuel_kg)
    t = state.epoch_s

    n = _solar_system_substeps(state, dt_s, max_step_s=3600.0) if state.mu != 0.0 else 50
    h = dt_s / n
    mdot = mass_flow_rate(throttle, max_thrust_n, ve_mps) if ve_mps > 0 else 0.0

    for _ in range(n):
        dm_half = 0.0
        if throttle > 0.0 and fuel > 0.0 and mdot > 0.0:
            t_burn = min(h, fuel / mdot)
            dm_half = 0.5 * mdot * t_burn
            m_start = dry_mass_kg + fuel
            v = v + ve_mps * np.log(m_start / (m_start - dm_half)) * thrust_dir_unit
            fuel = max(0.0, fuel - dm_half)
        if state.mu != 0.0:
            a0 = solar_system_accel(r, t)
            v_half = v + 0.5 * a0 * h
            r = r + v_half * h
            t = t + h
            a1 = solar_system_accel(r, t)
            v = v_half + 0.5 * a1 * h
        else:
            r = r + v * h
            t = t + h
        if dm_half > 0.0:
            m_mid = dry_mass_kg + fuel
            v = v + ve_mps * np.log(m_mid / (m_mid - dm_half)) * thrust_dir_unit
            fuel = max(0.0, fuel - dm_half)

    return StateVector(r=r, v=v, mu=state.mu, epoch_s=state.epoch_s + dt_s), float(fuel)
