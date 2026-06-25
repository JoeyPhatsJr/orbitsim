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
