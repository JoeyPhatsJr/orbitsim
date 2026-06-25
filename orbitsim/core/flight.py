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


def _accel(r, v, fuel, dry_mass_kg, thrust_dir_unit, throttle, max_thrust_n,
           ve_mps, mu):
    """Acceleration [m/s^2] = two-body gravity + thrust (thrust off when fuel
    <= 0)."""
    a = np.zeros(3)
    rn = np.linalg.norm(r)
    if mu != 0.0 and rn > 0.0:
        a = a - mu * r / rn**3
    if fuel > 0.0 and throttle > 0.0:
        mass = dry_mass_kg + fuel
        a = a + (throttle * max_thrust_n / mass) * thrust_dir_unit
    return a


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
    """Integrate r, v, fuel over dt_s under two-body gravity + thrust
    (fixed-step RK4).

    Mass decreases as fuel burns (real rocket equation). Thrust direction is
    held constant over the interval (the sim layer slews attitude separately).
    When fuel reaches zero, thrust stops mid-interval.

    Returns
    -------
    (StateVector, float)
        New state (same mu/epoch_s+dt) and remaining fuel [kg].
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

    def deriv(r_, v_, fuel_):
        a = _accel(r_, v_, fuel_, dry_mass_kg, thrust_dir_unit, throttle,
                   max_thrust_n, ve_mps, mu)
        df = -mdot if fuel_ > 0.0 else 0.0
        return v_, a, df

    for _ in range(substeps):
        # If fuel is zero, no more thrust; just coast.
        if fuel <= 0.0:
            fuel = 0.0
            k1r, k1v, _ = deriv(r, v, 0.0)
            k2r, k2v, _ = deriv(r + 0.5 * h * k1r, v + 0.5 * h * k1v, 0.0)
            k3r, k3v, _ = deriv(r + 0.5 * h * k2r, v + 0.5 * h * k2v, 0.0)
            k4r, k4v, _ = deriv(r + h * k3r, v + h * k3v, 0.0)
            r = r + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
            v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        else:
            # Standard RK4 step, but check if fuel will go negative.
            k1r, k1v, k1f = deriv(r, v, fuel)
            # Predict if we would go negative at any RK stage.
            test_f1 = fuel + 0.5 * h * k1f
            if test_f1 < 0.0:
                # Fuel will deplete this step; calculate time to depletion.
                # mdot is constant, so t_deplete = fuel / mdot.
                t_burn = fuel / mdot if mdot > 0.0 else h
                # Burn for t_burn, then coast for (h - t_burn).
                h_burn = min(t_burn, h)
                # One substep of burning.
                k1r_b, k1v_b, k1f_b = deriv(r, v, fuel)
                k2r_b, k2v_b, k2f_b = deriv(r + 0.5 * h_burn * k1r_b,
                                            v + 0.5 * h_burn * k1v_b,
                                            fuel + 0.5 * h_burn * k1f_b)
                k3r_b, k3v_b, k3f_b = deriv(r + 0.5 * h_burn * k2r_b,
                                            v + 0.5 * h_burn * k2v_b,
                                            fuel + 0.5 * h_burn * k2f_b)
                k4r_b, k4v_b, k4f_b = deriv(r + h_burn * k3r_b,
                                            v + h_burn * k3v_b,
                                            fuel + h_burn * k3f_b)
                r = r + (h_burn / 6.0) * (k1r_b + 2 * k2r_b + 2 * k3r_b + k4r_b)
                v = v + (h_burn / 6.0) * (k1v_b + 2 * k2v_b + 2 * k3v_b + k4v_b)
                fuel = max(0.0, fuel + (h_burn / 6.0) * (k1f_b + 2 * k2f_b +
                                                         2 * k3f_b + k4f_b))
                # Coast for the remainder.
                h_coast = h - h_burn
                if h_coast > 0.0:
                    k1r_c, k1v_c, _ = deriv(r, v, 0.0)
                    k2r_c, k2v_c, _ = deriv(r + 0.5 * h_coast * k1r_c,
                                            v + 0.5 * h_coast * k1v_c, 0.0)
                    k3r_c, k3v_c, _ = deriv(r + 0.5 * h_coast * k2r_c,
                                            v + 0.5 * h_coast * k2v_c, 0.0)
                    k4r_c, k4v_c, _ = deriv(r + h_coast * k3r_c,
                                            v + h_coast * k3v_c, 0.0)
                    r = r + (h_coast / 6.0) * (k1r_c + 2 * k2r_c + 2 * k3r_c +
                                               k4r_c)
                    v = v + (h_coast / 6.0) * (k1v_c + 2 * k2v_c + 2 * k3v_c +
                                               k4v_c)
            else:
                # Normal RK4 step.
                k2r, k2v, k2f = deriv(r + 0.5 * h * k1r, v + 0.5 * h * k1v,
                                      test_f1)
                test_f2 = fuel + 0.5 * h * k2f
                k3r, k3v, k3f = deriv(r + 0.5 * h * k2r, v + 0.5 * h * k2v,
                                      test_f2)
                test_f3 = fuel + h * k3f
                k4r, k4v, k4f = deriv(r + h * k3r, v + h * k3v, test_f3)
                r = r + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
                v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
                fuel = max(0.0, fuel + (h / 6.0) * (k1f + 2 * k2f + 2 * k3f +
                                                    k4f))

    return (
        StateVector(r=r, v=v, mu=mu, epoch_s=state.epoch_s + dt_s),
        max(0.0, fuel),
    )
