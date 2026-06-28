"""Body registry + vessels; per-tick propagation."""
from dataclasses import dataclass, field
import numpy as np

from orbitsim.core.bodies import CelestialBody
from orbitsim.core.state import StateVector
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.attitude import quat_identity
from orbitsim.core.flight import tsiolkovsky_dv


@dataclass
class Vessel:
    """A point-mass vessel: orbital state, propulsion, and attitude.

    Delta-V is derived from the rocket equation via `delta_v_remaining`; there
    is no stored budget — fuel mass is the single source of truth.

    Attributes
    ----------
    name : str
    state : StateVector
        Current inertial state (mutable here in the sim layer; updated each tick).
    nodes : list[ManeuverNode]
        Pending maneuver nodes for the sandbox (each its own list; default empty).
    dry_mass_kg : float
        Dry (no-fuel) mass [kg].
    fuel_mass_kg : float
        Current propellant mass [kg].
    max_thrust_n : float
        Maximum engine thrust [N].
    exhaust_velocity_mps : float
        Specific impulse equivalent: effective exhaust velocity [m/s].
    max_turn_rate_radps : float
        Maximum rotation rate for attitude slew [rad/s].
    throttle : float
        Current throttle [0, 1].
    sas_mode : str
        Stability augmentation mode ("OFF", "STABILITY", or SAS_MODES).
    orientation : np.ndarray
        Attitude quaternion [w, x, y, z], unit norm.
    """

    name: str
    state: StateVector
    nodes: list = field(default_factory=list)
    # Propulsion (SI).
    dry_mass_kg: float = 1000.0
    fuel_mass_kg: float = 0.0
    max_thrust_n: float = 0.0
    exhaust_velocity_mps: float = 3000.0
    # Attitude / control.
    max_turn_rate_radps: float = 0.6
    throttle: float = 0.0
    sas_mode: str = "OFF"
    unlimited_dv: bool = False
    orientation: np.ndarray = field(default_factory=quat_identity)
    sas_target_pos: object = None   # inertial target position [m] for TARGET/ANTITARGET, or None

    @property
    def mass_kg(self) -> float:
        """Current total mass = dry + fuel [kg]."""
        return self.dry_mass_kg + self.fuel_mass_kg

    @property
    def delta_v_remaining(self) -> float:
        """Remaining delta-V from the rocket equation [m/s]; inf if unlimited, 0 if no fuel."""
        if self.unlimited_dv:
            return float("inf")
        if self.fuel_mass_kg <= 0.0:
            return 0.0
        return tsiolkovsky_dv(self.exhaust_velocity_mps, self.mass_kg,
                              self.dry_mass_kg)


class World:
    """Holds the central body and all vessels; advances them analytically or
    numerically.

    Parameters
    ----------
    central : CelestialBody
        The body all vessel states are referenced to.
    vessels : list[Vessel]
    solar_system : bool
        When True, propagation uses the full inner solar system N-body model
        (Sun + Mercury + Venus + Moon + Mars as perturbers in the geocentric frame).
    """

    def __init__(self, central: CelestialBody, vessels: list[Vessel],
                 solar_system: bool = False) -> None:
        self.central = central
        self.vessels = vessels
        self.solar_system = solar_system

    def step(self, sim_dt_s: float) -> None:
        """Advance every vessel by sim_dt_s: slew attitude, then translate.

        When solar_system is True, propagation includes the Sun and inner planets
        as gravitational perturbers alongside the Moon.
        """
        from orbitsim.core.attitude import (
            slew_toward, sas_target_dir, nose_direction,
        )

        if self.solar_system:
            from orbitsim.core.nbody import propagate_solar_system
            from orbitsim.core.flight import integrate_powered_solar
            propagate_fn = propagate_solar_system
            powered_fn = integrate_powered_solar
        else:
            from orbitsim.core.nbody import propagate_earth_moon
            from orbitsim.core.flight import integrate_powered_nbody
            propagate_fn = propagate_earth_moon
            powered_fn = integrate_powered_nbody

        for vessel in self.vessels:
            # 1) Attitude: slew toward the SAS hold direction (if any) each tick.
            if vessel.sas_mode not in ("OFF", "STABILITY"):
                try:
                    target = sas_target_dir(vessel.sas_mode, vessel.state, vessel.sas_target_pos)
                except ValueError:
                    target = None
                if target is not None:
                    vessel.orientation = slew_toward(
                        vessel.orientation, target, vessel.max_turn_rate_radps, sim_dt_s)
            # 2) Translation.
            if vessel.throttle > 0.0 and (vessel.fuel_mass_kg > 0.0 or vessel.unlimited_dv):
                new_state, new_fuel = powered_fn(
                    vessel.state,
                    dry_mass_kg=vessel.dry_mass_kg,
                    fuel_kg=vessel.fuel_mass_kg,
                    thrust_dir_unit=nose_direction(vessel.orientation),
                    throttle=vessel.throttle,
                    max_thrust_n=vessel.max_thrust_n,
                    ve_mps=vessel.exhaust_velocity_mps,
                    dt_s=sim_dt_s,
                )
                vessel.state = new_state
                if not vessel.unlimited_dv:
                    vessel.fuel_mass_kg = new_fuel
            else:
                vessel.state = propagate_fn(vessel.state, sim_dt_s)

    def any_thrusting(self) -> bool:
        """True if any vessel is currently producing thrust (throttle>0 and
        fuel)."""
        return any(v.throttle > 0.0 and (v.fuel_mass_kg > 0.0 or v.unlimited_dv)
                   for v in self.vessels)
