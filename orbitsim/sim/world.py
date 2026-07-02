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
    landed_on : str or None
        Name of the body this vessel is resting on, or None in flight.
        Maintained by World.step's surface-contact resolution.
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
    sas_maneuver_dir: object = None  # unit burn direction for the MANEUVER hold, or None
    landed_on: object = None         # body name while resting on a surface, or None

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

    # Propellant assumed present for the impulse math while unlimited_dv is on
    # and the tank reads empty (fuel never depletes under unlimited, so this
    # only shapes the thrust acceleration, not the budget).
    UNLIMITED_FUEL_FLOOR_KG = 1000.0

    def step(self, sim_dt_s: float) -> None:
        """Advance every vessel by sim_dt_s: slew attitude, translate, then
        resolve surface contact.

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
                if vessel.sas_mode == "MANEUVER":
                    # The planned burn direction is computed by the maneuver UI
                    # and mirrored onto the vessel each frame.
                    target = vessel.sas_maneuver_dir
                else:
                    try:
                        target = sas_target_dir(
                            vessel.sas_mode, vessel.state, vessel.sas_target_pos)
                    except ValueError:
                        target = None
                if target is not None:
                    vessel.orientation = slew_toward(
                        vessel.orientation, target, vessel.max_turn_rate_radps, sim_dt_s)
            # 2) Translation.
            thrusting = vessel.throttle > 0.0 and (
                vessel.fuel_mass_kg > 0.0 or vessel.unlimited_dv)
            if thrusting:
                fuel_for_impulse = vessel.fuel_mass_kg
                if vessel.unlimited_dv:
                    fuel_for_impulse = max(fuel_for_impulse, self.UNLIMITED_FUEL_FLOOR_KG)
                new_state, new_fuel = powered_fn(
                    vessel.state,
                    dry_mass_kg=vessel.dry_mass_kg,
                    fuel_kg=fuel_for_impulse,
                    thrust_dir_unit=nose_direction(vessel.orientation),
                    throttle=vessel.throttle,
                    max_thrust_n=vessel.max_thrust_n,
                    ve_mps=vessel.exhaust_velocity_mps,
                    dt_s=sim_dt_s,
                )
                vessel.state = new_state
                if not vessel.unlimited_dv:
                    vessel.fuel_mass_kg = new_fuel
            elif vessel.landed_on is not None:
                # Resting on a surface with no thrust: ride the surface point
                # (exact and free) instead of integrating a free fall that the
                # contact clamp would immediately undo.
                self._ride_surface(vessel, sim_dt_s)
                continue
            else:
                vessel.state = propagate_fn(vessel.state, sim_dt_s)
            # 3) Surface contact: a vessel below the dominant body's surface is
            # placed on it with the body's velocity. This is what physically
            # bounds the integrator too — the singular r -> 0 region of the
            # gravity field is unreachable.
            self._resolve_surface_contact(vessel)

    def _contact_candidates(self, vessel):
        """(name, center_m, velocity_mps, radius_m) of bodies the vessel could
        currently be inside."""
        from orbitsim.core.moon import moon_state_at
        from orbitsim.core.bodies import MOON

        t = vessel.state.epoch_s
        if self.solar_system:
            from orbitsim.core.nbody import dominant_body_solar, geocentric_body_state
            body, _pos = dominant_body_solar(vessel.state.r, t)
            st = geocentric_body_state(body.name, t)
            return [(body.name, st.r, st.v, body.radius_m)]
        moon = moon_state_at(t)
        return [
            (self.central.name, np.zeros(3), np.zeros(3), self.central.radius_m),
            (MOON.name, moon.r, moon.v, MOON.radius_m),
        ]

    def _resolve_surface_contact(self, vessel) -> None:
        """Clamp a vessel that ended the tick below a surface onto that surface."""
        for name, center, velocity, radius in self._contact_candidates(vessel):
            rel = vessel.state.r - center
            dist = float(np.linalg.norm(rel))
            if dist >= radius:
                continue
            up = rel / dist if dist > 0.0 else np.array([0.0, 0.0, 1.0])
            vessel.state = StateVector(
                r=center + up * radius, v=np.asarray(velocity, dtype=np.float64),
                mu=vessel.state.mu, epoch_s=vessel.state.epoch_s)
            vessel.landed_on = name
            return
        vessel.landed_on = None

    def _ride_surface(self, vessel, sim_dt_s: float) -> None:
        """Advance a landed vessel by keeping it glued to its surface point
        (bodies don't rotate in this sim, so the inertial radial is fixed)."""
        from orbitsim.core.nbody import geocentric_body_state

        t0 = vessel.state.epoch_s
        t1 = t0 + sim_dt_s
        b0 = geocentric_body_state(vessel.landed_on, t0)
        b1 = geocentric_body_state(vessel.landed_on, t1)
        rel = vessel.state.r - b0.r
        dist = float(np.linalg.norm(rel))
        up = rel / dist if dist > 0.0 else np.array([0.0, 0.0, 1.0])
        radius = dist if dist > 0.0 else 1.0
        vessel.state = StateVector(
            r=b1.r + up * radius, v=b1.v, mu=vessel.state.mu, epoch_s=t1)

    def any_thrusting(self) -> bool:
        """True if any vessel is currently producing thrust (throttle>0 and
        fuel)."""
        return any(v.throttle > 0.0 and (v.fuel_mass_kg > 0.0 or v.unlimited_dv)
                   for v in self.vessels)
