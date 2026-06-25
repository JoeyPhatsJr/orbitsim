"""Body registry + vessels; per-tick propagation."""
from dataclasses import dataclass, field
import numpy as np

from orbitsim.core.bodies import CelestialBody
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.attitude import quat_identity
from orbitsim.core.flight import tsiolkovsky_dv


@dataclass
class Vessel:
    """A point-mass vessel: orbital state, propulsion, and attitude.

    delta_v_budget_mps is retained for display/back-compat; the authoritative
    delta-V is the derived `delta_v_remaining` (rocket equation).

    Attributes
    ----------
    name : str
    state : StateVector
        Current inertial state (mutable here in the sim layer; updated each tick).
    delta_v_budget_mps : float
        Retained for display/back-compat; see delta_v_remaining.
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
    delta_v_budget_mps: float = 0.0
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
    orientation: np.ndarray = field(default_factory=quat_identity)

    @property
    def mass_kg(self) -> float:
        """Current total mass = dry + fuel [kg]."""
        return self.dry_mass_kg + self.fuel_mass_kg

    @property
    def delta_v_remaining(self) -> float:
        """Remaining delta-V from the rocket equation [m/s]; 0 if no fuel."""
        if self.fuel_mass_kg <= 0.0:
            return 0.0
        return tsiolkovsky_dv(self.exhaust_velocity_mps, self.mass_kg,
                              self.dry_mass_kg)


class World:
    """Holds the central body and all vessels; advances them analytically.

    Parameters
    ----------
    central : CelestialBody
        The body all vessel states are referenced to.
    vessels : list[Vessel]
    """

    def __init__(self, central: CelestialBody, vessels: list[Vessel]) -> None:
        self.central = central
        self.vessels = vessels

    def step(self, sim_dt_s: float) -> None:
        """Propagate every vessel forward by sim_dt_s seconds (on-rails)."""
        for vessel in self.vessels:
            vessel.state = propagate_kepler(vessel.state, sim_dt_s)
