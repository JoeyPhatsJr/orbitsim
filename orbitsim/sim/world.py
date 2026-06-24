"""Body registry + vessels; per-tick propagation."""
from dataclasses import dataclass

from orbitsim.core.bodies import CelestialBody
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler


@dataclass
class Vessel:
    """A point-mass vessel with a current state and a delta-V budget.

    Attributes
    ----------
    name : str
    state : StateVector
        Current inertial state (mutable here in the sim layer; updated each tick).
    delta_v_budget_mps : float
        Remaining delta-V budget [m/s].
    """

    name: str
    state: StateVector
    delta_v_budget_mps: float = 0.0


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
