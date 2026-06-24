"""Impulsive delta-V maneuvers in the vessel's local orbital (RTN/LVLH) frame."""
from dataclasses import dataclass
import numpy as np

from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler


@dataclass(frozen=True)
class ManeuverNode:
    """An impulsive burn defined in the local RTN frame at a given epoch.

    Attributes
    ----------
    epoch_s : float
        When the burn happens [s past J2000 TDB].
    dv_prograde_mps : float
        Component along velocity (+ speeds up) [m/s].
    dv_normal_mps : float
        Component along orbital angular momentum h [m/s].
    dv_radial_mps : float
        Component along the radial-out RTN axis [m/s].
    """

    epoch_s: float
    dv_prograde_mps: float
    dv_normal_mps: float
    dv_radial_mps: float

    @property
    def magnitude_mps(self) -> float:
        """Total delta-V magnitude [m/s]."""
        return float(
            np.sqrt(
                self.dv_prograde_mps**2
                + self.dv_normal_mps**2
                + self.dv_radial_mps**2
            )
        )


def apply_maneuver(state: StateVector, node: ManeuverNode) -> StateVector:
    """Propagate to the node epoch and apply the impulsive burn.

    Parameters
    ----------
    state : StateVector
        Current state (its epoch_s is the start time).
    node : ManeuverNode

    Returns
    -------
    StateVector
        Post-burn state at node.epoch_s (same position, new velocity).

    Notes
    -----
    Local RTN basis at the burn point:
        v_hat = v / |v|                  (prograde)
        h_hat = (r x v) / |r x v|        (orbit-normal)
        r_hat = h_hat x v_hat            (radial-out, orthonormal — NOT r/|r|)
    """
    dt = node.epoch_s - state.epoch_s
    burn_state = propagate_kepler(state, dt)

    r = burn_state.r
    v = burn_state.v
    v_hat = v / np.linalg.norm(v)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    r_hat = np.cross(h_hat, v_hat)

    dv = (
        node.dv_prograde_mps * v_hat
        + node.dv_normal_mps * h_hat
        + node.dv_radial_mps * r_hat
    )
    new_v = v + dv

    return StateVector(
        r=np.array(r, dtype=np.float64),
        v=np.array(new_v, dtype=np.float64),
        mu=burn_state.mu,
        epoch_s=burn_state.epoch_s,
    )
