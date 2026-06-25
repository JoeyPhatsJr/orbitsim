"""Closed-form transfers (Hohmann, bi-elliptic, plane change) and Lambert."""
from dataclasses import dataclass
import numpy as np

from orbitsim.core.maneuvers import ManeuverNode


@dataclass(frozen=True)
class TransferSolution:
    """A transfer described by its burns, total cost, and flight time.

    Attributes
    ----------
    burns : list[ManeuverNode]
    dv_total_mps : float
        Sum of burn magnitudes [m/s].
    time_of_flight_s : float
    kind : str
        "hohmann" | "bielliptic" | "plane_change" | "lambert".
    """

    burns: list[ManeuverNode]
    dv_total_mps: float
    time_of_flight_s: float
    kind: str


def hohmann(r1_m: float, r2_m: float, mu: float) -> TransferSolution:
    """Two-burn Hohmann transfer between coplanar circular orbits.

    Parameters
    ----------
    r1_m, r2_m : float
        Initial and final circular radii [m].
    mu : float
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    TransferSolution
    """
    a_t = (r1_m + r2_m) / 2.0
    v1 = np.sqrt(mu / r1_m)
    v2 = np.sqrt(mu / r2_m)
    dv1 = v1 * (np.sqrt(2.0 * r2_m / (r1_m + r2_m)) - 1.0)
    dv2 = v2 * (1.0 - np.sqrt(2.0 * r1_m / (r1_m + r2_m)))
    tof = np.pi * np.sqrt(a_t**3 / mu)

    burns = [
        ManeuverNode(epoch_s=0.0, dv_prograde_mps=float(dv1), dv_normal_mps=0.0, dv_radial_mps=0.0),
        ManeuverNode(epoch_s=float(tof), dv_prograde_mps=float(dv2), dv_normal_mps=0.0, dv_radial_mps=0.0),
    ]
    return TransferSolution(
        burns=burns,
        dv_total_mps=float(abs(dv1) + abs(dv2)),
        time_of_flight_s=float(tof),
        kind="hohmann",
    )
