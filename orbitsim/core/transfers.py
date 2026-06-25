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


def bielliptic(r1_m: float, r2_m: float, rb_m: float, mu: float) -> TransferSolution:
    """Three-burn bi-elliptic transfer via an intermediate apoapsis rb.

    Parameters
    ----------
    r1_m, r2_m : float
        Initial and final circular radii [m].
    rb_m : float
        Intermediate apoapsis radius [m], should satisfy rb_m >= r2_m.
    mu : float

    Returns
    -------
    TransferSolution
    """
    a1 = (r1_m + rb_m) / 2.0   # first transfer ellipse
    a2 = (r2_m + rb_m) / 2.0   # second transfer ellipse

    v_c1 = np.sqrt(mu / r1_m)
    v_c2 = np.sqrt(mu / r2_m)

    # Burn 1: at r1, raise apoapsis to rb.
    v_peri1 = np.sqrt(mu * (2.0 / r1_m - 1.0 / a1))
    dv1 = v_peri1 - v_c1
    # Burn 2: at rb, raise periapsis from r1-ellipse to r2-ellipse.
    v_apo1 = np.sqrt(mu * (2.0 / rb_m - 1.0 / a1))
    v_apo2 = np.sqrt(mu * (2.0 / rb_m - 1.0 / a2))
    dv2 = v_apo2 - v_apo1
    # Burn 3: at r2, circularize (decelerate).
    v_peri2 = np.sqrt(mu * (2.0 / r2_m - 1.0 / a2))
    dv3 = v_c2 - v_peri2

    t1 = np.pi * np.sqrt(a1**3 / mu)
    t2 = np.pi * np.sqrt(a2**3 / mu)
    tof = t1 + t2

    burns = [
        ManeuverNode(epoch_s=0.0, dv_prograde_mps=float(dv1), dv_normal_mps=0.0, dv_radial_mps=0.0),
        ManeuverNode(epoch_s=float(t1), dv_prograde_mps=float(dv2), dv_normal_mps=0.0, dv_radial_mps=0.0),
        ManeuverNode(epoch_s=float(tof), dv_prograde_mps=float(dv3), dv_normal_mps=0.0, dv_radial_mps=0.0),
    ]
    return TransferSolution(
        burns=burns,
        dv_total_mps=float(abs(dv1) + abs(dv2) + abs(dv3)),
        time_of_flight_s=float(tof),
        kind="bielliptic",
    )


def plane_change(speed_mps: float, delta_i_rad: float) -> float:
    """delta-V for a simple plane change of delta_i at orbital speed v [m/s].

    dv = 2 v sin(delta_i / 2)
    """
    return float(2.0 * speed_mps * np.sin(delta_i_rad / 2.0))
