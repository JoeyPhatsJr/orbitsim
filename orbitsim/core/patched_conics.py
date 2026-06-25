"""Patched-conic interplanetary flight: SOI containment + frame shifts."""
import numpy as np

from orbitsim.core.bodies import CelestialBody, SUN
from orbitsim.core.ephemeris import body_state
from orbitsim.core.state import StateVector

# Approximate semi-major axes [m] for SOI sizing (heliocentric).
_SMA_M = {
    "Mercury": 5.79e10, "Venus": 1.082e11, "Earth": 1.496e11, "Mars": 2.279e11,
    "Jupiter": 7.785e11, "Saturn": 1.434e12, "Uranus": 2.871e12, "Neptune": 4.495e12,
}


def dominant_body(
    pos_m_helio: np.ndarray,
    t_sim_s: float,
    bodies: list[CelestialBody],
) -> CelestialBody:
    """Return the body whose SOI currently contains the heliocentric position.

    Picks the smallest enclosing SOI; defaults to the Sun if no planet's SOI
    contains the point.

    Parameters
    ----------
    pos_m_helio : np.ndarray
        Heliocentric position [m], shape (3,).
    t_sim_s : float
        Seconds past J2000 TDB.
    bodies : list[CelestialBody]
        Candidate bodies (should include SUN).
    """
    best = SUN
    best_soi = float("inf")
    for body in bodies:
        if body.parent is None:
            continue  # the Sun has no finite SOI
        sma = _SMA_M.get(body.name)
        if sma is None:
            continue
        soi = body.soi_radius_m(sma)
        body_pos = body_state(body.name.upper(), t_sim_s, center="SUN").r
        dist = np.linalg.norm(pos_m_helio - body_pos)
        if dist < soi and soi < best_soi:
            best = body
            best_soi = soi
    return best


def shift_frame(
    state: StateVector,
    from_center: str,
    to_center: str,
    t_sim_s: float,
    to_mu: float,
) -> StateVector:
    """Re-express a state from one central body's frame to another's.

    Galilean shift: subtract/add the relative body state from the ephemeris.
    r_new = r_old + (from_center - to_center) position
    v_new = v_old + (from_center - to_center) velocity

    Parameters
    ----------
    state : StateVector
        State referenced to `from_center`.
    from_center, to_center : str
        Body names ("EARTH", "SUN", ...).
    t_sim_s : float
    to_mu : float
        mu of the new central body for the returned StateVector.
    """
    # Position/velocity of from_center relative to to_center.
    rel = body_state(from_center.upper(), t_sim_s, center=to_center.upper())
    return StateVector(
        r=np.asarray(state.r, dtype=np.float64) + rel.r,
        v=np.asarray(state.v, dtype=np.float64) + rel.v,
        mu=to_mu,
        epoch_s=t_sim_s,
    )
