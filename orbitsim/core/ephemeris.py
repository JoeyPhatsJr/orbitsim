"""Skyfield / DE440 ephemeris wrapper returning SI StateVectors.

Sim time is seconds past J2000 TDB. J2000 epoch = JD 2451545.0 (TDB).
"""
import os
import numpy as np

from skyfield.api import Loader
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_SUN, MU_EARTH

_J2000_JD_TDB = 2451545.0
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
)

# Cache the loader, kernel, and timescale at module load.
_loader = Loader(_DATA_DIR)
_ts = _loader.timescale()
_kernel = _loader("de440s.bsp")

# Map our names to Skyfield kernel targets.
_TARGETS = {
    "SUN": "sun",
    "MERCURY": "mercury barycenter",
    "VENUS": "venus barycenter",
    "EARTH": "earth",
    "MOON": "moon",
    "MARS": "mars barycenter",
    "JUPITER": "jupiter barycenter",
    "SATURN": "saturn barycenter",
    "URANUS": "uranus barycenter",
    "NEPTUNE": "neptune barycenter",
}

# mu lookup for the returned StateVector's central body.
_CENTER_MU = {"SUN": MU_SUN, "EARTH": MU_EARTH}


def sim_time_to_skyfield(t_sim_s: float):
    """Convert seconds past J2000 TDB to a Skyfield Time."""
    jd_tdb = _J2000_JD_TDB + t_sim_s / 86400.0
    return _ts.tdb_jd(jd_tdb)


def body_state(name: str, t_sim_s: float, center: str = "SUN") -> StateVector:
    """Position + velocity of `name` relative to `center` as an SI StateVector.

    Parameters
    ----------
    name : str
        One of the keys in the module target map (e.g. "EARTH", "MARS").
    t_sim_s : float
        Seconds past J2000 TDB.
    center : str
        Reference body ("SUN" or "EARTH").

    Returns
    -------
    StateVector
        r, v in J2000/ICRF [m, m/s]; mu is the central body's mu (0.0 if unknown).
    """
    t = sim_time_to_skyfield(t_sim_s)
    target = _kernel[_TARGETS[name.upper()]]
    origin = _kernel[_TARGETS[center.upper()]]
    rel = (target - origin).at(t)
    r_m = rel.position.m                     # meters, shape (3,)
    v_m_s = rel.velocity.m_per_s             # m/s, shape (3,)
    return StateVector(
        r=np.asarray(r_m, dtype=np.float64),
        v=np.asarray(v_m_s, dtype=np.float64),
        mu=_CENTER_MU.get(center.upper(), 0.0),
        epoch_s=t_sim_s,
    )
