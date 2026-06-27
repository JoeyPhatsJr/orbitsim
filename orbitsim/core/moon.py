"""An idealized Keplerian Moon orbiting Earth (geocentric, no perturbations).

Used by the sandbox as an intercept target. Real lunar elements vary; these are
fixed mean values, accurate enough for approach planning, not ephemeris-grade.
"""
import numpy as np

from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.elements import KeplerianElements, elements_to_state
from orbitsim.core.state import StateVector

MOON_ORBIT = KeplerianElements(
    a=3.844e8, e=0.0, i=0.0898, raan=0.0, argp=0.0, nu=0.0,
    mu=MU_EARTH + MU_MOON, epoch_s=0.0,
)
_MOON_EPOCH_STATE = elements_to_state(MOON_ORBIT)
_MEAN_MOTION = np.sqrt(MOON_ORBIT.mu / MOON_ORBIT.a**3)


def moon_state_at(t_s: float) -> StateVector:
    """Geocentric Moon state at ``t_s`` using its exact circular-orbit solution."""
    dt = t_s - MOON_ORBIT.epoch_s
    angle = _MEAN_MOTION * dt
    cosine = np.cos(angle)
    sine = np.sin(angle)
    r = _MOON_EPOCH_STATE.r * cosine + (_MOON_EPOCH_STATE.v / _MEAN_MOTION) * sine
    v = -_MOON_EPOCH_STATE.r * _MEAN_MOTION * sine + _MOON_EPOCH_STATE.v * cosine
    return StateVector(r=r, v=v, mu=MOON_ORBIT.mu, epoch_s=t_s)
