"""An idealized Keplerian Moon orbiting Earth (geocentric, no perturbations).

Used by the sandbox as an intercept target. Real lunar elements vary; these are
fixed mean values, accurate enough for approach planning, not ephemeris-grade.
"""
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.elements import KeplerianElements, elements_to_state
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.state import StateVector

MOON_ORBIT = KeplerianElements(
    a=3.844e8, e=0.0549, i=0.0898, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH, epoch_s=0.0,
)
_MOON_EPOCH_STATE = elements_to_state(MOON_ORBIT)


def moon_state_at(t_s: float) -> StateVector:
    """Geocentric Moon state at sim time ``t_s`` [s past J2000], by two-body propagation."""
    return propagate_kepler(_MOON_EPOCH_STATE, t_s - MOON_ORBIT.epoch_s)
