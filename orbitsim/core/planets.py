"""Circular orbit approximations for solar system bodies.

All state functions return GEOCENTRIC positions (Earth at origin) to match the
sandbox frame. Heliocentric positions are computed first, then Earth's
heliocentric position is subtracted.

Orbital elements are mean values in the ecliptic plane (i=0 for all bodies for
simplicity; real inclinations are <7° and don't affect the sandbox experience).
Epochs are seconds past J2000 TDB, consistent with the rest of core/.
"""
import numpy as np

from orbitsim.core.constants import (
    MU_SUN, MU_EARTH, MU_MERCURY, MU_VENUS, MU_MARS,
    MU_JUPITER, MU_SATURN, MU_URANUS, MU_NEPTUNE,
)
from orbitsim.core.state import StateVector

# Semi-major axes [m] (NASA planetary fact sheet, mean values).
A_MERCURY = 5.7909e10
A_VENUS = 1.0821e11
A_EARTH = 1.4960e11
A_MARS = 2.2794e11
A_JUPITER = 7.7857e11
A_SATURN = 1.4335e12
A_URANUS = 2.8725e12
A_NEPTUNE = 4.4951e12

# Mean motions n = sqrt(mu_sun / a^3) [rad/s] — precomputed for speed.
_N_MERCURY = np.sqrt(MU_SUN / A_MERCURY**3)
_N_VENUS = np.sqrt(MU_SUN / A_VENUS**3)
_N_EARTH = np.sqrt(MU_SUN / A_EARTH**3)
_N_MARS = np.sqrt(MU_SUN / A_MARS**3)
_N_JUPITER = np.sqrt(MU_SUN / A_JUPITER**3)
_N_SATURN = np.sqrt(MU_SUN / A_SATURN**3)
_N_URANUS = np.sqrt(MU_SUN / A_URANUS**3)
_N_NEPTUNE = np.sqrt(MU_SUN / A_NEPTUNE**3)

# SOI radii [m]: r_SOI = a * (mu_planet / mu_sun)^(2/5).
MERCURY_SOI_M = A_MERCURY * (MU_MERCURY / MU_SUN) ** 0.4
VENUS_SOI_M = A_VENUS * (MU_VENUS / MU_SUN) ** 0.4
EARTH_SOI_M = A_EARTH * (MU_EARTH / MU_SUN) ** 0.4
MARS_SOI_M = A_MARS * (MU_MARS / MU_SUN) ** 0.4
JUPITER_SOI_M = A_JUPITER * (MU_JUPITER / MU_SUN) ** 0.4
SATURN_SOI_M = A_SATURN * (MU_SATURN / MU_SUN) ** 0.4
URANUS_SOI_M = A_URANUS * (MU_URANUS / MU_SUN) ** 0.4
NEPTUNE_SOI_M = A_NEPTUNE * (MU_NEPTUNE / MU_SUN) ** 0.4


def _circular_state_helio(a: float, n: float, t_s: float) -> tuple:
    """Heliocentric position and velocity of a body on a circular ecliptic orbit.

    Returns (r, v) as float64 (3,) arrays.
    """
    angle = n * t_s
    c, s = np.cos(angle), np.sin(angle)
    r = a * np.array([c, s, 0.0], dtype=np.float64)
    v = a * n * np.array([-s, c, 0.0], dtype=np.float64)
    return r, v


def earth_position_helio(t_s: float) -> np.ndarray:
    """Heliocentric position of Earth [m] at epoch t_s."""
    r, _ = _circular_state_helio(A_EARTH, _N_EARTH, t_s)
    return r


def _geocentric(a: float, n: float, t_s: float) -> tuple:
    """Geocentric position and velocity of a heliocentric body."""
    r_body, v_body = _circular_state_helio(a, n, t_s)
    r_earth, v_earth = _circular_state_helio(A_EARTH, _N_EARTH, t_s)
    return r_body - r_earth, v_body - v_earth


def sun_state_at(t_s: float) -> StateVector:
    """Geocentric state of the Sun at epoch t_s."""
    r_earth, v_earth = _circular_state_helio(A_EARTH, _N_EARTH, t_s)
    return StateVector(r=-r_earth, v=-v_earth, mu=MU_SUN, epoch_s=t_s)


def mercury_state_at(t_s: float) -> StateVector:
    """Geocentric state of Mercury at epoch t_s."""
    r, v = _geocentric(A_MERCURY, _N_MERCURY, t_s)
    return StateVector(r=r, v=v, mu=MU_MERCURY, epoch_s=t_s)


def venus_state_at(t_s: float) -> StateVector:
    """Geocentric state of Venus at epoch t_s."""
    r, v = _geocentric(A_VENUS, _N_VENUS, t_s)
    return StateVector(r=r, v=v, mu=MU_VENUS, epoch_s=t_s)


def mars_state_at(t_s: float) -> StateVector:
    """Geocentric state of Mars at epoch t_s."""
    r, v = _geocentric(A_MARS, _N_MARS, t_s)
    return StateVector(r=r, v=v, mu=MU_MARS, epoch_s=t_s)


def jupiter_state_at(t_s: float) -> StateVector:
    """Geocentric state of Jupiter at epoch t_s."""
    r, v = _geocentric(A_JUPITER, _N_JUPITER, t_s)
    return StateVector(r=r, v=v, mu=MU_JUPITER, epoch_s=t_s)


def saturn_state_at(t_s: float) -> StateVector:
    """Geocentric state of Saturn at epoch t_s."""
    r, v = _geocentric(A_SATURN, _N_SATURN, t_s)
    return StateVector(r=r, v=v, mu=MU_SATURN, epoch_s=t_s)


def uranus_state_at(t_s: float) -> StateVector:
    """Geocentric state of Uranus at epoch t_s."""
    r, v = _geocentric(A_URANUS, _N_URANUS, t_s)
    return StateVector(r=r, v=v, mu=MU_URANUS, epoch_s=t_s)


def neptune_state_at(t_s: float) -> StateVector:
    """Geocentric state of Neptune at epoch t_s."""
    r, v = _geocentric(A_NEPTUNE, _N_NEPTUNE, t_s)
    return StateVector(r=r, v=v, mu=MU_NEPTUNE, epoch_s=t_s)


# Registry for iteration: (state_fn, mu, soi_m, name).
INNER_PLANETS = [
    (sun_state_at, MU_SUN, float("inf"), "Sun"),
    (mercury_state_at, MU_MERCURY, MERCURY_SOI_M, "Mercury"),
    (venus_state_at, MU_VENUS, VENUS_SOI_M, "Venus"),
    (mars_state_at, MU_MARS, MARS_SOI_M, "Mars"),
]

OUTER_PLANETS = [
    (jupiter_state_at, MU_JUPITER, JUPITER_SOI_M, "Jupiter"),
    (saturn_state_at, MU_SATURN, SATURN_SOI_M, "Saturn"),
    (uranus_state_at, MU_URANUS, URANUS_SOI_M, "Uranus"),
    (neptune_state_at, MU_NEPTUNE, NEPTUNE_SOI_M, "Neptune"),
]

ALL_PLANETS = INNER_PLANETS + OUTER_PLANETS
