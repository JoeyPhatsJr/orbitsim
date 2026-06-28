"""Circular orbit approximations for inner solar system bodies.

All state functions return GEOCENTRIC positions (Earth at origin) to match the
sandbox frame. Heliocentric positions are computed first, then Earth's
heliocentric position is subtracted.

Orbital elements are mean values in the ecliptic plane (i=0 for all bodies for
simplicity; real inclinations are <7° and don't affect the sandbox experience).
Epochs are seconds past J2000 TDB, consistent with the rest of core/.
"""
import numpy as np

from orbitsim.core.constants import MU_SUN, MU_EARTH, MU_MERCURY, MU_VENUS, MU_MARS
from orbitsim.core.state import StateVector

# Semi-major axes [m] (NASA planetary fact sheet, mean values).
A_MERCURY = 5.7909e10
A_VENUS = 1.0821e11
A_EARTH = 1.4960e11
A_MARS = 2.2794e11

# Mean motions n = sqrt(mu_sun / a^3) [rad/s] — precomputed for speed.
_N_MERCURY = np.sqrt(MU_SUN / A_MERCURY**3)
_N_VENUS = np.sqrt(MU_SUN / A_VENUS**3)
_N_EARTH = np.sqrt(MU_SUN / A_EARTH**3)
_N_MARS = np.sqrt(MU_SUN / A_MARS**3)

# SOI radii [m]: r_SOI = a * (mu_planet / mu_sun)^(2/5).
MERCURY_SOI_M = A_MERCURY * (MU_MERCURY / MU_SUN) ** 0.4
VENUS_SOI_M = A_VENUS * (MU_VENUS / MU_SUN) ** 0.4
EARTH_SOI_M = A_EARTH * (MU_EARTH / MU_SUN) ** 0.4
MARS_SOI_M = A_MARS * (MU_MARS / MU_SUN) ** 0.4


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


# Registry for iteration: (state_fn, mu, soi_m, name).
INNER_PLANETS = [
    (sun_state_at, MU_SUN, float("inf"), "Sun"),
    (mercury_state_at, MU_MERCURY, MERCURY_SOI_M, "Mercury"),
    (venus_state_at, MU_VENUS, VENUS_SOI_M, "Venus"),
    (mars_state_at, MU_MARS, MARS_SOI_M, "Mars"),
]
