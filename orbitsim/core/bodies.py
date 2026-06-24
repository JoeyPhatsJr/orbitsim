"""CelestialBody dataclass and sphere-of-influence calculations."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CelestialBody:
    """A celestial body with gravitational parameters and orbital properties.

    Parameters
    ----------
    name : str
        Name of the body (e.g., "Earth", "Sun").
    mu : float
        Standard gravitational parameter μ = GM [m^3/s^2].
    radius_m : float
        Mean equatorial radius [m].
    j2 : float, optional
        Oblateness coefficient J2 (dimensionless). Defaults to 0.0.
    rotation_period_s : float, optional
        Sidereal rotation period [s]. Defaults to inf (no rotation).
    parent : CelestialBody | None, optional
        The body this body orbits. None if the primary body (e.g., the Sun).

    Attributes
    ----------
    All attributes are immutable (frozen dataclass).
    """

    name: str
    mu: float  # standard gravitational parameter [m^3/s^2]
    radius_m: float  # mean equatorial radius [m]
    j2: float = 0.0  # oblateness coefficient (dimensionless)
    rotation_period_s: float = float("inf")  # sidereal rotation period [s]
    parent: Optional["CelestialBody"] = None  # body it orbits

    def soi_radius_m(self, semi_major_axis_m: float) -> float:
        """Sphere-of-influence radius.

        r_SOI = a · (m/M_parent)^(2/5) ≈ a · (μ/μ_parent)^(2/5)

        Parameters
        ----------
        semi_major_axis_m : float
            Semi-major axis of the orbit around the parent body [m].

        Returns
        -------
        float
            Sphere-of-influence radius [m]. Returns inf if no parent.

        Notes
        -----
        Uses μ ∝ m to approximate the mass ratio.
        """
        if self.parent is None:
            return float("inf")

        # r_SOI = a · (μ/μ_parent)^(2/5)
        mass_ratio = (self.mu / self.parent.mu) ** (2.0 / 5.0)
        return semi_major_axis_m * mass_ratio


from orbitsim.core.constants import (
    MU_SUN,
    MU_EARTH,
    MU_MOON,
    R_SUN,
    R_EARTH,
    R_MOON,
    J2_EARTH,
)

# Sidereal rotation periods [s] (IAU): Earth 86164.0905 s, Sun ~25.05 d, Moon ~27.32 d.
SUN = CelestialBody(
    name="Sun",
    mu=MU_SUN,
    radius_m=R_SUN,
    rotation_period_s=25.05 * 86400.0,
    parent=None,
)
EARTH = CelestialBody(
    name="Earth",
    mu=MU_EARTH,
    radius_m=R_EARTH,
    j2=J2_EARTH,
    rotation_period_s=86164.0905,
    parent=SUN,
)
MOON = CelestialBody(
    name="Moon",
    mu=MU_MOON,
    radius_m=R_MOON,
    rotation_period_s=27.321661 * 86400.0,
    parent=EARTH,
)
