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
