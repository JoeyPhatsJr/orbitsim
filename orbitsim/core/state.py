"""StateVector: position & velocity in inertial frame."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StateVector:
    """Position and velocity in an inertial frame (J2000/ICRF).

    Parameters
    ----------
    r : np.ndarray
        Position vector [m], shape (3,), dtype float64.
    v : np.ndarray
        Velocity vector [m/s], shape (3,), dtype float64.
    mu : float
        Standard gravitational parameter μ = GM of central body [m^3/s^2].
    epoch_s : float, optional
        Seconds past J2000 TDB. Defaults to 0.0.

    Notes
    -----
    - Arrays are immutable (frozen dataclass, arrays set as read-only).
    - All input arrays are converted to dtype float64.
    - Central body is at the origin.
    """

    r: np.ndarray  # position [m], shape (3,)
    v: np.ndarray  # velocity [m/s], shape (3,)
    mu: float  # standard gravitational parameter [m^3/s^2]
    epoch_s: float = 0.0  # seconds past J2000 (TDB)

    def __post_init__(self) -> None:
        """Validate and immutabilize arrays."""
        # Validate shapes and dtypes
        if self.r.shape != (3,):
            raise ValueError(f"r must have shape (3,), got {self.r.shape}")
        if self.v.shape != (3,):
            raise ValueError(f"v must have shape (3,), got {self.v.shape}")

        # Convert to float64 if needed
        r_f64 = np.asarray(self.r, dtype=np.float64)
        v_f64 = np.asarray(self.v, dtype=np.float64)

        # Check for NaN/inf
        if not np.isfinite(r_f64).all():
            raise ValueError("r contains NaN or inf")
        if not np.isfinite(v_f64).all():
            raise ValueError("v contains NaN or inf")

        # Make arrays read-only via object.__setattr__ (needed for frozen dataclass)
        r_f64.flags.writeable = False
        v_f64.flags.writeable = False

        object.__setattr__(self, "r", r_f64)
        object.__setattr__(self, "v", v_f64)

    @property
    def r_mag(self) -> float:
        """Magnitude of position vector |r| [m]."""
        return float(np.linalg.norm(self.r))

    @property
    def v_mag(self) -> float:
        """Magnitude of velocity vector |v| [m/s]."""
        return float(np.linalg.norm(self.v))

    @property
    def specific_energy(self) -> float:
        """Specific orbital energy ε = v²/2 − μ/r [m²/s²].

        Notes
        -----
        For a bound orbit (ellipse): ε = −μ/(2a) < 0.
        For a parabolic orbit: ε = 0.
        For a hyperbolic orbit: ε > 0.
        """
        v_sq = np.dot(self.v, self.v)
        r_mag = self.r_mag
        return v_sq / 2.0 - self.mu / r_mag

    @property
    def angular_momentum(self) -> np.ndarray:
        """Specific angular momentum h = r × v [m²/s].

        Returns
        -------
        np.ndarray
            Angular momentum vector [m²/s], shape (3,), dtype float64.

        Notes
        -----
        For Keplerian orbits, this is constant in magnitude and direction.
        The magnitude h = |r × v| appears in Kepler's equation.
        """
        h = np.cross(self.r, self.v)
        return np.asarray(h, dtype=np.float64)
