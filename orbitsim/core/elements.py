"""KeplerianElements and state ↔ elements conversions."""
from dataclasses import dataclass
import numpy as np
from orbitsim.core.state import StateVector
from orbitsim.core.frames import perifocal_rotation, perifocal_to_inertial, inertial_to_perifocal  # noqa: F401


@dataclass(frozen=True)
class KeplerianElements:
    """Classical Keplerian orbital elements.

    Parameters
    ----------
    a : float
        Semi-major axis [m]. Positive for ellipse, negative for hyperbola, zero for parabola.
    e : float
        Eccentricity (dimensionless). 0 ≤ e < ∞.
    i : float
        Inclination [rad], 0 ≤ i ≤ π.
    raan : float
        Right ascension of ascending node Ω [rad], 0 ≤ Ω < 2π.
    argp : float
        Argument of periapsis ω [rad], 0 ≤ ω < 2π.
    nu : float
        True anomaly ν [rad], 0 ≤ ν < 2π.
    mu : float
        Standard gravitational parameter μ = GM of central body [m^3/s^2].
    epoch_s : float, optional
        Seconds past J2000 TDB. Defaults to 0.0.

    Attributes
    ----------
    All attributes are immutable (frozen dataclass).
    """

    a: float  # semi-major axis [m]
    e: float  # eccentricity [-]
    i: float  # inclination [rad], 0..π
    raan: float  # right ascension of ascending node [rad], 0..2π
    argp: float  # argument of periapsis [rad], 0..2π
    nu: float  # true anomaly [rad], 0..2π
    mu: float  # standard gravitational parameter [m^3/s^2]
    epoch_s: float = 0.0

    @property
    def period_s(self) -> float:
        """Orbital period T = 2π√(a³/μ) [s].

        Returns
        -------
        float
            Period [s] for bound orbits (a > 0).

        Raises
        ------
        ValueError
            If a ≤ 0 (parabolic or hyperbolic).
        """
        if self.a <= 0:
            raise ValueError(f"period_s undefined for a={self.a} ≤ 0 (unbound orbit)")
        return 2.0 * np.pi * np.sqrt(self.a**3 / self.mu)

    @property
    def periapsis_radius(self) -> float:
        """Periapsis radius r_p = a(1 − e) [m]."""
        return self.a * (1.0 - self.e)

    @property
    def apoapsis_radius(self) -> float:
        """Apoapsis radius r_a = a(1 + e) [m]. Only meaningful for bound orbits (e < 1)."""
        return self.a * (1.0 + self.e)

    @property
    def semi_latus_rectum(self) -> float:
        """Semi-latus rectum p = a(1 − e²) [m].

        Returns
        -------
        float
            Semi-latus rectum [m]. Also equal to h²/μ where h is specific angular momentum.
        """
        return self.a * (1.0 - self.e**2)


def _wrap_2pi(angle_rad: float) -> float:
    """Map an atan2 result in (-π, π] onto [0, 2π).

    A tiny negative angle (-1e-17) mod 2π rounds to exactly 2π in float64; fold that
    back to 0 so the range stays half-open.
    """
    wrapped = angle_rad % (2.0 * np.pi)
    return 0.0 if wrapped >= 2.0 * np.pi else float(wrapped)


def state_to_elements(state: StateVector) -> KeplerianElements:
    """Convert state vector to Keplerian elements (Curtis Algorithm 4.1).

    Parameters
    ----------
    state : StateVector
        State vector (position r, velocity v, gravitational parameter μ).

    Returns
    -------
    KeplerianElements
        Keplerian orbital elements.

    Notes
    -----
    Implements Curtis, Orbital Mechanics for Engineering Students, Algorithm 4.1.
    Handles edge cases: circular orbits (e ≈ 0), equatorial orbits (i ≈ 0).
    """
    r = state.r
    v = state.v
    mu = state.mu

    r_mag = state.r_mag
    v_mag = state.v_mag
    v_sq = np.dot(v, v)

    # Radial velocity
    vr = np.dot(r, v) / r_mag

    # Specific angular momentum h = r × v
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Every angle below is atan2(sin, cos), never arccos(cos) + a quadrant flip.
    # arccos has infinite slope at ±1, so at 0/π (periapsis, apoapsis, Ω or ω of 0/π,
    # near-equatorial i) a 1-ulp error in cos becomes a ~1.5e-8 rad error in the
    # angle, and the separate sign test can pick the wrong side of the wrap.
    # atan2 is well-conditioned everywhere. Signs match Curtis Alg. 4.1's flips:
    # the in-plane sine is taken along ĥ, i.e. in the direction of motion.
    h_hat = h / h_mag

    # Inclination i ∈ [0, π]: angle between ĥ and k̂
    i = np.arctan2(np.hypot(h[0], h[1]), h[2])

    # Node line N = k × h
    N = np.cross([0.0, 0.0, 1.0], h)
    N_mag = np.linalg.norm(N)

    # Right ascension of ascending node Ω
    if N_mag < 1e-11:
        # Equatorial orbit: Ω = 0
        raan = 0.0
    else:
        raan = _wrap_2pi(np.arctan2(N[1], N[0]))

    # Eccentricity vector: evec = (1/μ) * ((v² − μ/r)*r − r*vr*v)
    evec = (1.0 / mu) * ((v_sq - mu / r_mag) * r - r_mag * vr * v)
    e = np.linalg.norm(evec)

    # Argument of periapsis ω
    if e < 1e-11:
        # Circular orbit: ω = 0
        argp = 0.0
    elif N_mag < 1e-11:
        # Equatorial: longitude of periapsis, measured from the x-axis
        argp = _wrap_2pi(np.arctan2(evec[1], evec[0]))
    else:
        # In-plane angle from N to evec, positive in the direction of motion
        argp = _wrap_2pi(np.arctan2(np.dot(np.cross(N, evec), h_hat), np.dot(N, evec)))

    # True anomaly ν
    if e < 1e-11:
        # Circular orbit: measure ν from node line or x-axis
        if N_mag < 1e-11:
            # Circular equatorial: true longitude, from the x-axis
            nu = _wrap_2pi(np.arctan2(r[1], r[0]))
        else:
            # Argument of latitude, from the node line
            nu = _wrap_2pi(np.arctan2(np.dot(np.cross(N, r), h_hat), np.dot(N, r)))
    else:
        # In-plane angle from evec to r, positive in the direction of motion
        nu = _wrap_2pi(np.arctan2(np.dot(np.cross(evec, r), h_hat), np.dot(evec, r)))

    # Semi-major axis a = 1 / (2/r − v²/μ)
    a = 1.0 / (2.0 / r_mag - v_sq / mu)

    return KeplerianElements(
        a=a,
        e=e,
        i=i,
        raan=raan,
        argp=argp,
        nu=nu,
        mu=mu,
        epoch_s=state.epoch_s,
    )


def elements_to_state(elements: KeplerianElements) -> StateVector:
    """Convert Keplerian elements to state vector (Curtis Algorithm 4.2).

    Parameters
    ----------
    elements : KeplerianElements
        Keplerian orbital elements.

    Returns
    -------
    StateVector
        State vector (position r, velocity v) in inertial frame.

    Notes
    -----
    Implements Curtis, Orbital Mechanics for Engineering Students, Algorithm 4.2.
    Works for elliptic, parabolic, and hyperbolic orbits.
    """
    return elements_to_state_rotated(
        elements, perifocal_rotation(elements.raan, elements.i, elements.argp)
    )


def elements_to_state_rotated(elements: KeplerianElements, q: np.ndarray) -> StateVector:
    """``elements_to_state`` with the perifocal→inertial rotation ``q`` precomputed.

    ``q`` must be ``perifocal_rotation(elements.raan, elements.i, elements.argp)``; it is
    constant along an orbit, so callers propagating one orbit to many epochs build it once.
    Bit-identical to ``elements_to_state`` (which delegates here).
    """
    mu = elements.mu
    p = elements.semi_latus_rectum
    h = np.sqrt(mu * p)

    # Position and velocity in perifocal frame (PQW)
    r_pqw_mag = p / (1.0 + elements.e * np.cos(elements.nu))
    r_pqw = r_pqw_mag * np.array(
        [np.cos(elements.nu), np.sin(elements.nu), 0.0],
        dtype=np.float64,
    )

    v_pqw = (mu / h) * np.array(
        [-np.sin(elements.nu), elements.e + np.cos(elements.nu), 0.0],
        dtype=np.float64,
    )

    # Rotate to inertial frame (same arithmetic as frames.perifocal_to_inertial)
    r_inertial = np.asarray(q @ r_pqw, dtype=np.float64)
    v_inertial = np.asarray(q @ v_pqw, dtype=np.float64)

    return StateVector(
        r=r_inertial,
        v=v_inertial,
        mu=mu,
        epoch_s=elements.epoch_s,
    )
