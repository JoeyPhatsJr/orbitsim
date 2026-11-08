"""The geocentric Moon: real JPL/DE440 ephemeris, with a circular offline fallback.

``moon_state_at`` is what the sandbox uses everywhere (gravity, SOI, landing,
targeting). With the DE440 kernel it returns the real Moon, read once per sim-hour
and cubic-Hermite interpolated in between (~1 cm error; a raw Skyfield call per
integrator substep would be ~35x slower). Without the kernel -- offline on first
run, or outside the kernel's 1849-2150 range -- it falls back to
``moon_circular_state_at``: a circular orbit fitted to the real Moon's plane and
phase at J2000, so going offline costs accuracy (the real orbit's e ~ 0.055),
not a Moon on the wrong side of Earth.
"""
import numpy as np

from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.elements import KeplerianElements, elements_to_state
from orbitsim.core.hermite import hermite_state
from orbitsim.core.state import StateVector

try:
    from orbitsim.core import ephemeris as _ephemeris
except ImportError:  # skyfield not installed: circular fallback only
    _ephemeris = None

# Mean circular orbit in J2000/ICRF (equatorial). Plane and phase are the real
# Moon's at J2000 from DE440: i = 20.94 deg to the equator, RAAN = 12.24 deg,
# argument of latitude = 211.9 deg (circular => argp is undefined; it is folded
# into nu). The 5.14 deg often quoted is the inclination to the *ecliptic*.
MOON_ORBIT = KeplerianElements(
    a=3.844e8, e=0.0, i=0.3655121569158447, raan=0.21356613982226338,
    argp=0.0, nu=3.6990783528285305,
    mu=MU_EARTH + MU_MOON, epoch_s=0.0,
)
_MOON_EPOCH_STATE = elements_to_state(MOON_ORBIT)
_MEAN_MOTION = np.sqrt(MOON_ORBIT.mu / MOON_ORBIT.a**3)

_SAMPLE_STEP_S = 3600.0
_SAMPLE_CACHE_MAX = 8760          # one sim-year of hourly DE440 samples (~2 MB)
_sample_cache: dict = {}


def moon_circular_state_at(t_s: float) -> StateVector:
    """Geocentric Moon state on the fitted mean circular orbit (exact solution)."""
    dt = t_s - MOON_ORBIT.epoch_s
    angle = _MEAN_MOTION * dt
    cosine = np.cos(angle)
    sine = np.sin(angle)
    r = _MOON_EPOCH_STATE.r * cosine + (_MOON_EPOCH_STATE.v / _MEAN_MOTION) * sine
    v = -_MOON_EPOCH_STATE.r * _MEAN_MOTION * sine + _MOON_EPOCH_STATE.v * cosine
    return StateVector(r=r, v=v, mu=MOON_ORBIT.mu, epoch_s=t_s)


def _ephemeris_ready() -> bool:
    return _ephemeris is not None and _ephemeris.available()


_SAMPLE_BATCH = 256              # knots per vectorized Skyfield call (~10.7 sim-days)


def _sample(index: int) -> StateVector:
    """DE440 geocentric Moon state at the ``index``-th hourly knot (cached).

    A missing knot fetches its whole aligned block of _SAMPLE_BATCH knots in one
    vectorized call (bit-identical to per-knot calls): a 1e8x frame spans ~463
    knots, and one call each made a deep-space frame cost ~100 ms.
    """
    state = _sample_cache.get(index)
    if state is None:
        batch = max(1, min(_SAMPLE_BATCH, _SAMPLE_CACHE_MAX))
        if len(_sample_cache) + batch > _SAMPLE_CACHE_MAX:
            _sample_cache.clear()
        first = (index // batch) * batch
        try:
            knots = _ephemeris.body_states(
                "MOON", np.arange(first, first + batch) * _SAMPLE_STEP_S, center="EARTH")
        except ValueError:
            # The block runs past the kernel's range: fetch this knot alone (it
            # raises the range error itself when it is out too).
            knots = None
        if knots is None:
            state = _ephemeris.body_states(
                "MOON", np.array([index * _SAMPLE_STEP_S]), center="EARTH")[0]
            _sample_cache[index] = state
        else:
            for offset, knot in enumerate(knots):
                _sample_cache[first + offset] = knot
            # From the local block, not the shared dict: another thread may clear it.
            state = knots[index - first]
    return state


def moon_state_at(t_s: float) -> StateVector:
    """Geocentric Moon state at ``t_s`` [s past J2000 TDB]: real when possible.

    ``mu`` is Earth+Moon, matching ``MOON_ORBIT``, so callers that Kepler-propagate
    the Moon as an intercept target see a consistent two-body parameter.
    """
    if not _ephemeris_ready():
        return moon_circular_state_at(t_s)
    t = float(t_s)
    bucket = int(np.floor(t / _SAMPLE_STEP_S))
    try:
        start = _sample(bucket)
        end = _sample(bucket + 1)
    except ValueError:
        # Outside the kernel's date range (EphemerisRangeError is a ValueError).
        return moon_circular_state_at(t_s)
    return hermite_state(start, end, bucket * _SAMPLE_STEP_S, _SAMPLE_STEP_S, t,
                         mu=MOON_ORBIT.mu)
