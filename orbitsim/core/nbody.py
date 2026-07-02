"""Restricted N-body (CR3BP) core: idealized circular Earth+Moon, a velocity-Verlet
ship propagator, the Jacobi constant, and the Lagrange points. Barycentric inertial,
SI, float64. The ship is a massless test particle; the bodies are on rails."""
from contextlib import contextmanager
import threading

import numpy as np
from scipy.optimize import brentq

from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.state import StateVector
from orbitsim.core.moon import moon_state_at
from orbitsim.core.elements import state_to_elements

D_EM = 3.844e8                          # Earth-Moon separation [m]
MU_TOTAL = MU_EARTH + MU_MOON
OMEGA_EM = np.sqrt(MU_TOTAL / D_EM**3)  # mean motion [rad/s]
MASS_RATIO = MU_MOON / MU_TOTAL         # ~0.01215
EARTH_X = -MASS_RATIO * D_EM            # Earth rotating-frame x [m]
MOON_X = (1.0 - MASS_RATIO) * D_EM      # Moon rotating-frame x [m]


class _CircularBody:
    """A point mass on a circular barycentric orbit (signed radius along the
    rotating x-axis at t=0; rotates at OMEGA_EM)."""

    def __init__(self, mu: float, signed_radius_m: float):
        self.mu = mu
        self._R = signed_radius_m

    def state_at(self, t_s: float) -> StateVector:
        th = OMEGA_EM * t_s
        u = np.array([np.cos(th), np.sin(th), 0.0])
        n = np.array([-np.sin(th), np.cos(th), 0.0])
        return StateVector(r=self._R * u, v=self._R * OMEGA_EM * n,
                           mu=self.mu, epoch_s=t_s)


EARTH = _CircularBody(MU_EARTH, EARTH_X)
MOON = _CircularBody(MU_MOON, MOON_X)
EARTH_MOON = [EARTH, MOON]


def gravity_accel(r_m, t_s, attractors=EARTH_MOON):
    """Summed gravitational acceleration on a test particle at r_m, time t_s [m/s^2]."""
    r = np.asarray(r_m, dtype=np.float64)
    a = np.zeros(3)
    for body in attractors:
        d = r - body.state_at(t_s).r
        a += -body.mu * d / np.linalg.norm(d)**3
    return a


def earth_moon_accel(r_m, t_s):
    """Ship acceleration in the EARTH-CENTERED frame [m/s^2]: central Earth plus the
    Moon's third-body perturbation. The indirect term (+mu_M * r_M/|r_M|^3) is required
    because Earth is held fixed at the origin (non-inertial frame); it is also what makes
    the Lagrange points balance."""
    r = np.asarray(r_m, dtype=np.float64)
    rM = moon_state_at(t_s).r
    a = -MU_EARTH * r / np.linalg.norm(r)**3
    a += -MU_MOON * ((r - rM) / np.linalg.norm(r - rM)**3 + rM / np.linalg.norm(rM)**3)
    return a


def _substep_count(state, dt_s, attractors, max_step_s):
    """Number of uniform Verlet sub-steps for |dt_s|: small enough to resolve the
    closest body's local orbital timescale (1/200 of 2*pi*sqrt(r^3/mu))."""
    r = np.asarray(state.r, dtype=np.float64)
    cap = max_step_s
    for body in attractors:
        rb = np.linalg.norm(r - body.state_at(state.epoch_s).r)
        cap = min(cap, (2 * np.pi * np.sqrt(rb**3 / body.mu)) / 200.0)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


def _verlet(r, v, t, dt_s, accel_fn, n_sub):
    """Velocity-Verlet (kick-drift-kick) for n_sub uniform steps. accel_fn(r, t)->a."""
    r = np.asarray(r, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    h = dt_s / n_sub
    a = accel_fn(r, t)
    for _ in range(n_sub):
        v_half = v + 0.5 * a * h
        r = r + v_half * h
        t = t + h
        a = accel_fn(r, t)
        v = v_half + 0.5 * a * h
    return r, v, t


def propagate_nbody(state, dt_s, attractors=EARTH_MOON, max_step_s=3600.0):
    """Advance the ship by dt_s with velocity Verlet under summed attractors. Reversible."""
    n = _substep_count(state, dt_s, attractors, max_step_s)
    r, v, t = _verlet(state.r, state.v, state.epoch_s, dt_s,
                      lambda rr, tt: gravity_accel(rr, tt, attractors), n)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=t)


def _earth_moon_substeps(state, dt_s, max_step_s):
    """Sub-steps for propagate_earth_moon: cap by 1/200 of the local orbital timescale
    at Earth (origin) and at the Moon."""
    r = np.asarray(state.r, dtype=np.float64)
    cap = max_step_s
    rE = np.linalg.norm(r)
    cap = min(cap, (2 * np.pi * np.sqrt(rE**3 / MU_EARTH)) / 200.0)
    rM = np.linalg.norm(r - moon_state_at(state.epoch_s).r)
    cap = min(cap, (2 * np.pi * np.sqrt(rM**3 / MU_MOON)) / 200.0)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


def propagate_earth_moon(state, dt_s, max_step_s=3600.0):
    """Advance the ship by dt_s under earth_moon_accel (central Earth + Moon + indirect)."""
    n = _earth_moon_substeps(state, dt_s, max_step_s)
    r, v, t = _verlet(state.r, state.v, state.epoch_s, dt_s, earth_moon_accel, n)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=t)


MOON_SOI_M = 3.844e8 * (MU_MOON / MU_EARTH)**0.4   # Moon sphere of influence [m]


def osculating_elements(state, t_s):
    """Instantaneous Keplerian elements about the dominant body (Moon if the ship is
    within MOON_SOI_M of it, else Earth). Used for the HUD; drifts under perturbation."""
    rM = moon_state_at(t_s)
    if np.linalg.norm(state.r - rM.r) < MOON_SOI_M:
        rel = StateVector(state.r - rM.r, state.v - rM.v, MU_MOON, state.epoch_s)
    else:
        rel = StateVector(state.r, state.v, MU_EARTH, state.epoch_s)
    return state_to_elements(rel)


# ---------------------------------------------------------------------------
# Solar-system extension: Sun + all planets as geocentric perturbers.
# ---------------------------------------------------------------------------
from orbitsim.core.constants import (
    MU_SUN, MU_MERCURY, MU_VENUS, MU_MARS,
    MU_JUPITER, MU_SATURN, MU_URANUS, MU_NEPTUNE,
)
from orbitsim.core.planets import (
    sun_state_at, mercury_state_at, venus_state_at, mars_state_at,
    jupiter_state_at, saturn_state_at, uranus_state_at, neptune_state_at,
    EARTH_SOI_M, MERCURY_SOI_M, VENUS_SOI_M, MARS_SOI_M,
    JUPITER_SOI_M, SATURN_SOI_M, URANUS_SOI_M, NEPTUNE_SOI_M,
)
from orbitsim.core.bodies import SUN as SUN_BODY, MERCURY as MERCURY_BODY
from orbitsim.core.bodies import VENUS as VENUS_BODY, MARS as MARS_BODY, EARTH as EARTH_BODY
from orbitsim.core.bodies import JUPITER as JUPITER_BODY, SATURN as SATURN_BODY
from orbitsim.core.bodies import URANUS as URANUS_BODY, NEPTUNE as NEPTUNE_BODY

# ---------------------------------------------------------------------------
# Ephemeris cache: real JPL/Skyfield planet positions, cached once per frame.
# When populated, solar_system_accel uses real positions instead of circular
# approximations. Falls back gracefully if the DE440 kernel is unavailable.
# ---------------------------------------------------------------------------
_ephemeris_cache = {}
_ephemeris_context = threading.local()

try:
    from orbitsim.core.ephemeris import body_state as _ephem_body_state
    _EPHEMERIS_AVAILABLE = True
except Exception:
    _EPHEMERIS_AVAILABLE = False

_EPHEM_BODY_NAMES = ("SUN", "MERCURY", "VENUS", "MARS",
                     "JUPITER", "SATURN", "URANUS", "NEPTUNE")


def refresh_ephemeris_cache(t_s: float) -> bool:
    """Snapshot all planet geocentric positions from JPL/DE440 ephemeris.

    Call once per frame before physics stepping. Positions are constant within
    the frame's substeps (planet motion over one frame is negligible).
    Returns True if real ephemeris was used, False on fallback.
    """
    global _ephemeris_cache
    if not _EPHEMERIS_AVAILABLE:
        _ephemeris_cache = {}
        return False
    try:
        cache = {}
        for name in _EPHEM_BODY_NAMES:
            cache[name] = _ephem_body_state(name, t_s, center="EARTH")
        _ephemeris_cache = cache
        return True
    except Exception:
        _ephemeris_cache = {}
        return False


def ephemeris_available() -> bool:
    """Whether the JPL ephemeris cache is currently populated."""
    return bool(_ephemeris_cache)


@contextmanager
def stable_prediction_ephemeris():
    """Give a background prediction its own time-indexed JPL ephemeris cache."""
    previous = getattr(_ephemeris_context, "prediction_cache", None)
    _ephemeris_context.prediction_cache = {}
    try:
        yield
    finally:
        _ephemeris_context.prediction_cache = previous


_PREDICTION_EPHEMERIS_STEP_S = 86400.0


def _prediction_ephemeris_state(name, fallback_fn, t_s):
    """Cubic-Hermite interpolation of thread-local daily JPL samples."""
    cache = _ephemeris_context.prediction_cache
    step = _PREDICTION_EPHEMERIS_STEP_S
    bucket = int(np.floor(float(t_s) / step))

    def sample(index):
        key = (name, index)
        if key not in cache:
            epoch = index * step
            if _EPHEMERIS_AVAILABLE:
                try:
                    cache[key] = _ephem_body_state(name, epoch, center="EARTH")
                except Exception:
                    cache[key] = fallback_fn(epoch)
            else:
                cache[key] = fallback_fn(epoch)
        return cache[key]

    start = sample(bucket)
    end = sample(bucket + 1)
    u = (float(t_s) - bucket * step) / step
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    r = h00 * start.r + h10 * step * start.v + h01 * end.r + h11 * step * end.v
    dh00 = (6.0 * u2 - 6.0 * u) / step
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = (-6.0 * u2 + 6.0 * u) / step
    dh11 = 3.0 * u2 - 2.0 * u
    v = dh00 * start.r + dh10 * start.v + dh01 * end.r + dh11 * end.v
    return StateVector(r=r, v=v, mu=start.mu, epoch_s=float(t_s))


def _make_cached_state_fn(name, fallback_fn):
    """Create a state function that returns cached ephemeris when available."""
    def fn(t_s):
        prediction_cache = getattr(_ephemeris_context, "prediction_cache", None)
        if prediction_cache is not None:
            return _prediction_ephemeris_state(name, fallback_fn, t_s)
        cached = _ephemeris_cache.get(name)
        if cached is not None:
            return cached
        return fallback_fn(t_s)
    return fn


_csun = _make_cached_state_fn("SUN", sun_state_at)
_cmercury = _make_cached_state_fn("MERCURY", mercury_state_at)
_cvenus = _make_cached_state_fn("VENUS", venus_state_at)
_cmars = _make_cached_state_fn("MARS", mars_state_at)
_cjupiter = _make_cached_state_fn("JUPITER", jupiter_state_at)
_csaturn = _make_cached_state_fn("SATURN", saturn_state_at)
_curanus = _make_cached_state_fn("URANUS", uranus_state_at)
_cneptune = _make_cached_state_fn("NEPTUNE", neptune_state_at)

_SOLAR_PERTURBERS = [
    (_csun, MU_SUN),
    (_cmercury, MU_MERCURY),
    (_cvenus, MU_VENUS),
    (_cmars, MU_MARS),
    (_cjupiter, MU_JUPITER),
    (_csaturn, MU_SATURN),
    (_curanus, MU_URANUS),
    (_cneptune, MU_NEPTUNE),
]


def solar_system_accel(r_m, t_s):
    """Geocentric N-body acceleration [m/s^2]: central Earth + Moon + Sun + all planets.

    Extends earth_moon_accel with the Sun and all seven other planets as
    third-body perturbers. Each gets a direct term and an indirect term (the
    indirect term accounts for the non-inertial geocentric frame — Earth is
    accelerated by every body, and the frame tracks Earth).

    When the ephemeris cache is populated (via refresh_ephemeris_cache), uses
    real JPL/DE440 planet positions; otherwise falls back to circular
    approximations.
    """
    r = np.asarray(r_m, dtype=np.float64)
    a = -MU_EARTH * r / np.linalg.norm(r) ** 3
    rM = moon_state_at(t_s).r
    a += -MU_MOON * ((r - rM) / np.linalg.norm(r - rM) ** 3 + rM / np.linalg.norm(rM) ** 3)
    for state_fn, mu in _SOLAR_PERTURBERS:
        rB = state_fn(t_s).r
        dr = r - rB
        a += -mu * (dr / np.linalg.norm(dr) ** 3 + rB / np.linalg.norm(rB) ** 3)
    return a


def _solar_system_substeps(state, dt_s, max_step_s):
    """Adaptive sub-step count for the solar system propagator.

    Caps the sub-step at 1/200 of the local orbital timescale at every body
    (Earth, Moon, Sun, and each planet). Near a body the timescale is short
    and many sub-steps are needed; in deep space the timescale is long.
    """
    r = np.asarray(state.r, dtype=np.float64)
    cap = max_step_s
    rE = np.linalg.norm(r)
    if rE > 0:
        cap = min(cap, (2 * np.pi * np.sqrt(rE ** 3 / MU_EARTH)) / 200.0)
    rM = np.linalg.norm(r - moon_state_at(state.epoch_s).r)
    if rM > 0:
        cap = min(cap, (2 * np.pi * np.sqrt(rM ** 3 / MU_MOON)) / 200.0)
    for state_fn, mu, soi in [
        (_csun, MU_SUN, float("inf")),
        (_cmercury, MU_MERCURY, MERCURY_SOI_M),
        (_cvenus, MU_VENUS, VENUS_SOI_M),
        (_cmars, MU_MARS, MARS_SOI_M),
        (_cjupiter, MU_JUPITER, JUPITER_SOI_M),
        (_csaturn, MU_SATURN, SATURN_SOI_M),
        (_curanus, MU_URANUS, URANUS_SOI_M),
        (_cneptune, MU_NEPTUNE, NEPTUNE_SOI_M),
    ]:
        rB = np.linalg.norm(r - state_fn(state.epoch_s).r)
        if rB > 0 and rB < 10 * soi:
            cap = min(cap, (2 * np.pi * np.sqrt(rB ** 3 / mu)) / 200.0)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


def propagate_solar_system(state, dt_s, max_step_s=6.0 * 3600.0):
    """Propagate a vessel under full solar system gravity (geocentric)."""
    n = _solar_system_substeps(state, dt_s, max_step_s)
    r, v, t = _verlet(state.r, state.v, state.epoch_s, dt_s, solar_system_accel, n)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=t)


def dominant_body_solar(r_m, t_s):
    """Return (CelestialBody, geocentric_position) of the body whose SOI contains the
    vessel, preferring the smallest SOI (most specific body). Falls back to Earth."""
    r = np.asarray(r_m, dtype=np.float64)
    rM = moon_state_at(t_s).r
    if np.linalg.norm(r - rM) < MOON_SOI_M:
        from orbitsim.core.bodies import MOON as MOON_BODY
        return MOON_BODY, rM
    planet_checks = [
        (_cmercury, MERCURY_SOI_M, MERCURY_BODY),
        (_cvenus, VENUS_SOI_M, VENUS_BODY),
        (_cmars, MARS_SOI_M, MARS_BODY),
        (_curanus, URANUS_SOI_M, URANUS_BODY),
        (_cneptune, NEPTUNE_SOI_M, NEPTUNE_BODY),
        (_csaturn, SATURN_SOI_M, SATURN_BODY),
        (_cjupiter, JUPITER_SOI_M, JUPITER_BODY),
    ]
    for state_fn, soi, body in planet_checks:
        rB = state_fn(t_s).r
        if np.linalg.norm(r - rB) < soi:
            return body, rB
    if np.linalg.norm(r) > EARTH_SOI_M:
        rS = _csun(t_s).r
        return SUN_BODY, rS
    return EARTH_BODY, np.zeros(3)


_CACHED_BODY_LOOKUP = {
    "Sun": _csun, "Mercury": _cmercury, "Venus": _cvenus, "Mars": _cmars,
    "Jupiter": _cjupiter, "Saturn": _csaturn, "Uranus": _curanus, "Neptune": _cneptune,
}


def osculating_elements_solar(state, t_s):
    """Osculating Keplerian elements relative to the dominant body in the solar system.

    Checks the full body hierarchy: Moon → planets → Earth/Sun.
    """
    body, r_body = dominant_body_solar(state.r, t_s)
    if body.name == "Moon":
        rM = moon_state_at(t_s)
        rel = StateVector(state.r - rM.r, state.v - rM.v, MU_MOON, state.epoch_s)
    elif body.name == "Earth":
        rel = StateVector(state.r, state.v, MU_EARTH, state.epoch_s)
    else:
        cached_fn = _CACHED_BODY_LOOKUP.get(body.name)
        if cached_fn is not None:
            st = cached_fn(t_s)
            rel = StateVector(state.r - st.r, state.v - st.v, body.mu, state.epoch_s)
        else:
            rel = StateVector(state.r, state.v, MU_EARTH, state.epoch_s)
    return state_to_elements(rel)


def max_safe_warp_solar(state, t_s, warp_steps, real_dt_s=1 / 60, budget_substeps=200):
    """Largest warp whose per-frame integration stays within budget_substeps (solar system)."""
    allowed = [w for w in warp_steps
               if _solar_system_substeps(state, real_dt_s * w, 6.0 * 3600.0) <= budget_substeps]
    return max(allowed) if allowed else min(warp_steps)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # inertial->rotating


def rotating_frame(r_m, v_mps, t_s):
    """Map an inertial state into the frame co-rotating at OMEGA_EM (Moon fixed +x)."""
    R = _rot_z(OMEGA_EM * t_s)
    r_rot = R @ np.asarray(r_m, dtype=np.float64)
    w = np.array([0.0, 0.0, OMEGA_EM])
    v_rot = R @ np.asarray(v_mps, dtype=np.float64) - np.cross(w, r_rot)
    return r_rot, v_rot


def jacobi_constant(state, t_s):
    """Jacobi constant C = 2*Omega - |v_rot|^2 (conserved along a coast)."""
    r_rot, v_rot = rotating_frame(state.r, state.v, t_s)
    x, y = r_rot[0], r_rot[1]
    r1 = np.linalg.norm(r_rot - np.array([EARTH_X, 0.0, 0.0]))
    r2 = np.linalg.norm(r_rot - np.array([MOON_X, 0.0, 0.0]))
    omega2 = 0.5 * OMEGA_EM**2 * (x**2 + y**2) + MU_EARTH / r1 + MU_MOON / r2
    return float(2.0 * omega2 - np.dot(v_rot, v_rot))


def _collinear_accel_x(x):
    """Net rotating-frame x-acceleration for a point on the Earth-Moon axis at x."""
    ax_g = (-MU_EARTH * (x - EARTH_X) / abs(x - EARTH_X)**3
            - MU_MOON * (x - MOON_X) / abs(x - MOON_X)**3)
    return OMEGA_EM**2 * x + ax_g


def lagrange_points(t_s):
    """Inertial positions of L1..L5 at t_s [m]."""
    eps = 1e-3 * D_EM
    x1 = brentq(_collinear_accel_x, EARTH_X + eps, MOON_X - eps)      # between bodies
    x2 = brentq(_collinear_accel_x, MOON_X + eps, MOON_X + 0.4 * D_EM)  # beyond Moon
    x3 = brentq(_collinear_accel_x, -1.6 * D_EM, EARTH_X - eps)       # beyond Earth
    h = np.sqrt(3.0) / 2.0 * D_EM
    xtri = (0.5 - MASS_RATIO) * D_EM
    rot = {
        "L1": np.array([x1, 0.0, 0.0]),
        "L2": np.array([x2, 0.0, 0.0]),
        "L3": np.array([x3, 0.0, 0.0]),
        "L4": np.array([xtri, h, 0.0]),
        "L5": np.array([xtri, -h, 0.0]),
    }
    Rinv = _rot_z(OMEGA_EM * t_s).T   # rotating -> inertial
    return {k: Rinv @ v for k, v in rot.items()}


def _earth_fixed_collinear_accel_along(s, u, t_s):
    """Net rotating-frame acceleration at the point p = s*u (signed distance s along the
    Earth-Moon unit vector u), projected onto u: live gravity (with the indirect term) plus
    the centrifugal term OMEGA_EM**2 * p about the origin."""
    p = s * np.asarray(u, dtype=np.float64)
    a = earth_moon_accel(p, t_s) + OMEGA_EM**2 * p
    return float(np.dot(a, u))


def earth_fixed_lagrange_points(t_s):
    """Inertial positions of L1..L5 [m] in the live Earth-fixed frame, consistent with
    earth_moon_accel (indirect term) and the Moon's actual geometry at t_s (e=0 but inclined).

    Collinear points solve net_along(s)=0 along the Earth-Moon line; the equilateral points are
    the Moon position rotated +/-60 deg about the orbit normal."""
    m = moon_state_at(t_s)
    rM = np.asarray(m.r, dtype=np.float64)
    d = np.linalg.norm(rM)
    u = rM / d
    n_hat = np.cross(rM, m.v)
    n_hat = n_hat / np.linalg.norm(n_hat)
    eps = 1e-3 * d
    s1 = brentq(_earth_fixed_collinear_accel_along, eps, d - eps, args=(u, t_s))           # L1
    s2 = brentq(_earth_fixed_collinear_accel_along, d + eps, d + 0.4 * d, args=(u, t_s))   # L2
    s3 = brentq(_earth_fixed_collinear_accel_along, -1.6 * d, -eps, args=(u, t_s))         # L3

    def _rot(vec, ang):   # Rodrigues rotation of vec by ang about n_hat
        c, sn = np.cos(ang), np.sin(ang)
        return vec * c + np.cross(n_hat, vec) * sn + n_hat * np.dot(n_hat, vec) * (1.0 - c)

    return {
        "L1": s1 * u,
        "L2": s2 * u,
        "L3": s3 * u,
        "L4": _rot(rM, np.radians(60.0)),
        "L5": _rot(rM, np.radians(-60.0)),
    }


def max_safe_warp(state, t_s, warp_steps, real_dt_s=1 / 60, budget_substeps=200):
    """Largest warp in warp_steps whose frame integrates within budget_substeps Verlet
    sub-steps at the current proximity (so time-warp stays accurate near bodies)."""
    allowed = [w for w in warp_steps
               if _earth_moon_substeps(state, real_dt_s * w, 3600.0) <= budget_substeps]
    return max(allowed) if allowed else min(warp_steps)
