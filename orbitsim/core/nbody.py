"""Restricted N-body (CR3BP) core: idealized circular Earth+Moon, a velocity-Verlet
ship propagator, the Jacobi constant, and the Lagrange points. Barycentric inertial,
SI, float64. The ship is a massless test particle; the bodies are on rails."""
from contextlib import contextmanager
import math
import threading

import numpy as np
from scipy.optimize import brentq

from orbitsim.core.constants import MU_EARTH, MU_MOON, R_EARTH, R_MOON
from orbitsim.core.state import StateVector
from orbitsim.core.moon import moon_state_at
from orbitsim.core.hermite import hermite_state
from orbitsim.core.elements import state_to_elements

def _norm3(x) -> float:
    """|x| for a real 3-vector; bit-identical to np.linalg.norm (which is sqrt(x.dot(x)))
    without its n-d overhead. The substep loop calls this ~3,000x per frame at warp."""
    return math.sqrt(x.dot(x))


def _cross3(a, b) -> np.ndarray:
    """a x b for 3-vectors; same operations and order as np.cross, so bit-identical."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


D_EM = 3.844e8                          # Earth-Moon separation [m]
MU_TOTAL = MU_EARTH + MU_MOON
OMEGA_EM = np.sqrt(MU_TOTAL / D_EM**3)  # mean motion [rad/s]
MASS_RATIO = MU_MOON / MU_TOTAL         # ~0.01215
EARTH_X = -MASS_RATIO * D_EM            # Earth rotating-frame x [m]
MOON_X = (1.0 - MASS_RATIO) * D_EM      # Moon rotating-frame x [m]


class _CircularBody:
    """A point mass on a circular barycentric orbit (signed radius along the
    rotating x-axis at t=0; rotates at OMEGA_EM). body_radius_m floors the
    substep sizing: below the surface the physics is over anyway."""

    def __init__(self, mu: float, signed_radius_m: float, body_radius_m: float = 0.0):
        self.mu = mu
        self._R = signed_radius_m
        self.body_radius_m = body_radius_m

    def state_at(self, t_s: float) -> StateVector:
        th = OMEGA_EM * t_s
        u = np.array([np.cos(th), np.sin(th), 0.0])
        n = np.array([-np.sin(th), np.cos(th), 0.0])
        return StateVector(r=self._R * u, v=self._R * OMEGA_EM * n,
                           mu=self.mu, epoch_s=t_s)


EARTH = _CircularBody(MU_EARTH, EARTH_X, R_EARTH)
MOON = _CircularBody(MU_MOON, MOON_X, R_MOON)
EARTH_MOON = [EARTH, MOON]


def gravity_accel(r_m, t_s, attractors=EARTH_MOON):
    """Summed gravitational acceleration on a test particle at r_m, time t_s [m/s^2]."""
    r = np.asarray(r_m, dtype=np.float64)
    a = np.zeros(3)
    for body in attractors:
        d = r - body.state_at(t_s).r
        a += -body.mu * d / _norm3(d)**3
    return a


def earth_moon_accel(r_m, t_s):
    """Ship acceleration in the EARTH-CENTERED frame [m/s^2]: central Earth plus the
    Moon's third-body perturbation. The indirect term (+mu_M * r_M/|r_M|^3) is required
    because Earth is held fixed at the origin (non-inertial frame); it is also what makes
    the Lagrange points balance."""
    r = np.asarray(r_m, dtype=np.float64)
    rM = moon_state_at(t_s).r
    a = -MU_EARTH * r / _norm3(r)**3
    a += -MU_MOON * ((r - rM) / _norm3(r - rM)**3 + rM / _norm3(rM)**3)
    return a


# Substep sizing. The dominant body's substep is 1/SUBSTEP_DIVISOR of the
# orbital timescale 2*pi*sqrt(rp^3/mu) at the *osculating periapsis* rp, not
# at the current radius: on a coast rp is (nearly) constant, so the step is
# uniform around the whole orbit — velocity Verlet stays symplectic (no
# secular energy drift) — while periapsis passages are resolved by
# construction, no matter where inside a long warp step they occur. Other
# bodies contribute a cap from the current distance (they only matter when
# close). rp is floored at the body's surface radius: below the surface the
# flight is over (sim/world.py clamps to the surface), so resolving deeper
# is pointless. MAX_SUBSTEPS_PER_CALL is a safety bound (not a target) so a
# pathological state can never hang the caller.
#
# 100 with the 4th-order (Yoshida) step: on the pinned Moon transfer this flies
# within ~25 km of the converged closest approach (Verlet at 200 missed by
# ~2,500 km). NOTE: _timescale_cap_s / _solar_cap_s bind this value as a default
# argument when the module loads — patching the constant at runtime changes nothing;
# patch their __defaults__ (or pass divisor=) instead.
SUBSTEP_DIVISOR = 100.0
MAX_SUBSTEPS_PER_CALL = 100_000
_MIN_SUBSTEP_S = 1e-6      # floor so a zero cap (r exactly at a body) can't stall


def _conic_periapsis_m(r_rel, v_rel, mu):
    """Periapsis radius [m] of the osculating conic — valid for any e (>=0)."""
    r = _norm3(r_rel)
    if r == 0.0:
        return 0.0
    h_vec = _cross3(r_rel, v_rel)
    p = np.dot(h_vec, h_vec) / mu
    e_vec = _cross3(v_rel, h_vec) / mu - r_rel / r
    return p / (1.0 + _norm3(e_vec))


def _timescale_cap_s(radius_m, mu, divisor=SUBSTEP_DIVISOR):
    """1/divisor of the circular-orbit timescale at radius_m (default SUBSTEP_DIVISOR)."""
    return (2 * np.pi * np.sqrt(radius_m**3 / mu)) / divisor


def _attractor_cap_s(r, v, t_s, attractors, max_step_s):
    """Largest safe substep at state (r, v) [s].

    The body with the shortest current-distance timescale is treated as
    dominant and sized by its osculating periapsis; the others by current
    distance.
    """
    caps = []
    dists = []
    for body in attractors:
        rb = _norm3(r - body.state_at(t_s).r)
        dists.append(rb)
        caps.append(_timescale_cap_s(rb, body.mu) if rb > 0.0 else np.inf)
    cap = max_step_s
    if caps:
        dom = int(np.argmin(caps))
        for i, body in enumerate(attractors):
            if dists[i] <= 0.0:
                continue
            if i == dom:
                st = body.state_at(t_s)
                rp = _conic_periapsis_m(r - st.r, v - st.v, body.mu)
                rp = max(rp, body.body_radius_m)
                if rp > 0.0:
                    cap = min(cap, _timescale_cap_s(rp, body.mu))
            else:
                cap = min(cap, caps[i])
    return cap


def _substep_count(state, dt_s, attractors, max_step_s):
    """Estimated sub-step count for |dt_s| (used for warp budgeting; must use
    the same cap the integrator uses so warp budgets stay honest)."""
    cap = _attractor_cap_s(np.asarray(state.r, dtype=np.float64),
                           np.asarray(state.v, dtype=np.float64),
                           state.epoch_s, attractors, max_step_s)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


# Yoshida (1990) 4th-order symplectic composition: one step of size h is three
# velocity-Verlet sub-steps of w1*h, w0*h, w1*h (w0 < 0, 2*w1 + w0 = 1). The
# composition is symmetric, so it stays symplectic and time-reversible, and the
# error drops from O(h^2) to O(h^4). Each sub-step reuses the previous end
# acceleration (FSAL), so a step costs EVALS_PER_STEP gravity evaluations.
YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) * YOSHIDA_W1
_YOSHIDA_WEIGHTS = (YOSHIDA_W1, YOSHIDA_W0, YOSHIDA_W1)
EVALS_PER_STEP = len(_YOSHIDA_WEIGHTS)


def _symplectic_adaptive(r, v, t, dt_s, accel_fn, cap_fn, base_step_s,
                         min_substep_s=_MIN_SUBSTEP_S):
    """Yoshida 4th-order symplectic integrator with a block-quantized adaptive step.

    Each step is three velocity-Verlet (kick-drift-kick) sub-steps weighted
    YOSHIDA_W1, YOSHIDA_W0, YOSHIDA_W1. cap_fn(r, v, t) returns the largest
    allowed step [s] at that state.
    With periapsis-based caps the cap is (nearly) constant along a coast, so
    the steps are uniform and the integrator keeps its symplectic
    bounded-energy behavior; when the cap does move (SOI hand-offs, thrust
    history), the step is quantized to base_step_s / 2^k — the largest such
    value not exceeding the local cap — so step changes are rare, discrete
    block switches rather than continuous drift-inducing jitter.

    ``min_substep_s`` floors the substep. The accurate on-rails sim leaves it at
    the tiny default; the visual trajectory prediction raises it so a near-Earth
    climb-out (deep in the gravity well, where the periapsis cap would otherwise
    force ~25 s steps) stays cheap — it trades sub-periapsis detail the coarse
    prediction line never draws anyway for a large speed-up.
    """
    r = np.asarray(r, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    if dt_s == 0.0:
        return r, v, t
    sign = 1.0 if dt_s > 0.0 else -1.0
    remaining = abs(dt_s)
    a = accel_fn(r, t)
    steps = 0
    while remaining > 0.0:
        steps += 1
        if steps >= MAX_SUBSTEPS_PER_CALL:
            h_mag = remaining              # budget exhausted: finish coarsely
        else:
            cap = max(cap_fn(r, v, t), min_substep_s)
            k = max(0.0, np.ceil(np.log2(base_step_s / cap)))
            h_mag = min(remaining, base_step_s / 2.0**k)
        remaining -= h_mag
        for w in _YOSHIDA_WEIGHTS:
            h = sign * h_mag * w
            v_half = v + 0.5 * a * h
            r = r + v_half * h
            t = t + h
            a = accel_fn(r, t)
            v = v_half + 0.5 * a * h
    return r, v, t


_verlet_adaptive = _symplectic_adaptive   # historical name; callers and tests keep it


def propagate_nbody(state, dt_s, attractors=EARTH_MOON, max_step_s=3600.0,
                    min_substep_s=_MIN_SUBSTEP_S):
    """Advance the ship by dt_s with adaptive velocity Verlet under summed attractors."""
    r, v, t = _verlet_adaptive(
        state.r, state.v, state.epoch_s, dt_s,
        lambda rr, tt: gravity_accel(rr, tt, attractors),
        lambda rr, vv, tt: _attractor_cap_s(rr, vv, tt, attractors, max_step_s),
        max_step_s, min_substep_s=min_substep_s)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=state.epoch_s + dt_s)


def _earth_moon_cap_s(r, v, t_s, max_step_s):
    """Largest safe substep at (r, v) for the geocentric Earth+Moon field [s].

    Inside the Moon's SOI the Moon is dominant (periapsis-based cap) and Earth
    contributes a current-distance cap; outside, the roles swap.
    """
    cap = max_step_s
    moon = moon_state_at(t_s)
    rM = _norm3(r - moon.r)
    rE = _norm3(r)
    if rM < MOON_SOI_M:
        rp = max(_conic_periapsis_m(r - moon.r, v - moon.v, MU_MOON), R_MOON)
        cap = min(cap, _timescale_cap_s(rp, MU_MOON))
        if rE > 0.0:
            cap = min(cap, _timescale_cap_s(rE, MU_EARTH))
    else:
        if rE > 0.0:
            rp = max(_conic_periapsis_m(r, v, MU_EARTH), R_EARTH)
            cap = min(cap, _timescale_cap_s(rp, MU_EARTH))
        if rM > 0.0:
            cap = min(cap, _timescale_cap_s(rM, MU_MOON))
    return cap


def _earth_moon_substeps(state, dt_s, max_step_s):
    """Estimated sub-steps for propagate_earth_moon (used for warp budgeting;
    must use the same cap the integrator uses so warp budgets stay honest)."""
    cap = _earth_moon_cap_s(np.asarray(state.r, dtype=np.float64),
                            np.asarray(state.v, dtype=np.float64),
                            state.epoch_s, max_step_s)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


def propagate_earth_moon(state, dt_s, max_step_s=3600.0, min_substep_s=_MIN_SUBSTEP_S):
    """Advance the ship by dt_s under earth_moon_accel (central Earth + Moon + indirect)."""
    r, v, t = _verlet_adaptive(
        state.r, state.v, state.epoch_s, dt_s, earth_moon_accel,
        lambda rr, vv, tt: _earth_moon_cap_s(rr, vv, tt, max_step_s), max_step_s,
        min_substep_s=min_substep_s)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=state.epoch_s + dt_s)


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
    from orbitsim.core.ephemeris import body_states as _ephem_body_states
    _EPHEMERIS_AVAILABLE = True
except ImportError:
    # skyfield itself is missing; keep the attribute defined so callers (and
    # tests) can always reference/patch nbody._ephem_body_state.
    _ephem_body_state = None
    _ephem_body_states = None
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
    except ValueError:
        # Expected ephemeris failures are ValueErrors: EphemerisUnavailableError
        # (kernel missing/offline) and Skyfield's EphemerisRangeError (date outside
        # DE440's 1849-2150). Anything else is a bug and must not be masked by the
        # circular fallback.
        _ephemeris_cache = {}
        return False


def ephemeris_available() -> bool:
    """Whether the JPL ephemeris cache is currently populated."""
    return bool(_ephemeris_cache)


@contextmanager
def stable_prediction_ephemeris():
    """Give a background prediction its own time-indexed JPL ephemeris cache."""
    previous = getattr(_ephemeris_context, "prediction_cache", None)
    previous_last = getattr(_ephemeris_context, "prediction_last", None)
    _ephemeris_context.prediction_cache = {}
    # {name: (t_s, state)}: the accel and the substep cap ask for every body at the
    # same t; reuse the interpolated state instead of computing it twice (bit-identical).
    _ephemeris_context.prediction_last = {}
    try:
        yield
    finally:
        _ephemeris_context.prediction_cache = previous
        _ephemeris_context.prediction_last = previous_last


_PREDICTION_EPHEMERIS_STEP_S = 86400.0

# Daily knots kept across frames for deep-space flight (bounded: 8 bodies x ~1 yr).
_flight_samples: dict = {}
_FLIGHT_SAMPLES_MAX = 8 * 400
# A missing knot fetches this many days at once (one vectorized Skyfield call,
# bit-identical to scalar calls): a 1e8x frame spans ~19 days.
_FLIGHT_KNOT_BATCH = 64


@contextmanager
def time_correct_flight_ephemeris():
    """Planet positions at each substep's own time for one flight tick.

    The per-frame cache freezes every planet for a whole frame. Near Earth that is
    harmless (Earth and ship feel almost the same Sun), but outside Earth's SOI the
    direct and indirect solar terms no longer cancel and a frozen Sun puts a
    heliocentric coast ~2,000,000 km off after 60 days at 1e7x. Inside this context
    the cached wrappers interpolate daily DE440 knots (cubic Hermite, metres of
    error), exactly like a background prediction, but the knots persist across
    frames and are fetched 64 days at a time.
    """
    previous = getattr(_ephemeris_context, "prediction_cache", None)
    previous_last = getattr(_ephemeris_context, "prediction_last", None)
    previous_batch = getattr(_ephemeris_context, "prediction_batch", 1)
    if len(_flight_samples) > _FLIGHT_SAMPLES_MAX:
        _flight_samples.clear()
    _ephemeris_context.prediction_cache = _flight_samples
    _ephemeris_context.prediction_last = {}
    _ephemeris_context.prediction_batch = _FLIGHT_KNOT_BATCH
    try:
        yield
    finally:
        _ephemeris_context.prediction_cache = previous
        _ephemeris_context.prediction_last = previous_last
        _ephemeris_context.prediction_batch = previous_batch


def _prediction_ephemeris_state(name, fallback_fn, t_s):
    """Cubic-Hermite interpolation of thread-local daily JPL samples."""
    last = _ephemeris_context.prediction_last
    hit = last.get(name)
    if hit is not None and hit[0] == t_s:
        return hit[1]
    cache = _ephemeris_context.prediction_cache
    step = _PREDICTION_EPHEMERIS_STEP_S
    bucket = int(np.floor(float(t_s) / step))

    def sample(index):
        key = (name, index)
        if key not in cache:
            epoch = index * step
            batch = getattr(_ephemeris_context, "prediction_batch", 1)
            if batch > 1 and _EPHEMERIS_AVAILABLE:
                try:
                    knots = _ephem_body_states(
                        name, np.arange(index, index + batch) * step, center="EARTH")
                except ValueError:   # offline / out of range: knot by knot below
                    knots = None
                if knots is not None:
                    for offset, st in enumerate(knots):
                        cache.setdefault((name, index + offset), st)
                    return cache[key]
            if _EPHEMERIS_AVAILABLE:
                try:
                    cache[key] = _ephem_body_state(name, epoch, center="EARTH")
                except ValueError:   # offline / out of DE440 range (see refresh)
                    cache[key] = fallback_fn(epoch)
            else:
                cache[key] = fallback_fn(epoch)
        return cache[key]

    start = sample(bucket)
    end = sample(bucket + 1)
    if getattr(_ephemeris_context, "prediction_batch", 1) > 1:
        state = _lean_hermite(start, end, bucket * step, step, t_s)   # flight
    else:
        state = hermite_state(start, end, bucket * step, step, t_s, mu=start.mu)
    last[name] = (t_s, state)
    return state


class _LeanState:
    """r/v/mu/epoch_s without StateVector's validation (flight's per-substep planet
    positions: interpolated from validated knots, read-only consumers)."""
    __slots__ = ("r", "v", "mu", "epoch_s")

    def __init__(self, r, v, mu, epoch_s):
        self.r, self.v, self.mu, self.epoch_s = r, v, mu, epoch_s


def _lean_hermite(start, end, t0_s, step_s, t_s):
    """core.hermite.hermite_state's arithmetic, returning a _LeanState."""
    u = (float(t_s) - t0_s) / step_s
    u2 = u * u
    u3 = u2 * u
    r = ((2.0 * u3 - 3.0 * u2 + 1.0) * start.r + (u3 - 2.0 * u2 + u) * step_s * start.v
         + (-2.0 * u3 + 3.0 * u2) * end.r + (u3 - u2) * step_s * end.v)
    v = (((6.0 * u2 - 6.0 * u) / step_s) * start.r + (3.0 * u2 - 4.0 * u + 1.0) * start.v
         + ((-6.0 * u2 + 6.0 * u) / step_s) * end.r + (3.0 * u2 - 2.0 * u) * end.v)
    return _LeanState(r, v, start.mu, float(t_s))


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
    a = -MU_EARTH * r / _norm3(r) ** 3
    rM = moon_state_at(t_s).r
    a += -MU_MOON * ((r - rM) / _norm3(r - rM) ** 3 + rM / _norm3(rM) ** 3)
    for state_fn, mu in _SOLAR_PERTURBERS:
        rB = state_fn(t_s).r
        dr = r - rB
        a += -mu * (dr / _norm3(dr) ** 3 + rB / _norm3(rB) ** 3)
    return a


from orbitsim.core.constants import (
    R_SUN, R_MERCURY, R_VENUS, R_MARS,
    R_JUPITER, R_SATURN, R_URANUS, R_NEPTUNE,
)

# (state_fn, mu, soi_m, body_radius_m) for the heliocentric perturbers.
_SOLAR_CAP_BODIES = [
    (_csun, MU_SUN, float("inf"), R_SUN),
    (_cmercury, MU_MERCURY, MERCURY_SOI_M, R_MERCURY),
    (_cvenus, MU_VENUS, VENUS_SOI_M, R_VENUS),
    (_cmars, MU_MARS, MARS_SOI_M, R_MARS),
    (_cjupiter, MU_JUPITER, JUPITER_SOI_M, R_JUPITER),
    (_csaturn, MU_SATURN, SATURN_SOI_M, R_SATURN),
    (_curanus, MU_URANUS, URANUS_SOI_M, R_URANUS),
    (_cneptune, MU_NEPTUNE, NEPTUNE_SOI_M, R_NEPTUNE),
]


def _solar_cap_s(r, v, t_s, max_step_s, periapsis_fn=_conic_periapsis_m,
                divisor=SUBSTEP_DIVISOR):
    """Largest safe substep at (r, v) for the full solar-system field [s].

    The SOI-dominant body (Moon -> planet -> Earth -> Sun) gets a
    periapsis-based cap; every other body within 10 SOI contributes a
    current-distance cap. Near a body the timescale is short and many
    sub-steps are needed; in deep space the timescale is long.
    """
    cap = max_step_s
    moon = moon_state_at(t_s)
    rM = _norm3(r - moon.r)
    rE = _norm3(r)
    moon_dominant = rM < MOON_SOI_M
    if moon_dominant:
        rp = max(periapsis_fn(r - moon.r, v - moon.v, MU_MOON), R_MOON)
        cap = min(cap, _timescale_cap_s(rp, MU_MOON, divisor))
    elif rM > 0:
        cap = min(cap, _timescale_cap_s(rM, MU_MOON, divisor))

    planet_dominant = False
    for state_fn, mu, soi, radius in _SOLAR_CAP_BODIES[1:]:   # planets only
        st = state_fn(t_s)
        rB = _norm3(r - st.r)
        if rB <= 0 or rB >= 10 * soi:
            continue
        if rB < soi and not moon_dominant:
            planet_dominant = True
            rp = max(periapsis_fn(r - st.r, v - st.v, mu), radius)
            cap = min(cap, _timescale_cap_s(rp, mu, divisor))
        else:
            cap = min(cap, _timescale_cap_s(rB, mu, divisor))

    earth_dominant = not moon_dominant and not planet_dominant and rE < EARTH_SOI_M
    if earth_dominant:
        rp = max(periapsis_fn(r, v, MU_EARTH), R_EARTH)
        cap = min(cap, _timescale_cap_s(rp, MU_EARTH, divisor))
    elif rE > 0:
        cap = min(cap, _timescale_cap_s(rE, MU_EARTH, divisor))

    sun = _csun(t_s)
    rS = _norm3(r - sun.r)
    if rS > 0:
        if moon_dominant or planet_dominant or earth_dominant:
            cap = min(cap, _timescale_cap_s(rS, MU_SUN, divisor))
        else:
            rp = max(periapsis_fn(r - sun.r, v - sun.v, MU_SUN), R_SUN)
            cap = min(cap, _timescale_cap_s(rp, MU_SUN, divisor))
    return cap


def _solar_system_substeps(state, dt_s, max_step_s):
    """Estimated sub-step count for the solar system propagator (used for warp
    budgeting; must use the same cap the integrator uses)."""
    cap = _solar_cap_s(np.asarray(state.r, dtype=np.float64),
                       np.asarray(state.v, dtype=np.float64),
                       state.epoch_s, max_step_s)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


# Flight's largest coast step [s]. It binds only in deep space (every body's cap is
# longer); steps are quantized to FLIGHT_MAX_STEP_S / 2^k, and 12 h / 2^(k+1) equals
# the old 6 h / 2^k, so a coast whose cap is under 12 h steps exactly as before. At
# 6 h, Yoshida's 3 evaluations per step pushed 100,000,000x over the 200-evaluation
# frame budget in deep space; at 12 h it fits (117 evaluations).
FLIGHT_MAX_STEP_S = 12.0 * 3600.0


def propagate_solar_system(state, dt_s, max_step_s=FLIGHT_MAX_STEP_S,
                           min_substep_s=_MIN_SUBSTEP_S):
    """Propagate a vessel under full solar system gravity (geocentric)."""
    r, v, t = _verlet_adaptive(
        state.r, state.v, state.epoch_s, dt_s, solar_system_accel,
        lambda rr, vv, tt: _solar_cap_s(rr, vv, tt, max_step_s), max_step_s,
        min_substep_s=min_substep_s)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=state.epoch_s + dt_s)


# Drawn predictions (live trajectory + maneuver preview) only. Flight's periapsis-sized
# steps with the old 120 s preview floor mispredicted a Moon flyby by 7x; stepping by the
# *current* distance to each body at 1/PREDICTION_SUBSTEP_DIVISOR of its timescale matches the converged path
# within ~1% at ~6x the old speed. Never used by World.step (the flown sim).
PREDICTION_SUBSTEP_DIVISOR = 300.0   # 4th-order step: 0.02% Moon-pass error at ~0.6 s (1000 cost ~2 s)


def _current_distance_m(r_rel, v_rel, mu):
    """Step-size radius for predictions: the current distance (not the conic periapsis)."""
    return _norm3(r_rel)


def propagate_solar_prediction(state, dt_s, max_step_s=24.0 * 3600.0, min_substep_s=1.0):
    """Propagate under full solar-system gravity for a *drawn prediction*.

    Same Verlet integrator and forces as propagate_solar_system, but each body's
    substep cap uses its current distance at 1/PREDICTION_SUBSTEP_DIVISOR of the local
    timescale: fine near Earth and the Moon, coarse in between. Not bit-identical to the
    flown sim (it is closer to the converged path); never call it from World.step.
    """
    r, v, t = _verlet_adaptive(
        state.r, state.v, state.epoch_s, dt_s, solar_system_accel,
        lambda rr, vv, tt: _solar_cap_s(rr, vv, tt, max_step_s,
                                        periapsis_fn=_current_distance_m,
                                        divisor=PREDICTION_SUBSTEP_DIVISOR),
        max_step_s, min_substep_s=min_substep_s)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=state.epoch_s + dt_s)


def dominant_body_solar(r_m, t_s):
    """Return (CelestialBody, geocentric_position) of the body whose SOI contains the
    vessel, preferring the smallest SOI (most specific body). Falls back to Earth."""
    r = np.asarray(r_m, dtype=np.float64)
    rM = moon_state_at(t_s).r
    if _norm3(r - rM) < MOON_SOI_M:
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
        if _norm3(r - rB) < soi:
            return body, rB
    if _norm3(r) > EARTH_SOI_M:
        rS = _csun(t_s).r
        return SUN_BODY, rS
    return EARTH_BODY, np.zeros(3)


_CACHED_BODY_LOOKUP = {
    "Sun": _csun, "Mercury": _cmercury, "Venus": _cvenus, "Mars": _cmars,
    "Jupiter": _cjupiter, "Saturn": _csaturn, "Uranus": _curanus, "Neptune": _cneptune,
}


def geocentric_body_state(name: str, t_s: float) -> StateVector:
    """Geocentric StateVector of any named body ('Earth' is the frame origin).

    Uses the per-frame ephemeris cache (with circular fallback) for the Sun
    and planets, and the on-rails Moon.
    """
    if name == "Earth":
        return StateVector(r=np.zeros(3), v=np.zeros(3), mu=MU_EARTH, epoch_s=t_s)
    if name == "Moon":
        return moon_state_at(t_s)
    fn = _CACHED_BODY_LOOKUP.get(name)
    if fn is None:
        raise ValueError(f"unknown body: {name!r}")
    return fn(t_s)


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


def max_safe_warp_solar(state, t_s, warp_steps, real_dt_s=1 / 60, budget_evals=200):
    """Largest warp whose per-frame integration stays within budget_evals gravity
    evaluations (solar system; a coast step costs EVALS_PER_STEP)."""
    allowed = [w for w in warp_steps
               if _solar_system_substeps(state, real_dt_s * w, FLIGHT_MAX_STEP_S)
               * EVALS_PER_STEP
               <= budget_evals]
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


def _earth_fixed_collinear_accel_along(s, u, t_s, omega_sq):
    """Net rotating-frame acceleration at the point p = s*u (signed distance s along the
    Earth-Moon unit vector u), projected onto u: live gravity (with the indirect term) plus
    the centrifugal term omega_sq * p about the origin."""
    p = s * np.asarray(u, dtype=np.float64)
    a = earth_moon_accel(p, t_s) + omega_sq * p
    return float(np.dot(a, u))


def earth_fixed_lagrange_points(t_s):
    """Inertial positions of L1..L5 [m] in the live Earth-fixed frame, consistent with
    earth_moon_accel (indirect term) and the Moon's actual (real, eccentric) geometry at t_s.

    The Moon's orbit is eccentric, so there is no rigidly rotating frame; the points are
    defined in the pulsating frame of the elliptic restricted 3-body problem: the CR3BP
    geometry rescaled to the current separation d, i.e. balance points of gravity against
    the Kepler rate for that separation, sqrt(mu/d^3). Collinear points solve net_along(s)=0
    along the Earth-Moon line; the equilateral points are the Moon position rotated +/-60
    deg about the orbit normal (exact for that rate)."""
    m = moon_state_at(t_s)
    rM = np.asarray(m.r, dtype=np.float64)
    d = np.linalg.norm(rM)
    u = rM / d
    n_hat = np.cross(rM, m.v)
    n_hat = n_hat / np.linalg.norm(n_hat)
    omega_sq = MU_TOTAL / d**3
    eps = 1e-3 * d
    args = (u, t_s, omega_sq)
    s1 = brentq(_earth_fixed_collinear_accel_along, eps, d - eps, args=args)           # L1
    s2 = brentq(_earth_fixed_collinear_accel_along, d + eps, d + 0.4 * d, args=args)   # L2
    s3 = brentq(_earth_fixed_collinear_accel_along, -1.6 * d, -eps, args=args)         # L3

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


def max_safe_warp(state, t_s, warp_steps, real_dt_s=1 / 60, budget_evals=200):
    """Largest warp in warp_steps whose frame integrates within budget_evals gravity
    evaluations (EVALS_PER_STEP per coast step) at the current proximity (so
    time-warp stays accurate and affordable near bodies)."""
    allowed = [w for w in warp_steps
               if _earth_moon_substeps(state, real_dt_s * w, 3600.0) * EVALS_PER_STEP
               <= budget_evals]
    return max(allowed) if allowed else min(warp_steps)
