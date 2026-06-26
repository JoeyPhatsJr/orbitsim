"""Restricted N-body (CR3BP) core: idealized circular Earth+Moon, a velocity-Verlet
ship propagator, the Jacobi constant, and the Lagrange points. Barycentric inertial,
SI, float64. The ship is a massless test particle; the bodies are on rails."""
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
