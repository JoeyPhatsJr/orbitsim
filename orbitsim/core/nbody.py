"""Restricted N-body (CR3BP) core: idealized circular Earth+Moon, a velocity-Verlet
ship propagator, the Jacobi constant, and the Lagrange points. Barycentric inertial,
SI, float64. The ship is a massless test particle; the bodies are on rails."""
import numpy as np

from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.state import StateVector

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


def _substep_count(state, dt_s, attractors, max_step_s):
    """Number of uniform Verlet sub-steps for |dt_s|: small enough to resolve the
    closest body's local orbital timescale (1/200 of 2*pi*sqrt(r^3/mu))."""
    r = np.asarray(state.r, dtype=np.float64)
    cap = max_step_s
    for body in attractors:
        rb = np.linalg.norm(r - body.state_at(state.epoch_s).r)
        cap = min(cap, (2 * np.pi * np.sqrt(rb**3 / body.mu)) / 200.0)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


def propagate_nbody(state, dt_s, attractors=EARTH_MOON, max_step_s=3600.0):
    """Advance the ship by dt_s using velocity Verlet (kick-drift-kick). Reversible."""
    n = _substep_count(state, dt_s, attractors, max_step_s)
    h = dt_s / n
    r = np.asarray(state.r, dtype=np.float64).copy()
    v = np.asarray(state.v, dtype=np.float64).copy()
    t = state.epoch_s
    a = gravity_accel(r, t, attractors)
    for _ in range(n):
        v_half = v + 0.5 * a * h
        r = r + v_half * h
        t = t + h
        a = gravity_accel(r, t, attractors)
        v = v_half + 0.5 * a * h
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=t)
