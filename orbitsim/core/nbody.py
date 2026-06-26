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
