"""Targetable bodies for maneuver planning (pure; no Panda3D).

A Target answers 'where is it at time t' in the same inertial, Earth-centered
frame as the vessel. Ships become Targets in a later cycle.
"""
import numpy as np

from orbitsim.core.moon import moon_state_at
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.nbody import earth_fixed_lagrange_points, OMEGA_EM


class MoonTarget:
    name = "Moon"
    supports_closest_approach = True   # the Moon is ~Keplerian -> CA prediction is meaningful

    def state_at(self, t_s: float) -> StateVector:
        return moon_state_at(t_s)


class LagrangePointTarget:
    """An Earth-Moon Lagrange point as a navigation target. It rotates rigidly with the Moon,
    so it is NOT on a Keplerian orbit; closest-approach prediction is not applicable (the render
    layer shows a live distance/relative-speed readout instead)."""

    supports_closest_approach = False

    def __init__(self, name: str, point_id: str) -> None:
        self.name = name            # display label shown in the HUD/picker
        self.point_id = point_id    # key into earth_fixed_lagrange_points(t): "L1".."L5"
        # (kept distinct so a future display name like "EML4" can map to point id "L4")

    def state_at(self, t_s: float) -> StateVector:
        r = earth_fixed_lagrange_points(t_s)[self.point_id]
        m = moon_state_at(t_s)
        n_hat = np.cross(m.r, m.v)
        n_hat = n_hat / np.linalg.norm(n_hat)
        v = np.cross(OMEGA_EM * n_hat, r)   # rigid-rotation velocity about the Moon's normal
        return StateVector(r=np.asarray(r, dtype=np.float64), v=v, mu=MU_EARTH, epoch_s=t_s)
