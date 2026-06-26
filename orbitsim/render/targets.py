"""Targetable bodies for maneuver planning (pure; no Panda3D).

A Target answers 'where is it at time t' in the same inertial, Earth-centered
frame as the vessel. Ships become Targets in a later cycle.
"""
from orbitsim.core.moon import moon_state_at
from orbitsim.core.state import StateVector


class MoonTarget:
    name = "Moon"

    def state_at(self, t_s: float) -> StateVector:
        return moon_state_at(t_s)
