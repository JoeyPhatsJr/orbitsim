"""Tests for the sim-layer World and Vessel."""
import numpy as np
from orbitsim.sim.world import Vessel, World
from orbitsim.core.state import StateVector
from orbitsim.core.elements import state_to_elements
from orbitsim.core.bodies import EARTH
from orbitsim.core.constants import MU_EARTH


def _circular_vessel(r_m: float = 7.0e6) -> Vessel:
    v = np.sqrt(MU_EARTH / r_m)
    state = StateVector(r=np.array([r_m, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)
    return Vessel(name="test", state=state, delta_v_budget_mps=1000.0)


def test_world_step_period_closure():
    vessel = _circular_vessel()
    world = World(central=EARTH, vessels=[vessel])
    period = state_to_elements(vessel.state).period_s
    world.step(period)
    pos_error = np.linalg.norm(world.vessels[0].state.r - np.array([7.0e6, 0.0, 0.0]))
    assert pos_error < 1e-3  # 1 mm closure (analytic)


def test_world_step_updates_state_object():
    vessel = _circular_vessel()
    world = World(central=EARTH, vessels=[vessel])
    before = world.vessels[0].state
    world.step(100.0)
    assert world.vessels[0].state is not before  # new immutable instance


def test_vessel_carries_budget():
    vessel = _circular_vessel()
    assert vessel.delta_v_budget_mps == 1000.0
