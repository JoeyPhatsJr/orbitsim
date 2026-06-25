"""Tests for sandbox save/load persistence (Phase 6.1)."""
import json

import numpy as np
import pytest

from orbitsim.core.bodies import EARTH
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.state import StateVector
from orbitsim.sim.clock import SimClock
from orbitsim.sim.persistence import save_scenario, load_scenario
from orbitsim.sim.world import Vessel, World


def _sample_world() -> tuple[World, SimClock]:
    r0 = 7.0e6
    v_circ = float(np.sqrt(MU_EARTH / r0))
    state = StateVector(
        r=np.array([r0, 1.0e5, -2.0e5]),
        v=np.array([10.0, v_circ * 1.02, v_circ * 0.13]),
        mu=MU_EARTH,
    )
    vessel = Vessel(
        name="Sandbox-1",
        state=state,
        dry_mass_kg=1234.0,
        fuel_mass_kg=678.0,
        max_thrust_n=30000.0,
        exhaust_velocity_mps=3100.0,
        max_turn_rate_radps=0.8,
        throttle=0.4,
        sas_mode="PROGRADE",
        orientation=np.array([0.5, 0.5, 0.5, 0.5]),  # non-identity, unit norm
    )
    vessel.nodes.append(ManeuverNode(epoch_s=120.0, dv_prograde_mps=50.0,
                                     dv_normal_mps=-3.0, dv_radial_mps=1.5))
    vessel.nodes.append(ManeuverNode(epoch_s=900.0, dv_prograde_mps=-12.0,
                                     dv_normal_mps=0.0, dv_radial_mps=4.0))
    world = World(central=EARTH, vessels=[vessel])
    clock = SimClock(sim_time_s=4567.0, warp=100.0)
    return world, clock


def test_save_load_round_trip(tmp_path):
    world, clock = _sample_world()
    path = tmp_path / "quicksave.json"

    save_scenario(world, clock, path)
    world2, clock2 = load_scenario(path)

    assert world2.central is EARTH
    assert clock2.sim_time_s == clock.sim_time_s
    assert clock2.warp == clock.warp

    v0, v1 = world.vessels[0], world2.vessels[0]
    assert v1.name == v0.name
    assert np.array_equal(v1.state.r, v0.state.r)
    assert np.array_equal(v1.state.v, v0.state.v)
    assert v1.state.mu == EARTH.mu
    assert v1.dry_mass_kg == v0.dry_mass_kg
    assert v1.fuel_mass_kg == v0.fuel_mass_kg
    assert v1.max_thrust_n == v0.max_thrust_n
    assert v1.exhaust_velocity_mps == v0.exhaust_velocity_mps
    assert v1.max_turn_rate_radps == v0.max_turn_rate_radps
    assert v1.throttle == v0.throttle
    assert v1.sas_mode == v0.sas_mode
    assert np.array_equal(v1.orientation, v0.orientation)
    assert len(v1.nodes) == len(v0.nodes)
    for n0, n1 in zip(v0.nodes, v1.nodes):
        assert n1.epoch_s == n0.epoch_s
        assert n1.dv_prograde_mps == n0.dv_prograde_mps
        assert n1.dv_normal_mps == n0.dv_normal_mps
        assert n1.dv_radial_mps == n0.dv_radial_mps
