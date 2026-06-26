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
    return Vessel(name="test", state=state)


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


def test_vessel_has_node_list():
    from orbitsim.core.maneuvers import ManeuverNode

    vessel = _circular_vessel()
    assert vessel.nodes == []
    node = ManeuverNode(epoch_s=10.0, dv_prograde_mps=5.0, dv_normal_mps=0.0, dv_radial_mps=0.0)
    vessel.nodes.append(node)
    assert vessel.nodes[0].dv_prograde_mps == 5.0


def test_vessel_node_lists_are_independent():
    # default_factory must give each Vessel its own list, not a shared one.
    a = _circular_vessel()
    b = _circular_vessel()
    from orbitsim.core.maneuvers import ManeuverNode

    a.nodes.append(ManeuverNode(epoch_s=0.0, dv_prograde_mps=1.0, dv_normal_mps=0.0, dv_radial_mps=0.0))
    assert b.nodes == []


def test_vessel_propulsion_defaults_and_derived():
    from orbitsim.core.attitude import quat_identity
    v = _circular_vessel()
    # New fields exist with defaults.
    assert v.throttle == 0.0
    assert v.sas_mode == "OFF"
    assert np.allclose(v.orientation, quat_identity())
    # Configure a rocket and check derived quantities.
    v.dry_mass_kg = 1000.0
    v.fuel_mass_kg = 1000.0
    v.exhaust_velocity_mps = 3000.0
    assert v.mass_kg == 2000.0
    assert abs(v.delta_v_remaining - 3000.0 * np.log(2.0)) < 1e-6


def test_vessel_delta_v_zero_without_fuel():
    v = _circular_vessel()
    v.dry_mass_kg = 1000.0
    v.fuel_mass_kg = 0.0
    assert v.delta_v_remaining == 0.0


def test_world_step_burn_drains_fuel_and_adds_speed():
    v = _circular_vessel()
    v.dry_mass_kg = 1000.0
    v.fuel_mass_kg = 500.0
    v.max_thrust_n = 30000.0
    v.exhaust_velocity_mps = 3000.0
    v.throttle = 1.0
    v.sas_mode = "PROGRADE"
    # Point the nose prograde so the burn adds energy.
    world = World(central=EARTH, vessels=[v])
    speed0 = v.state.v_mag
    fuel0 = v.fuel_mass_kg
    # Slew first so the nose is prograde, then a short burn.
    for _ in range(60):
        world.step(0.1)
    assert v.fuel_mass_kg < fuel0          # fuel burned
    assert v.state.v_mag > speed0          # prograde burn sped us up
    assert world.any_thrusting() is True


def test_world_step_coast_is_on_rails():
    v = _circular_vessel()
    v.throttle = 0.0
    world = World(central=EARTH, vessels=[v])
    period = state_to_elements(v.state).period_s
    world.step(period)
    pos_err = np.linalg.norm(world.vessels[0].state.r - np.array([7.0e6, 0.0, 0.0]))
    assert pos_err < 1e-3                   # analytic period closure preserved
    assert world.any_thrusting() is False


def test_world_step_slews_attitude_toward_prograde():
    from orbitsim.core.attitude import nose_direction
    v = _circular_vessel()
    v.sas_mode = "PROGRADE"
    v.throttle = 0.0                        # slewing works while coasting
    world = World(central=EARTH, vessels=[v])
    for _ in range(100):
        world.step(0.1)
    prograde = v.state.v / v.state.v_mag
    # Nose should have turned to (near) prograde.
    assert np.dot(nose_direction(v.orientation), prograde) > 0.999


def test_unlimited_dv_is_infinite_regardless_of_fuel():
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.5e3, 0]),
                     mu=3.986e14, epoch_s=0.0)
    v = Vessel(name="x", state=st, fuel_mass_kg=0.0, unlimited_dv=True)
    assert v.delta_v_remaining == float("inf")
    v2 = Vessel(name="y", state=st, fuel_mass_kg=500.0, unlimited_dv=False)
    assert v2.delta_v_remaining < float("inf")
