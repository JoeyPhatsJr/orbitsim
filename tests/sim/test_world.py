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


def _coast(world: World, total_s: float, dt_s: float = 0.5) -> float:
    """Advance the world by total_s in flight-cadence increments, the way the render
    loop drives it. Under N-body the coast is numerically integrated (not on rails), so
    a single giant World.step is far less accurate than many small ones; stepping at
    flight cadence exercises the fidelity the sandbox actually runs at. Returns the
    elapsed sim time (n * dt_s)."""
    n = int(round(total_s / dt_s))
    for _ in range(n):
        world.step(dt_s)
    return n * dt_s


def test_world_step_period_closure():
    vessel = _circular_vessel()
    world = World(central=EARTH, vessels=[vessel])
    period = state_to_elements(vessel.state).period_s
    _coast(world, period)
    pos_error = np.linalg.norm(world.vessels[0].state.r - np.array([7.0e6, 0.0, 0.0]))
    # Under N-body the Moon perturbs the LEO orbit by ~130 m over one period — a real
    # physical effect that converges as dt->0, NOT integrator slop. Exact two-body
    # period closure (< 1 mm) is covered by tests/core/test_propagate.py.
    assert pos_error < 500.0  # within the physical Moon perturbation over one period


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


def test_world_step_coast_propagates_without_thrust():
    # Full-orbit N-body closure is covered by test_world_step_period_closure; this guards
    # the coast path itself: a throttle-0 vessel propagates along its orbit and never
    # registers as thrusting.
    v = _circular_vessel()
    v.throttle = 0.0
    world = World(central=EARTH, vessels=[v])
    start = world.vessels[0].state.r.copy()
    _coast(world, 600.0, dt_s=1.0)          # 10 min coast at flight cadence
    moved = np.linalg.norm(world.vessels[0].state.r - start)
    assert moved > 1.0e6                     # actually moved along the orbit
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


def test_unlimited_dv_step_thrusts_without_draining_fuel():
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, dry_mass_kg=1000.0, fuel_mass_kg=10.0,
               max_thrust_n=5.0e4, exhaust_velocity_mps=3000.0,
               throttle=1.0, unlimited_dv=True)
    # point the nose prograde so thrust does something
    v.sas_mode = "PROGRADE"
    w = World(central=EARTH, vessels=[v])
    speed0 = v.state.v_mag
    w.step(1.0)
    assert v.fuel_mass_kg == 10.0          # fuel not drained
    assert v.state.v_mag != speed0          # thrust applied (speed changed)


def test_unlimited_dv_locks_warp_even_with_zero_fuel():
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, fuel_mass_kg=0.0, max_thrust_n=5.0e4,
               throttle=1.0, unlimited_dv=True)
    w = World(central=EARTH, vessels=[v])
    assert w.any_thrusting() is True


def test_target_sas_slews_nose_toward_target():
    from orbitsim.core.attitude import nose_direction
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, sas_mode="TARGET", max_turn_rate_radps=1.0)
    v.sas_target_pos = np.array([7.0e6, 1.0e8, 0.0])   # far +Y of the ship
    w = World(central=EARTH, vessels=[v])
    want = v.sas_target_pos - v.state.r
    want = want / np.linalg.norm(want)
    a0 = float(np.dot(nose_direction(v.orientation), want))
    for _ in range(120):
        w.step(0.05)
    a1 = float(np.dot(nose_direction(v.orientation), want))
    assert a1 > a0           # nose turned toward the target
    assert a1 > 0.9          # and got close to pointing at it


def test_target_sas_with_no_target_does_not_crash():
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, sas_mode="TARGET")   # sas_target_pos stays None
    w = World(central=EARTH, vessels=[v])
    q0 = v.orientation.copy()
    w.step(1.0)
    np.testing.assert_array_equal(v.orientation, q0)    # attitude held, no error


def test_world_step_coast_uses_nbody_near_moon():
    """A vessel near the Moon drifts differently from Keplerian after N-body step."""
    from orbitsim.core.moon import moon_state_at
    from orbitsim.core.propagate import propagate_kepler
    rM = moon_state_at(0.0).r
    r_ship = rM + np.array([5.0e6, 0.0, 0.0])
    v_ship = np.array([0.0, 500.0, 0.0])
    st = StateVector(r=r_ship, v=v_ship, mu=MU_EARTH, epoch_s=0.0)
    vessel = Vessel(name="test", state=st)
    world = World(central=EARTH, vessels=[vessel])
    dt = 3600.0
    world.step(dt)
    # N-body result should diverge from Keplerian by > 1 km near the Moon.
    kep = propagate_kepler(st, dt)
    divergence = np.linalg.norm(world.vessels[0].state.r - kep.r)
    assert divergence > 1000.0, f"expected N-body divergence, got {divergence:.1f} m"


def test_world_step_coast_leo_close_to_kepler():
    """In LEO the Moon barely perturbs the orbit: stepped at flight cadence, N-body coast
    tracks two-body Kepler to a few metres over one quarter-orbit."""
    from orbitsim.core.propagate import propagate_kepler
    r = 7.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    vessel = Vessel(name="test", state=st)
    world = World(central=EARTH, vessels=[vessel])
    period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)
    elapsed = _coast(world, period / 4, dt_s=1.0)
    kep = propagate_kepler(st, elapsed)
    assert np.linalg.norm(world.vessels[0].state.r - kep.r) < 100.0


def test_world_step_thrust_nbody_fuel_drains():
    """Thrusting under N-body drains fuel (same contract as two-body)."""
    r = 7.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    v = Vessel(name="test", state=st, dry_mass_kg=1000.0, fuel_mass_kg=500.0,
               max_thrust_n=30000.0, exhaust_velocity_mps=3000.0,
               throttle=1.0, sas_mode="PROGRADE")
    world = World(central=EARTH, vessels=[v])
    fuel0 = v.fuel_mass_kg
    speed0 = v.state.v_mag
    for _ in range(60):
        world.step(0.1)
    assert v.fuel_mass_kg < fuel0
    assert v.state.v_mag > speed0


# ---------------------------------------------------------------------------
# Surface contact / landing
# ---------------------------------------------------------------------------
from orbitsim.core.constants import R_EARTH


def _dropped_vessel(altitude_m: float = 2.0e5) -> Vessel:
    """A vessel released at rest above Earth's surface (plunging trajectory)."""
    state = StateVector(r=np.array([R_EARTH + altitude_m, 0.0, 0.0]),
                        v=np.zeros(3), mu=MU_EARTH, epoch_s=0.0)
    return Vessel(name="lander", state=state)


def test_falling_vessel_lands_on_surface_instead_of_diving_to_center():
    vessel = _dropped_vessel()
    world = World(central=EARTH, vessels=[vessel])
    for _ in range(1000):                       # 500 s at flight cadence
        world.step(0.5)
        if vessel.landed_on is not None:
            break
    assert vessel.landed_on == "Earth"
    assert abs(np.linalg.norm(vessel.state.r) - R_EARTH) < 1.0
    assert np.linalg.norm(vessel.state.v) < 1e-9   # at rest on the surface


def test_landed_vessel_stays_put_while_coasting():
    vessel = _dropped_vessel()
    world = World(central=EARTH, vessels=[vessel])
    for _ in range(1000):
        world.step(0.5)
        if vessel.landed_on is not None:
            break
    site = vessel.state.r.copy()
    epoch = vessel.state.epoch_s
    world.step(3600.0)                          # one big warp step while landed
    assert vessel.landed_on == "Earth"
    assert np.linalg.norm(vessel.state.r - site) < 1.0
    assert vessel.state.epoch_s == epoch + 3600.0


def test_landed_vessel_lifts_off_under_thrust():
    vessel = _dropped_vessel()
    vessel.fuel_mass_kg = 1000.0
    vessel.max_thrust_n = 50000.0               # TWR ~ 2.5 on the pad
    world = World(central=EARTH, vessels=[vessel])
    for _ in range(1000):
        world.step(0.5)
        if vessel.landed_on is not None:
            break
    # Point the nose radially out (body +Z rotated to +x, the landing site's up).
    from orbitsim.core.attitude import quat_from_axis_angle
    vessel.orientation = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    vessel.throttle = 1.0
    for _ in range(20):
        world.step(0.5)
    assert vessel.landed_on is None
    assert np.linalg.norm(vessel.state.r) > R_EARTH + 10.0


def test_maneuver_sas_slews_toward_stored_direction():
    vessel = _circular_vessel()
    world = World(central=EARTH, vessels=[vessel])
    from orbitsim.core.attitude import nose_direction
    target = np.array([0.0, 0.0, 1.0])
    vessel.sas_mode = "MANEUVER"
    vessel.sas_maneuver_dir = target
    before = float(np.dot(nose_direction(vessel.orientation), target))
    world.step(0.5)
    after = float(np.dot(nose_direction(vessel.orientation), target))
    assert after >= before  # nose turns toward the maneuver direction
    for _ in range(20):
        world.step(0.5)
    assert float(np.dot(nose_direction(vessel.orientation), target)) > 0.99


def test_unlimited_dv_thrusts_with_empty_tank():
    vessel = _circular_vessel()
    vessel.fuel_mass_kg = 0.0
    vessel.unlimited_dv = True
    vessel.max_thrust_n = 30000.0
    vessel.throttle = 1.0
    world = World(central=EARTH, vessels=[vessel])
    world.step(1.0)
    # Nose is +Z (identity orientation): thrust accel = 30 kN / 2000 kg = 15 m/s^2.
    assert vessel.state.v[2] > 10.0             # thrust produced acceleration
    assert vessel.fuel_mass_kg == 0.0           # tank untouched under unlimited
