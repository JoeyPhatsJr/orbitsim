"""Tests for impulsive maneuvers in the local RTN frame."""
import numpy as np
from orbitsim.core.maneuvers import ManeuverNode, apply_maneuver
from orbitsim.core.state import StateVector
from orbitsim.core.elements import state_to_elements
from orbitsim.core.constants import MU_EARTH


def _periapsis_state() -> StateVector:
    """Elliptical orbit positioned at periapsis (r along +x, v along +y)."""
    rp = 7.0e6
    a = 8.0e6
    # vis-viva at periapsis
    v = np.sqrt(MU_EARTH * (2.0 / rp - 1.0 / a))
    return StateVector(r=np.array([rp, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)


def test_node_magnitude():
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=3.0, dv_normal_mps=4.0, dv_radial_mps=0.0)
    assert abs(node.magnitude_mps - 5.0) < 1e-12


def test_prograde_burn_raises_apoapsis_only():
    state = _periapsis_state()
    elem0 = state_to_elements(state)
    ra0 = elem0.a * (1 + elem0.e)
    rp0 = elem0.a * (1 - elem0.e)

    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=100.0, dv_normal_mps=0.0, dv_radial_mps=0.0)
    new_state = apply_maneuver(state, node)
    elem1 = state_to_elements(new_state)
    ra1 = elem1.a * (1 + elem1.e)
    rp1 = elem1.a * (1 - elem1.e)

    assert ra1 > ra0                      # apoapsis raised
    assert abs(rp1 - rp0) < 1.0           # periapsis unchanged (< 1 m)


def test_normal_burn_changes_inclination_and_adds_energy_in_quadrature():
    state = _periapsis_state()
    elem0 = state_to_elements(state)
    dv = 50.0
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=0.0, dv_normal_mps=dv, dv_radial_mps=0.0)
    new_state = apply_maneuver(state, node)
    elem1 = state_to_elements(new_state)
    # A normal impulse is perpendicular to v, so speed adds in quadrature *exactly*:
    # |v_new|^2 = |v_old|^2 + dv^2. This DOES raise energy (only a speed-preserving
    # rotation of v leaves energy fixed; a pure normal impulse is not that). Here the
    # resulting Δa/a ≈ 5e-5 — second-order small but real, not zero.
    assert abs(new_state.v_mag**2 - (state.v_mag**2 + dv**2)) < 1e-3
    assert elem1.i > elem0.i + 1e-6                          # inclination changed (main effect)
    assert 0.0 < abs(elem1.a - elem0.a) / elem0.a < 1e-3    # energy rises slightly, not zero


def test_total_dv_added_matches_magnitude():
    state = _periapsis_state()
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=30.0, dv_normal_mps=40.0, dv_radial_mps=0.0)
    new_state = apply_maneuver(state, node)
    dv_vec = new_state.v - state.v
    assert abs(np.linalg.norm(dv_vec) - node.magnitude_mps) < 1e-6


def test_node_epoch_propagates_before_burn():
    state = _periapsis_state()
    period = state_to_elements(state).period_s
    node = ManeuverNode(
        epoch_s=period / 2.0, dv_prograde_mps=10.0, dv_normal_mps=0.0, dv_radial_mps=0.0
    )
    new_state = apply_maneuver(state, node)
    # Burn happens at apoapsis (half a period later): position is on the far side.
    assert new_state.r[0] < 0
    assert abs(new_state.epoch_s - (state.epoch_s + period / 2.0)) < 1e-6
