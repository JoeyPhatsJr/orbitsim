"""Tests for impulsive maneuvers in the local RTN frame."""
import numpy as np
import pytest
from hypothesis import given, strategies as st
from orbitsim.core.maneuvers import (
    ManeuverNode, apply_maneuver, time_to_periapsis, time_to_apoapsis,
)
from orbitsim.core.state import StateVector
from orbitsim.core.elements import KeplerianElements, state_to_elements, elements_to_state
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


def test_predict_elements_after_matches_apply_then_convert():
    from orbitsim.core.maneuvers import predict_elements_after

    state = _periapsis_state()
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=120.0, dv_normal_mps=0.0, dv_radial_mps=0.0)
    predicted = predict_elements_after(state, node)
    expected = state_to_elements(apply_maneuver(state, node))
    assert abs(predicted.a - expected.a) < 1.0
    assert abs(predicted.e - expected.e) < 1e-9


def _state_at_nu(nu):
    elem = KeplerianElements(a=7.0e6, e=0.2, i=0.5, raan=0.3, argp=0.4, nu=nu, mu=MU_EARTH)
    return elements_to_state(elem)


def _period():
    return KeplerianElements(a=7.0e6, e=0.2, i=0.5, raan=0.3, argp=0.4, nu=0.0, mu=MU_EARTH).period_s


def test_time_to_periapsis_from_apoapsis_is_half_period():
    T = _period()
    assert abs(time_to_periapsis(_state_at_nu(np.pi)) - T / 2.0) < 1e-3


def test_time_to_apoapsis_at_apoapsis_is_zero():
    assert time_to_apoapsis(_state_at_nu(np.pi)) < 1e-3


def test_time_to_periapsis_at_periapsis_is_zero():
    assert time_to_periapsis(_state_at_nu(0.0)) < 1e-3


def test_time_to_apoapsis_from_periapsis_is_half_period():
    T = _period()
    assert abs(time_to_apoapsis(_state_at_nu(0.0)) - T / 2.0) < 1e-3


def test_timing_within_one_period_for_arbitrary_nu():
    T = _period()
    for nu in (0.1, 1.0, 2.5, 4.0, 6.0):
        for f in (time_to_periapsis, time_to_apoapsis):
            t = f(_state_at_nu(nu))
            assert 0.0 <= t < T


def test_timing_raises_on_hyperbolic():
    hyp = StateVector(r=np.array([7.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 12000.0, 0.0]), mu=MU_EARTH)  # > escape -> e>1
    with pytest.raises(ValueError):
        time_to_periapsis(hyp)
    with pytest.raises(ValueError):
        time_to_apoapsis(hyp)


@given(nu=st.floats(min_value=0.0, max_value=2.0 * np.pi, exclude_max=True),
       e=st.floats(min_value=0.0, max_value=0.9))
def test_timing_invariant_zero_to_period(nu, e):
    elem = KeplerianElements(a=8.0e6, e=e, i=0.4, raan=0.2, argp=0.3, nu=nu, mu=MU_EARTH)
    state = elements_to_state(elem)
    T = elem.period_s
    assert 0.0 <= time_to_periapsis(state) < T + 1e-6
    assert 0.0 <= time_to_apoapsis(state) < T + 1e-6
