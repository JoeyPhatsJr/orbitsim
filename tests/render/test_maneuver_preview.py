"""Regression tests for asynchronous maneuver-preview invalidation."""
from orbitsim.core.maneuvers import ManeuverNode
import numpy as np

from orbitsim.core.constants import MU_EARTH
from orbitsim.core.state import StateVector
from orbitsim.render.app import (
    _coast_chunks,
    _localize_polyline,
    _maneuver_preview_key,
    _trajectory_horizon_s,
)


def _node(epoch_s, prograde=120.0):
    return ManeuverNode(epoch_s, prograde, 0.0, 0.0)


def test_burn_now_key_ignores_advancing_state_epoch():
    assert _maneuver_preview_key(_node(10.0), None) == _maneuver_preview_key(
        _node(11.0), None
    )


def test_key_changes_with_dv_or_scheduled_epoch():
    base = _maneuver_preview_key(_node(10.0), 600.0)
    assert base != _maneuver_preview_key(_node(10.0, prograde=121.0), 600.0)
    assert base != _maneuver_preview_key(_node(10.0), 630.0)


def test_escape_trajectory_keeps_interplanetary_horizon_before_soi_exit():
    radius = 7.0e6
    escape_speed = np.sqrt(2.0 * MU_EARTH / radius)
    state = StateVector(
        r=np.array([radius, 0.0, 0.0]),
        v=np.array([0.0, 1.01 * escape_speed, 0.0]),
        mu=MU_EARTH,
    )
    assert _trajectory_horizon_s(state, solar_system=True) == 400.0 * 86400.0


def test_bound_leo_uses_short_reusable_horizon():
    radius = 7.0e6
    state = StateVector(
        r=np.array([radius, 0.0, 0.0]),
        v=np.array([0.0, np.sqrt(MU_EARTH / radius), 0.0]),
        mu=MU_EARTH,
    )
    assert _trajectory_horizon_s(state, solar_system=True) < 86400.0


def test_teleport_coast_chunks_land_exactly_without_oversized_steps():
    duration = 250.25 * 86400.0
    chunks = _coast_chunks(duration)
    assert np.isclose(sum(chunks), duration)
    assert max(chunks) <= 86400.0
    assert all(step > 0.0 for step in chunks)


def test_teleport_coast_ignores_nonpositive_duration():
    assert _coast_chunks(0.0) == []
    assert _coast_chunks(-1.0) == []


def test_interplanetary_polyline_is_localized_before_float32_conversion():
    origin = np.array([1.5e11, -2.2e11, 7.0e10])
    offsets = np.array([[0.0, 0.0, 0.0], [12.0, -7.0, 3.0], [31.0, 5.0, -9.0]])

    local, actual_origin = _localize_polyline(origin + offsets)

    np.testing.assert_array_equal(actual_origin, origin)
    np.testing.assert_array_equal(local, offsets)
    np.testing.assert_array_equal(local.astype(np.float32), offsets.astype(np.float32))
