"""Regression tests for asynchronous maneuver-preview invalidation."""
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.render.app import _maneuver_preview_key


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
