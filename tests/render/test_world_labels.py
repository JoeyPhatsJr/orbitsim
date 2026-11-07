"""Small structural checks for readable world-space labels."""
from panda3d.core import NodePath, TextNode

from orbitsim.render.world_labels import (
    build_labeled_marker, build_ring_marker, build_tick_marker, build_world_label,
)


def test_world_label_is_billboarded_and_keeps_requested_text():
    parent = NodePath("root")
    label = build_world_label(
        parent, "L1", color=(0.5, 1.0, 0.9, 1.0), scale=12.0
    )
    assert isinstance(label.node(), TextNode)
    assert label.node().get_text() == "L1"
    assert label.get_scale().x == 12.0
    assert label.has_billboard()


def test_labeled_marker_builds_separate_marker_and_text_nodes():
    parent = NodePath("root")
    marker, label = build_labeled_marker(
        parent, "PE", color=(0.2, 1.0, 0.5, 1.0), marker_scale=6.0
    )
    assert marker.get_scale().x == 6.0
    assert label.node().get_text() == "PE"
    assert marker.get_parent() == parent
    assert label.get_parent() == parent


def test_tick_marker_is_constant_size_unlit():
    # A short bar in render units: the rig keeps render units a constant on-screen size.
    parent = NodePath("root")
    tick = build_tick_marker(parent, color=(1, 1, 1, 1), length=6.0)
    assert tick.get_parent() == parent
    assert tick.has_light_off() and tick.has_billboard()
    lo, hi = tick.get_tight_bounds()
    assert abs((hi - lo).length() - 6.0) < 1e-3


def test_ring_marker_is_a_billboarded_unit_ring():
    parent = NodePath("root")
    ring = build_ring_marker(parent, color=(0.3, 1.0, 1.0, 1.0))
    assert ring.has_light_off() and ring.has_billboard()
    lo, hi = ring.get_tight_bounds()
    assert abs(hi.x - 1.0) < 1e-3 and abs(lo.x + 1.0) < 1e-3 and abs(hi.y - lo.y) < 1e-6
