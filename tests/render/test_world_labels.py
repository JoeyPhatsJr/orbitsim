"""Small structural checks for readable world-space labels."""
from panda3d.core import NodePath, TextNode

from orbitsim.render.world_labels import build_labeled_marker, build_world_label


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
