"""Consistent high-contrast styling for labels anchored in the 3D world."""


def build_world_label(parent, text, *, color, scale):
    """Create a billboard label with a subtle card and shadow.

    It remains depth-tested, so bodies can occlude it; only depth writes are
    disabled to avoid labels punching holes in later transparent geometry.
    """
    from panda3d.core import TextNode, TransparencyAttrib

    node = TextNode(f"label_{text}")
    node.set_text(text)
    node.set_align(TextNode.A_center)
    node.set_text_color(*color)
    node.set_shadow(0.055, 0.055)
    node.set_shadow_color(0.0, 0.0, 0.0, 0.95)
    node.set_card_as_margin(0.22, 0.22, 0.10, 0.10)
    node.set_card_color(0.015, 0.025, 0.055, 0.72)

    label = parent.attach_new_node(node)
    label.set_scale(scale)
    label.set_billboard_point_eye()
    label.set_light_off()
    label.set_depth_test(True)
    label.set_depth_write(False)
    label.set_transparency(TransparencyAttrib.M_alpha)
    return label


def build_labeled_marker(
    parent,
    text,
    *,
    color,
    marker_scale: float = 5.0,
    label_scale: float = 11.0,
):
    """Build a constant-screen-size fullbright marker and matching label."""
    from panda3d.core import TransparencyAttrib

    from orbitsim.render.geometry import make_uv_sphere

    marker = make_uv_sphere(1.0, 8, 12)
    marker.reparent_to(parent)
    marker.set_color(*color)
    marker.set_scale(marker_scale)
    marker.set_light_off()
    marker.set_depth_test(True)
    marker.set_depth_write(False)
    marker.set_transparency(TransparencyAttrib.M_alpha)
    label = build_world_label(parent, text, color=color, scale=label_scale)
    return marker, label
