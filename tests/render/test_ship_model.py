"""Tests for ship-view zoom->visibility helpers (pure math, no graphics)."""
import pytest

from orbitsim.render.ship_model import (
    view_blend, model_node_scale, SHIP_VIEW_NEAR_M, SHIP_VIEW_FAR_M,
)


def test_map_only_beyond_far():
    assert view_blend(SHIP_VIEW_FAR_M) == (1.0, 0.0)
    assert view_blend(1.0e6) == (1.0, 0.0)


def test_ship_only_within_near():
    assert view_blend(SHIP_VIEW_NEAR_M) == (0.0, 1.0)
    assert view_blend(10.0) == (0.0, 1.0)


def test_crossfade_sums_to_one_and_monotonic():
    mid = 0.5 * (SHIP_VIEW_NEAR_M + SHIP_VIEW_FAR_M)
    m, s = view_blend(mid)
    assert m == pytest.approx(1.0 - s)
    assert 0.0 < s < 1.0
    # closer => more ship
    _, s_near = view_blend(mid - 1.0)
    _, s_far = view_blend(mid + 1.0)
    assert s_near > s > s_far


def test_alphas_bounded():
    for d in (1.0, 150.0, 200.0, 1000.0, 5000.0, 1e7):
        m, s = view_blend(d)
        assert 0.0 <= m <= 1.0 and 0.0 <= s <= 1.0


def test_model_node_scale_is_inverse():
    assert model_node_scale(0.05) == pytest.approx(20.0)
    assert model_node_scale(2.0e4) == pytest.approx(5.0e-5)


def test_build_ship_model_nose_points_plus_z():
    # Headless: building geometry needs Panda but no window.
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "window-type none")
    from orbitsim.render.ship_model import build_ship_model

    np_ = build_ship_model()
    lo, hi = np_.get_tight_bounds()
    # Nose extends further along +Z than the tail does along -Z.
    assert hi.z > abs(lo.z)
    # Roughly ship-sized (metres), not a unit primitive.
    assert (hi.z - lo.z) > 5.0


def test_ship_icon_is_fixed_render_length_and_outlined():
    # Pure geometry: no ShowBase/window needed (one ShowBase per process is fragile).
    from orbitsim.render.ship_model import build_ship_icon, SHIP_ICON_RENDER_UNITS
    icon = build_ship_icon()
    lo, hi = icon.get_tight_bounds()
    assert (hi - lo).z == pytest.approx(SHIP_ICON_RENDER_UNITS * 1.18, rel=0.02)  # outline is the outer hull
    assert not icon.find("**/outline").is_empty()
    assert icon.has_light_off() or icon.find("**/+GeomNode").has_light_off()


def test_ship_icon_body_always_draws_over_its_outline():
    # The model's faces aren't consistently wound, so an inverted-hull outline drew
    # *over* the body (the icon looked like a solid cyan blob). The outline must be
    # two-sided, write no depth, and sort before the body, so only its rim shows.
    from panda3d.core import CullFaceAttrib, DepthWriteAttrib
    from orbitsim.render.ship_model import build_ship_icon
    icon = build_ship_icon()
    outline = icon.find("**/outline")
    body = icon.find("**/body")
    assert outline.get_attrib(DepthWriteAttrib).get_mode() == DepthWriteAttrib.M_off
    assert outline.get_attrib(CullFaceAttrib).get_effective_mode() == CullFaceAttrib.M_cull_none
    assert body.get_attrib(CullFaceAttrib).get_effective_mode() == CullFaceAttrib.M_cull_none
    assert outline.get_bin_draw_order() < body.get_bin_draw_order()


def test_ship_icon_outline_contrasts_with_own_orbit_line():
    # The icon always sits on its own (cyan) orbit line, so a cyan rim would vanish.
    from orbitsim.render.line_style import ORBIT_STYLE
    from orbitsim.render.ship_model import SHIP_ICON_OUTLINE_COLOR
    assert sum(SHIP_ICON_OUTLINE_COLOR[:3]) < 0.5 < sum(ORBIT_STYLE.color[:3])
