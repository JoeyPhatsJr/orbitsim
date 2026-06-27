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
