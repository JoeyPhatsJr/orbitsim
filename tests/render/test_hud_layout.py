"""The one screen-layout table: no two panels overlap, none leaves the screen."""
import pytest

from orbitsim.render.ui.hud_layout import (
    FOCUS_SCREEN_Y, MAX_ASPECT, MIN_ASPECT, PANELS, to_absolute, to_pixels,
)

ASPECTS = (1.5, 16 / 10, 16 / 9, 21 / 9)
SCALES = (0.84, 1.0, 1.08)        # ResponsiveLayout range, capped at 1.08 as app.py does


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("aspect", ASPECTS)
@pytest.mark.parametrize("scale", SCALES)
def test_panels_never_overlap(aspect, scale):
    rects = {n: to_absolute(r, aspect, scale) for n, r in PANELS.items()}
    names = sorted(rects)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not _overlap(rects[a], rects[b]), (a, b, aspect, scale)


@pytest.mark.parametrize("aspect", ASPECTS)
def test_panels_stay_on_screen(aspect):
    for n, r in PANELS.items():
        x0, y0, x1, y1 = to_absolute(r, aspect, 1.08)
        assert -aspect <= x0 < x1 <= aspect and -1.0 <= y0 < y1 <= 1.0, n


def test_navball_rect_matches_display_region():
    from orbitsim.render.navball import _BALL_FRAC_H
    r = PANELS["navball"]
    assert abs((r.y1 - r.y0) - 2.0 * _BALL_FRAC_H) < 0.03


@pytest.mark.parametrize("scale", SCALES)
def test_navball_rect_ignores_ui_scale(scale):
    # The ball's display region is a fixed fraction of the window height; its rect
    # (and the speed chip sitting on top of it) must not shrink with the UI scale.
    for name in ("navball", "speed_chip"):
        assert to_absolute(PANELS[name], 16 / 9, scale) == to_absolute(PANELS[name], 16 / 9, 1.0)


def test_to_pixels_bottom_left_origin():
    x0, y0, x1, y1 = to_pixels(PANELS["orbit_fuel"], 1600, 900)
    assert x0 < 100 and y0 < 50 and x1 > x0 and y1 > y0


def test_to_pixels_matches_aspect2d():
    # aspect2d x in [-aspect, aspect], y in [-1, 1]; pixels bottom-up.
    r = PANELS["time_warp"]
    ax0, ay0, ax1, ay1 = to_absolute(r, 1600 / 900, 1.0)
    px = to_pixels(r, 1600, 900)
    assert px[0] == pytest.approx((ax0 / (1600 / 900) + 1.0) * 800.0)
    assert px[3] == pytest.approx((ay1 + 1.0) * 450.0)


def test_focus_band_and_aspect_limits():
    assert 0.0 < FOCUS_SCREEN_Y < 0.5
    assert MIN_ASPECT == 1.5 and MAX_ASPECT == pytest.approx(21 / 9)


def test_anchor_names():
    assert {r.anchor for r in PANELS.values()} <= {
        "top_left", "top_right", "bottom_left", "bottom_center", "bottom_right"}
    assert set(PANELS) == {"time_warp", "target", "orbit_fuel", "navball", "speed_chip",
                           "sas", "burn"}
