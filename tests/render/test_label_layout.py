"""Pure screen-space label layout: pixel sizing, slots, priority, pinned, obstacles."""
import math
import random

import pytest

from orbitsim.render.label_layout import (
    SLOTS,
    LabelRequest,
    label_origin_offset_px,
    layout_labels,
    rects_overlap,
    render2d_rect_px,
    slot_rect,
    units_per_pixel,
)

VIEW = (400.0, 300.0)


def req(key, x, y, prio=50, w=40.0, h=20.0, r=6.0):
    return LabelRequest(key, None if x is None else (x, y), (w, h), r, prio)


def test_units_per_pixel_matches_lens_geometry():
    assert units_per_pixel(1000.0, 30.0, 900) == pytest.approx(
        2000.0 * math.tan(math.radians(15)) / 900)


def test_units_per_pixel_scales_with_depth_and_viewport():
    base = units_per_pixel(1000.0, 30.0, 900)
    assert units_per_pixel(2000.0, 30.0, 900) == pytest.approx(2 * base)
    assert units_per_pixel(1000.0, 30.0, 1800) == pytest.approx(base / 2)


def test_units_per_pixel_rejects_bad_input():
    for args in ((0.0, 30.0, 900), (-1.0, 30.0, 900), (1.0, 0.0, 900), (1.0, 180.0, 900),
                 (1.0, 30.0, 0)):
        with pytest.raises(ValueError):
            units_per_pixel(*args)


def test_slot_rects_clear_the_marker():
    assert SLOTS == ("above", "right", "left", "below")
    assert slot_rect((100, 100), (40, 20), 8, "above") == (80, 108, 120, 128)
    assert slot_rect((100, 100), (40, 20), 8, "right") == (108, 90, 148, 110)
    assert slot_rect((100, 100), (40, 20), 8, "left") == (52, 90, 92, 110)
    assert slot_rect((100, 100), (40, 20), 8, "below") == (80, 72, 120, 92)
    with pytest.raises(ValueError):
        slot_rect((100, 100), (40, 20), 8, "diagonal")


def test_isolated_labels_go_above():
    out = layout_labels([req("a", 100, 100), req("b", 300, 200)], VIEW)
    assert out["a"][0] == "above" and out["b"][0] == "above"


def test_crowded_label_moves_to_another_slot_instead_of_stacking():
    out = layout_labels([req("hi", 200, 150, 90), req("lo", 215, 150, 10)], VIEW)
    assert out["hi"][0] == "above" and out["lo"] is not None and out["lo"][0] != "above"
    assert not rects_overlap(out["hi"][1], out["lo"][1])


def test_label_never_covers_another_marker():
    out = layout_labels([req("a", 200, 150, 90), req("b", 200, 175, 10)], VIEW)
    b_dot = (194, 169, 206, 181)
    assert not rects_overlap(out["a"][1], b_dot)


def test_lowest_priority_label_hidden_when_no_slot_fits():
    reqs = [req(k, 200 + i, 150, p) for i, (k, p) in enumerate(
        [("Earth", 100), ("Venus", 90), ("Mars", 85), ("Mercury", 70), ("Sun", 10)])]
    out = layout_labels(reqs, VIEW)
    assert out["Earth"] is not None and out["Sun"] is None
    shown = [v[1] for v in out.values() if v is not None]
    assert all(not rects_overlap(a, b) for i, a in enumerate(shown) for b in shown[i + 1:])


def test_pinned_label_always_shown_and_others_avoid_it():
    reqs = [req(k, 200 + i, 150, p) for i, (k, p) in enumerate(
        [("Earth", 100), ("Venus", 90), ("Mars", 85), ("Mercury", 70), ("Sun", 10)])]
    out = layout_labels(reqs, VIEW, pinned={"Sun"})
    assert out["Sun"] is not None
    assert all(v is None or k == "Sun" or not rects_overlap(v[1], out["Sun"][1])
               for k, v in out.items())


def test_offscreen_anchor_is_hidden_and_no_obstacle():
    out = layout_labels([req("gone", None, None, 100), req("a", 200, 150)], VIEW)
    assert out["gone"] is None and out["a"][0] == "above"


def test_label_must_fit_inside_viewport():
    out = layout_labels([req("top", 200, 295)], VIEW)
    assert out["top"][0] != "above"
    x0, y0, x1, y1 = out["top"][1]
    assert 0 <= x0 and x1 <= VIEW[0] and 0 <= y0 and y1 <= VIEW[1]


def test_obstacles_block_labels():
    out = layout_labels([req("a", 200, 150)], VIEW, obstacles=[(150, 155, 250, 200)])
    assert out["a"][0] != "above"


def test_preferred_slot_is_kept_when_still_free():
    out = layout_labels([req("a", 200, 150)], VIEW, preferred_slots={"a": "left"})
    assert out["a"][0] == "left"


def test_random_layouts_never_overlap():
    rng = random.Random(5)
    for _ in range(200):
        reqs = [req(f"k{i}", rng.uniform(0, 400), rng.uniform(0, 300), rng.randint(0, 100),
                    rng.uniform(20, 70), 22.0, rng.choice([6.0, 7.0, 10.0])) for i in range(9)]
        out = layout_labels(reqs, VIEW)
        shown = [(k, v[1]) for k, v in out.items() if v is not None]
        dots = {r.key: (r.anchor_px[0] - r.marker_radius_px, r.anchor_px[1] - r.marker_radius_px,
                        r.anchor_px[0] + r.marker_radius_px, r.anchor_px[1] + r.marker_radius_px)
                for r in reqs}
        for i, (k, a) in enumerate(shown):
            assert 0 <= a[0] and a[2] <= 400 and 0 <= a[1] and a[3] <= 300
            assert all(not rects_overlap(a, b) for _, b in shown[i + 1:])
            assert all(not rects_overlap(a, d) for kk, d in dots.items() if kk != k)


def test_label_origin_offset_puts_card_on_rect():
    assert label_origin_offset_px((80, 108, 120, 128), (-1.0, 1.0, -0.35, 1.12), 10.0,
                                  (100, 100)) == pytest.approx((0.0, 14.15))


def test_render2d_rect_px():
    assert render2d_rect_px((-1, -1), (1, 1), (1600, 900)) == (0, 0, 1600, 900)
    assert render2d_rect_px((-0.5, 0.8), (0.5, 1.0), (1600, 900)) == pytest.approx(
        (400, 810, 1200, 900))


def test_solar_frame_skips_a_zero_size_framebuffer():
    # Review (plan 5): a minimised window can report a 0-height framebuffer;
    # units_per_pixel raises on it, which would kill the frame task.
    from types import SimpleNamespace
    from orbitsim.render.controllers.solar_view import SolarView
    win = SimpleNamespace(get_fb_x_size=lambda: 1600, get_fb_y_size=lambda: 0)
    SolarView(SimpleNamespace(win=win)).frame_markers_and_labels(None)   # no raise
