"""Pure math for editing a burn by keyboard (no Panda3D)."""
import math

import pytest

from orbitsim.render.node_edit import (
    NUDGE_DV, NUDGE_KEYS, NUDGE_TIME, jog_rate, nudge_rate, snap_dv, snap_epoch,
)


def test_nudge_rate_zero_before_hold_delay():
    assert nudge_rate(0.29, NUDGE_DV) == 0.0


def test_nudge_rate_starts_at_base_and_doubles():
    assert nudge_rate(0.3, NUDGE_DV) == 10.0 and nudge_rate(0.8, NUDGE_DV) == pytest.approx(20.0)


def test_nudge_rate_is_capped():
    assert nudge_rate(99.0, NUDGE_DV) == 1000.0 and nudge_rate(99.0, NUDGE_TIME) == 21600.0


def test_nudge_keys_cover_each_axis_both_ways():
    assert sorted(NUDGE_KEYS.values()) == sorted(
        (a, s) for a in ("pro", "nrm", "rad", "time") for s in (1, -1))


def test_jog_rate_is_the_slider_curve():
    # Same expression order as the original slider code (divide, then scale): the
    # node_edit_execute golden pins its exact delta-V, so this must be bit-identical.
    assert jog_rate(0.6, 400.0, 4.0, 0.02) == 400.0 * ((math.exp(2.4) - 1) / (math.exp(4.0) - 1))
    assert jog_rate(0.01, 400.0, 4.0, 0.02) == 0.0 and jog_rate(-1.0, 400.0, 4.0, 0.02) == -400.0


def test_snaps():
    assert snap_dv(12.345) == 12.3 and snap_epoch(2200.6) == 2201.0


# --- Placement windows and screen sizing (plan 3, Task 2) ---
from orbitsim.render.node_edit import (  # noqa: E402
    menu_pos, node_epoch_ok, node_epoch_window, screen_constant_scale,
)


def test_window_is_one_period_for_a_bound_orbit():
    assert node_epoch_window(100.0, 5000.0, [], 20000.0) == (110.0, 5100.0)


def test_window_stops_at_first_encounter():
    assert node_epoch_window(100.0, None, [9000.0, 3000.0], 40000.0) == (110.0, 3000.0)


def test_window_runs_to_line_end_when_unbound():
    assert node_epoch_window(100.0, None, [], 40000.0) == (110.0, 40000.0)


def test_node_epoch_ok_needs_lead():
    assert node_epoch_ok(110.0, 100.0) and not node_epoch_ok(109.9, 100.0)


def test_screen_constant_scale():
    assert screen_constant_scale(1000.0, 10.0, 30.0, 600) == pytest.approx(
        10 * 2000 * math.tan(math.radians(15)) / 600)


def test_menu_pos_centre_and_clamp():
    assert menu_pos((480.0, 300.0), 960, 600) == pytest.approx((0.03, -0.03))
    x, z = menu_pos((959.0, 2.0), 960, 600)
    assert x == pytest.approx(1.6 - 0.46) and z == pytest.approx(-1 + 0.17)


def test_menu_moves_above_a_rect_it_would_overlap():
    # The navball has its own display region drawn over all 2D GUI: a menu under it
    # is invisible, so the menu jumps above it.
    from orbitsim.render.node_edit import menu_clear_of
    nav = (-0.3, 0.3, -1.0, -0.4)                 # (left, right, bottom, top), aspect2d
    assert menu_clear_of((-0.1, -0.5), nav) == pytest.approx((-0.1, -0.4 + 0.17 + 0.02))
    assert menu_clear_of((0.5, -0.5), nav) == (0.5, -0.5)          # beside it: unchanged


# --- Drag handles (plan 3, Task 3) ---
from orbitsim.render.node_edit import HANDLES, axis_unit_px, handle_drag_value  # noqa: E402


def test_axis_unit_px():
    assert axis_unit_px((0, 0), (30, 40), 8) == pytest.approx((0.6, 0.8))


def test_axis_unit_px_none_when_foreshortened():
    assert axis_unit_px((0, 0), (3, 4), 8) is None


def test_handle_drag_value_along_against_and_across():
    assert handle_drag_value((0, 0), (75, 0), (1, 0), 150) == 0.5
    assert handle_drag_value((0, 0), (-300, 0), (1, 0), 150) == -1.0
    assert handle_drag_value((0, 0), (0, 80), (1, 0), 150) == 0.0


def test_handles_cover_each_axis_both_ways():
    assert sorted((a, s) for _k, a, s, _c in HANDLES) == sorted(
        (a, s) for a in ("pro", "nrm", "rad") for s in (1, -1))


# --- Review fixes (plan 3) ---
def test_window_ignores_an_encounter_already_in_progress():
    # Orbiting the Moon: the line starts inside its SOI, reported as an "encounter"
    # entered ~now. That must not close the window (review: no click placement there).
    assert node_epoch_window(100.0, None, [99.85, 3000.0], 40000.0) == (110.0, 3000.0)


def test_window_ignores_earth_period_inside_another_soi():
    # The geocentric period means nothing in lunar orbit.
    assert node_epoch_window(100.0, 5000.0, [], 40000.0, in_other_soi=True) == (110.0, 40000.0)


def test_radial_keys_push_the_way_their_labels_say():
    # F1 says "; / '  Burn radial in / out". apply_maneuver's radial axis (rtn_basis[2])
    # points away from the central body, so "out" is a positive radial component.
    import numpy as np
    from orbitsim.core.maneuvers import rtn_basis
    r, v = np.array([7e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0])
    radial_axis = rtn_basis(r, v)[2]
    push = {key: sign * radial_axis for key, (axis, sign) in NUDGE_KEYS.items() if axis == "rad"}
    assert np.dot(push["'"], r) > 0 and np.dot(push[";"], r) < 0      # ' = out, ; = in
