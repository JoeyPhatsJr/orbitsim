"""Tests for pure world-marker readability helpers."""
import numpy as np
import pytest

from orbitsim.render.world_markers import (
    apsis_indices,
    apsis_points_on_path,
    declutter_indices,
    distance_fade,
)


def test_apsis_indices_choose_minimum_and_maximum_radius():
    points = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-2.0, 0.0, 0.0]])
    assert apsis_indices(points) == (1, 0)


def test_apsis_indices_reject_empty_or_malformed_input():
    with pytest.raises(ValueError):
        apsis_indices(np.empty((0, 3)))
    with pytest.raises(ValueError):
        apsis_indices(np.empty((3, 2)))


def test_apsis_points_interpolate_on_adjacent_line_segments():
    # Radius-squared minima/maxima are deliberately asymmetric around their
    # winning samples, producing non-zero sub-sample offsets.
    points = np.array([
        [3.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.3, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    pe, ap = apsis_points_on_path(points)
    assert 1.0 <= pe[0] <= 1.3
    assert 3.8 <= ap[0] <= 4.0
    assert pe[1] == pe[2] == ap[1] == ap[2] == 0.0


def test_apsis_points_keep_open_path_endpoint_exact():
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    pe, ap = apsis_points_on_path(points)
    assert np.array_equal(pe, points[0])
    assert np.array_equal(ap, points[-1])


def test_distance_fade_is_smooth_bounded_and_monotonic():
    values = [distance_fade(d, 10.0, 100.0) for d in (0.0, 10.0, 55.0, 100.0, 200.0)]
    assert values[0] == values[1] == 1.0
    assert values[-2] == values[-1] == pytest.approx(0.22)
    assert values == sorted(values, reverse=True)


def test_declutter_keeps_highest_priority_when_labels_overlap():
    points = [(100.0, 100.0), (105.0, 104.0), (300.0, 300.0)]
    visible = declutter_indices(points, [10, 50, 1], min_separation_px=20.0)
    assert visible == {1, 2}


def test_declutter_ignores_offscreen_points_and_validates_lengths():
    assert declutter_indices([None, (5.0, 5.0)], [100, 1], min_separation_px=10.0) == {1}
    with pytest.raises(ValueError):
        declutter_indices([(0.0, 0.0)], [], min_separation_px=10.0)


def test_box_blocked_by_panels_tests_the_whole_label():
    # Review (4b): labels are wide and centre-aligned, so an anchor clear of a panel
    # can still put most of the text under it. The whole box counts.
    from orbitsim.render.world_markers import box_blocked_by_panels
    rects = [(100.0, 0.0, 400.0, 300.0)]
    assert box_blocked_by_panels((380.0, 140.0, 620.0, 162.0), rects)   # anchor at x=500
    assert box_blocked_by_panels((402.0, 140.0, 620.0, 162.0), rects)   # within the margin
    assert not box_blocked_by_panels((420.0, 140.0, 620.0, 162.0), rects)
    assert not box_blocked_by_panels((150.0, 310.0, 300.0, 330.0), rects)
    assert box_blocked_by_panels((150.0, 301.0, 300.0, 330.0), rects)
    assert not box_blocked_by_panels(None, rects)                       # behind the camera
    assert not box_blocked_by_panels((150.0, 0.0, 300.0, 30.0), [])


def test_label_box_px_bounds_projected_corners():
    import numpy as np
    from orbitsim.render.world_markers import label_box_px
    corners = np.array([[10.0, 5.0], [50.0, 5.0], [10.0, 25.0], [50.0, 25.0]])
    assert label_box_px(corners) == (10.0, 5.0, 50.0, 25.0)
    corners[2] = np.nan                                                 # a corner behind the camera
    assert label_box_px(corners) is None
