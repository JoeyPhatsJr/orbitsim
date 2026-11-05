from orbitsim.render.picking import nearest_marker


def test_hit_within_tolerance():
    assert nearest_marker((100.0, 100.0), [(105.0, 102.0)], tol_px=10.0) == 0


def test_miss_beyond_tolerance():
    assert nearest_marker((100.0, 100.0), [(200.0, 200.0)], tol_px=10.0) is None


def test_nearest_of_several():
    assert nearest_marker((0.0, 0.0), [(50.0, 0.0), (8.0, 0.0), (9.0, 0.0)],
                          tol_px=10.0) == 1


def test_empty_list():
    assert nearest_marker((0.0, 0.0), [], tol_px=10.0) is None


# --- Screen-space orbit picking (plan 3, Task 2) ---
import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from orbitsim.render.picking import pick_polyline_epoch, project_points_px  # noqa: E402

PTS = np.array([[0, 0], [100, 0], [200, 0], [300, 0]], float)
EP = np.array([0., 10., 20., 30.])


def test_pick_interpolates_epoch_along_segment():
    p = pick_polyline_epoch((150.0, 5.0), PTS, EP, 0.0, 30.0, tol_px=10.0)
    assert p.index == 1 and p.fraction == pytest.approx(0.5)
    assert p.epoch_s == pytest.approx(15.0) and p.distance_px == pytest.approx(5.0)


def test_pick_misses_beyond_tolerance():
    assert pick_polyline_epoch((150.0, 20.0), PTS, EP, 0.0, 30.0, tol_px=10.0) is None


def test_pick_clamps_to_time_window():
    assert pick_polyline_epoch((50.0, 0.0), PTS, EP, 12.0, 30.0, tol_px=math.inf).epoch_s == pytest.approx(12.0)


def test_pick_prefers_earliest_of_overlapping_loops():
    loop = np.array([[0, 0], [100, 0], [100, 1], [0, 1]], float)   # segment 2 runs back over segment 0
    assert pick_polyline_epoch((50.0, 0.9), loop, EP, 0.0, 30.0, tol_px=10.0).epoch_s == pytest.approx(5.0)


def test_pick_skips_points_behind_camera():
    pts = PTS.copy()
    pts[1] = np.nan
    assert pick_polyline_epoch((50.0, 0.0), pts, EP, 0.0, 30.0, tol_px=10.0) is None


def test_pick_none_for_empty_window():
    assert pick_polyline_epoch((50.0, 0.0), PTS, EP, 25.0, 20.0, tol_px=math.inf) is None


def test_project_identity_and_behind_camera():
    px = project_points_px(np.array([[0, 0, 0], [1, 1, 0]], float), np.eye(4), 960, 600)
    assert px.tolist() == [[480.0, 300.0], [960.0, 600.0]]
    m = np.eye(4)
    m[:, 3] = [0, 1, 0, 0]                          # clip w = y
    assert np.isnan(project_points_px(np.array([[0, -1, 0]], float), m, 960, 600)).all()
