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
