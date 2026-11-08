"""When the solar viewer rebuilds its orbit lines (pure decision)."""
from orbitsim.render.controllers.solar_view import (
    SOLAR_ORBIT_MIN_REBUILD_S, solar_orbits_need_rebuild,
)


def _need(bucket=5, last_bucket=4, px=1.0, last_px=1.0, now=10.0, last=9.0):
    return solar_orbits_need_rebuild(bucket, last_bucket, px, last_px, now, last)


def test_first_frame_builds():
    assert _need(last_bucket=None, last=None)


def test_a_new_bucket_rebuilds_once_enough_real_time_passed():
    assert _need(now=10.0, last=10.0 - SOLAR_ORBIT_MIN_REBUILD_S)


def test_a_new_bucket_waits_at_max_warp():
    # At 1e8x a 30-day bucket passes every ~1.5 frames; the osculating orbits barely
    # move, so rebuild at most every SOLAR_ORBIT_MIN_REBUILD_S of real time.
    assert not _need(now=10.0, last=10.0 - 0.5 * SOLAR_ORBIT_MIN_REBUILD_S)


def test_same_bucket_never_rebuilds():
    assert not _need(bucket=4, last_bucket=4, now=100.0, last=0.0)


def test_a_line_scale_change_rebuilds_at_once():
    # Moving the window to a display with another pixel density changes line widths.
    assert _need(bucket=4, last_bucket=4, px=2.0, last_px=1.0, now=10.0, last=9.99)
