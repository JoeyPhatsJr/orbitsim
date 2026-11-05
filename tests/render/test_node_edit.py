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
