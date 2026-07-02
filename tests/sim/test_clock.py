"""Tests for SimClock (sim time + time-warp)."""
import pytest
from orbitsim.sim.clock import SimClock


def test_advance_returns_scaled_dt():
    clock = SimClock(sim_time_s=0.0, warp=10.0)
    sim_dt = clock.advance(2.0)
    assert sim_dt == 20.0


def test_advance_accumulates_sim_time():
    clock = SimClock(sim_time_s=100.0, warp=5.0)
    clock.advance(2.0)
    assert clock.sim_time_s == 110.0


def test_warp_up_steps_through_table():
    clock = SimClock(warp=1.0)
    clock.warp_up()
    assert clock.warp == 5.0
    clock.warp_up()
    assert clock.warp == 10.0


def test_warp_down_steps_back():
    clock = SimClock(warp=10.0)
    clock.warp_down()
    assert clock.warp == 5.0


def test_warp_up_clamps_at_max():
    clock = SimClock(warp=float(SimClock.WARP_STEPS[-1]))
    clock.warp_up()
    assert clock.warp == float(SimClock.WARP_STEPS[-1])


def test_warp_table_reaches_deep_space_rate():
    # The solar-system sandbox promises up to 100,000,000x in deep space.
    assert SimClock.WARP_STEPS[-1] == 100_000_000


def test_warp_down_clamps_at_min():
    clock = SimClock(warp=1.0)
    clock.warp_down()
    assert clock.warp == 1.0


def test_warp_value_must_be_in_table():
    # An off-table warp snaps to the nearest table value on the next step.
    clock = SimClock(warp=7.0)
    clock.warp_up()
    assert clock.warp in SimClock.WARP_STEPS
