"""A transfer plan must still be flyable when it lands (review: Part 4a, Important 1).

The Moon planner takes 1-2.5 s of wall time. At the sandbox's default x100 warp that is
100-250 s of sim time, about one departure-grid step, so a grid starting at "now" was
in the past when the plan landed and set_node_epoch clamped it to now: burning 10 s
late impacts the Moon, 250 s late misses it entirely, while the toast said "Pe 198 km".
"""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from orbitsim.core.constants import R_MOON
from orbitsim.core.flyby_targeting import FlybyPlan, MoonPass
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.render.controllers.targeting import Targeting
from orbitsim.render.node_edit import MIN_NODE_LEAD_S, PLAN_LEAD_S, departure_grid_s
from orbitsim.render.targets import MoonTarget


def test_departure_grid_starts_a_lead_ahead():
    grid = departure_grid_s(6000.0, 24)
    assert len(grid) == 24
    assert grid[0] == PLAN_LEAD_S and grid[-1] == pytest.approx(6000.0 + PLAN_LEAD_S)
    assert PLAN_LEAD_S > MIN_NODE_LEAD_S


class _Editor:
    def __init__(self):
        self.calls = []

    def set_node_epoch(self, epoch_s):
        self.calls.append(("epoch", epoch_s))

    def set_dv(self, axis, value):
        self.calls.append((axis, value))


class _InlineExecutor:
    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:          # noqa: BLE001 -- a future carries it
            fut.set_exception(exc)
        return fut


def _fake_app(now_s, warp=100.0):
    from orbitsim.__main__ import _default_world
    world = _default_world()
    v0 = world.vessels[0]
    v0.state = type(v0.state)(v0.state.r, v0.state.v, v0.state.mu, now_s)
    toasts = []
    app = SimpleNamespace(
        world=world, clock=SimpleNamespace(sim_time_s=now_s, warp=warp),
        _target=MoonTarget(), _targets=[], _planning_future=None, _grid_executor=None,
        _planning_executor=_InlineExecutor(), _auto_execute_node=False,
        maneuver_editor=_Editor(), _flash_message=toasts.append)
    return app, toasts


def edits_of(app):
    """What the planner asked the (fake) burn editor to do."""
    editor = app.maneuver_editor
    return editor.calls


def _plan(epoch_s):
    node = ManeuverNode(epoch_s, 3131.0, -25.0, -81.0)
    return FlybyPlan(node, MoonPass(R_MOON + 198e3, epoch_s + 1e6, np.zeros(2), 1000.0, True),
                     True)


def test_planning_drops_warp_and_grids_from_a_lead(monkeypatch):
    import orbitsim.core.optimize as opt
    seen = {}

    def fake_intercept(ship_state, target_now, mu, dep, tof, executor=None):
        seen["dep"] = np.asarray(dep)
        seen["ship_epoch"] = ship_state.epoch_s
        raise ValueError("stop after the grid")
    monkeypatch.setattr(opt, "intercept_node", fake_intercept)
    app, toasts = _fake_app(now_s=500.0, warp=100.0)
    Targeting(app).plan_intercept()
    assert app.clock.warp == 1.0                  # planning time ~ sim time
    assert seen["ship_epoch"] == 500.0
    assert seen["dep"][0] >= PLAN_LEAD_S          # relative to the ship epoch (= now)


def test_stale_plan_is_refused_with_an_honest_toast():
    app, toasts = _fake_app(now_s=4000.0)
    Targeting(app)._apply_planned_node(_plan(4000.0 + MIN_NODE_LEAD_S - 1.0))
    assert edits_of(app) == []        # the editor's burn is untouched
    assert toasts == ["Plan went stale — press Plan again"]


def test_fresh_plan_is_installed():
    app, toasts = _fake_app(now_s=4000.0)
    Targeting(app)._apply_planned_node(_plan(4000.0 + 60.0))
    assert ("epoch", 4060.0) in edits_of(app)
    assert toasts and toasts[0].startswith("Moon transfer planned · Pe 198 km")


def _plan_moon_with(monkeypatch, plan):
    import orbitsim.core.optimize as opt
    import orbitsim.core.flyby_targeting as ft
    monkeypatch.setattr(opt, "intercept_node", lambda *a, **k: plan.node)
    monkeypatch.setattr(ft, "moon_flyby_node", lambda *a, **k: plan)


def test_plan_for_a_target_no_longer_selected_is_dropped(monkeypatch):
    # The planner runs for seconds; if the player switches or clears the target
    # meanwhile, the burn it returns aims at something else. Don't install it.
    app, toasts = _fake_app(now_s=4000.0)
    _plan_moon_with(monkeypatch, _plan(4000.0 + 60.0))
    t = Targeting(app)
    t.plan_intercept()
    app._target = None                       # cleared while planning
    t.poll_planning()
    assert edits_of(app) == []
    assert toasts[-1] == "Target changed — plan dropped"


def test_plan_for_the_same_target_is_still_installed(monkeypatch):
    app, toasts = _fake_app(now_s=4000.0)
    _plan_moon_with(monkeypatch, _plan(4000.0 + 60.0))
    t = Targeting(app)
    t.plan_intercept()
    app._target = MoonTarget()               # re-selected: same body
    t.poll_planning()
    assert ("epoch", 4060.0) in edits_of(app)


def test_unconverged_plan_that_hits_the_moon_says_impact():
    app, toasts = _fake_app(now_s=4000.0)
    node = ManeuverNode(4060.0, 3131.0, -25.0, 81.0)
    plan = FlybyPlan(node, MoonPass(R_MOON - 30e3, 4060.0 + 1e6, np.zeros(2), 1000.0, True),
                     False)
    Targeting(app)._apply_planned_node(plan)
    assert "impact" in toasts[-1].lower() and "Pe -" not in toasts[-1], toasts[-1]
