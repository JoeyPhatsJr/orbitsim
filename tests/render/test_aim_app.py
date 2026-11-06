"""In the real app the target status box follows the exact path, never a stale one.

Subprocess probe (one ShowBase per process), inline executors and a fixed 60 fps clock.
"""
import ast
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE = r"""
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX"); h.step(5)
out = {}
hooks.select_target(app, "Moon")
h.step(150)                                   # the live line refreshes every 2 s
out["no_burn"] = hooks.aim_status(app)
h.send("i")                                   # plan the transfer
h.step(30)                                    # applied, settled (0.25 s), exact drawn
out["planned"] = hooks.aim_status(app)
out["planned_markers"] = hooks.encounter_marker_count(app)
h.press("Plan Moon transfer")                 # the panel button re-plans the same state
h.step(30)
out["replanned"] = hooks.aim_status(app)
pro, nrm, rad = hooks.node_state(app)[1:]
hooks.set_burn(app, pro + 25.0, nrm, rad)     # the player reshapes the burn
h.step(1)
out["editing"] = hooks.aim_status(app)
out["editing_markers"] = hooks.encounter_marker_count(app)
print("RESULT", repr(out))
app.destroy()
"""


def _pe_km(text):
    """The Pe altitude [km] in 'Moon encounter · Pe 1,234 km'."""
    return float(text.split("Pe ")[1].split(" km")[0].replace(",", ""))


def _run(probe):
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return ast.literal_eval(line[len("RESULT "):])


def test_status_follows_the_exact_path():
    out = _run(PROBE)
    assert out["no_burn"][0] == "miss", out           # LEO never reaches the Moon
    assert out["no_burn"][1].startswith("No encounter · closest "), out
    assert out["planned"][0] == "encounter", out       # the plan aims a ~200 km pass
    assert 150.0 <= _pe_km(out["planned"][1]) <= 250.0, out
    assert out["planned_markers"] == 1, out
    assert out["replanned"][0] == "encounter", out     # re-planning is idempotent
    assert 150.0 <= _pe_km(out["replanned"][1]) <= 250.0, out
    assert out["editing"][0] == "checking", out       # new burn: grey until its exact result
    assert out["editing_markers"] == 0, out           # never the un-burned path's markers


PLAN_AT_WARP = r"""
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
from orbitsim.render.node_edit import MIN_NODE_LEAD_S
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX"); h.step(5)
hooks.select_target(app, "Moon")
h.step(5)
before = float(app.clock.warp)
h.press("Plan Moon transfer")
during = float(app.clock.warp)
h.step(2)                                     # the plan lands (inline executors)
epoch = hooks.node_state(app)[0]
print("RESULT", repr([before, during, float(epoch) - float(app.clock.sim_time_s),
                      float(MIN_NODE_LEAD_S)]))
app.destroy()
"""


def test_planning_at_warp_lands_a_future_burn():
    # Review (Part 4a): at x100 the 1-2.5 s planner run was ~100-250 s of sim time;
    # the departure was past when the plan landed and the burn got clamped to now.
    before, during, lead_s, min_lead_s = _run(PLAN_AT_WARP)
    assert before == 100.0 and during == 1.0
    assert lead_s >= min_lead_s
