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


SWITCH = r"""
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX"); h.step(5)
hooks.select_target(app, "Moon")
h.send("i"); h.step(30)                       # Moon transfer planned, exact path drawn
before = hooks.preview_lines(app)
hooks.select_target(app, "Mars")              # switch target with the burn unchanged
shown, statuses = [], []
for _ in range(40):
    h.step(1)
    shown.append(hooks.preview_lines(app))
    statuses.append(hooks.aim_status(app))
hooks.select_target(app, "Venus"); h.step(1)   # A -> B -> C -> A, each a frame apart
shown.append(hooks.preview_lines(app))
hooks.select_target(app, "Moon"); h.step(1)
shown.append(hooks.preview_lines(app))
print("RESULT", repr({"before": before, "shown": shown, "statuses": statuses}))
app.destroy()
"""


def test_switching_target_keeps_the_exact_path_drawn():
    # The drawn path doesn't depend on the target; only the approach does. Switching
    # target must not drop the exact line (flicker) or fall back to the estimate.
    out = _run(SWITCH)
    assert out["before"] == (False, True), out["before"]
    assert all(s == (False, True) for s in out["shown"]), out["shown"]
    assert out["statuses"][0] is None or out["statuses"][0][0] == "checking", out["statuses"][0]
    assert out["statuses"][-1] is not None and out["statuses"][-1][0] != "checking", \
        out["statuses"][-1]


EXECUTED = r"""
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX"); h.step(5)
hooks.select_target(app, "Moon")
h.step(150)                                   # live line: LEO misses the Moon
before = hooks.aim_status(app)
h.send("i"); h.step(30)                       # plan the transfer
h.press("Jump & Execute")                     # burn it: no burn left, a new path
after = []
for _ in range(150):
    h.step(1)
    after.append(hooks.aim_status(app))
print("RESULT", repr({"before": before, "after": after}))
app.destroy()
"""


def test_status_never_shows_the_pre_burn_path_after_a_burn():
    # With no burn planned the box follows the live line; right after a burn (or a
    # quickload) that line still shows the old orbit for up to its 2 s refresh.
    out = _run(EXECUTED)
    assert out["before"][0] == "miss", out["before"]
    kinds = [s[0] for s in out["after"] if s is not None]
    assert "miss" not in kinds, kinds[:20]
    assert kinds and kinds[-1] == "encounter", kinds[-5:]


THRUST = r"""
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX"); h.step(5)
hooks.select_target(app, "Moon")
h.step(150)
vessel = app.world.vessels[0]
vessel.throttle = 1.0
h.step(30)                                    # a short manual burn
vessel.throttle = 0.0
frames = 0
while frames < 300 and (hooks.aim_status(app) or ("checking",))[0] == "checking":
    h.step(1)
    frames += 1
print("RESULT", repr(frames))
app.destroy()
"""


def test_status_recovers_right_after_a_manual_burn():
    # Review of PR #20: the stale line's completion left the submit time alone, so
    # the box sat on "checking" ~1.5-2 s after the throttle was cut.
    assert _run(THRUST) <= 10
