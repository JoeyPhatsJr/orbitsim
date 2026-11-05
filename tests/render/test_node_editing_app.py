"""Editing a burn in the real app: keyboard nudges (and later handles / drags).

Subprocess probes (one ShowBase per process), inline executors, fixed 60 fps clock.
"""
import ast
import math
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(probe):
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return ast.literal_eval(line[len("RESULT "):])


NUDGES = r"""
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX"); h.step(10)
for k in ("]", "j", ";"):                 # taps: node later, +1 prograde, -1 radial
    h.send(k); h.send(k + "-up")
a = hooks.node_state(app)
h.send("l"); h.step(60); h.send("l-up")  # 1 s hold: tap + 0.7 s of ramp
b = hooks.node_state(app)
h.send("j"); hooks.set_key_held(app, "j", False); h.step(30)   # release swallowed
c = hooks.node_state(app)
print("RESULT", [a, b, c])
app.destroy()
"""


def test_nudge_taps_set_exact_steps():
    a, b, c = _run(NUDGES)
    assert a[1:] == (1.0, 0.0, -1.0) and a[0] is not None          # node created, +1 pro, -1 rad
    ramp = 10 * 0.5 / math.log(2) * (2 ** 1.4 - 1)
    assert b[2] == pytest.approx(ramp + 1.0, abs=0.5)               # tap + 0.7 s ramp


def test_nudge_stops_when_key_no_longer_held():
    # A modifier pressed mid-hold turns the release into "shift-j-up"; the key must
    # not ramp forever once the device says it isn't held.
    a, b, c = _run(NUDGES)
    assert c[1] == b[1] + 1.0                                       # only the tap, no ramp


# --- Hover ghost and the click-orbit menu (plan 3, Task 2) ---
PRELUDE = r"""
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX"); h.step(30)
def click(px):
    hooks.set_mouse_px(app, px); h.step(1); h.send("mouse1"); h.send("mouse1-up"); h.step(1)
"""


def test_hover_shows_ghost_with_lead_time():
    out = _run(PRELUDE + r"""
hooks.set_mouse_px(app, hooks.orbit_point_px(app, 40)); h.step(2)
labels = [np_.node().get_text() for np_ in app.render.find_all_matches("**/+TextNode")
          if not np_.is_hidden() and np_.node().get_text().startswith("in ")]
print("RESULT", labels)
app.destroy()
""")
    assert len(out) == 1 and out[0].endswith(" min"), out


def test_click_orbit_opens_menu_and_adds_burn():
    labels, after, state, epoch = _run(PRELUDE + r"""
click(hooks.orbit_point_px(app, 40))
labels = hooks.orbit_menu_labels(app)
epoch = hooks.orbit_point_epoch(app, 40)
h.press("Add burn here"); h.step(1)
print("RESULT", [labels, hooks.orbit_menu_labels(app), hooks.node_state(app), epoch])
app.destroy()
""")
    assert labels == ["Add burn here", "Warp to here"] and after is None
    assert abs(state[0] - epoch) < 60.0 and state[1:] == (0.0, 0.0, 0.0)


def test_menu_says_move_when_a_burn_exists():
    labels = _run(PRELUDE + r"""
click(hooks.orbit_point_px(app, 40)); h.press("Add burn here"); h.step(1)
click(hooks.orbit_point_px(app, 60))
print("RESULT", hooks.orbit_menu_labels(app))
app.destroy()
""")
    assert labels[0] == "Move burn here"


def test_escape_closes_menu_not_settings():
    menu, settings = _run(PRELUDE + r"""
click(hooks.orbit_point_px(app, 40)); h.send("escape"); h.step(1)
print("RESULT", [hooks.orbit_menu_labels(app), hooks.settings_visible(app)])
app.destroy()
""")
    assert menu is None and settings is False


def test_warp_to_here_coasts():
    advanced = _run(PRELUDE + r"""
t0 = app.clock.sim_time_s
click(hooks.orbit_point_px(app, 40)); h.press("Warp to here"); h.step(3)
print("RESULT", app.clock.sim_time_s - t0)
app.destroy()
""")
    assert advanced >= 2000.0


def test_warp_refused_while_thrusting():
    advanced = _run(PRELUDE + r"""
h.send("z"); h.step(2)
t0 = app.clock.sim_time_s
click(hooks.orbit_point_px(app, 40)); h.press("Warp to here"); h.step(2)
print("RESULT", app.clock.sim_time_s - t0)
app.destroy()
""")
    assert advanced < 10.0


def test_add_burn_refused_once_passed():
    state = _run(PRELUDE + r"""
click(hooks.orbit_point_px(app, hooks.first_sample_after(app, 60.0)))
h.step(80)                                  # ~130 s of sim time at x100: the point passes
h.press("Add burn here"); h.step(1)
print("RESULT", hooks.node_state(app))
app.destroy()
""")
    assert state[0] is None


def test_click_body_still_targets():
    target, menu, target2, menu2 = _run(PRELUDE + r"""
for _ in range(24):          # Plan 1 opening view: zoom out until the Moon is on screen
    h.send("wheel_down")
h.step(90)
click(hooks.body_px(app, "Moon"))
a = [hooks.current_target_name(app), hooks.orbit_menu_labels(app)]
click((5.0, 5.0))
print("RESULT", a + [hooks.current_target_name(app), hooks.orbit_menu_labels(app)])
app.destroy()
""")
    assert target == "Moon" and menu is None and target2 == "Moon" and menu2 is None


# --- 3D handles (plan 3, Task 3) ---
DRAG = PRELUDE + r"""
click(hooks.orbit_point_px(app, 40)); h.press("Add burn here"); h.step(5)
def drag(kind, hold_frames=60, release=True):
    start = hooks.handle_px(app, kind); node = hooks.node_px(app)
    hooks.set_mouse_px(app, start); h.step(1); h.send("mouse1")
    end = (start[0] + 1.5 * (start[0] - node[0]), start[1] + 1.5 * (start[1] - node[1]))
    hooks.set_mouse_px(app, end); h.step(hold_frames)
    if release:
        h.send("mouse1-up"); h.step(30)
"""


def test_prograde_handle_adds_only_prograde():
    epoch0, state = _run(DRAG + r"""
epoch0 = hooks.node_state(app)[0]
drag("pro+")
print("RESULT", [epoch0, hooks.node_state(app)])
app.destroy()
""")
    epoch, pro, nrm, rad = state
    assert pro > 0 and nrm == 0.0 and rad == 0.0 and epoch == epoch0 and pro == round(pro, 1)


def test_radial_in_handle_lowers_radial():
    state = _run(DRAG + r"""
drag("rad-")
print("RESULT", hooks.node_state(app))
app.destroy()
""")
    assert state[3] < 0 and state[1] == 0.0 and state[2] == 0.0


def test_drag_ends_when_button_no_longer_held():
    # Mouse released over a panel / outside the window: DirectGUI swallows mouse1-up,
    # so the drag must end when the device says the button is up.
    dragging, a, b = _run(DRAG + r"""
drag("pro+", hold_frames=5, release=False)
hooks.set_mouse1_held(app, False); h.step(2)
a = hooks.node_state(app)
h.step(30)
print("RESULT", [hooks.dragging(app), a, hooks.node_state(app)])
app.destroy()
""")
    assert dragging is None and a == b


def test_handles_hidden_without_node():
    node = _run(PRELUDE + r"""
print("RESULT", hooks.node_px(app))
app.destroy()
""")
    assert node is None


# --- Drag the node along the orbit (plan 3, Task 4) ---
SLIDE = PRELUDE + r"""
click(hooks.orbit_point_px(app, 40)); h.press("Add burn here")
h.send("j"); h.send("j-up"); h.step(5)                 # +1 m/s prograde
def slide_to(i):
    hooks.set_mouse_px(app, hooks.node_px(app)); h.step(1); h.send("mouse1")
    target = hooks.orbit_point_px(app, i)
    hooks.set_mouse_px(app, target); h.step(10)
    h.send("mouse1-up"); h.step(5)
"""


def test_dragging_node_slides_its_time():
    state, epoch80 = _run(SLIDE + r"""
epoch80 = hooks.orbit_point_epoch(app, 80)
slide_to(80)
print("RESULT", [hooks.node_state(app), epoch80])
app.destroy()
""")
    epoch, pro, nrm, rad = state
    assert abs(epoch - epoch80) <= 60.0 and epoch == float(int(epoch))
    assert (pro, nrm, rad) == (1.0, 0.0, 0.0)


def test_node_drag_stays_on_current_orbit():
    state, now, period = _run(SLIDE + r"""
from orbitsim.core.elements import state_to_elements
slide_to(250)                                          # second loop: past one period
period = float(state_to_elements(app.world.vessels[0].state).period_s)
print("RESULT", [hooks.node_state(app), app.clock.sim_time_s, period])
app.destroy()
""")
    assert now + 10.0 <= state[0] <= now + period
