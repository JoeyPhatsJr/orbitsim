"""A burn whose time slipped past says so, and Execute burns from where you are.

Review of PR #20: frame_maneuver_node clears a passed node's time every frame, so the
panel silently turned it into a burn-now with no hint (and "Burn Now" never showed).
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
hooks.set_burn(app, pro=50.0)
app.maneuver_editor.place_node(app.clock.sim_time_s + 30.0)
h.step(2)
app.clock.sim_time_s += 60.0                   # the burn time slips past unexecuted
h.step(2)
out = {"panel": hooks.burn_panel(app), "toast": app.hud.toast.getText()}
speed0 = float((app.world.vessels[0].state.v ** 2).sum() ** 0.5)
h.press("Burn Now")
h.step(1)
out["after"] = hooks.burn_panel(app)
out["dv_applied"] = float((app.world.vessels[0].state.v ** 2).sum() ** 0.5) - speed0
hooks.set_burn(app, pro=10.0)                  # a new burn: no longer "passed"
app.maneuver_editor.place_node(app.clock.sim_time_s + 600.0)
h.step(1)
out["replanned"] = hooks.burn_panel(app)
print("RESULT", repr(out))
app.destroy()
"""


def test_passed_burn_says_so_and_burns_now():
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    out = ast.literal_eval(line[len("RESULT "):])
    readout, buttons = out["panel"]
    assert "node passed, burns now" in readout, out["panel"]
    assert "Burn Now" in buttons, out["panel"]
    assert "passed" in (out["toast"] or "").lower(), out["toast"]
    assert out["dv_applied"] > 40.0, out                       # it burned
    readout, buttons = out["replanned"]
    assert "passed" not in readout and "Burn Now" not in buttons, out["replanned"]
