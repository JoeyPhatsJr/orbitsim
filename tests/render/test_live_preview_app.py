"""In the real app the burn line follows edits instantly, then settles to the exact path.

Subprocess probe (one ShowBase per process), inline executors and a fixed 60 fps clock.
"""
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
h.press("LAUNCH SANDBOX")
h.step(5)
out = []
hooks.set_jog(app, "pro", 0.6)          # player drags the prograde control
h.step(1)
out.append(hooks.preview_lines(app))    # editing: estimate only
h.step(3)
out.append(hooks.preview_lines(app))    # still editing: estimate only
hooks.set_jog(app, "pro", 0.0)          # released
h.step(30)                              # 0.5 s at 60 fps: settled, exact computed + drawn
out.append(hooks.preview_lines(app))
h.send("wheel_up")                      # zoom: the (no longer resubmitted) line must rescale
h.step(40)
out.append(hooks.preview_at_current_scale(app))
h.press("Execute Burn")                  # burn done: delta-V back to 0, both lines go
h.step(2)
out.append(hooks.preview_lines(app))
print("RESULT", out)
app.destroy()
"""


def test_burn_line_is_live_while_editing_and_exact_after():
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    states = eval(line[len("RESULT "):])
    # (estimate_shown, exact_shown) at: editing, still editing, settled; zoomed line rescaled; executed (dV 0)
    assert states == [(True, False), (True, False), (False, True), True, (False, False)], states
