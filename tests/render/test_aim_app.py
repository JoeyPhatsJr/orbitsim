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
pro, nrm, rad = hooks.node_state(app)[1:]
hooks.set_burn(app, pro + 25.0, nrm, rad)     # the player reshapes the burn
h.step(1)
out["editing"] = hooks.aim_status(app)
out["editing_markers"] = hooks.encounter_marker_count(app)
print("RESULT", repr(out))
app.destroy()
"""


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
    assert out["planned"][0] in ("encounter", "miss", "impact"), out   # coloured = exact
    assert out["editing"][0] == "checking", out       # new burn: grey until its exact result
    assert out["editing_markers"] == 0, out           # never the un-burned path's markers
