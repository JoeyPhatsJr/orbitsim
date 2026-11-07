"""In the real app no world label sits on a HUD panel, and the opening view keeps your
orbit clear of the navball (the camera focus sits above the bottom cluster).

Subprocess probe (one ShowBase per process), inline executors and a fixed 60 fps clock.
"""
import ast
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE = r"""
import sys
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism(win_size=sys.argv[1])
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX")
h.step(120)
out = {"boot_over": hooks.labels_over_panels(app),
       "boot_shown": len(hooks.shown_world_labels(app)),
       "panels": sorted(hooks.shown_panel_rects_px(app)),
       "under_navball": hooks.orbit_points_under_navball(app)}
hooks.select_target(app, "Moon")
h.step(10)
h.send("i")
h.step(10)
app.rig.set_distance(1.0e9)
h.step(60)
out["planned_over"] = hooks.labels_over_panels(app)
out["planned_shown"] = hooks.shown_world_labels(app)
# Review (4b): camera poses where a wide label anchored just off a panel ran under it.
posed = []
for d, az_deg, el in ((1.0e9, 45, 0.6), (2.0e9, 330, 1.2), (2.0e9, 345, 0.6), (2.0e9, 345, 1.2)):
    app.rig.set_distance(d)
    app.rig.target_azimuth = app.rig.azimuth = az_deg * 0.2618 / 15
    app.rig.target_elevation = app.rig.elevation = el
    h.step(2)
    posed += hooks.labels_over_panels(app)
out["posed_over"] = posed
print("RESULT", repr(out))
app.destroy()
"""


@pytest.mark.parametrize("size", ["1600 900", "1280 720"])
def test_no_label_on_a_panel_and_orbit_clear_of_the_navball(size):
    proc = subprocess.run([sys.executable, "-c", PROBE, size], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    out = ast.literal_eval(line[len("RESULT "):])
    # Not vacuous: the panels and labels the checks look at are there.
    assert {"time_warp", "orbit_fuel", "navball", "speed_chip", "sas", "burn"} <= set(out["panels"])
    assert out["boot_shown"] >= 5
    assert out["boot_over"] == [], out
    assert out["under_navball"] == 0, out
    assert out["planned_over"] == [], out
    assert out["posed_over"] == [], out
    assert any("Moon" in t or "MOON" in t for t in out["planned_shown"]), out
