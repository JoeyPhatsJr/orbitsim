"""Map markers say what they mean: Pe/Ap ticks with altitudes, one burn line, the
encounter in words, and Lagrange points faded unless targeted.

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
install_determinism(win_size="1600 900")
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX")
h.step(60)
out = {"boot": hooks.world_label_texts(app), "lagrange": hooks.lagrange_state(app)}
hooks.select_target(app, "Moon")
h.step(10)
h.press("Plan Moon transfer")
h.step(30)
out["planned"] = hooks.world_label_texts(app)
out["summary"] = app.maneuver_editor.burn_summary_text()
hooks.select_target(app, "L4")
app.rig.set_distance(5.0e9)             # L4 is ~384,000 km out: off-lens below ~4e9 m
h.step(5)
out["l4"] = hooks.lagrange_state(app)
# Re-run the declutter alone (the frame's last step may hide a label that lands on
# a panel at this zoom): the targeted point competes for a label again.
app.readouts.update_marker_readability(app.world.vessels[0].state.r)
out["l4_shown"] = hooks.shown_world_labels(app)
print("RESULT", repr(out))
app.destroy()
"""


def test_markers_say_what_they_mean():
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    out = ast.literal_eval(line[len("RESULT "):])
    boot = out["boot"]
    assert "Pe 500 km" in boot and "Ap 2,465 km" in boot, boot
    assert "PE" not in boot and "AP" not in boot
    # Lagrange points: faded markers, no labels, until one is the target.
    for name, (alpha, label_shown) in out["lagrange"].items():
        assert alpha <= 0.25 + 1e-9 and not label_shown, (name, alpha, label_shown)
    # The targeted point is back to full strength and labelled (its own "L4" label
    # stands aside for the higher-priority "TARGET: L4" at the same spot).
    alpha, _shown = out["l4"]["L4"]
    assert alpha > 0.25, out["l4"]
    assert "TARGET: L4" in out["l4_shown"] or "L4" in out["l4_shown"], out["l4_shown"]
    assert all(a <= 0.25 + 1e-9 for n, (a, _s) in out["l4"].items() if n != "L4")
    planned = out["planned"]
    assert out["summary"] in planned, (out["summary"], planned)        # the burn's one line
    assert out["summary"].startswith("Burn 3,1") and " · in " in out["summary"]
    assert any(t.startswith("Moon encounter · Pe ") for t in planned), planned
