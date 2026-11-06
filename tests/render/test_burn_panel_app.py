"""The burn panel replaces the jog-slider panel: it shows only what exists, and every
action the slider panel had stays reachable (Next Pe / Next Ap / Delete burn / Execute).

Subprocess probe (one ShowBase per process), inline executors and a fixed 60 fps clock.
"""
import ast
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE = r"""
from direct.gui.DirectGuiBase import DirectGuiWidget
from direct.gui.DirectSlider import DirectSlider
from tests.render.app_harness import hooks
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
h = Harness(app)
h.press("LAUNCH SANDBOX")
h.step(5)
out = {"sliders": sum(isinstance(w, DirectSlider) for w in DirectGuiWidget.guiDict.values())}
out["idle"] = hooks.burn_panel(app)
hooks.set_burn(app, pro=37.4)
h.step(2)
out["burn_now"] = hooks.burn_panel(app)
h.press("Next Ap")
h.step(2)
out["scheduled"] = hooks.burn_panel(app)
h.press("Delete burn")
h.step(2)
out["deleted"] = hooks.burn_panel(app)
out["node"] = [None if v is None else float(v) for v in hooks.node_state(app)]
print("RESULT", repr(out))
app.destroy()
"""


def _run():
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return ast.literal_eval(line[len("RESULT "):])


def test_burn_panel_shows_what_exists_and_keeps_every_action():
    out = _run()
    assert out["sliders"] == 0                                  # the jog sliders are gone
    assert out["idle"] == (None, ["Next Pe", "Next Ap"])         # no burn: no readout/execute
    readout, buttons = out["burn_now"]
    assert readout.startswith("Burn 37 m/s · ") and readout.endswith("s burn"), readout
    assert " in " not in readout                                  # burn-now: no "in ..." part
    assert buttons == ["Next Pe", "Next Ap", "Delete burn", "Execute Burn"]
    readout, buttons = out["scheduled"]
    assert " · in " in readout, readout
    assert buttons == ["Next Pe", "Next Ap", "Delete burn", "Jump & Execute"]
    assert out["deleted"] == (None, ["Next Pe", "Next Ap"])
    assert out["node"] == [None, 0.0, 0.0, 0.0]                  # delete clears time AND dV
