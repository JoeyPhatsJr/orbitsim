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
