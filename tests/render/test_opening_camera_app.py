"""The real app's opening camera must lean toward the real (DE440) Sun.

apply_opening_camera ran before the first refresh_ephemeris_cache, so it read the
offline circular-Sun fallback (~100 deg off the DE440 Sun that lights Earth) and
framed Earth's night side. Runs the app in a subprocess (one ShowBase per process).
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE = r"""
import math
import numpy as np
from tests.render.app_harness.driver import Harness, build_app, install_determinism
install_determinism()
app = build_app("sandbox")
Harness(app).press("LAUNCH SANDBOX")
from orbitsim.core.nbody import _csun, ephemeris_available, refresh_ephemeris_cache
t = app.clock.sim_time_s
refresh_ephemeris_cache(t)
if not ephemeris_available():
    print("RESULT SKIP")
else:
    s = np.array(_csun(t).r, dtype=float)
    s /= np.linalg.norm(s)
    az, el = app.rig.azimuth, app.rig.elevation
    c = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    print(f"RESULT {float(c @ s):.6f}")
app.destroy()
"""


def test_opening_camera_leans_toward_the_real_sun():
    proc = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    value = line.split()[1]
    if value == "SKIP":
        pytest.skip("DE440 ephemeris not available offline")
    assert float(value) > 0.0, f"camera faces away from the real Sun (cos={value})"
