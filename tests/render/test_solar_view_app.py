"""The solar viewer, driven for real: orbit lines, pixel-sized markers and names.

Each probe runs the app in a subprocess (one ShowBase per process), as
test_opening_camera_app.py does, and prints one ``RESULT <json>`` line.
"""
import json
import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PLANETS = ["Earth", "Jupiter", "Mars", "Mercury", "Neptune", "Saturn", "Uranus", "Venus"]

PRELUDE = """
import json, math
from tests.render.app_harness.driver import Harness, build_app, install_determinism
from tests.render.app_harness import hooks
install_determinism()
{before_build}
app = build_app("solar")
h = Harness(app)
h.press("OPEN SOLAR VIEWER")
"""


def _probe(body: str, before_build: str = ""):
    code = PRELUDE.format(before_build=before_build) + textwrap.dedent(body) + "\napp.destroy()\n"
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_solar_viewer_draws_every_planet_orbit():
    names, all_shown, all_named = _probe("""
        h.step(5)
        nodes = hooks.solar_orbit_nodes(app)
        print("RESULT " + json.dumps([sorted(nodes),
              all(not n.is_hidden() for n in nodes.values()),
              all(n.get_name() == f"solar_orbit_{k}" for k, n in nodes.items())]))
    """)
    assert names == PLANETS and all_shown and all_named


def test_solar_orbits_rebuild_per_bucket_not_on_zoom():
    keys0, keys1, scale_ok, keys2 = _probe("""
        def keys():
            return {k: n.get_key() for k, n in hooks.solar_orbit_nodes(app).items()}
        h.step(5)
        keys0 = keys()
        app.rig.set_distance(1.0e13)
        h.step(3)
        keys1 = keys()
        scale_ok = math.isclose(hooks.solar_orbit_frame_scale(app),
                                1.0e9 / app.transform.scale_m_per_unit, rel_tol=1e-6)  # NodePath scale is float32
        app.clock.sim_time_s += 30 * 86400.0
        h.step(1)
        keys2 = keys()
        print("RESULT " + json.dumps([keys0, keys1, scale_ok, keys2]))
    """)
    assert sorted(keys0) == PLANETS
    assert keys1 == keys0          # zooming never rebuilds
    assert scale_ok                # the frame carries the zoom instead
    assert all(keys2[k] != keys0[k] for k in keys0)   # a new 30-day bucket rebuilds


def test_solar_viewer_offline_draws_orbits():
    names, offline_text = _probe("""
        h.step(5)
        print("RESULT " + json.dumps([sorted(hooks.solar_orbit_nodes(app)),
              "circular approx (offline)" in app.hud.text.getText()]))
    """, before_build="import orbitsim.core.ephemeris as e; e._load_failed = True")
    assert names == PLANETS and offline_text
