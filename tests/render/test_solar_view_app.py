"""The solar viewer, driven for real: orbit lines, pixel-sized markers and names.

Each probe runs the app in a subprocess (one ShowBase per process), as
test_opening_camera_app.py does, and prints one ``RESULT <json>`` line.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

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


def test_markers_and_labels_keep_their_pixel_size_at_any_zoom():
    # Default 960x600 offscreen window, so line_px_scale() == 1.0.
    rows = _probe("""
        h.step(5)
        rows = []
        for d in (4.0e11, 2.0e12, 1.0e13):
            app.rig.set_distance(d)
            h.step(3)
            rows.append({name: (hooks.screen_radius_px(app, f"solar_marker_{name}"),
                                hooks.label_screen_height_px(app, name))
                         for name in ("Earth", "Jupiter", "Neptune", "Sun")})
        print("RESULT " + json.dumps(rows))
    """)
    card_h = (1.12 + 0.355) * 15.0
    for row in rows:
        assert row["Earth"][0] is not None and row["Sun"][0] is not None   # always on screen
        for name, (radius, height) in row.items():
            if radius is None:            # off-screen / behind the camera at this zoom: skip
                continue
            want = {"Sun": 10.0, "Jupiter": 7.0}.get(name, 6.0)
            assert radius == pytest.approx(want, abs=0.3), (name, row)
            assert height == pytest.approx(card_h, abs=0.5), (name, row)


def test_bodies_off_screen_or_behind_the_camera_hide_their_names():
    # Zoomed in to 0.33 AU, most planets are off-screen or behind the camera. Their names
    # must stay hidden through the whole frame (the panel pass must not re-show them).
    gone, shown = _probe("""
        h.step(5)
        app.rig.set_distance(5.0e10)
        h.step(3)
        gone = [k for k, v in hooks.solar_marker_px(app).items() if v is None]
        print("RESULT " + json.dumps([gone, hooks.shown_world_labels(app)]))
    """)
    assert len(gone) >= 3
    assert not set(gone) & set(shown)
