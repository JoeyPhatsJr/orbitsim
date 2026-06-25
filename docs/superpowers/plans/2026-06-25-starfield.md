# Starfield / Skybox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the black void with a real star background — a downloaded Milky Way star map on an inertial, camera-centered sky sphere (procedural point-star fallback when offline), always rendered behind everything.

**Architecture:** A `"stars"` texture key reuses the existing download/cache; `render/skybox.py` builds either a textured inside-out sky sphere or a procedural `GeomPoints` field; `app.py` parents it to the scene and re-centers it on the camera each frame. Render-only; no `core/`/`sim/` changes. Design spec: `docs/superpowers/specs/2026-06-25-starfield-design.md`.

**Tech Stack:** Python 3.10, Panda3D, numpy, pytest.

## Global Constraints

- All new code in `render/`. No `core/`/`sim/` changes. The app must never crash for a missing texture — fall back to procedural stars.
- **Verified star map URL (use exactly this):** `https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/galaxy_starfield.png`
- Load textures with `Filename.from_os_specific(path)` (Panda cannot resolve raw Windows backslash paths — this bit the Earth textures).
- Sky must render behind everything: `set_bin("background", 0)`, `set_depth_write(False)`, `set_depth_test(False)`, `set_light_off()`.
- `black` is NOT installed; clean code at line length ≤ 100.
- Always use `.venv/Scripts/python`. Full suite: `.venv/Scripts/python -m pytest tests/ -q`.
- Commit after each task with the exact message given. Use ONLY `git add <specific files>` — NEVER `git add -A`. Never stage: `data/`, `debug_curtis.py`, `kickbacks.vsix`, `.hypothesis/`, `CLAUDE.md`, `porkchop.png`.
- Render tasks end with a HUMAN VISUAL CHECKPOINT; verify headlessly first (`window-type offscreen` + `taskMgr.step()` + screenshot).

## Gate

Realistic-Earth graphics complete (131 tests green). Independent of it.

## Existing API available

```python
from orbitsim.render.textures import texture_path, TEXTURE_URLS   # download/cache, offline-safe
from orbitsim.render.geometry import make_uv_sphere               # supports with_uv=True
# app.py: _start_sim builds the scene for both modes; _update (sandbox) and
# _update_solar_system (solar) run each frame and call self.rig.apply().
# self.camera is the world camera NodePath; self.render is the scene root.
```

---

## Task 1: Register the star texture

**Files:**
- Modify: `orbitsim/render/textures.py`
- Test: `tests/render/test_textures.py` (append)

**Interfaces:**
- Produces: `TEXTURE_URLS["stars"]` → the verified galaxy starfield URL.

- [ ] **Step 1: Write the failing test**

Append to `tests/render/test_textures.py`:
```python
def test_registry_has_stars_key():
    assert "stars" in textures.TEXTURE_URLS
    assert textures.TEXTURE_URLS["stars"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_textures.py -k stars -q`
Expected: FAIL (KeyError / assertion — "stars" not in registry).

- [ ] **Step 3: Add the entry**

In `orbitsim/render/textures.py`, add to the `TEXTURE_URLS` dict:
```python
    "stars": "https://raw.githubusercontent.com/jeromeetienne/threex.planets/master/images/galaxy_starfield.png",
```
(The PNG extension + magic checks already handle it; no other change.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_textures.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/textures.py tests/render/test_textures.py
git commit -m "Starfield Task 1: register the star texture URL"
```

---

## Task 2: Procedural star-direction helper

**Files:**
- Create: `orbitsim/render/skybox.py`
- Test: `tests/render/test_skybox.py`

**Interfaces:**
- Produces: `random_star_dirs(n: int, seed: int = 0) -> list[tuple[float, float, float]]`
  — `n` unit vectors, deterministic per seed.

- [ ] **Step 1: Write the failing tests**

Create `tests/render/test_skybox.py`:
```python
"""Tests for the procedural star-direction helper (offline, no graphics)."""
import math
from orbitsim.render.skybox import random_star_dirs


def test_returns_n_unit_vectors():
    dirs = random_star_dirs(100, seed=1)
    assert len(dirs) == 100
    for x, y, z in dirs:
        assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-9


def test_deterministic_for_seed():
    assert random_star_dirs(50, seed=7) == random_star_dirs(50, seed=7)


def test_different_seeds_differ():
    assert random_star_dirs(50, seed=1) != random_star_dirs(50, seed=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/render/test_skybox.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.render.skybox).

- [ ] **Step 3: Implement random_star_dirs**

Create `orbitsim/render/skybox.py`:
```python
"""Star background: a textured inside-out sky sphere, or a procedural point field
when the star texture is unavailable."""
import numpy as np


def random_star_dirs(n: int, seed: int = 0):
    """Return n unit direction vectors uniformly on the sphere (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [tuple(float(c) for c in row) for row in v]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/render/test_skybox.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/skybox.py tests/render/test_skybox.py
git commit -m "Starfield Task 2: deterministic procedural star directions"
```

---

## Task 3: build_starfield (textured sphere + procedural fallback)

**Files:**
- Modify: `orbitsim/render/skybox.py`
- Test: `tests/render/test_skybox.py` (append)

**Interfaces:**
- Consumes: `texture_path`, `make_uv_sphere`, `random_star_dirs`, Panda3D.
- Produces: `build_starfield(base) -> NodePath` (textured when available, else procedural;
  always a non-empty node with background/depth/light settings applied).

- [ ] **Step 1: Write the failing test**

Append to `tests/render/test_skybox.py`:
```python
def test_build_starfield_returns_a_node():
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", "audio-library-name null")
    from direct.showbase.ShowBase import ShowBase
    from orbitsim.render.skybox import build_starfield

    base = ShowBase()
    sky = build_starfield(base)
    assert sky is not None and not sky.is_empty()
    base.destroy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_skybox.py -k build -q`
Expected: FAIL (ImportError: cannot import name 'build_starfield').

- [ ] **Step 3: Implement build_starfield**

Append to `orbitsim/render/skybox.py`:
```python
from panda3d.core import (
    Filename, CullFaceAttrib, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    GeomPoints, Geom, GeomNode, NodePath,
)

from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.textures import texture_path

_SKY_RADIUS = 5000.0     # render units; depth-test is off, so this only clears the near plane
_STAR_COUNT = 3000


def _background(node: NodePath) -> NodePath:
    node.set_bin("background", 0)
    node.set_depth_write(False)
    node.set_depth_test(False)
    node.set_light_off()
    return node


def _procedural_points() -> NodePath:
    rng_dirs = random_star_dirs(_STAR_COUNT, seed=42)
    import numpy as np

    bright = np.random.default_rng(42).uniform(0.4, 1.0, size=_STAR_COUNT)
    fmt = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData("stars", fmt, Geom.UHStatic)
    vdata.set_num_rows(_STAR_COUNT)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    for (x, y, z), b in zip(rng_dirs, bright):
        vw.add_data3(x * _SKY_RADIUS, y * _SKY_RADIUS, z * _SKY_RADIUS)
        cw.add_data4(b, b, b, 1.0)
    pts = GeomPoints(Geom.UHStatic)
    pts.add_consecutive_vertices(0, _STAR_COUNT)
    geom = Geom(vdata)
    geom.add_primitive(pts)
    gnode = GeomNode("stars")
    gnode.add_geom(geom)
    np_ = NodePath(gnode)
    np_.set_render_mode_thickness(2)
    return np_


def build_starfield(base) -> NodePath:
    """A textured inside-out sky sphere, or a procedural point field if offline."""
    path = texture_path("stars")
    if path is not None:
        sky = make_uv_sphere(1.0, 32, 64, with_uv=True)
        sky.set_scale(_SKY_RADIUS)
        sky.set_texture(base.loader.load_texture(Filename.from_os_specific(path)))
        sky.set_attrib(CullFaceAttrib.make_reverse())   # see it from the inside
        sky.set_color(1, 1, 1, 1)
        return _background(sky)
    return _background(_procedural_points())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_skybox.py -q`
Expected: PASS. (If online, downloads the star map once into `data/textures/`.)

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/skybox.py tests/render/test_skybox.py
git commit -m "Starfield Task 3: build_starfield (textured sphere + procedural fallback)"
```

---

## Task 4: Wire the starfield into the app — HUMAN VISUAL CHECKPOINT

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `build_starfield`.
- Produces: a star background in both sandbox and solar modes, centered on the camera each
  frame (inertial orientation), always behind the scene.

- [ ] **Step 1: Build the starfield in `_start_sim`**

In `orbitsim/render/app.py`, add the import near the top:
```python
from orbitsim.render.skybox import build_starfield
```
In `_start_sim`, after the transform/rig/hud are created and before the task is added
(runs for BOTH modes), add:
```python
        self.starfield = build_starfield(self)
        self.starfield.reparent_to(self.render)
```

- [ ] **Step 2: Recenter on the camera each frame**

Add a tiny helper to `OrbitApp`:
```python
    def _update_starfield(self):
        if getattr(self, "starfield", None) is not None:
            self.starfield.set_pos(self.camera.get_pos(self.render))
```
Call `self._update_starfield()` immediately before `self.rig.apply()` in BOTH the sandbox
`_update` and `_update_solar_system`.

- [ ] **Step 3: Smoke-check imports**

Run: `.venv/Scripts/python -c "import orbitsim.render.app; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Headless screenshot**

Create `tmp_sky_check.py`:
```python
from panda3d.core import loadPrcFileData, Filename
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 900 700")
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp
app = OrbitApp(_default_world(), SimClock(0.0, 1.0))
app._on_play()
for _ in range(6):
    app.taskMgr.step()
print("has starfield:", getattr(app, "starfield", None) is not None)
app.win.save_screenshot(Filename.from_os_specific("tmp_sky.png"))
app.destroy()
print("OK")
```
Run: `PYTHONPATH=. .venv/Scripts/python tmp_sky_check.py`, open `tmp_sky.png`. Confirm stars
fill the background while Earth + orbit + vessel still draw on top. Delete temp files.

- [ ] **Step 5: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: HUMAN VISUAL CHECKPOINT**

Run `.venv/Scripts/python -m orbitsim` (and `--solar`). Reviewer confirms: a star field
fills the background; right-dragging the camera pans across a fixed star field (stars
don't move with the ship); Earth, orbit, and vessel always render in front of the stars.

- [ ] **Step 7: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "Starfield Task 4: camera-centered inertial star background"
```

---

## Exit Criteria

- `"stars"` texture key downloads/caches like the others, with offline fallback.
- `random_star_dirs` is deterministic and returns unit vectors (unit-tested).
- `build_starfield` returns a usable node textured-or-procedural (headless test).
- Star background renders behind everything in both modes and stays inertial as the
  camera orbits (visual checkpoint).
- `pytest tests/ -q` fully green; no `core/`/`sim/` changes.

## Self-Review Notes

- Spec coverage: registry (Task 1), procedural helper (Task 2), builder + fallback +
  background layering (Task 3), app integration + camera-centering (Task 4). All mapped.
- Offline-safety: `texture_path("stars")` returns `None` → `build_starfield` uses the
  procedural points path; no crash. `_update_starfield` guards a missing attribute.
- Type consistency: `random_star_dirs(n, seed)`, `build_starfield(base) -> NodePath`,
  `_update_starfield()` used identically across tasks. `Filename.from_os_specific` used
  for the texture load, per the documented Panda path gotcha.
- Unit-testable pieces (registry key, star dirs) are offline pytest tests; the textured
  sky and layering are checkpoint-verified per project convention.
```
