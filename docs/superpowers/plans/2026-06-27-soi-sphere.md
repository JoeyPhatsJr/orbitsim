# Moon SOI Sphere Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw a faint translucent wireframe sphere at the Moon's sphere-of-influence radius in the sandbox, centered on the moving Moon at true world scale, brightening when the vessel is inside and distance-fading when the camera is far.

**Architecture:** Render-only. A `make_wireframe_sphere` builder in `render/geometry.py` makes a unit-radius lat/long `LineSegs` sphere; `app.py` places it each frame at `to_render(moon.r)`, scales it by `MOON_SOI_M / scale_m_per_unit` (true SOI size), recolors/brightens it on the inside test, and fades it by camera distance. Reuses `core.nbody.MOON_SOI_M` and `render.world_markers.distance_fade` — no physics changes.

**Tech Stack:** Python 3, Panda3D, pytest. Venv interpreter only: `.venv/Scripts/python`.

## Global Constraints

- **Layering:** all new code in `orbitsim/render/`. No `core/` changes — import `MOON_SOI_M` from `core.nbody`.
- **Sandbox-only:** the SOI sphere is built and updated only in the sandbox path
  (`if not self.solar_system and self.world.vessels:` / the existing Moon-visuals guards). The `--solar`
  viewer must be unaffected.
- **True scale via floating origin:** position with `self.transform.to_render(world_m)`; scale with
  `MOON_SOI_M / self.transform.scale_m_per_unit`. Do NOT bake the SOI radius into the mesh (build a
  unit sphere and node-scale it).
- **Render attributes** on the sphere node: light off, transparency `M_alpha`, depth test on, depth
  write off (matches the orbit-line treatment so it never punches through the Moon/trajectory).
- **Venv:** run everything with `.venv/Scripts/python -m pytest ...`. Bare `python` lacks deps.
- **Headless visual checks:** prepend `loadPrcFileData("", "window-type offscreen")` before
  constructing `OrbitApp(world, clock, solar_system=False)`, call `app._on_play()` to build the scene,
  step with `app.taskMgr.step()`, capture `app.win.save_screenshot(Filename.from_os_specific(path))`.
  Set `PYTHONPATH=.`. Build the sandbox world/clock the way `orbitsim/__main__.py::_default_world` does.
- **Commits:** stage explicit paths only (never `git add -A`/`.`). End messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`, then `git push`. Do NOT stage `data/`,
  `saves/`, screenshots, `.hypothesis/`, `.claude/settings.local.json`, `CLAUDE.md`, `orbitsim.egg-info/`.

---

### Task 1: `make_wireframe_sphere` geometry builder

**Files:**
- Modify: `orbitsim/render/geometry.py` (add the function below `make_uv_sphere`)
- Test: `tests/render/test_geometry.py` (add a structural test; create if absent)

**Interfaces:**
- Produces: `make_wireframe_sphere(n_lat: int = 9, n_lon: int = 12, color=(0.55, 0.75, 1.0, 1.0), segments: int = 48) -> NodePath`
  — a **unit-radius** wireframe sphere (`n_lat`−1 latitude rings + `n_lon` longitude meridians) built
  with `LineSegs`. The returned `NodePath` has light off, transparency `M_alpha`, depth test on, depth
  write off. Tight bounds ≈ the unit cube (radius 1).

- [ ] **Step 1: Write the failing structural test**

```python
# tests/render/test_geometry.py  (add; keep existing tests if the file exists)
def test_make_wireframe_sphere_is_unit_radius_nonempty():
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "window-type none")
    from orbitsim.render.geometry import make_wireframe_sphere

    np_ = make_wireframe_sphere()
    assert not np_.is_empty()
    lo, hi = np_.get_tight_bounds()
    # Unit sphere: extents reach ~+/-1 on each axis.
    for hicomp in (hi.x, hi.y, hi.z):
        assert 0.9 <= hicomp <= 1.0001
    for locomp in (lo.x, lo.y, lo.z):
        assert -1.0001 <= locomp <= -0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_geometry.py::test_make_wireframe_sphere_is_unit_radius_nonempty -q`
Expected: FAIL — `ImportError: cannot import name 'make_wireframe_sphere'`.

- [ ] **Step 3: Implement `make_wireframe_sphere`**

Add to `orbitsim/render/geometry.py` (the module already imports panda3d geometry types at the top; add
`LineSegs` and `TransparencyAttrib` to the import, plus `import math`):

```python
def make_wireframe_sphere(n_lat: int = 9, n_lon: int = 12,
                          color=(0.55, 0.75, 1.0, 1.0), segments: int = 48) -> NodePath:
    """Unit-radius wireframe sphere (latitude rings + longitude meridians).

    Light off, alpha-transparent, depth-tested with depth-write off so it reads as a
    boundary without punching through the body or trajectory behind it.
    """
    import math
    from panda3d.core import LineSegs, TransparencyAttrib

    ls = LineSegs("wireframe_sphere")
    ls.set_color(*color)
    ls.set_thickness(1.2)

    # Latitude rings: constant z = cos(theta), radius sin(theta).
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        z = math.cos(theta)
        r = math.sin(theta)
        for j in range(segments + 1):
            phi = 2.0 * math.pi * j / segments
            x, y = r * math.cos(phi), r * math.sin(phi)
            (ls.move_to if j == 0 else ls.draw_to)(x, y, z)

    # Longitude meridians: constant phi, theta sweeps pole to pole.
    for k in range(n_lon):
        phi = 2.0 * math.pi * k / n_lon
        for i in range(segments + 1):
            theta = math.pi * i / segments
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            (ls.move_to if i == 0 else ls.draw_to)(x, y, z)

    node = NodePath(ls.create())
    node.set_light_off()
    node.set_transparency(TransparencyAttrib.M_alpha)
    node.set_depth_test(True)
    node.set_depth_write(False)
    return node
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_geometry.py -q`
Expected: PASS (the new test + any existing geometry tests).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/geometry.py tests/render/test_geometry.py
git commit -m "SOI sphere: make_wireframe_sphere geometry builder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 2: Wire the SOI sphere into the sandbox

**Files:**
- Modify: `orbitsim/render/app.py` (top imports; sandbox build in `_start_sim`; per-frame update in `_update`)

**Interfaces:**
- Consumes: `make_wireframe_sphere` (Task 1); `core.nbody.MOON_SOI_M`;
  `render.world_markers.distance_fade`; `self.transform.to_render` / `.scale_m_per_unit`;
  `self.rig.distance_m`; `moon_now` (already computed in `_update`); `self.world.vessels[0].state.r`.
- Produces: `self._soi_np` (the SOI sphere NodePath), updated each frame; sandbox-only.

- [ ] **Step 1: Add imports + class constants**

At the top of `app.py`, add to the imports:

```python
from orbitsim.core.nbody import MOON_SOI_M
from orbitsim.render.world_markers import distance_fade
from orbitsim.render.geometry import make_uv_sphere, make_wireframe_sphere
```

(If `make_uv_sphere` is already imported from `orbitsim.render.geometry`, just extend that line to also
import `make_wireframe_sphere`, and add the two other imports.)

Add class constants on `OrbitApp` (near other render constants, e.g. by `SHIP_VIEW_DISTANCE_M`):

```python
    SOI_COLOR = (0.55, 0.75, 1.0, 1.0)        # cool blue-white wireframe (outside)
    SOI_INSIDE_COLOR = (0.55, 1.0, 0.70, 1.0)  # greenish "captured by the Moon"
    SOI_BASE_ALPHA = 0.45
    SOI_INSIDE_ALPHA = 0.75
    SOI_FADE_NEAR_M = 1.5e9   # camera distance: full alpha when closer than this
    SOI_FADE_FAR_M = 1.5e10   # ... fading to zero past this (tune by screenshot)
```

- [ ] **Step 2: Build the sphere in `_start_sim` (sandbox branch)**

In the sandbox branch of `_start_sim`, right after the Moon marker is created
(`self._moon_np = make_uv_sphere(1.0, 12, 16)` … `self._moon_np.set_scale(...)`), add:

```python
            # Moon sphere-of-influence: faint true-scale wireframe boundary.
            self._soi_np = make_wireframe_sphere(color=self.SOI_COLOR)
            self._soi_np.reparent_to(self.render)
            self._soi_np.hide()  # shown + placed each frame in _update
```

- [ ] **Step 3: Place + style the sphere each frame in `_update`**

In `_update`, right after the Moon marker is positioned
(`self._moon_np.set_pos(*self.transform.to_render(moon_now.r))`, ~app.py:1283), add:

```python
        # Moon SOI wireframe: true-scale, brighter when the vessel is inside, camera-distance fade.
        soi_scale = MOON_SOI_M / self.transform.scale_m_per_unit
        self._soi_np.set_pos(*self.transform.to_render(moon_now.r))
        self._soi_np.set_scale(soi_scale)
        inside = float(np.linalg.norm(self.world.vessels[0].state.r - moon_now.r)) < MOON_SOI_M
        color = self.SOI_INSIDE_COLOR if inside else self.SOI_COLOR
        base_alpha = self.SOI_INSIDE_ALPHA if inside else self.SOI_BASE_ALPHA
        self._soi_np.set_color(color[0], color[1], color[2], base_alpha)
        fade = distance_fade(self.rig.distance_m, self.SOI_FADE_NEAR_M, self.SOI_FADE_FAR_M, minimum=0.0)
        self._soi_np.set_alpha_scale(fade)
        self._soi_np.show()
```

Note: `set_color` overrides the mesh's baked vertex color with a flat RGBA (so inside/outside recolor
works); `set_alpha_scale` then multiplies in the camera fade. `np` is already imported in `app.py`.

- [ ] **Step 4: Headless visual check — sphere around the Moon + inside brighten**

Scratch script (not committed): start the sandbox offscreen (`_on_play`), zoom out so the Moon is in
frame (`app.rig.set_distance(1.2e9)`), step, screenshot. Then move the vessel inside the SOI and
screenshot.

```python
# scratch_soi.py
from panda3d.core import loadPrcFileData, Filename
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "win-size 700 700")
import numpy as np
from orbitsim.core.bodies import EARTH
from orbitsim.core.constants import MU_EARTH, R_EARTH
from orbitsim.core.state import StateVector
from orbitsim.core.moon import moon_state_at
from orbitsim.core.nbody import MOON_SOI_M
from orbitsim.sim.clock import SimClock
from orbitsim.sim.world import Vessel, World
from orbitsim.render.app import OrbitApp
OUT = r"<scratchpad>"

r0 = R_EARTH + 500e3
v_circ = np.sqrt(MU_EARTH / r0)
state = StateVector(r=np.array([r0, 0.0, 0.0]), v=np.array([0.0, v_circ * 1.1, 0.0]), mu=MU_EARTH)
vessel = Vessel(name="S", state=state, dry_mass_kg=1000.0, fuel_mass_kg=800.0,
                max_thrust_n=30000.0, exhaust_velocity_mps=3000.0, max_turn_rate_radps=0.8)
app = OrbitApp(World(central=EARTH, vessels=[vessel]), SimClock(sim_time_s=0.0, warp=1.0),
               solar_system=False)
app._on_play()
app.rig.set_distance(1.2e9)   # frame the Moon
for _ in range(5): app.taskMgr.step()
app.win.save_screenshot(Filename.from_os_specific(OUT + r"\soi_outside.png"))

# Put the vessel inside the Moon SOI (just inside the boundary).
m = moon_state_at(app.clock.sim_time_s)
app.world.vessels[0].state = StateVector(
    r=m.r + np.array([0.3 * MOON_SOI_M, 0.0, 0.0]), v=m.v, mu=MU_EARTH)
app.rig.set_distance(2.0e8)
for _ in range(5): app.taskMgr.step()
app.win.save_screenshot(Filename.from_os_specific(OUT + r"\soi_inside.png"))
print("saved")
```

Run: `PYTHONPATH=. .venv/Scripts/python <scratchpad>/scratch_soi.py`
Expected: `soi_outside.png` shows a faint blue-white wireframe sphere centered on the Moon (not
obscuring it); `soi_inside.png` shows the greenish brighter "inside" sphere. Read both PNGs. Tune
`SOI_FADE_NEAR_M`/`SOI_FADE_FAR_M`/alphas if the sphere is invisible or too strong at the framed
distance.

- [ ] **Step 5: Full suite + commit**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all pass.

```bash
git add orbitsim/render/app.py
git commit -m "SOI sphere: true-scale Moon SOI wireframe in sandbox (inside brighten + fade)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

## Self-Review notes

- **Spec coverage:** wireframe builder with correct render attributes (Task 1); true-scale placement
  via floating origin + `MOON_SOI_M / scale_m_per_unit` (Task 2 Step 3); inside-brighten using the
  `MOON_SOI_M` distance test (Task 2 Step 3); camera-distance fade via `distance_fade` (Task 2 Step 3);
  sandbox-only build/guard (Task 2 Steps 2–3); always-on, no toggle (nothing added); structural test +
  headless screenshots (both tasks). Out-of-scope items (Earth/planet SOIs, solid shell, ring, toggle,
  HUD text) intentionally absent.
- **Type consistency:** `make_wireframe_sphere(n_lat, n_lon, color, segments) -> NodePath`;
  `self._soi_np`; constants `SOI_COLOR`/`SOI_INSIDE_COLOR`/`SOI_BASE_ALPHA`/`SOI_INSIDE_ALPHA`/
  `SOI_FADE_NEAR_M`/`SOI_FADE_FAR_M`; `distance_fade(distance_m, near_m, far_m, *, minimum)` (matches
  `render/world_markers.py`) — used consistently.
- **Verification point (Task 2 Step 4):** the fade near/far and alphas are screenshot-tuned; the
  placement/scale/inside logic is independent of the final values. Confirm the exact existing import
  line for `make_uv_sphere` in `app.py` and extend it rather than duplicating the import.
```
