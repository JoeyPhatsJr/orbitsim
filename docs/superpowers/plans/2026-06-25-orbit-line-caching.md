# Orbit-Line Caching + Orbit Frame (Phase 6.3 B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** Task 1 is a pure helper (TDD). Task 2 is the render refactor, executed **inline by the controller** with headless math/identity verification per project convention.

**Goal:** Build orbit geometry in world meters under a single per-frame "orbit frame" anchor, rebuild the vessel orbit only when its elements change, and fix the stale A3 Moon orbit ring.

**Architecture:** A node with `pos = to_render(0)`, `scale = 1/scale_m_per_unit` holding world-meter vertices renders each vertex at `to_render(p)`. All Earth-centered orbit lines (vessel orbit, Moon ring, preview) hang off this frame; the vessel orbit polyline is rebuilt only on element change.

**Tech Stack:** Python 3, numpy, Panda3D. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- The render identity the approach relies on: `to_render(0) + p / scale_m_per_unit == to_render(p)` (float32). (verbatim from spec)
- Orbit-line geometry vertices are **world meters** (raw `sample_orbit_points` output), parented under the orbit frame — NOT pre-transformed render points. (verbatim from spec)
- Vessel orbit rebuilt only when `(a, e, i, raan, argp)` changes beyond tolerance (`1e-9` rel on `a`, `1e-9` abs on `e`/angles). (verbatim from spec)
- Markers, Earth body, camera rig, physics: unchanged. (verbatim from spec)
- float32 line precision (~32 m at lunar distance) is an accepted tradeoff. (user-approved)
- Run tests with `.venv/Scripts/python -m pytest`. Commits: explicit paths; end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. (verbatim)

---

## File Structure

- `orbitsim/render/orbit_lines.py` — add pure `orbit_shape_changed(a, b, tol=1e-9) -> bool`.
- `tests/render/test_orbit_lines.py` — add its unit tests (file exists).
- `orbitsim/render/app.py` — orbit frame, world-meter orbit geometry, vessel-orbit cache, Moon-ring fix, preview move.

Reference shapes (existing):
- `RenderTransform.to_render(p) -> (x,y,z)`; `transform.scale_m_per_unit`, `transform.set_origin`.
- `KeplerianElements(a,e,i,raan,argp,nu,mu,epoch_s)`; `state_to_elements(state)`.
- `sample_orbit_points(elements, n) -> (n,3) float64 world meters`; `build_orbit_node(points, color) -> NodePath`.
- App: `_rebuild_orbit(idx, vessel)` (~714), `self.orbit_nps` list, the per-frame `transform.set_origin` + `central_np` placement (~744–750), the Moon ring `self._moon_orbit_np` (~_start_sim), the preview block (~maneuver block).

---

## Task 1: Pure `orbit_shape_changed` helper

**Files:**
- Modify: `orbitsim/render/orbit_lines.py`
- Test: `tests/render/test_orbit_lines.py`

**Interfaces:**
- Produces: `orbit_shape_changed(a: KeplerianElements, b: KeplerianElements, tol: float = 1e-9) -> bool` — True if any of `(a, e, i, raan, argp)` differs beyond tolerance (`a` compared relative, the rest absolute), or if either side is None.

- [ ] **Step 1: Write the failing test**

Add to `tests/render/test_orbit_lines.py`:

```python
from orbitsim.core.elements import KeplerianElements
from orbitsim.core.constants import MU_EARTH
from orbitsim.render.orbit_lines import orbit_shape_changed


def _elem(**over):
    base = dict(a=7.0e6, e=0.1, i=0.5, raan=0.3, argp=0.4, nu=0.0, mu=MU_EARTH)
    base.update(over)
    return KeplerianElements(**base)


def test_shape_unchanged_for_identical():
    assert orbit_shape_changed(_elem(), _elem()) is False


def test_shape_unchanged_ignores_true_anomaly():
    # nu (position along the orbit) is not part of the shape.
    assert orbit_shape_changed(_elem(nu=0.0), _elem(nu=1.7)) is False


def test_shape_changed_on_semimajor_axis():
    assert orbit_shape_changed(_elem(a=7.0e6), _elem(a=7.0e6 + 1.0)) is True


def test_shape_changed_on_angles():
    assert orbit_shape_changed(_elem(argp=0.4), _elem(argp=0.4 + 1e-6)) is True


def test_none_counts_as_changed():
    assert orbit_shape_changed(None, _elem()) is True
    assert orbit_shape_changed(_elem(), None) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_orbit_lines.py -q -k shape`
Expected: FAIL — `ImportError: cannot import name 'orbit_shape_changed'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/render/orbit_lines.py` (after the imports):

```python
def orbit_shape_changed(a, b, tol: float = 1e-9) -> bool:
    """True if the orbit *shape* (a, e, i, raan, argp) differs beyond tolerance.

    True anomaly (position along the orbit) is ignored. A None on either side
    counts as changed (forces an initial build).
    """
    if a is None or b is None:
        return True
    if abs(a.a - b.a) > tol * max(abs(a.a), 1.0):
        return True
    return (abs(a.e - b.e) > tol or abs(a.i - b.i) > tol
            or abs(a.raan - b.raan) > tol or abs(a.argp - b.argp) > tol)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_orbit_lines.py -q`
Expected: PASS (existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/orbit_lines.py tests/render/test_orbit_lines.py
git commit -m "$(cat <<'EOF'
Orbit lines: pure orbit_shape_changed helper

True when (a, e, i, raan, argp) shifts beyond tolerance (nu ignored);
drives the rebuild-only-on-orbit-change cache.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Orbit frame + world-meter geometry + vessel-orbit cache + Moon-ring fix

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `orbit_shape_changed` (Task 1), `build_orbit_node`, `sample_orbit_points`, `state_to_elements`.
- Produces (app state): `self._orbit_frame: NodePath`, `self._orbit_elem_cache: list`; modified `_rebuild_orbit`; per-frame orbit-frame update; preview + Moon ring under the frame.

- [ ] **Step 1: Create the orbit frame + caches; build Moon ring in world meters under it**

In `_start_sim`, near where `self.orbit_nps`/`self.vessel_nps` are set up (sandbox path), add:

```python
        self._orbit_frame = self.render.attach_new_node("orbit_frame")
        self._orbit_elem_cache = [None for _ in world.vessels]
```

Change the Moon ring build (in `_start_sim`) from render-space to world-meter under the frame:

```python
            moon_pts = [tuple(p) for p in sample_orbit_points(MOON_ORBIT, n=256)]
            self._moon_orbit_np = build_orbit_node(moon_pts, color=(0.5, 0.5, 0.55, 1.0))
            self._moon_orbit_np.reparent_to(self._orbit_frame)
```

(Note: `self._orbit_frame` must be created before the Moon ring build — ensure ordering.)

- [ ] **Step 2: Per-frame orbit-frame placement**

In the sandbox update loop, right after `self.central_np.set_pos(cx, cy, cz)` (origin already set this frame), add:

```python
        self._orbit_frame.set_pos(cx, cy, cz)  # = to_render(0)
        self._orbit_frame.set_scale(1.0 / self.transform.scale_m_per_unit)
```

(`cx, cy, cz` is `to_render(zeros)`, already computed at ~747.)

- [ ] **Step 3: Cache `_rebuild_orbit` — rebuild geometry only on shape change, world-meter under frame**

Replace `_rebuild_orbit`:

```python
    def _rebuild_orbit(self, idx, vessel) -> None:
        elem = state_to_elements(vessel.state)
        if not orbit_shape_changed(self._orbit_elem_cache[idx], elem):
            return  # coasting: shape unchanged, keep the cached geometry
        self._orbit_elem_cache[idx] = elem
        pts = [tuple(p) for p in sample_orbit_points(elem, n=256)]
        if self.orbit_nps[idx] is not None:
            self.orbit_nps[idx].remove_node()
        node = build_orbit_node(pts)
        node.reparent_to(self._orbit_frame)
        self.orbit_nps[idx] = node
```

Add the import at the top of `app.py`: `from orbitsim.render.orbit_lines import sample_orbit_points, build_orbit_node, orbit_shape_changed` (extend the existing import line).

- [ ] **Step 4: Move the maneuver preview to world-meter under the frame**

In the preview block (maneuver section), change the preview build:

```python
        if node.magnitude_mps > 0.0:
            pred = predict_elements_after(v0.state, node)
            ppts = [tuple(p) for p in sample_orbit_points(pred, n=256)]
            if self._preview_np is not None:
                self._preview_np.remove_node()
            self._preview_np = build_orbit_node(ppts, color=(1.0, 0.2, 1.0, 1.0))
            self._preview_np.reparent_to(self._orbit_frame)
        elif self._preview_np is not None:
            self._preview_np.remove_node()
            self._preview_np = None
```

- [ ] **Step 5: Headless math + caching + Moon-ring verification**

```bash
PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
import numpy as np
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.core.moon import MOON_ORBIT
from orbitsim.render.orbit_lines import sample_orbit_points

app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app._start_sim()
app.taskMgr.step()

# (a) Math identity: to_render(0) + p/scale == to_render(p) (the frame relies on this).
T = app.transform
for p in sample_orbit_points(MOON_ORBIT, n=8):
    lhs = np.array(T.to_render(np.zeros(3))) + np.asarray(p) / T.scale_m_per_unit
    rhs = np.array(T.to_render(p))
    assert np.linalg.norm(lhs - rhs) < 1e-3, (lhs, rhs)

# (b) Cache: coasting vessel -> orbit NodePath identity stable across frames.
o0 = app.orbit_nps[0]
for _ in range(5):
    app.taskMgr.step()
assert app.orbit_nps[0] is o0, "coasting orbit should not be rebuilt"

# (c) Burn changes the orbit -> rebuild happens.
v = app.world.vessels[0]
app._dv["pro"] = 100.0
app._execute_burn()
app.taskMgr.step()
assert app.orbit_nps[0] is not o0, "post-burn orbit should rebuild"

# (d) Moon ring stays correctly placed: a world-meter vertex under the frame maps to to_render(p)
#     for the CURRENT (moved) origin. The frame pos/scale encode that.
mp = sample_orbit_points(MOON_ORBIT, n=4)[1]
fr = app._orbit_frame
rendered = np.array(fr.get_pos()) + fr.get_scale()[0] * np.asarray(mp)
assert np.linalg.norm(rendered - np.array(app.transform.to_render(mp))) < 1e-2, (rendered,)
print("OK: math identity, coast cache hit, burn rebuild, Moon ring placement")
PY
```

Expected: `OK: math identity, coast cache hit, burn rebuild, Moon ring placement`.

- [ ] **Step 6: Full suite + commit**

Run: `.venv/Scripts/python -m pytest tests/ -q` → all pass.

```bash
git add orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
Render: orbit frame + cache; rebuild orbit only on change; fix Moon ring

Orbit lines (vessel, Moon ring, preview) now hold world-meter vertices
under a single per-frame orbit-frame node (pos=to_render(0),
scale=1/scale). The vessel orbit polyline is rebuilt only when its
elements change (coast = no rebuild). This also fixes the A3 Moon orbit
ring, which was built once in render space and never repositioned.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Orbit frame (pos=to_render(0), scale=1/scale) → Task 2 Steps 1–2. ✓
- World-meter orbit geometry under the frame (vessel, Moon, preview) → Task 2 Steps 1, 3, 4. ✓
- Vessel orbit rebuilt only on element change → Task 1 helper + Task 2 Step 3. ✓
- Moon-ring staleness fixed → Task 2 Steps 1–2 (built under frame, frame repositions). ✓
- Math identity test, caching unit test, headless Moon-ring check → Task 1 tests + Task 2 Step 5. ✓
- Markers/Earth/rig/physics unchanged → not touched. ✓

**Placeholder scan:** No TBD/TODO; all code shown; headless check is a concrete script with expected output.

**Type consistency:** `orbit_shape_changed(a, b, tol=1e-9)` defined Task 1, used Task 2 Step 3. `_orbit_frame`, `_orbit_elem_cache` introduced Step 1, used Steps 2–4. `build_orbit_node(points, color)` reused unchanged with world-meter tuples. `sample_orbit_points(elements, n)` returns world meters (existing).
