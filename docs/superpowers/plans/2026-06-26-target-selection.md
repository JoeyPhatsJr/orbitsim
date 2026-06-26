# Target Selection + Planning Readouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the Moon "Target" toggle into a click-selected current target that drives the closest-approach + relative-velocity readout, the TARGET/ANTITARGET SAS hold, and navball target markers.

**Architecture:** A pure `Target` value object (Moon today) and a pure `nearest_marker` hit-test live in `render/` but import only `core`; the sim layer gains a per-tick `Vessel.sas_target_pos` that `World.step` feeds to the existing `sas_target_dir`; `app.py` wires click-selection, the target position, and the readouts.

**Tech Stack:** Python 3, numpy, Panda3D. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- SI everywhere; float64; `core/` never imports render/panda3d. (project rule)
- `render/targets.py` and `render/picking.py` must NOT import panda3d (pure, unit-tested). (this plan)
- Run tests with `.venv/Scripts/python -m pytest`. Commits: explicit paths; end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; then `git push`. (repo discipline)
- Tasks 1–3 pure (Haiku TDD). Tasks 4–5 render, controller-executed with headless screenshots.

## File Structure

- `orbitsim/render/picking.py` (new) — `nearest_marker(click_px, markers_px, tol_px)`.
- `orbitsim/render/targets.py` (new) — `Target` protocol + `MoonTarget` wrapping `core.moon.moon_state_at`.
- `orbitsim/sim/world.py` — `Vessel.sas_target_pos`; `World.step` passes it to `sas_target_dir`.
- `orbitsim/render/navball.py` — add `TARGET`/`ANTITARGET` to `_MARKER_COLORS`.
- `orbitsim/render/app.py` — `self._targets`/`self._target`, click-to-pick, "Clear Target" button, CA generalization, set `sas_target_pos` each frame, pass `target_pos` to navball, HUD target name.
- Tests: `tests/render/test_picking.py`, `tests/render/test_targets.py`, `tests/sim/test_world.py`.

---

## Task 1: `nearest_marker` pure hit-test

**Files:**
- Create: `orbitsim/render/picking.py`
- Test: `tests/render/test_picking.py`

**Interfaces:**
- Produces: `nearest_marker(click_px: tuple[float, float], markers_px: list[tuple[float, float]], tol_px: float) -> int | None` — index of the nearest marker within `tol_px` pixels, else `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/render/test_picking.py`:

```python
from orbitsim.render.picking import nearest_marker


def test_hit_within_tolerance():
    assert nearest_marker((100.0, 100.0), [(105.0, 102.0)], tol_px=10.0) == 0


def test_miss_beyond_tolerance():
    assert nearest_marker((100.0, 100.0), [(200.0, 200.0)], tol_px=10.0) is None


def test_nearest_of_several():
    assert nearest_marker((0.0, 0.0), [(50.0, 0.0), (8.0, 0.0), (9.0, 0.0)],
                          tol_px=10.0) == 1


def test_empty_list():
    assert nearest_marker((0.0, 0.0), [], tol_px=10.0) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_picking.py -q`
Expected: FAIL — `ModuleNotFoundError: orbitsim.render.picking`.

- [ ] **Step 3: Implement**

Create `orbitsim/render/picking.py`:

```python
"""Pure screen-space marker hit-testing (no Panda3D)."""
from math import hypot


def nearest_marker(click_px, markers_px, tol_px):
    """Index of the nearest marker within tol_px of the click, else None.

    Parameters
    ----------
    click_px : (float, float)        Click position in pixels.
    markers_px : list[(float, float)] Marker positions in pixels.
    tol_px : float                    Max hit distance in pixels.
    """
    best_i, best_d = None, tol_px
    for i, (mx, my) in enumerate(markers_px):
        d = hypot(mx - click_px[0], my - click_px[1])
        if d <= best_d:
            best_i, best_d = i, d
    return best_i
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_picking.py -q`
Expected: PASS (4).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/picking.py tests/render/test_picking.py
git commit -m "$(cat <<'EOF'
Picking: pure nearest_marker screen-space hit-test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: `MoonTarget` wrapping the Moon ephemeris

**Files:**
- Create: `orbitsim/render/targets.py`
- Test: `tests/render/test_targets.py`

**Interfaces:**
- Consumes: `core.moon.moon_state_at(t_s) -> StateVector`.
- Produces: `MoonTarget()` with `.name == "Moon"` and `.state_at(t_s) -> StateVector` delegating to `moon_state_at`.

- [ ] **Step 1: Write the failing test**

Create `tests/render/test_targets.py`:

```python
import numpy as np
from orbitsim.render.targets import MoonTarget
from orbitsim.core.moon import moon_state_at


def test_name_and_delegation():
    t = MoonTarget()
    assert t.name == "Moon"
    for ts in (0.0, 1.0e5, 3.0e5):
        np.testing.assert_allclose(t.state_at(ts).r, moon_state_at(ts).r, rtol=0, atol=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_targets.py -q`
Expected: FAIL — `ModuleNotFoundError: orbitsim.render.targets`.

- [ ] **Step 3: Implement**

Create `orbitsim/render/targets.py`:

```python
"""Targetable bodies for maneuver planning (pure; no Panda3D).

A Target answers 'where is it at time t' in the same inertial, Earth-centered
frame as the vessel. Ships become Targets in a later cycle.
"""
from orbitsim.core.moon import moon_state_at
from orbitsim.core.state import StateVector


class MoonTarget:
    name = "Moon"

    def state_at(self, t_s: float) -> StateVector:
        return moon_state_at(t_s)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_targets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/targets.py tests/render/test_targets.py
git commit -m "$(cat <<'EOF'
Targets: MoonTarget value object wrapping moon_state_at

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: `Vessel.sas_target_pos` + `World.step` slews to target

**Files:**
- Modify: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py`

**Interfaces:**
- Produces: `Vessel.sas_target_pos: np.ndarray | None = None`; when `sas_mode in ("TARGET","ANTITARGET")`, `World.step` feeds it to `sas_target_dir`. `None` → that tick's slew is skipped (no crash).

Read `World.step`'s attitude block (`orbitsim/sim/world.py:96-110`, where it calls `slew_toward(..., sas_target_dir(vessel.sas_mode, vessel.state), ...)`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/sim/test_world.py`:

```python
def test_target_sas_slews_nose_toward_target():
    from orbitsim.sim.world import Vessel, World
    from orbitsim.core.bodies import EARTH
    from orbitsim.core.state import StateVector
    from orbitsim.core.attitude import nose_direction
    import numpy as np
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, sas_mode="TARGET", max_turn_rate_radps=1.0)
    v.sas_target_pos = np.array([7.0e6, 1.0e8, 0.0])   # far +Y of the ship
    w = World(central=EARTH, vessels=[v])
    want = v.sas_target_pos - v.state.r
    want = want / np.linalg.norm(want)
    a0 = float(np.dot(nose_direction(v.orientation), want))
    for _ in range(120):
        w.step(0.05)
    a1 = float(np.dot(nose_direction(v.orientation), want))
    assert a1 > a0           # nose turned toward the target
    assert a1 > 0.9          # and got close to pointing at it


def test_target_sas_with_no_target_does_not_crash():
    from orbitsim.sim.world import Vessel, World
    from orbitsim.core.bodies import EARTH
    from orbitsim.core.state import StateVector
    import numpy as np
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, sas_mode="TARGET")   # sas_target_pos stays None
    w = World(central=EARTH, vessels=[v])
    q0 = v.orientation.copy()
    w.step(1.0)
    np.testing.assert_array_equal(v.orientation, q0)    # attitude held, no error
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q -k target`
Expected: FAIL — `sas_target_pos` not a field / `TARGET requires target_pos` raised.

- [ ] **Step 3: Implement**

Add the field to `Vessel` (after `unlimited_dv` or near `orientation`):

```python
    sas_target_pos: object = None   # inertial target position [m] for TARGET/ANTITARGET, or None
```

In `World.step`'s attitude block, compute the target direction defensively:

```python
            try:
                target = sas_target_dir(vessel.sas_mode, vessel.state, vessel.sas_target_pos)
            except ValueError:
                target = None
            if target is not None:
                vessel.orientation = slew_toward(
                    vessel.orientation, target, vessel.max_turn_rate_radps, sim_dt_s)
```

(Match the existing variable names for the slew call — read the current block and keep its `slew_toward(...)` signature; only thread `vessel.sas_target_pos` into `sas_target_dir` and guard the `ValueError`.)

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "$(cat <<'EOF'
World: Vessel.sas_target_pos drives TARGET/ANTITARGET slew (None-safe)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 4: Navball TARGET/ANTITARGET markers (controller)

**Files:**
- Modify: `orbitsim/render/navball.py`

**Interfaces:**
- Produces: navball draws TARGET/ANTITARGET markers when `update(..., target_pos=...)` is non-None.

- [ ] **Step 1: Add the colors**

In `orbitsim/render/navball.py`, add to `_MARKER_COLORS`:

```python
    "TARGET": (1.0, 0.2, 1.0, 1),
    "ANTITARGET": (0.6, 0.2, 0.6, 1),
```

The existing `update` loop iterates `_MARKER_COLORS` and already calls
`sas_target_dir(mode, state, target_pos)`, hiding any marker whose mode raises
`ValueError` (i.e. when `target_pos is None`). No other change needed.

- [ ] **Step 2: Headless check**

```bash
cd "C:/AI/Claude/Orbital Mechanics Sim" && PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
import numpy as np
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app._on_play(); app.taskMgr.step()
st = app.world.vessels[0].state
app.navball.update(orientation_q=app.world.vessels[0].orientation, state=st,
                   target_pos=st.r + np.array([0.0, 1.0e8, 0.0]))
assert "TARGET" in app.navball._markers and not app.navball._markers["TARGET"].is_hidden()
print("OK: navball TARGET marker visible with a target")
PY
```

Expected: `OK: navball TARGET marker visible with a target`.

- [ ] **Step 3: Commit**

```bash
git add orbitsim/render/navball.py
git commit -m "$(cat <<'EOF'
Navball: TARGET/ANTITARGET markers (shown when a target is set)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 5: App integration — click-to-target, CA generalization, wiring (controller)

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `nearest_marker` (T1), `MoonTarget` (T2), `Vessel.sas_target_pos` (T3), navball markers (T4).
- Produces: `self._targets: list`, `self._target` (Target|None), `_pick_target_at_click`, `_clear_target`; CA block + SAS + navball + HUD driven by `self._target`.

- [ ] **Step 1: Replace Moon-target state with the registry**

In `_build_maneuver_ui`/sandbox setup (where `self._target_moon = False` is today, ~app.py:224), introduce:

```python
        from orbitsim.render.targets import MoonTarget
        self._targets = [MoonTarget()]
        self._target = None     # current Target or None
```

Remove `self._target_moon`. The Moon body marker (`_moon_np`) and orbit ring stay.

- [ ] **Step 2: Repurpose the "Target" button to "Clear Target"**

In `_build_maneuver_ui`'s `node_btns`, change `("Target", self._toggle_target)` to `("Clear Tgt", self._clear_target)`. Replace `_toggle_target` with:

```python
    def _clear_target(self):
        """Deselect the current target; remove its CA markers + readout."""
        self._target = None
        for attr in ("_ca_marker_ship", "_ca_marker_moon"):
            np_ = getattr(self, attr, None)
            if np_ is not None:
                np_.remove_node()
                setattr(self, attr, None)
        self._ca = None
        self._target_text.setText("Target: none")
```

- [ ] **Step 3: Click-to-pick (left-click tap)**

`mouse1-up` is already accepted for `_release_jogs`. Track press position and extend the release. In `_setup_input` (sandbox block) add:

```python
            self._mouse1_down_px = None
            self.accept("mouse1", self._on_mouse1_down)
```

Add methods:

```python
    def _on_mouse1_down(self):
        mw = self.mouseWatcherNode
        self._mouse1_down_px = self._mouse_px() if (mw and mw.has_mouse()) else None

    def _mouse_px(self):
        mw = self.mouseWatcherNode
        w, h = self.win.get_x_size(), self.win.get_y_size()
        return ((mw.get_mouse_x() * 0.5 + 0.5) * w, (mw.get_mouse_y() * 0.5 + 0.5) * h)

    def _marker_px(self, world_r):
        """Project an inertial position to pixels, or None if behind the camera."""
        from panda3d.core import Point2
        rp = self.transform.to_render(world_r)
        p = self.cam.get_relative_point(self.render, rp)
        proj = Point2()
        if not self.camLens.project(p, proj):
            return None
        w, h = self.win.get_x_size(), self.win.get_y_size()
        return ((proj.x * 0.5 + 0.5) * w, (proj.y * 0.5 + 0.5) * h)

    def _try_pick_target(self):
        from orbitsim.render.picking import nearest_marker
        if self._mouse1_down_px is None:
            return
        click = self._mouse_px()
        if (abs(click[0] - self._mouse1_down_px[0]) > 6.0
                or abs(click[1] - self._mouse1_down_px[1]) > 6.0):
            return  # was a drag, not a tap
        now = self.clock.sim_time_s
        px = [self._marker_px(t.state_at(now).r) for t in self._targets]
        idxs = [i for i, p in enumerate(px) if p is not None]
        hit = nearest_marker(click, [px[i] for i in idxs], tol_px=22.0)
        if hit is not None:
            self._target = self._targets[idxs[hit]]
```

In `_release_jogs`, call `self._try_pick_target()` first (read current `_release_jogs`; add the call at the top), so a tap both springs the jogs and attempts a pick.

- [ ] **Step 4: Generalize the CA block + wire SAS/navball/HUD**

In `_update`, replace `if self._target_moon:` with `if self._target is not None:` and replace the two `moon_state_at(...)` calls inside that block with `self._target.state_at(base_epoch)` and `self._target.state_at(self._ca_abs_epoch)`; change the readout text to use `self._target.name`. Just before `self.world.step(sim_dt)`, set the per-tick SAS target:

```python
        tgt = self._target.state_at(self.clock.sim_time_s).r if self._target else None
        for v in self.world.vessels:
            v.sas_target_pos = tgt
```

Change the navball call to pass the target position:

```python
        self.navball.update(orientation_q=v0.orientation, state=v0.state, target_pos=tgt)
```

(`tgt` is already computed above this line in `_update`; if out of scope there, recompute it inline.)

- [ ] **Step 5: Headless verification**

```bash
cd "C:/AI/Claude/Orbital Mechanics Sim" && PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "win-size 900 700")
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app._on_play()
for _ in range(4): app.taskMgr.step()
# Directly select the Moon target (bypassing the literal mouse tap).
app._target = app._targets[0]
app.world.vessels[0].sas_mode = "TARGET"
for _ in range(6): app.taskMgr.step()
assert app._target is not None and app._target.name == "Moon"
assert app.world.vessels[0].sas_target_pos is not None
assert app._ca is not None                      # closest-approach computed
app._clear_target()
for _ in range(2): app.taskMgr.step()
assert app._target is None and app.world.vessels[0].sas_target_pos is None
print("OK: target select drives CA + SAS target; clear works")
PY
```

Expected: `OK: target select drives CA + SAS target; clear works`. Also run `.venv/Scripts/python -m pytest tests/ -q` and take a screenshot (Moon targeted, navball TARGET marker + CA readout) to eyeball.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
Render: click-to-target selection driving CA/rel-vel, SAS, navball

Generalizes the Moon toggle to a Target registry; left-click tap selects
the nearest body marker, 'Clear Tgt' deselects. Feeds the target position
into the closest-approach readout, TARGET/ANTITARGET SAS, and the navball.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Self-Review

- Spec: target abstraction (T2) ✓; pure picking (T1) ✓; click tap-select + clear (T5.2–5.3) ✓; CA + rel-vel generalized (T5.4; rel-vel already in readout) ✓; working TARGET SAS (T3, T5.4) ✓; navball markers (T4, T5.4) ✓; HUD target name (T5.2/5.4) ✓.
- Refinement vs spec: clearing is via the "Clear Tgt" button (not clear-on-empty-click), to avoid clicks on other UI buttons clearing the target. Selection stays click-to-target as specified.
- No placeholders; pure modules import only `core`. Types consistent: `nearest_marker(...) -> int|None`; `Target.state_at(t)->StateVector`; `Vessel.sas_target_pos`.
