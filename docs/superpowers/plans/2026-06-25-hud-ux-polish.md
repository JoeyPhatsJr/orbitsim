# HUD/UX Polish (Phase 6.2 A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** this cycle is entirely render-layer/visual work. Per project convention (CLAUDE.md: "render/visual tasks are done by the controller with headless screenshots"), it is executed **inline by the controller**, not dispatched to implementer subagents. Pure logic (unit conversion, panel-line assembly, keybind data) is unit-tested; DirectGUI widgets and key wiring are verified with a headless smoke run.

**Goal:** Add a keybind help overlay (F1), an inclination readout, an on-screen toast message channel, and a settings panel with a km↔mi units toggle.

**Architecture:** Two new small DirectGUI components (`keybind_overlay.py`, `settings_panel.py`); the inclination readout, units handling, and toast extend the existing `Hud`. Pure string/units logic is extracted into testable module-level functions; widgets and F1/Esc wiring live in `app.py`. No `core`/`sim` changes.

**Tech Stack:** Python 3, Panda3D DirectGUI (`OnscreenText`, `DirectFrame`, `DirectButton`), numpy. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- All changes in `orbitsim/render/` only — NO `core`/`sim`/`panda3d`-core physics changes. (verbatim from spec)
- Unit conversion stays at the HUD boundary: SI in, km/mi out. (verbatim from project rule)
- Miles conversion factor: `1 km = 0.621371 mi`; labels exactly `"km"`/`"mi"` and `"km/s"`/`"mi/s"`. (verbatim from spec)
- Inclination from `KeplerianElements.i` (radians); display in degrees, one decimal, with a `°` suffix. (verbatim from spec)
- Esc is not otherwise bound — safe to bind to settings toggle. (verified)
- Run tests with `.venv/Scripts/python -m pytest` — bare `python` lacks deps. (verbatim)
- Commits: stage explicit paths only (never `git add -A`/`.`); end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. (verbatim from repo git discipline)

---

## File Structure

- `orbitsim/render/hud/__init__.py` — extend `Hud`: pure `orbit_panel_lines(...)` helper (inclination + unit conversion), `units`/`set_units`, and a `flash()` toast.
- `orbitsim/render/keybind_overlay.py` — new `KeybindOverlay` widget + `SANDBOX_BINDINGS`/`SOLAR_BINDINGS` data.
- `orbitsim/render/settings_panel.py` — new `SettingsPanel` widget (units toggle).
- `orbitsim/render/app.py` — construct the overlay + settings panel; bind F1/Esc; pass `elem.i` + units into `hud.update`; reimplement `_flash_message` to drive the toast.
- `tests/render/test_hud_format.py` — new; unit tests for `orbit_panel_lines` (pure).
- `tests/render/test_keybind_overlay.py` — new; unit test for the binding data.

Reference shapes (existing, do not change):
- `Hud` (`render/hud/__init__.py`): `Hud(base)`; `.update(*, sim_time_s, warp, altitude_m, speed_mps, periapsis_m, apoapsis_m, period_s)`; `.update_flight(...)`; holds `.text`, `.flight` `OnscreenText`.
- App sandbox update loop (`render/app.py:680`) already computes `elem = state_to_elements(v0.state)` (so `elem.i` is available) and calls `self.hud.update(...)`.
- App `_flash_message(text)` (`render/app.py`, added in save/load cycle) currently does `print(f"[orbitsim] {text}")`.

---

## Task 1: Inclination readout + km/mi units (pure `orbit_panel_lines` + Hud wiring)

**Files:**
- Modify: `orbitsim/render/hud/__init__.py`
- Modify: `orbitsim/render/app.py` (the `self.hud.update(...)` call site at ~680)
- Test: `tests/render/test_hud_format.py` (create)

**Interfaces:**
- Produces: module-level pure function in `render/hud/__init__.py`:
  `orbit_panel_lines(*, sim_time_s: float, warp: float, altitude_m: float, speed_mps: float, periapsis_m: float, apoapsis_m: float, period_s: float, inclination_rad: float, units: str) -> list[str]`
- Produces: `Hud.units` (str, default `"km"`), `Hud.set_units(unit: str) -> None`, and `Hud.update(...)` gains a required `inclination_rad: float` keyword.

- [ ] **Step 1: Write the failing test**

Create `tests/render/test_hud_format.py`:

```python
"""Pure-logic tests for the HUD orbit panel line builder (no DirectGUI needed)."""
from orbitsim.render.hud import orbit_panel_lines


def _lines(**over):
    base = dict(
        sim_time_s=0.0, warp=1.0, altitude_m=500_000.0, speed_mps=7600.0,
        periapsis_m=400_000.0, apoapsis_m=600_000.0, period_s=5400.0,
        inclination_rad=0.5, units="km",
    )
    base.update(over)
    return orbit_panel_lines(**base)


def test_inclination_line_in_degrees():
    text = "\n".join(_lines(inclination_rad=0.5))
    assert "Inclination: 28.6°" in text  # 0.5 rad -> 28.6 deg


def test_km_units_default():
    text = "\n".join(_lines(units="km", altitude_m=500_000.0))
    assert "Altitude: 500.0 km" in text
    assert "Speed: 7.600 km/s" in text


def test_mi_units_conversion():
    text = "\n".join(_lines(units="mi", altitude_m=500_000.0, speed_mps=7600.0))
    # 500 km * 0.621371 = 310.7 mi ; 7.6 km/s * 0.621371 = 4.722 mi/s
    assert "Altitude: 310.7 mi" in text
    assert "Speed: 4.722 mi/s" in text


def test_period_always_minutes_unit_agnostic():
    text = "\n".join(_lines(period_s=5400.0))
    assert "Period: 90.0 min" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_hud_format.py -q`
Expected: FAIL — `ImportError: cannot import name 'orbit_panel_lines'`.

- [ ] **Step 3: Implement the pure helper + Hud wiring**

In `orbitsim/render/hud/__init__.py`, add at module level (after the imports, before `class Hud`):

```python
_MI_PER_KM = 0.621371


def _dist(meters: float, units: str) -> str:
    """Format a distance [m] as km or mi (1 km = 0.621371 mi), one decimal."""
    km = meters / 1000.0
    if units == "mi":
        return f"{km * _MI_PER_KM:,.1f} mi"
    return f"{km:,.1f} km"


def _speed(mps: float, units: str) -> str:
    """Format a speed [m/s] as km/s or mi/s, three decimals."""
    kms = mps / 1000.0
    if units == "mi":
        return f"{kms * _MI_PER_KM:,.3f} mi/s"
    return f"{kms:,.3f} km/s"


def orbit_panel_lines(
    *, sim_time_s: float, warp: float, altitude_m: float, speed_mps: float,
    periapsis_m: float, apoapsis_m: float, period_s: float,
    inclination_rad: float, units: str,
) -> list[str]:
    """Build the orbit-info panel text lines. Pure (no DirectGUI) so it is unit-testable."""
    import numpy as np
    return [
        f"Sim time: {sim_time_s:,.0f} s past J2000",
        f"Warp: x{warp:,.0f}",
        f"Altitude: {_dist(altitude_m, units)}",
        f"Speed: {_speed(speed_mps, units)}",
        f"Periapsis: {_dist(periapsis_m, units)}",
        f"Apoapsis: {_dist(apoapsis_m, units)}",
        f"Inclination: {np.degrees(inclination_rad):,.1f}°",
        f"Period: {period_s / 60.0:,.1f} min",
    ]
```

Then change `Hud.__init__` to add `self.units = "km"` (after creating `self.text`/`self.flight`), add a setter, and rewrite `Hud.update` to use the helper:

```python
    def set_units(self, units: str) -> None:
        """Set distance units for HUD readouts ('km' or 'mi')."""
        self.units = units

    def update(
        self, *, sim_time_s: float, warp: float, altitude_m: float, speed_mps: float,
        periapsis_m: float, apoapsis_m: float, period_s: float, inclination_rad: float,
    ) -> None:
        lines = orbit_panel_lines(
            sim_time_s=sim_time_s, warp=warp, altitude_m=altitude_m, speed_mps=speed_mps,
            periapsis_m=periapsis_m, apoapsis_m=apoapsis_m, period_s=period_s,
            inclination_rad=inclination_rad, units=self.units,
        )
        self.text.setText("\n".join(lines))
```

(Add `self.units = "km"` in `__init__`.) In `orbitsim/render/app.py`, update the sandbox `self.hud.update(...)` call (~680) to pass inclination:

```python
        self.hud.update(
            sim_time_s=self.clock.sim_time_s,
            warp=self.clock.warp,
            altitude_m=v0.state.r_mag - self.world.central.radius_m,
            speed_mps=v0.state.v_mag,
            periapsis_m=rp - self.world.central.radius_m,
            apoapsis_m=ra - self.world.central.radius_m,
            period_s=period,
            inclination_rad=elem.i,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/render/test_hud_format.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Headless check + full suite**

Run the full suite: `.venv/Scripts/python -m pytest tests/ -q` → all pass (the existing `test_navball`/etc. don't call `Hud.update`, so the new required kwarg won't break them; the only caller is `app.py`).

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/hud/__init__.py orbitsim/render/app.py tests/render/test_hud_format.py
git commit -m "$(cat <<'EOF'
HUD: inclination readout + km/mi units toggle

Extract orbit_panel_lines() as a pure, unit-tested helper; add Hud.units /
set_units; show inclination (deg) in the orbit panel.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: HUD toast (`Hud.flash`) + `_flash_message` rewire

**Files:**
- Modify: `orbitsim/render/hud/__init__.py` (add `toast` text + `flash`)
- Modify: `orbitsim/render/app.py` (`_flash_message` → `self.hud.flash`)

**Interfaces:**
- Consumes: `Hud` from Task 1.
- Produces: `Hud.flash(text: str, seconds: float = 2.0) -> None`; a `Hud.toast` `OnscreenText` (centered). The app's `_flash_message(text)` calls `self.hud.flash(text)`.

- [ ] **Step 1: Implement the toast**

In `Hud.__init__`, store the base and add a centered toast text + a handle for the pending clear task:

```python
        self._base = base
        self._toast_task = None
        self.toast = OnscreenText(
            text="", pos=(0.0, 0.6), scale=0.07, fg=(1.0, 1.0, 0.6, 1),
            shadow=(0, 0, 0, 1), mayChange=True, parent=base.aspect2d,
        )
```

Add the method:

```python
    def flash(self, text: str, seconds: float = 2.0) -> None:
        """Show a transient center-screen message that clears after `seconds`."""
        if self._toast_task is not None:
            self._base.taskMgr.remove(self._toast_task)
            self._toast_task = None
        self.toast.setText(text)
        self._toast_task = self._base.taskMgr.doMethodLater(
            seconds, self._clear_toast, "hud-toast-clear"
        )

    def _clear_toast(self, task):
        self.toast.setText("")
        self._toast_task = None
        return task.done
```

- [ ] **Step 2: Rewire `_flash_message` in app.py**

Replace the body of `_flash_message`:

```python
    def _flash_message(self, text: str) -> None:
        """Transient on-screen user feedback (toast)."""
        self.hud.flash(text)
```

- [ ] **Step 3: Headless smoke test**

Run this scratch check (offscreen) to confirm flash sets the toast text and quicksave/quickload drive it:

```bash
PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app.taskMgr.step()
app.hud.flash("Hello")
assert app.hud.toast.getText() == "Hello", app.hud.toast.getText()
app._flash_message("Quicksaved")
assert app.hud.toast.getText() == "Quicksaved"
print("OK: toast works")
PY
```

Expected: prints `OK: toast works`.

- [ ] **Step 4: Full suite + commit**

Run: `.venv/Scripts/python -m pytest tests/ -q` → all pass.

```bash
git add orbitsim/render/hud/__init__.py orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
HUD: on-screen toast; quicksave/quickload now show on screen

Hud.flash(text) shows a transient centered message; _flash_message drives
it instead of printing, closing the loop the save/load cycle left open.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Keybind help overlay (F1)

**Files:**
- Create: `orbitsim/render/keybind_overlay.py`
- Modify: `orbitsim/render/app.py` (construct overlay; bind F1; pick bindings by mode)
- Test: `tests/render/test_keybind_overlay.py` (create)

**Interfaces:**
- Produces: `SANDBOX_BINDINGS: list[tuple[str, str]]`, `SOLAR_BINDINGS: list[tuple[str, str]]`, and
  `class KeybindOverlay` with `KeybindOverlay(parent, lines: list[tuple[str, str]])`, `.toggle()`, `.show()`, `.hide()`, and a `.visible` bool.

- [ ] **Step 1: Write the failing test (binding data is pure)**

Create `tests/render/test_keybind_overlay.py`:

```python
"""Pure-data tests for keybind overlay content (no DirectGUI needed)."""
from orbitsim.render.keybind_overlay import SANDBOX_BINDINGS, SOLAR_BINDINGS


def test_sandbox_bindings_cover_key_controls():
    keys = {k for k, _ in SANDBOX_BINDINGS}
    for expected in ("F5", "F9", "F1", "Esc", "Z", "X", "T"):
        assert expected in keys, expected


def test_solar_bindings_minimal_but_present():
    keys = {k for k, _ in SOLAR_BINDINGS}
    assert "F1" in keys and "Esc" in keys
    # No flight controls in the solar viewer.
    assert "Z" not in keys and "T" not in keys


def test_every_binding_has_a_description():
    for k, desc in SANDBOX_BINDINGS + SOLAR_BINDINGS:
        assert isinstance(k, str) and k
        assert isinstance(desc, str) and desc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_keybind_overlay.py -q`
Expected: FAIL — `ModuleNotFoundError: orbitsim.render.keybind_overlay`.

- [ ] **Step 3: Implement the overlay**

Create `orbitsim/render/keybind_overlay.py`:

```python
"""Toggleable on-screen keybind help panel (F1)."""
from direct.gui.DirectFrame import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

SANDBOX_BINDINGS = [
    ("Right-drag", "Orbit camera"),
    ("Wheel", "Zoom"),
    ("Arrows", "Orbit camera"),
    ("W / S", "Pitch"),
    ("A / D", "Yaw"),
    ("Q / E", "Roll"),
    ("Shift / Ctrl", "Throttle up / down"),
    ("Z", "Full throttle"),
    ("X", "Cut throttle"),
    ("T", "SAS on/off"),
    ("1-7", "SAS mode (pro/retro/normal/...)"),
    (", / .", "Warp down / up"),
    ("F5 / F9", "Quicksave / Quickload"),
    ("Esc", "Settings"),
    ("F1", "Toggle this help"),
]

SOLAR_BINDINGS = [
    ("Right-drag", "Orbit camera"),
    ("Wheel", "Zoom"),
    ("Arrows", "Orbit camera"),
    (", / .", "Warp down / up"),
    ("Esc", "Settings"),
    ("F1", "Toggle this help"),
]


class KeybindOverlay:
    """A hidden-by-default panel listing key bindings; toggled with F1."""

    def __init__(self, parent, lines):
        self.visible = False
        self._frame = DirectFrame(
            frameColor=(0, 0, 0, 0.6), frameSize=(-0.7, 0.7, -0.75, 0.75),
            pos=(0, 0, 0), parent=parent,
        )
        body = "\n".join(f"{k:>14}   {desc}" for k, desc in lines)
        self._text = OnscreenText(
            text="Controls\n\n" + body, scale=0.05, fg=(1, 1, 1, 1),
            align=TextNode.ALeft, pos=(-0.62, 0.66), parent=self._frame,
            mayChange=False,
        )
        self._frame.hide()

    def show(self):
        self._frame.show()
        self.visible = True

    def hide(self):
        self._frame.hide()
        self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/render/test_keybind_overlay.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire F1 in app.py**

In `app.py`, import at top: `from orbitsim.render.keybind_overlay import KeybindOverlay, SANDBOX_BINDINGS, SOLAR_BINDINGS`. Where the HUD is built (after `self.hud = Hud(self)`), construct the overlay with the mode-appropriate bindings:

```python
        bindings = SOLAR_BINDINGS if self.solar_system else SANDBOX_BINDINGS
        self.keybind_overlay = KeybindOverlay(self.aspect2d, bindings)
```

In `_setup_input` (outside the sandbox-only block, so it works in both modes), add:

```python
        self.accept("f1", self.keybind_overlay.toggle)
```

- [ ] **Step 6: Headless smoke + full suite + commit**

Headless check:

```bash
PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app.taskMgr.step()
o = app.keybind_overlay
assert o.visible is False
o.toggle(); assert o.visible is True
o.toggle(); assert o.visible is False
print("OK: overlay toggles")
PY
```

Expected: `OK: overlay toggles`. Then `.venv/Scripts/python -m pytest tests/ -q` → all pass.

```bash
git add orbitsim/render/keybind_overlay.py orbitsim/render/app.py tests/render/test_keybind_overlay.py
git commit -m "$(cat <<'EOF'
HUD: F1 keybind help overlay

Toggleable panel listing controls; mode-aware (sandbox vs solar) binding
lists are pure data and unit-tested.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Settings panel (Esc) + units toggle wiring

**Files:**
- Create: `orbitsim/render/settings_panel.py`
- Modify: `orbitsim/render/app.py` (construct panel; bind Esc; wire units callback to `hud.set_units`)

**Interfaces:**
- Consumes: `Hud.set_units` (Task 1), `KeybindOverlay` pattern (Task 3).
- Produces: `class SettingsPanel` with `SettingsPanel(parent, on_units_change: Callable[[str], None])`, `.toggle()`, `.visible`. The units button cycles `"km"`↔`"mi"` and calls `on_units_change(unit)`.

- [ ] **Step 1: Implement the settings panel**

Create `orbitsim/render/settings_panel.py`:

```python
"""Toggleable settings panel (Esc): currently a km/mi units toggle."""
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectButton import DirectButton
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class SettingsPanel:
    """A hidden-by-default settings panel; toggled with Esc."""

    def __init__(self, parent, on_units_change):
        self.visible = False
        self.units = "km"
        self._on_units_change = on_units_change
        self._frame = DirectFrame(
            frameColor=(0, 0, 0, 0.7), frameSize=(-0.45, 0.45, -0.25, 0.25),
            pos=(0, 0, 0), parent=parent,
        )
        OnscreenText(
            text="Settings", scale=0.06, fg=(1, 1, 1, 1), align=TextNode.ACenter,
            pos=(0, 0.15), parent=self._frame, mayChange=False,
        )
        self._units_btn = DirectButton(
            text="Units: km", scale=0.05, pos=(0, 0, 0.0),
            command=self._cycle_units, parent=self._frame,
        )
        self._frame.hide()

    def _cycle_units(self):
        self.units = "mi" if self.units == "km" else "km"
        self._units_btn["text"] = f"Units: {self.units}"
        self._on_units_change(self.units)

    def show(self):
        self._frame.show()
        self.visible = True

    def hide(self):
        self._frame.hide()
        self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()
```

- [ ] **Step 2: Wire Esc + units callback in app.py**

Import at top: `from orbitsim.render.settings_panel import SettingsPanel`. After the HUD/overlay construction:

```python
        self.settings_panel = SettingsPanel(self.aspect2d, self.hud.set_units)
```

In `_setup_input` (both modes), add:

```python
        self.accept("escape", self.settings_panel.toggle)
```

- [ ] **Step 3: Headless smoke test (units callback drives HUD)**

```bash
PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app.taskMgr.step()
sp = app.settings_panel
assert sp.visible is False and app.hud.units == "km"
sp.toggle(); assert sp.visible is True
sp._cycle_units(); assert app.hud.units == "mi", app.hud.units
sp._cycle_units(); assert app.hud.units == "km"
print("OK: settings units toggle drives HUD")
PY
```

Expected: `OK: settings units toggle drives HUD`.

- [ ] **Step 4: Final headless visual capture + full suite**

Step the sandbox app a few frames with the overlay shown and save a screenshot to eyeball the overlay/HUD/inclination, then run the full suite.

```bash
.venv/Scripts/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/settings_panel.py orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
HUD: Esc settings panel with km/mi units toggle

Settings panel cycles units and drives Hud.set_units so distance/speed
readouts re-label live.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Keybind help overlay (F1, mode-aware, hidden default) → Task 3. ✓
- Inclination readout (deg, from `elem.i`) → Task 1. ✓
- HUD toast replacing the print fallback → Task 2. ✓
- Settings panel (Esc) + km/mi units re-labelling HUD → Tasks 1 (units in HUD) + 4 (panel). ✓
- Render-only, conversions at HUD boundary, mi factor 0.621371, labels km/mi → Task 1 constants + Global Constraints. ✓
- Out-of-scope items (scheduling, rendezvous, perturbations toggle) → not present. ✓
- Tests: pure logic unit-tested (`orbit_panel_lines`, binding data); widgets/wiring headless-verified → Tasks 1, 3 unit tests; Tasks 2, 3, 4 headless smokes. ✓

**Placeholder scan:** No TBD/TODO; all code shown in full; every headless check has a concrete script + expected output.

**Type consistency:** `orbit_panel_lines(..., inclination_rad, units)` signature consistent between Task 1's helper, test, and `Hud.update`. `Hud.set_units(unit)` consumed identically by `SettingsPanel(on_units_change=...)` (Task 4) and defined in Task 1. `KeybindOverlay`/`SettingsPanel` `.toggle()`/`.visible` consistent across tasks. `flash(text, seconds=2.0)` consistent between Task 2 def and the `_flash_message` caller.
