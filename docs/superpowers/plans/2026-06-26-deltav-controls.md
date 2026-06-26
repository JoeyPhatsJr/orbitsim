# Δv Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bigger Δv cap and an unlimited-Δv toggle (title screen, Esc panel, `U` key) that makes `delta_v_remaining` infinite and stops fuel from draining.

**Architecture:** A new `unlimited_dv` flag on `Vessel` drives the rocket-equation derivation and the powered-flight step in the pure sim layer; the render layer only flips the flag and formats an infinite readout.

**Tech Stack:** Python 3, numpy, Panda3D. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- SI everywhere; float64; `core/` never imports render/panda3d. (project rule)
- Run tests with `.venv/Scripts/python -m pytest`. (verbatim)
- Commits: explicit paths only (never `git add -A`); keep `data/`, scratch files, `.claude/settings.local.json` out; `CLAUDE.md` untracked. End messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`, then `git push`. (repo discipline)
- Tasks 1–3 are pure sim (Haiku TDD). Task 4 is render, controller-executed with headless screenshots.

## File Structure

- `orbitsim/sim/world.py` — `Vessel.unlimited_dv` flag; `delta_v_remaining`, `World.step`, `World.any_thrusting` honor it.
- `orbitsim/sim/persistence.py` — serialize/restore `unlimited_dv`.
- `orbitsim/render/app.py` — title slider cap + checkbox, `_on_play`, settings callback, `U` keybind, `_execute_burn` skip, `_refresh_readout` ∞.
- `orbitsim/render/settings_panel.py` — "Unlimited Δv" toggle button.
- `orbitsim/render/hud/__init__.py` — `update_flight` shows `∞`.
- Tests: `tests/sim/test_world.py`, `tests/sim/test_persistence.py`.

---

## Task 1: `Vessel.unlimited_dv` → infinite Δv

**Files:**
- Modify: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py`

**Interfaces:**
- Produces: `Vessel.unlimited_dv: bool = False`; `Vessel.delta_v_remaining` returns `float("inf")` when set.

- [ ] **Step 1: Write the failing test**

Add to `tests/sim/test_world.py`:

```python
def test_unlimited_dv_is_infinite_regardless_of_fuel():
    from orbitsim.sim.world import Vessel
    from orbitsim.core.state import StateVector
    import numpy as np
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.5e3, 0]),
                     mu=3.986e14, epoch_s=0.0)
    v = Vessel(name="x", state=st, fuel_mass_kg=0.0, unlimited_dv=True)
    assert v.delta_v_remaining == float("inf")
    v2 = Vessel(name="y", state=st, fuel_mass_kg=500.0, unlimited_dv=False)
    assert v2.delta_v_remaining < float("inf")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q -k unlimited`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'unlimited_dv'`.

- [ ] **Step 3: Implement**

In `orbitsim/sim/world.py`, add the field to the `Vessel` dataclass (next to `sas_mode`):

```python
    unlimited_dv: bool = False
```

And guard `delta_v_remaining` (first line of the property body):

```python
    @property
    def delta_v_remaining(self) -> float:
        """Remaining delta-V from the rocket equation [m/s]; inf if unlimited, 0 if no fuel."""
        if self.unlimited_dv:
            return float("inf")
        if self.fuel_mass_kg <= 0.0:
            return 0.0
        return tsiolkovsky_dv(self.exhaust_velocity_mps, self.mass_kg, self.dry_mass_kg)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q -k unlimited`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "$(cat <<'EOF'
Vessel: unlimited_dv flag -> infinite delta_v_remaining

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: `World.step` doesn't drain fuel under unlimited; warp still locks

**Files:**
- Modify: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py`

**Interfaces:**
- Consumes: `Vessel.unlimited_dv` (Task 1).
- Produces: under `unlimited_dv`, a thrusting step changes velocity but leaves `fuel_mass_kg` unchanged; `any_thrusting()` is True with `throttle>0` even at zero fuel.

Read the current `World.step` (around `orbitsim/sim/world.py:88-126`) and `any_thrusting` (around `:128-131`) before editing.

- [ ] **Step 1: Write the failing tests**

Add to `tests/sim/test_world.py`:

```python
def test_unlimited_dv_step_thrusts_without_draining_fuel():
    from orbitsim.sim.world import Vessel, World
    from orbitsim.core.bodies import EARTH
    from orbitsim.core.state import StateVector
    import numpy as np
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, dry_mass_kg=1000.0, fuel_mass_kg=10.0,
               max_thrust_n=5.0e4, exhaust_velocity_mps=3000.0,
               throttle=1.0, unlimited_dv=True)
    # point the nose prograde so thrust does something
    v.sas_mode = "PROGRADE"
    w = World(central=EARTH, vessels=[v])
    speed0 = v.state.v_mag
    w.step(1.0)
    assert v.fuel_mass_kg == 10.0          # fuel not drained
    assert v.state.v_mag != speed0          # thrust applied (speed changed)


def test_unlimited_dv_locks_warp_even_with_zero_fuel():
    from orbitsim.sim.world import Vessel, World
    from orbitsim.core.bodies import EARTH
    from orbitsim.core.state import StateVector
    import numpy as np
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.546e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    v = Vessel(name="x", state=st, fuel_mass_kg=0.0, max_thrust_n=5.0e4,
               throttle=1.0, unlimited_dv=True)
    w = World(central=EARTH, vessels=[v])
    assert w.any_thrusting() is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q -k unlimited`
Expected: FAIL (fuel drains / `any_thrusting` False at zero fuel).

- [ ] **Step 3: Implement**

In `World.step`, the powered branch currently gates on `vessel.throttle > 0.0 and vessel.fuel_mass_kg > 0.0` and assigns `vessel.fuel_mass_kg = new_fuel`. Change the gate to allow unlimited, and skip the fuel write under unlimited:

```python
            if vessel.throttle > 0.0 and (vessel.fuel_mass_kg > 0.0 or vessel.unlimited_dv):
                # ... existing integrate_powered(...) call producing new_state, new_fuel ...
                vessel.state = new_state
                if not vessel.unlimited_dv:
                    vessel.fuel_mass_kg = new_fuel
```

(Keep the existing `integrate_powered` call and its other outputs exactly as-is; only the gate and the `fuel_mass_kg` assignment are conditional. Under unlimited, `integrate_powered` still runs at the real current mass, so acceleration is normal — we just don't persist the depletion.)

In `any_thrusting`:

```python
    def any_thrusting(self) -> bool:
        return any(v.throttle > 0.0 and (v.fuel_mass_kg > 0.0 or v.unlimited_dv)
                   for v in self.vessels)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "$(cat <<'EOF'
World: unlimited_dv thrusts without draining fuel; warp still locks

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: Persistence round-trips `unlimited_dv`

**Files:**
- Modify: `orbitsim/sim/persistence.py`
- Test: `tests/sim/test_persistence.py`

**Interfaces:**
- Consumes: `Vessel.unlimited_dv` (Task 1).
- Produces: `_vessel_to_dict` writes `"unlimited_dv"`; `_vessel_from_dict` reads it defaulting to `False`.

- [ ] **Step 1: Write the failing test**

Add to `tests/sim/test_persistence.py`:

```python
def test_unlimited_dv_round_trips(tmp_path):
    from orbitsim.sim.world import Vessel, World
    from orbitsim.sim.clock import SimClock
    from orbitsim.sim.persistence import save_scenario, load_scenario
    from orbitsim.core.bodies import EARTH
    from orbitsim.core.state import StateVector
    import numpy as np
    st = StateVector(r=np.array([7.0e6, 0, 0]), v=np.array([0, 7.5e3, 0]),
                     mu=EARTH.mu, epoch_s=0.0)
    w = World(central=EARTH, vessels=[Vessel(name="x", state=st, unlimited_dv=True)])
    p = tmp_path / "s.json"
    save_scenario(w, SimClock(0.0, 1.0), str(p))
    w2, _ = load_scenario(str(p))
    assert w2.vessels[0].unlimited_dv is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/sim/test_persistence.py -q -k unlimited`
Expected: FAIL — loaded vessel defaults `unlimited_dv` to False.

- [ ] **Step 3: Implement**

In `_vessel_to_dict`, add to the returned dict:

```python
        "unlimited_dv": vessel.unlimited_dv,
```

In `_vessel_from_dict`, pass it to the `Vessel(...)` constructor using `.get` so old saves load:

```python
            unlimited_dv=d.get("unlimited_dv", False),
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/sim/test_persistence.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/persistence.py tests/sim/test_persistence.py
git commit -m "$(cat <<'EOF'
Persistence: round-trip unlimited_dv (defaults False for old saves)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 4: Render wiring — cap, checkbox, Esc toggle, key, ∞ readout (controller)

**Files:**
- Modify: `orbitsim/render/app.py`, `orbitsim/render/settings_panel.py`, `orbitsim/render/hud/__init__.py`

**Interfaces:**
- Consumes: `Vessel.unlimited_dv` (Tasks 1–2).
- Produces: `OrbitApp._set_unlimited_dv(on: bool)`; settings panel calls back to it; HUD/readout render `∞`.

- [ ] **Step 1: HUD shows ∞**

In `orbitsim/render/hud/__init__.py`, add `import math` at the top if absent, and change the dV line in `update_flight`:

```python
            (f"dV left: ∞" if not math.isfinite(dv_remaining)
             else f"dV left: {dv_remaining:,.0f} m/s"),
```

- [ ] **Step 2: Maneuver readout shows ∞**

In `orbitsim/render/app.py::_refresh_readout`, replace the f-string with a finite-guard:

```python
    def _refresh_readout(self) -> None:
        import math
        node = self._current_node()
        budget = self.world.vessels[0].delta_v_remaining
        left = "∞" if not math.isfinite(budget) else f"{budget:,.0f} m/s"
        self._dv_readout.setText(
            f"Maneuver dV: {node.magnitude_mps:,.1f} m/s   (dV left {left})"
        )
```

- [ ] **Step 3: Title slider cap + Unlimited checkbox**

In `_build_title_screen`: change the slider range to `range=(0.0, 20000.0)`. Add a checkbox after the slider (import `from direct.gui.DirectCheckButton import DirectCheckButton` at top of file):

```python
        self._unlimited_check = DirectCheckButton(
            text="Unlimited dV", scale=0.05, pos=(0.0, 0.0, -0.32),
            text_fg=(1, 1, 1, 1), boxPlacement="left", parent=self.aspect2d,
        )
```

Append `self._unlimited_check` to `self._title_nodes`. In `_on_play`, after applying fuel, set the flag:

```python
        on = bool(self._unlimited_check["indicatorValue"])
        for vessel in self.world.vessels:
            vessel.unlimited_dv = on
```

- [ ] **Step 4: `_set_unlimited_dv` + Esc toggle + `U` key**

Add to `OrbitApp`:

```python
    def _set_unlimited_dv(self, on: bool) -> None:
        for vessel in self.world.vessels:
            vessel.unlimited_dv = on
        self._flash_message(f"Unlimited dV {'ON' if on else 'OFF'}")
        self._refresh_readout()

    def _toggle_unlimited_dv(self) -> None:
        cur = bool(self.world.vessels and self.world.vessels[0].unlimited_dv)
        self._set_unlimited_dv(not cur)
```

In `_setup_input`, inside the `if not self.solar_system and self.world.vessels:` block:

```python
            self.accept("u", self._toggle_unlimited_dv)
```

In `SettingsPanel.__init__`, accept an `on_unlimited_toggle` callback, widen the frame to `frameSize=(-0.45, 0.45, -0.32, 0.25)`, and add a button below the units button:

```python
        self._unlimited_on = False
        self._unlimited_btn = DirectButton(
            text="Unlimited dV: off", scale=0.05, pos=(0, 0, -0.12),
            command=self._toggle_unlimited, parent=self._frame,
        )
```

with:

```python
    def _toggle_unlimited(self):
        self._unlimited_on = not self._unlimited_on
        self._unlimited_btn["text"] = f"Unlimited dV: {'on' if self._unlimited_on else 'off'}"
        self._on_unlimited_toggle(self._unlimited_on)
```

Update the `SettingsPanel(...)` construction site in `app.py` to pass `on_unlimited_toggle=self._set_unlimited_dv`. (Find where `SettingsPanel(` is constructed; add the kwarg.)

- [ ] **Step 5: Execute-burn skips fuel under unlimited**

In `_execute_burn`, wrap the fuel deduction:

```python
            if not v0.unlimited_dv:
                burned = fuel_burned_for_dv(v0.exhaust_velocity_mps, v0.mass_kg, dv)
                v0.fuel_mass_kg = max(0.0, v0.fuel_mass_kg - burned)
```

- [ ] **Step 6: Headless verification**

```bash
cd "C:/AI/Claude/Orbital Mechanics Sim" && PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
import math
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app._on_play()
app.taskMgr.step()
app._toggle_unlimited_dv()
assert app.world.vessels[0].unlimited_dv is True
assert math.isinf(app.world.vessels[0].delta_v_remaining)
app._toggle_unlimited_dv()
assert app.world.vessels[0].unlimited_dv is False
print("OK: unlimited dV toggles, delta_v_remaining inf when on")
PY
```

Expected: `OK: unlimited dV toggles, delta_v_remaining inf when on`. Also run the full suite: `.venv/Scripts/python -m pytest tests/ -q`.

- [ ] **Step 7: Commit**

```bash
git add orbitsim/render/app.py orbitsim/render/settings_panel.py orbitsim/render/hud/__init__.py
git commit -m "$(cat <<'EOF'
Render: dV cap 20k, unlimited-dV checkbox/Esc toggle/U key, inf readout

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Self-Review

- Spec: bigger cap (Task 4.3) ✓; unlimited flag inf + no drain (Tasks 1–2) ✓; title + Esc + key (Task 4.3–4.4) ✓; persistence (Task 3) ✓; ∞ HUD/readout (Task 4.1–4.2) ✓; execute-burn skip (Task 4.5) ✓.
- No placeholders; all code shown. Keybind `u` verified free (`t`/`z`/`x` taken; `1`–`7` are SAS).
- Types consistent: `unlimited_dv: bool` used uniformly; `_set_unlimited_dv(on: bool)` is the single setter the panel/key/title funnel through.
