# Scheduled Maneuver Nodes (Phase 6.2 A2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** Task 1 is pure physics (TDD — dispatchable to a Haiku implementer). Task 2 is render-layer maneuver-UI work, executed **inline by the controller** with headless verification per project convention.

**Goal:** Turn the sandbox's immediate-execute maneuver editor into a scheduled single-node planner: place a node ahead on the orbit (manual time-to-node or Next Pe/Ap), preview it, auto-warp-down toward it, and manually execute the impulsive burn at the node.

**Architecture:** Two pure core functions (`time_to_periapsis`/`time_to_apoapsis`) power the Pe/Ap presets. The render maneuver UI gains a fixed absolute node epoch (`_node_epoch_s`) whose time-to-node counts down as the clock advances; new buttons, an orbit marker, a pending-node readout, auto-warp-down, and execute-gating. The node mirrors into `vessel.nodes` so save/load persists it for free.

**Tech Stack:** Python 3, numpy, Panda3D DirectGUI. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- Core (`orbitsim/core/`) imports no render/panda3d; SI units, float64. (verbatim from project layering)
- `time_to_periapsis`/`time_to_apoapsis` are for **bound** orbits only — raise `ValueError` when `e ≥ 1` or `a ≤ 0`. (verbatim from spec)
- Both timing helpers return a value in `[0, period)`. (verbatim from spec)
- Node epoch is stored **absolutely** and held fixed once placed; time-to-node = `node_epoch_s − clock.sim_time_s` counts down. NEVER recompute the epoch as `now + ttn` each frame. (verbatim from spec — the key correctness point)
- Execution: impulsive apply + fuel spend (reuse existing `_execute_burn` path); enabled only when no node is scheduled OR `time_to_node ≤ EXECUTE_TOLERANCE_S`. Auto-warp-down never warps up. (verbatim from spec)
- Constants: `AUTO_WARP_LEAD_S = 5.0`, `EXECUTE_TOLERANCE_S = 2.0`, `NODE_TIME_STEP_S = 30.0`.
- Run tests with `.venv/Scripts/python -m pytest`. (verbatim)
- Commits: explicit paths only; end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. (verbatim from repo git discipline)

---

## File Structure

- `orbitsim/core/maneuvers.py` — add `time_to_periapsis(state)` and `time_to_apoapsis(state)`.
- `tests/core/test_maneuvers.py` — add their known-answer + invariant + error tests.
- `orbitsim/render/app.py` — scheduled-node state, timing/preset/clear buttons, node marker, pending readout, auto-warp-down, execute-gating, `vessel.nodes` mirror (all within the existing maneuver-UI region).

Reference shapes (existing):
- `KeplerianElements` (`core/elements.py`): `.a .e .i .raan .argp .nu .mu`, property `.period_s` (raises `ValueError` if `a ≤ 0`).
- `core/kepler.py`: `true_to_eccentric_anomaly(nu, e) -> float`, `eccentric_to_mean_anomaly(E, e) -> float`.
- `core/elements.py`: `state_to_elements(state) -> KeplerianElements`, `elements_to_state(elements) -> StateVector`.
- `core/propagate.py`: `propagate_kepler(state, dt_s) -> StateVector` (used in the render task for the marker; no change).
- `core/maneuvers.py`: `ManeuverNode(epoch_s, dv_prograde_mps, dv_normal_mps, dv_radial_mps)`, `apply_maneuver(state, node)`, `predict_elements_after(state, node)`.
- App maneuver UI (`render/app.py`): `_build_maneuver_ui` (~273), `_current_node` (~360, currently `epoch_s=current`), `_refresh_readout` (~374), `_execute_burn` (~382), and the live preview in the update loop (~662).

---

## Task 1: Core `time_to_periapsis` / `time_to_apoapsis`

**Files:**
- Modify: `orbitsim/core/maneuvers.py`
- Test: `tests/core/test_maneuvers.py`

**Interfaces:**
- Produces: `time_to_periapsis(state: StateVector) -> float`, `time_to_apoapsis(state: StateVector) -> float` — seconds until the vessel next passes ν=0 (resp. ν=π); both in `[0, period)`; raise `ValueError` for unbound orbits.

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_maneuvers.py` (append; reuse existing imports of numpy and add what's needed):

```python
import pytest
from orbitsim.core.elements import KeplerianElements, elements_to_state
from orbitsim.core.maneuvers import time_to_periapsis, time_to_apoapsis
from orbitsim.core.constants import MU_EARTH


def _state_at_nu(nu):
    elem = KeplerianElements(a=7.0e6, e=0.2, i=0.5, raan=0.3, argp=0.4, nu=nu, mu=MU_EARTH)
    return elements_to_state(elem)


def _period():
    return KeplerianElements(a=7.0e6, e=0.2, i=0.5, raan=0.3, argp=0.4, nu=0.0, mu=MU_EARTH).period_s


def test_time_to_periapsis_from_apoapsis_is_half_period():
    T = _period()
    assert abs(time_to_periapsis(_state_at_nu(np.pi)) - T / 2.0) < 1e-3


def test_time_to_apoapsis_at_apoapsis_is_zero():
    assert time_to_apoapsis(_state_at_nu(np.pi)) < 1e-3


def test_time_to_periapsis_at_periapsis_is_zero():
    assert time_to_periapsis(_state_at_nu(0.0)) < 1e-3


def test_time_to_apoapsis_from_periapsis_is_half_period():
    T = _period()
    assert abs(time_to_apoapsis(_state_at_nu(0.0)) - T / 2.0) < 1e-3


def test_timing_within_one_period_for_arbitrary_nu():
    T = _period()
    for nu in (0.1, 1.0, 2.5, 4.0, 6.0):
        for f in (time_to_periapsis, time_to_apoapsis):
            t = f(_state_at_nu(nu))
            assert 0.0 <= t < T


def test_timing_raises_on_hyperbolic():
    from orbitsim.core.state import StateVector
    hyp = StateVector(r=np.array([7.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 12000.0, 0.0]), mu=MU_EARTH)  # > escape -> e>1
    with pytest.raises(ValueError):
        time_to_periapsis(hyp)
    with pytest.raises(ValueError):
        time_to_apoapsis(hyp)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_maneuvers.py -q -k "timing or periapsis or apoapsis"`
Expected: FAIL — `ImportError: cannot import name 'time_to_periapsis'`.

- [ ] **Step 3: Implement the helpers**

Add to `orbitsim/core/maneuvers.py` (it already imports `state_to_elements`; add the kepler imports):

```python
from orbitsim.core.kepler import true_to_eccentric_anomaly, eccentric_to_mean_anomaly


def _time_to_mean_anomaly(state: StateVector, target_M: float) -> float:
    """Seconds until the vessel's mean anomaly next reaches target_M (bound orbits only)."""
    elem = state_to_elements(state)
    if elem.e >= 1.0 or elem.a <= 0.0:
        raise ValueError("time-to-apsis is defined only for bound (elliptical) orbits")
    E = true_to_eccentric_anomaly(elem.nu, elem.e)
    M = eccentric_to_mean_anomaly(E, elem.e)
    n = 2.0 * np.pi / elem.period_s  # mean motion [rad/s]
    dM = (target_M - M) % (2.0 * np.pi)  # forward angle to the target, in [0, 2π)
    return dM / n


def time_to_periapsis(state: StateVector) -> float:
    """Seconds until the vessel next passes periapsis (ν=0). Bound orbits only."""
    return _time_to_mean_anomaly(state, 2.0 * np.pi)


def time_to_apoapsis(state: StateVector) -> float:
    """Seconds until the vessel next passes apoapsis (ν=π). Bound orbits only."""
    return _time_to_mean_anomaly(state, np.pi)
```

Note: for periapsis the target mean anomaly is `2π` (≡ 0 forward); `(2π − M) % 2π` gives 0 when `M = 0` (already at periapsis), matching the known-answer test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_maneuvers.py -q`
Expected: PASS (existing + 6 new).

- [ ] **Step 5: Add a hypothesis invariant test**

Append:

```python
from hypothesis import given, strategies as st


@given(nu=st.floats(min_value=0.0, max_value=2.0 * np.pi, exclude_max=True),
       e=st.floats(min_value=0.0, max_value=0.9))
def test_timing_invariant_zero_to_period(nu, e):
    elem = KeplerianElements(a=8.0e6, e=e, i=0.4, raan=0.2, argp=0.3, nu=nu, mu=MU_EARTH)
    state = elements_to_state(elem)
    T = elem.period_s
    assert 0.0 <= time_to_periapsis(state) < T + 1e-6
    assert 0.0 <= time_to_apoapsis(state) < T + 1e-6
```

Run: `.venv/Scripts/python -m pytest tests/core/test_maneuvers.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/core/maneuvers.py tests/core/test_maneuvers.py
git commit -m "$(cat <<'EOF'
Maneuvers: time_to_periapsis / time_to_apoapsis (bound orbits)

Seconds until the next apsis passage via mean-anomaly time-of-flight;
powers the Next Pe/Ap node presets. Raises on unbound orbits.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Render maneuver planner (scheduled node, presets, marker, auto-warp-down, execute gate)

**Files:**
- Modify: `orbitsim/render/app.py` (maneuver-UI region + sandbox update loop)

**Interfaces:**
- Consumes: `time_to_periapsis`, `time_to_apoapsis` (Task 1); existing `apply_maneuver`, `predict_elements_after`, `propagate_kepler`, `fuel_burned_for_dv`, `clock.warp_down`.
- Produces (app methods/state): `self._node_epoch_s: float | None`; `_step_node_time(delta_s)`, `_node_to_pe()`, `_node_to_ap()`, `_clear_node()`, `_time_to_node() -> float | None`; modified `_current_node`, `_execute_burn`, and update-loop maneuver block.

- [ ] **Step 1: Add scheduled-node state + buttons in `_build_maneuver_ui`**

Add `self._node_epoch_s = None` and `self._node_marker_np = None` near the top of `_build_maneuver_ui`. Add a row of buttons (place under the Execute button, in `self.a2dBottomRight`) and a time-to-node readout:

```python
        self._node_ttn_text = OnscreenText(
            text="", pos=(0.08, -0.42), scale=0.045, fg=(0.4, 1.0, 1.0, 1),
            shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True, parent=self.a2dTopLeft,
        )
        node_btns = [
            ("Node -", lambda: self._step_node_time(-self.NODE_TIME_STEP_S)),
            ("Node +", lambda: self._step_node_time(self.NODE_TIME_STEP_S)),
            ("Next Pe", self._node_to_pe),
            ("Next Ap", self._node_to_ap),
            ("Clear", self._clear_node),
        ]
        for i, (label, cmd) in enumerate(node_btns):
            DirectButton(text=label, scale=0.045, pos=(-0.95 + i * 0.34, 0.0, -0.06),
                         command=cmd, parent=self.a2dBottomRight)
```

Add the class constants near the other maneuver constants:

```python
    NODE_TIME_STEP_S = 30.0
    AUTO_WARP_LEAD_S = 5.0
    EXECUTE_TOLERANCE_S = 2.0
```

- [ ] **Step 2: Add the node-timing methods**

```python
    def _time_to_node(self):
        """Seconds until the scheduled node, or None if no node is scheduled."""
        if self._node_epoch_s is None:
            return None
        return self._node_epoch_s - self.clock.sim_time_s

    def _step_node_time(self, delta_s):
        """Nudge the node epoch by delta_s (creating one at now+delta if none), clamped >= now."""
        now = self.clock.sim_time_s
        base = self._node_epoch_s if self._node_epoch_s is not None else now
        self._node_epoch_s = max(now, base + delta_s)

    def _node_to_pe(self):
        from orbitsim.core.maneuvers import time_to_periapsis
        try:
            self._node_epoch_s = self.clock.sim_time_s + time_to_periapsis(self.world.vessels[0].state)
        except ValueError:
            pass  # unbound orbit: no apsis to target

    def _node_to_ap(self):
        from orbitsim.core.maneuvers import time_to_apoapsis
        try:
            self._node_epoch_s = self.clock.sim_time_s + time_to_apoapsis(self.world.vessels[0].state)
        except ValueError:
            pass

    def _clear_node(self):
        self._node_epoch_s = None
        if self._node_marker_np is not None:
            self._node_marker_np.remove_node()
            self._node_marker_np = None
```

- [ ] **Step 3: Point `_current_node` at the scheduled epoch**

Change `_current_node` to use the scheduled epoch when set (fall back to the current epoch — preserves immediate-burn behavior when no node is scheduled):

```python
    def _current_node(self) -> ManeuverNode:
        epoch = self._node_epoch_s if self._node_epoch_s is not None else self.world.vessels[0].state.epoch_s
        return ManeuverNode(
            epoch_s=epoch,
            dv_prograde_mps=self._dv["pro"],
            dv_normal_mps=self._dv["nrm"],
            dv_radial_mps=self._dv["rad"],
        )
```

- [ ] **Step 4: Gate `_execute_burn` and clear the node after executing**

Wrap the existing apply logic so it only fires when due, then clears the node:

```python
    def _execute_burn(self) -> None:
        from orbitsim.core.flight import fuel_burned_for_dv

        ttn = self._time_to_node()
        if ttn is not None and ttn > self.EXECUTE_TOLERANCE_S:
            return  # scheduled node not due yet
        v0 = self.world.vessels[0]
        node = self._current_node()
        dv = node.magnitude_mps
        if 0.0 < dv <= v0.delta_v_remaining:
            v0.state = apply_maneuver(v0.state, node)
            burned = fuel_burned_for_dv(v0.exhaust_velocity_mps, v0.mass_kg, dv)
            v0.fuel_mass_kg = max(0.0, v0.fuel_mass_kg - burned)
        self._clear_node()
        for axis in self._dv:
            self._dv[axis] = 0.0
            self._dv_value_text[axis].setText("+0")
        self._release_jogs()
        self._refresh_readout()
```

Note: `apply_maneuver` propagates to `node.epoch_s` before applying — when executed within tolerance, `node.epoch_s ≈ current epoch`, so the impulse lands at the vessel's actual position.

- [ ] **Step 5: Update-loop block — marker, auto-warp-down, readout, `vessel.nodes` mirror**

In the sandbox update loop where the magenta preview is built (~662), extend the maneuver block to also: mirror the node into `vessel.nodes` (for persistence), draw the node marker, run auto-warp-down, and update the readout. Replace/extend the existing preview block with:

```python
        node = self._current_node()
        v0 = self.world.vessels[0]
        ttn = self._time_to_node()
        # Persist the planned node (single-node list) so quicksave restores it.
        v0.nodes = [node] if (self._node_epoch_s is not None or node.magnitude_mps > 0.0) else []
        # Preview (magenta) of the post-burn orbit.
        if node.magnitude_mps > 0.0:
            pred = predict_elements_after(v0.state, node)
            ppts = [self.transform.to_render(p) for p in sample_orbit_points(pred, n=256)]
            if self._preview_np is not None:
                self._preview_np.remove_node()
            self._preview_np = build_orbit_node(ppts, color=(1.0, 0.2, 1.0, 1.0))
            self._preview_np.reparent_to(self.render)
        elif self._preview_np is not None:
            self._preview_np.remove_node()
            self._preview_np = None
        # Node marker (cyan) at the node's predicted position.
        if self._node_epoch_s is not None and ttn is not None and ttn >= 0.0:
            from orbitsim.core.propagate import propagate_kepler
            npos = propagate_kepler(v0.state, ttn).r
            mx, my, mz = self.transform.to_render(npos)
            if self._node_marker_np is None:
                self._node_marker_np = make_uv_sphere(1.0, 8, 12)
                self._node_marker_np.reparent_to(self.render)
                self._node_marker_np.set_color(0.3, 1.0, 1.0, 1.0)
                self._node_marker_np.set_light_off()
                self._node_marker_np.set_scale(6.0)
            self._node_marker_np.set_pos(mx, my, mz)
        elif self._node_marker_np is not None:
            self._node_marker_np.remove_node()
            self._node_marker_np = None
        # Auto-warp-down as the node nears (never warps up).
        if ttn is not None and 0.0 < ttn <= self.AUTO_WARP_LEAD_S * self.clock.warp and self.clock.warp > 1.0:
            self.clock.warp_down()
        # Pending-node readout.
        if ttn is not None and ttn >= 0.0:
            mm, ss = divmod(int(ttn), 60)
            self._node_ttn_text.setText(f"Node in T-{mm:02d}:{ss:02d}   dV {node.magnitude_mps:,.1f} m/s")
        else:
            self._node_ttn_text.setText("")
```

(Remove the old standalone preview block this replaces.)

- [ ] **Step 6: Headless smoke test**

```bash
PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
import numpy as np
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.core.maneuvers import time_to_periapsis

app = OrbitApp(_default_world(), SimClock(0.0, 1000.0), solar_system=False)
app._start_sim()
app.taskMgr.step()
v = app.world.vessels[0]

# Place a node via Next Pe; epoch should be ~ now + time_to_periapsis.
app._node_to_pe()
exp = app.clock.sim_time_s + time_to_periapsis(v.state)
assert abs(app._node_epoch_s - exp) < 1.0, (app._node_epoch_s, exp)

# Edit dV and confirm the planned node + persistence mirror.
app._dv["pro"] = 50.0
app.taskMgr.step()
assert len(v.nodes) == 1 and abs(v.nodes[0].dv_prograde_mps - 50.0) < 1e-9

# Not due yet -> execute is a no-op (node stays).
fuel0 = v.fuel_mass_kg
app._execute_burn()
assert app._node_epoch_s is not None and v.fuel_mass_kg == fuel0

# Jump time to the node, execute -> burn applied, node cleared, fuel spent.
app.clock.sim_time_s = app._node_epoch_s
app._dv["pro"] = 50.0
app._execute_burn()
assert app._node_epoch_s is None and v.fuel_mass_kg < fuel0
print("OK: scheduled node place / persist / gated-execute")
PY
```

Expected: `OK: scheduled node place / persist / gated-execute`.

- [ ] **Step 7: Auto-warp-down headless check**

```bash
PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app._start_sim()
app.taskMgr.step()
# Schedule a node 3 s ahead at high warp; the update loop should step warp down.
app._node_epoch_s = app.clock.sim_time_s + 3.0
app.clock.warp = 100.0
for _ in range(12):
    app.taskMgr.step()
assert app.clock.warp < 100.0, app.clock.warp
print("OK: auto-warp-down engaged, warp =", app.clock.warp)
PY
```

Expected: prints a reduced warp.

- [ ] **Step 8: Visual screenshot + full suite**

Capture a sandbox screenshot with a scheduled node (marker + preview + readout visible) and eyeball it; then run the full suite:

```bash
.venv/Scripts/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
Maneuver: scheduled node planner (time-to-node, Pe/Ap presets)

Place a single node ahead on the orbit (manual step or Next Pe/Ap), see a
cyan marker + magenta preview + T-MM:SS readout, auto-warp-down as it
nears, and execute the impulsive burn at the node. Node mirrors into
vessel.nodes so quicksave persists it.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Execution = auto-warp-down + manual execute (impulsive, fuel) → Task 2 Steps 4, 5. ✓
- Single node, fixed absolute epoch, counting-down ttn → `_node_epoch_s` + `_time_to_node` (Task 2). ✓
- Time-to-node control + Next Pe/Ap presets → Task 2 Steps 1, 2; core helpers Task 1. ✓
- Node marker on orbit + magenta preview at future epoch → Task 2 Step 5. ✓
- Pending-node readout (single entry) → Task 2 Step 5. ✓
- Auto-warp-down (never up), lead 5 s → Task 2 Step 5 + constant. ✓
- Execute gated by EXECUTE_TOLERANCE_S, clears node → Task 2 Step 4. ✓
- Persists via vessel.nodes → Task 2 Step 5 mirror; verified in Step 6 smoke. ✓
- Core helpers bound-only, [0,period), ValueError unbound, known-answer + invariant tests → Task 1. ✓

**Placeholder scan:** No TBD/TODO; all code shown; each headless check has a concrete script + expected output.

**Type consistency:** `time_to_periapsis(state)`/`time_to_apoapsis(state)` defined in Task 1, consumed in Task 2 (`_node_to_pe`/`_node_to_ap`). `_node_epoch_s` (float|None), `_time_to_node()`, `_current_node()`, `_clear_node()` consistent across Task 2 steps. Constants `NODE_TIME_STEP_S`/`AUTO_WARP_LEAD_S`/`EXECUTE_TOLERANCE_S` defined once (Step 1) and used in Steps 2/4/5.
