# Phase 3 — Sandbox & Maneuver Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user add a maneuver node on the orbit, dial in a delta-V (prograde/normal/radial), and see the predicted resulting orbit live before committing the burn — the KSP-style sandbox core.

**Architecture:** All burn math is pure functions in `core/maneuvers.py` (impulsive delta-V in the vessel's local RTN frame, fully unit-tested). The renderer draws a second orbit line for the preview and adds DirectGUI sliders; those parts are verified visually.

**Tech Stack:** Python 3.10, numpy, Panda3D, pytest.

## Global Constraints

- SI units in `core/`/`sim/`: meters, seconds, radians, m/s. Convert to km only at the HUD boundary.
- `core/` must NOT import `panda3d`/`sim`/`render`.
- `ManeuverNode`, `StateVector`, `KeplerianElements` are frozen dataclasses.
- `black` line length 100. Type hints + NumPy docstrings everywhere.
- `pytest tests/ -q` green after every task; render tasks end with a HUMAN VISUAL CHECKPOINT.

## Gate

Phase 2 renders one orbit with a working camera/clock. Do not start until that visual checkpoint passed.

## Phase 1/2 API available

```python
from orbitsim.core.state import StateVector       # frozen; .r_mag .v_mag .specific_energy .angular_momentum
from orbitsim.core.elements import (
    KeplerianElements, state_to_elements, elements_to_state,
)
from orbitsim.core.propagate import propagate_kepler   # (state, dt) -> StateVector
from orbitsim.core.constants import MU_EARTH
from orbitsim.sim.world import Vessel, World           # Vessel.state mutable; World.step(dt)
```

---

## File Structure

- `orbitsim/core/maneuvers.py` — CREATE: `ManeuverNode`, `apply_maneuver`, `predict_elements_after`.
- `orbitsim/render/app.py` — MODIFY: draw preview orbit line; add node create + slider UI; execute logic.
- `orbitsim/sim/world.py` — MODIFY: optional `Vessel.nodes: list[ManeuverNode]` field for the sandbox.
- Tests: `tests/core/test_maneuvers.py`.

---

## Task 1: ManeuverNode + apply_maneuver

**Files:**
- Create: `orbitsim/core/maneuvers.py`
- Test: `tests/core/test_maneuvers.py`

**Interfaces:**
- Consumes: `StateVector`, `propagate_kepler`, numpy.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ManeuverNode:
      epoch_s: float
      dv_prograde_mps: float
      dv_normal_mps: float
      dv_radial_mps: float
      @property
      def magnitude_mps(self) -> float:  # sqrt(p^2 + n^2 + r^2)
  def apply_maneuver(state: StateVector, node: ManeuverNode) -> StateVector:
      # propagate state to node.epoch_s, add dv in the local RTN basis, return new state
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_maneuvers.py`:
```python
"""Tests for impulsive maneuvers in the local RTN frame."""
import numpy as np
from orbitsim.core.maneuvers import ManeuverNode, apply_maneuver
from orbitsim.core.state import StateVector
from orbitsim.core.elements import state_to_elements
from orbitsim.core.constants import MU_EARTH


def _periapsis_state() -> StateVector:
    """Elliptical orbit positioned at periapsis (r along +x, v along +y)."""
    rp = 7.0e6
    a = 8.0e6
    # vis-viva at periapsis
    v = np.sqrt(MU_EARTH * (2.0 / rp - 1.0 / a))
    return StateVector(r=np.array([rp, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)


def test_node_magnitude():
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=3.0, dv_normal_mps=4.0, dv_radial_mps=0.0)
    assert abs(node.magnitude_mps - 5.0) < 1e-12


def test_prograde_burn_raises_apoapsis_only():
    state = _periapsis_state()
    elem0 = state_to_elements(state)
    ra0 = elem0.a * (1 + elem0.e)
    rp0 = elem0.a * (1 - elem0.e)

    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=100.0, dv_normal_mps=0.0, dv_radial_mps=0.0)
    new_state = apply_maneuver(state, node)
    elem1 = state_to_elements(new_state)
    ra1 = elem1.a * (1 + elem1.e)
    rp1 = elem1.a * (1 - elem1.e)

    assert ra1 > ra0                      # apoapsis raised
    assert abs(rp1 - rp0) < 1.0           # periapsis unchanged (< 1 m)


def test_normal_burn_changes_inclination_and_adds_energy_in_quadrature():
    state = _periapsis_state()
    elem0 = state_to_elements(state)
    dv = 50.0
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=0.0, dv_normal_mps=dv, dv_radial_mps=0.0)
    new_state = apply_maneuver(state, node)
    elem1 = state_to_elements(new_state)
    # A pure normal impulse is perpendicular to v, so speed adds in quadrature *exactly*:
    # |v_new|^2 = |v_old|^2 + dv^2. This DOES raise energy (only a speed-preserving rotation
    # of v leaves energy fixed). Here delta-a/a ~ 5e-5 — small but real, NOT zero.
    # (The original plan asserted delta-a/a < 1e-6, which is physically impossible.)
    assert abs(new_state.v_mag**2 - (state.v_mag**2 + dv**2)) < 1e-3
    assert elem1.i > elem0.i + 1e-6                          # inclination changed (main effect)
    assert 0.0 < abs(elem1.a - elem0.a) / elem0.a < 1e-3    # energy rises slightly, not zero


def test_total_dv_added_matches_magnitude():
    state = _periapsis_state()
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=30.0, dv_normal_mps=40.0, dv_radial_mps=0.0)
    new_state = apply_maneuver(state, node)
    dv_vec = new_state.v - state.v
    assert abs(np.linalg.norm(dv_vec) - node.magnitude_mps) < 1e-6


def test_node_epoch_propagates_before_burn():
    state = _periapsis_state()
    period = state_to_elements(state).period_s
    node = ManeuverNode(
        epoch_s=period / 2.0, dv_prograde_mps=10.0, dv_normal_mps=0.0, dv_radial_mps=0.0
    )
    new_state = apply_maneuver(state, node)
    # Burn happens at apoapsis (half a period later): position is on the far side.
    assert new_state.r[0] < 0
    assert abs(new_state.epoch_s - (state.epoch_s + period / 2.0)) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_maneuvers.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.core.maneuvers).

- [ ] **Step 3: Implement maneuvers.py**

Create `orbitsim/core/maneuvers.py`:
```python
"""Impulsive delta-V maneuvers in the vessel's local orbital (RTN/LVLH) frame."""
from dataclasses import dataclass
import numpy as np

from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler


@dataclass(frozen=True)
class ManeuverNode:
    """An impulsive burn defined in the local RTN frame at a given epoch.

    Attributes
    ----------
    epoch_s : float
        When the burn happens [s past J2000 TDB].
    dv_prograde_mps : float
        Component along velocity (+ speeds up) [m/s].
    dv_normal_mps : float
        Component along orbital angular momentum h [m/s].
    dv_radial_mps : float
        Component along the radial-out RTN axis [m/s].
    """

    epoch_s: float
    dv_prograde_mps: float
    dv_normal_mps: float
    dv_radial_mps: float

    @property
    def magnitude_mps(self) -> float:
        """Total delta-V magnitude [m/s]."""
        return float(
            np.sqrt(
                self.dv_prograde_mps**2
                + self.dv_normal_mps**2
                + self.dv_radial_mps**2
            )
        )


def apply_maneuver(state: StateVector, node: ManeuverNode) -> StateVector:
    """Propagate to the node epoch and apply the impulsive burn.

    Parameters
    ----------
    state : StateVector
        Current state (its epoch_s is the start time).
    node : ManeuverNode

    Returns
    -------
    StateVector
        Post-burn state at node.epoch_s (same position, new velocity).

    Notes
    -----
    Local RTN basis at the burn point:
        v_hat = v / |v|                  (prograde)
        h_hat = (r x v) / |r x v|        (orbit-normal)
        r_hat = h_hat x v_hat            (radial-out, orthonormal — NOT r/|r|)
    """
    dt = node.epoch_s - state.epoch_s
    burn_state = propagate_kepler(state, dt)

    r = burn_state.r
    v = burn_state.v
    v_hat = v / np.linalg.norm(v)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    r_hat = np.cross(h_hat, v_hat)

    dv = (
        node.dv_prograde_mps * v_hat
        + node.dv_normal_mps * h_hat
        + node.dv_radial_mps * r_hat
    )
    new_v = v + dv

    return StateVector(
        r=np.array(r, dtype=np.float64),
        v=np.array(new_v, dtype=np.float64),
        mu=burn_state.mu,
        epoch_s=burn_state.epoch_s,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_maneuvers.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/maneuvers.py tests/core/test_maneuvers.py
git commit -m "Phase 3 Task 1: ManeuverNode + apply_maneuver (RTN frame)"
```

---

## Task 2: Predicted-orbit elements after a maneuver

**Files:**
- Modify: `orbitsim/core/maneuvers.py` (add `predict_elements_after`)
- Test: `tests/core/test_maneuvers.py` (append)

**Interfaces:**
- Consumes: `apply_maneuver`, `state_to_elements`.
- Produces:
  ```python
  def predict_elements_after(state: StateVector, node: ManeuverNode) -> KeplerianElements:
      # apply_maneuver then state_to_elements; the renderer samples this for the preview line
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_maneuvers.py`:
```python
def test_predict_elements_after_matches_apply_then_convert():
    from orbitsim.core.maneuvers import predict_elements_after
    from orbitsim.core.elements import state_to_elements
    state = _periapsis_state()
    node = ManeuverNode(epoch_s=0.0, dv_prograde_mps=120.0, dv_normal_mps=0.0, dv_radial_mps=0.0)
    predicted = predict_elements_after(state, node)
    expected = state_to_elements(apply_maneuver(state, node))
    assert abs(predicted.a - expected.a) < 1.0
    assert abs(predicted.e - expected.e) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_maneuvers.py::test_predict_elements_after_matches_apply_then_convert -q`
Expected: FAIL (ImportError: cannot import name 'predict_elements_after').

- [ ] **Step 3: Implement predict_elements_after**

Append to `orbitsim/core/maneuvers.py`:
```python
from orbitsim.core.elements import KeplerianElements, state_to_elements


def predict_elements_after(state: StateVector, node: ManeuverNode) -> KeplerianElements:
    """Return the Keplerian elements that result from applying `node` to `state`."""
    return state_to_elements(apply_maneuver(state, node))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_maneuvers.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/maneuvers.py tests/core/test_maneuvers.py
git commit -m "Phase 3 Task 2: predict_elements_after for orbit preview"
```

---

## Task 3: Vessel maneuver-node list (sim layer)

**Files:**
- Modify: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py` (append)

**Interfaces:**
- Consumes: `ManeuverNode`.
- Produces: `Vessel.nodes: list[ManeuverNode]` (default empty), unchanged `World.step`.

- [ ] **Step 1: Write the failing test**

Append to `tests/sim/test_world.py`:
```python
def test_vessel_has_node_list():
    from orbitsim.core.maneuvers import ManeuverNode
    vessel = _circular_vessel()
    assert vessel.nodes == []
    node = ManeuverNode(epoch_s=10.0, dv_prograde_mps=5.0, dv_normal_mps=0.0, dv_radial_mps=0.0)
    vessel.nodes.append(node)
    assert vessel.nodes[0].dv_prograde_mps == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py::test_vessel_has_node_list -q`
Expected: FAIL (AttributeError: 'Vessel' object has no attribute 'nodes').

- [ ] **Step 3: Add the field**

In `orbitsim/sim/world.py`, modify the `Vessel` dataclass. Change its definition to:
```python
from dataclasses import dataclass, field
from orbitsim.core.maneuvers import ManeuverNode


@dataclass
class Vessel:
    name: str
    state: StateVector
    delta_v_budget_mps: float = 0.0
    nodes: list[ManeuverNode] = field(default_factory=list)
```
(Keep the existing imports; add `field` to the dataclass import and the `ManeuverNode` import at the top.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "Phase 3 Task 3: Vessel.nodes list for sandbox maneuvers"
```

---

## Task 4: Preview orbit line + node UI + execute (render) — HUMAN VISUAL CHECKPOINT

These steps build interactive DirectGUI and a second orbit line; verified visually.

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `ManeuverNode`, `predict_elements_after`, `apply_maneuver`, `sample_orbit_points`, `build_orbit_node`.
- Produces: an in-app maneuver editor: a node on vessel 0, three sliders (prograde/normal/radial) + a node-time slider, a live magenta preview orbit, an Execute button, and a delta-V-cost readout.

- [ ] **Step 1: Add maneuver state + sliders to OrbitApp**

In `orbitsim/render/app.py`, add imports near the top:
```python
from direct.gui.DirectSlider import DirectSlider
from direct.gui.DirectButton import DirectButton
from orbitsim.core.maneuvers import ManeuverNode, predict_elements_after, apply_maneuver
```

In `OrbitApp.__init__`, after the HUD is created, add:
```python
        # Maneuver editor state (operates on vessel 0).
        self._node = ManeuverNode(
            epoch_s=self.clock.sim_time_s + 600.0,  # 10 min from now
            dv_prograde_mps=0.0, dv_normal_mps=0.0, dv_radial_mps=0.0,
        )
        self._preview_np = None
        self._build_maneuver_ui()
```

- [ ] **Step 2: Add the UI builder + callbacks**

Add these methods to `OrbitApp`:
```python
    def _build_maneuver_ui(self) -> None:
        def mk_slider(y, command):
            return DirectSlider(
                pos=(0.0, 0.0, y), scale=0.4, range=(-200.0, 200.0), value=0.0,
                pageSize=10.0, command=command,
                parent=self.a2dBottomRight if hasattr(self, "a2dBottomRight") else None,
            )
        self._s_pro = mk_slider(0.30, self._on_slider)
        self._s_nrm = mk_slider(0.18, self._on_slider)
        self._s_rad = mk_slider(0.06, self._on_slider)
        self._exec_btn = DirectButton(
            text="Execute Burn", scale=0.05, pos=(-0.5, 0.0, -0.9),
            command=self._execute_burn,
        )

    def _on_slider(self) -> None:
        self._node = ManeuverNode(
            epoch_s=self._node.epoch_s,
            dv_prograde_mps=self._s_pro["value"],
            dv_normal_mps=self._s_nrm["value"],
            dv_radial_mps=self._s_rad["value"],
        )

    def _execute_burn(self) -> None:
        v0 = self.world.vessels[0]
        if self._node.magnitude_mps <= v0.delta_v_budget_mps:
            v0.state = apply_maneuver(v0.state, self._node)
            v0.delta_v_budget_mps -= self._node.magnitude_mps
        # Reset the node ahead of the (new) current time.
        self._node = ManeuverNode(
            epoch_s=self.clock.sim_time_s + 600.0,
            dv_prograde_mps=0.0, dv_normal_mps=0.0, dv_radial_mps=0.0,
        )
        self._s_pro["value"] = 0.0
        self._s_nrm["value"] = 0.0
        self._s_rad["value"] = 0.0
```

- [ ] **Step 3: Draw the live preview orbit in `_update`**

In `OrbitApp._update`, just before `self.rig.apply()`, add:
```python
        # Live maneuver preview (magenta) for vessel 0.
        if self._node.magnitude_mps > 0.0:
            pred = predict_elements_after(self.world.vessels[0].state, self._node)
            ppts = sample_orbit_points(pred, n=256)
            ppts_render = [self.transform.to_render(p) for p in ppts]
            if self._preview_np is not None:
                self._preview_np.remove_node()
            self._preview_np = build_orbit_node(ppts_render, color=(1.0, 0.2, 1.0, 1.0))
            self._preview_np.reparent_to(self.render)
        elif self._preview_np is not None:
            self._preview_np.remove_node()
            self._preview_np = None
```

- [ ] **Step 4: Smoke-check imports**

Run: `.venv/Scripts/python -c "import orbitsim.render.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Full suite stays green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: HUMAN VISUAL CHECKPOINT**

Run: `.venv/Scripts/python -m orbitsim`

Reviewer confirms:
1. Dragging the prograde slider draws a magenta preview orbit whose apoapsis grows/shrinks live.
2. Dragging the normal slider tilts the preview orbit out of the current orbit plane.
3. Clicking "Execute Burn" makes the real (blue) orbit jump to match the magenta preview, and the HUD Pe/Ap/period update accordingly. The preview disappears (sliders reset to 0).
4. After execute, the predicted orbit and the actual orbit coincide within a line width (preview matched reality).

If the executed orbit does NOT match the preview, the bug is a mismatch between `apply_maneuver` timing and the preview's reference state — both must use vessel 0's current state and the same node. Fix before proceeding.

- [ ] **Step 7: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "Phase 3 Task 4: maneuver node UI, live preview orbit, execute"
```

---

## Phase 3 Exit Criteria

- User can add a node, see the predicted orbit update live, execute the burn, and watch the real orbit become the predicted one.
- All `core/maneuvers.py` tests green; preview matches post-burn reality within a line width (and < 1 m in the `predict_elements_after` test).
- `pytest tests/ -q` fully green.

Then proceed to `docs/superpowers/plans/2026-06-24-phase4-transfers.md`.

## Self-Review Notes

- Spec coverage: maneuvers.py (3.1), preview function (3.2), node editing UI + execute (3.3), sandbox wiring via Vessel.nodes + default scenario from Phase 2 (3.4) — all mapped.
- All burn physics is red-green TDD; only the DirectGUI/preview drawing is visual.
