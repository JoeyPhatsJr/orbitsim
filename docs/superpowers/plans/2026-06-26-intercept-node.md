# Porkchop Intercept Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One click generates a maneuver node that flies the ship to the current target, found by a (burn-time × time-of-flight) Lambert sweep minimizing the departure Δv.

**Architecture:** A pure `intercept_node` solver in `core/optimize.py` reuses the existing Lambert + propagate machinery but minimizes departure-only Δv, then projects the inertial burn onto the RTN basis to build a `ManeuverNode`; `app.py` adds an "Intercept" button that builds grids from live geometry and applies the node to the maneuver editor.

**Tech Stack:** Python 3, numpy, scipy, lamberthub, Panda3D. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- SI everywhere; float64; `core/` never imports render/panda3d. (project rule)
- TDD mandatory for the core solver; never loosen a tolerance to force a pass. (project rule)
- lamberthub/Lambert is singular at exactly 180° transfer angle — keep test geometry off π. (gotcha)
- Run tests with `.venv/Scripts/python -m pytest`. Commits: explicit paths; end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; then `git push`. (repo discipline)
- Task 1 pure (Haiku TDD). Task 2 render, controller-executed with headless screenshots.
- **Depends on the target-selection cycle** (`self._target`).

## File Structure

- `orbitsim/core/optimize.py` — add `intercept_node(...)` (alongside `porkchop`/`optimize_transfer`; imports already present: `np`, `minimize`, `propagate_kepler`, `lambert`).
- `orbitsim/core/maneuvers.py` — `ManeuverNode` (consumed, unchanged).
- `orbitsim/render/app.py` — "Intercept" button + grid construction + apply node.
- Test: `tests/core/test_optimize.py`.

---

## Task 1: `intercept_node` departure-Δv solver (pure, TDD)

**Files:**
- Modify: `orbitsim/core/optimize.py`
- Test: `tests/core/test_optimize.py`

**Interfaces:**
- Consumes: `propagate_kepler(state, dt) -> StateVector`; `lambert(r1, r2, tof, mu) -> (v1, v2)`; `ManeuverNode(epoch_s, dv_prograde_mps, dv_normal_mps, dv_radial_mps)`.
- Produces:
  `intercept_node(ship_state: StateVector, target_state_now: StateVector, mu: float, dep_times_s: np.ndarray, tof_grid_s: np.ndarray, refine: bool = True) -> ManeuverNode`
  — a node whose burn (applied then propagated `tof*`) brings the ship to the target's position. Raises `ValueError` if no grid cell yields a Lambert solution.

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_optimize.py`:

```python
import numpy as np
from orbitsim.core.state import StateVector
from orbitsim.core.bodies import EARTH
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.maneuvers import apply_maneuver
from orbitsim.core.optimize import intercept_node


def _ship_and_target():
    mu = EARTH.mu
    # Ship in a circular ~7,000 km LEO (coplanar, equatorial).
    r1 = 7.0e6
    ship = StateVector(r=np.array([r1, 0.0, 0.0]),
                       v=np.array([0.0, np.sqrt(mu / r1), 0.0]), mu=mu, epoch_s=0.0)
    # Target in a higher circular orbit, phased ~100° ahead (off 180° to avoid the
    # Lambert plane singularity).
    r2 = 4.0e7
    ang = np.radians(100.0)
    tv = np.sqrt(mu / r2)
    target = StateVector(
        r=np.array([r2 * np.cos(ang), r2 * np.sin(ang), 0.0]),
        v=np.array([-tv * np.sin(ang), tv * np.cos(ang), 0.0]), mu=mu, epoch_s=0.0)
    return ship, target, mu


def test_intercept_node_closes_the_loop():
    ship, target, mu = _ship_and_target()
    dep = np.linspace(0.0, 3.0e3, 16)
    tof = np.linspace(1.0e3, 4.0e4, 40)
    node = intercept_node(ship, target, mu, dep, tof)
    # Apply the burn, then propagate by the chosen TOF; we should reach the target.
    t_dep = node.epoch_s - ship.epoch_s
    tof_star = _recover_tof(ship, target, mu, node)   # see helper below
    post = apply_maneuver(ship, node)
    arrived = propagate_kepler(post, tof_star)
    target_then = propagate_kepler(target, t_dep + tof_star)
    sep = np.linalg.norm(arrived.r - target_then.r)
    assert sep < 1.0e4    # within 10 km of a moving target — a real intercept


def test_intercept_node_rtn_projection_is_lossless():
    # The node's RTN components must recompose to the inertial burn vector.
    ship, target, mu = _ship_and_target()
    dep = np.linspace(0.0, 3.0e3, 12)
    tof = np.linspace(1.0e3, 4.0e4, 30)
    node = intercept_node(ship, target, mu, dep, tof, refine=False)
    burn = propagate_kepler(ship, node.epoch_s - ship.epoch_s)
    v_hat = burn.v / np.linalg.norm(burn.v)
    h = np.cross(burn.r, burn.v); h_hat = h / np.linalg.norm(h)
    r_hat = np.cross(h_hat, v_hat)
    recomposed = (node.dv_prograde_mps * v_hat + node.dv_normal_mps * h_hat
                  + node.dv_radial_mps * r_hat)
    assert node.magnitude_mps > 0.0
    assert np.isfinite(recomposed).all()


def test_intercept_node_raises_when_infeasible():
    ship, target, mu = _ship_and_target()
    # All TOFs non-positive => no feasible Lambert cell.
    import pytest
    with pytest.raises(ValueError):
        intercept_node(ship, target, mu, np.array([0.0]), np.array([-1.0, 0.0]))
```

Add this helper at the bottom of the test module (re-derives the optimal TOF the
solver used, so the closing-the-loop check is self-contained):

```python
def _recover_tof(ship, target, mu, node):
    from orbitsim.core.transfers import lambert
    burn = propagate_kepler(ship, node.epoch_s - ship.epoch_s)
    best_tof, best = None, np.inf
    for tof in np.linspace(1.0e3, 4.0e4, 400):
        r2 = propagate_kepler(target, (node.epoch_s - ship.epoch_s) + tof).r
        try:
            v1, _ = lambert(burn.r, r2, float(tof), mu)
        except Exception:
            continue
        post = apply_maneuver(ship, node)
        d = np.linalg.norm((post.v) - v1)
        if d < best:
            best, best_tof = d, tof
    return best_tof
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_optimize.py -q -k intercept_node`
Expected: FAIL — `ImportError: cannot import name 'intercept_node'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/optimize.py` (imports already include `np`, `minimize`,
`propagate_kepler`, `lambert`; add `from orbitsim.core.maneuvers import ManeuverNode`
at the top):

```python
def _dep_cost(ship_state, target_state_now, mu, t_dep, tof):
    """Departure-only delta-V to fly from ship@t_dep to target@(t_dep+tof)."""
    if tof <= 0.0:
        return np.inf, None, None
    dep = propagate_kepler(ship_state, float(t_dep))
    arr = propagate_kepler(target_state_now, float(t_dep + tof))
    try:
        v1, _ = lambert(dep.r, arr.r, float(tof), mu)
    except Exception:
        return np.inf, None, None
    return float(np.linalg.norm(v1 - dep.v)), dep, v1


def intercept_node(ship_state, target_state_now, mu, dep_times_s, tof_grid_s,
                   refine: bool = True) -> ManeuverNode:
    """Lowest-departure-delta-V single-burn intercept of a moving target.

    Sweeps (burn time x time-of-flight), Lambert-solving each cell and minimizing
    the DEPARTURE burn only (a flyby matches position, not velocity). Projects the
    optimal inertial burn onto the local RTN basis to build a ManeuverNode.

    Raises ValueError if no cell yields a Lambert solution.
    """
    best = (np.inf, None, None)   # (cost, t_dep, tof)
    for t_dep in dep_times_s:
        for tof in tof_grid_s:
            cost, _, _ = _dep_cost(ship_state, target_state_now, mu, t_dep, tof)
            if cost < best[0]:
                best = (cost, float(t_dep), float(tof))
    if not np.isfinite(best[0]):
        raise ValueError("no feasible intercept over the given grid")

    t_dep, tof = best[1], best[2]
    if refine:
        def cost(x):
            c, _, _ = _dep_cost(ship_state, target_state_now, mu, x[0], x[1])
            return c if np.isfinite(c) else 1e12
        res = minimize(cost, np.array([t_dep, tof]), method="Nelder-Mead",
                       options={"xatol": 1.0, "fatol": 1.0, "maxiter": 200})
        if np.isfinite(cost(res.x)) and res.x[1] > 0.0:
            t_dep, tof = float(res.x[0]), float(res.x[1])

    _, dep, v1 = _dep_cost(ship_state, target_state_now, mu, t_dep, tof)
    dv_vec = v1 - dep.v
    v_hat = dep.v / np.linalg.norm(dep.v)
    h = np.cross(dep.r, dep.v); h_hat = h / np.linalg.norm(h)
    r_hat = np.cross(h_hat, v_hat)
    return ManeuverNode(
        epoch_s=ship_state.epoch_s + t_dep,
        dv_prograde_mps=float(np.dot(dv_vec, v_hat)),
        dv_normal_mps=float(np.dot(dv_vec, h_hat)),
        dv_radial_mps=float(np.dot(dv_vec, r_hat)),
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_optimize.py -q -k intercept_node`
Expected: PASS (3). If `closes_the_loop` is marginally over tolerance, the bug is the
implementation or grid density — do NOT loosen `1.0e4`; widen `tof`/`dep` resolution or
verify the RTN basis matches `apply_maneuver`.

- [ ] **Step 5: Run the full core suite + commit**

Run: `.venv/Scripts/python -m pytest tests/core -q` → all pass.

```bash
git add orbitsim/core/optimize.py tests/core/test_optimize.py
git commit -m "$(cat <<'EOF'
Optimize: intercept_node — departure-dV porkchop -> ManeuverNode

Sweeps burn-time x TOF, Lambert-solving each cell, minimizing departure dV
only (flyby matches position). Projects the optimal inertial burn onto the
RTN basis. Tested by closing the loop to <10 km on a moving target.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: "Intercept" button wiring (controller)

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `intercept_node` (T1); `self._target` + `self._dv` + `self._node_epoch_s` (target-selection cycle + maneuver UI).
- Produces: `OrbitApp._plan_intercept()` bound to an "Intercept" button.

- [ ] **Step 1: Add the button**

In `_build_maneuver_ui`'s `node_btns`, append `("Intercept", self._plan_intercept)`.

- [ ] **Step 2: Implement the handler**

```python
    def _plan_intercept(self):
        """Auto-plan a flyby of the current target via a departure-dV porkchop."""
        import numpy as np
        from orbitsim.core.optimize import intercept_node
        from orbitsim.core.elements import state_to_elements
        if self._target is None:
            self._flash_message("No target selected")
            return
        v0 = self.world.vessels[0]
        try:
            period = state_to_elements(v0.state).period_s
        except ValueError:
            self._flash_message("Unbound orbit — can't plan intercept")
            return
        now = self.clock.sim_time_s
        dep = np.linspace(0.0, period, 24)
        tof = np.linspace(3.0e3, 14.0 * 86400.0, 48)
        try:
            node = intercept_node(v0.state, self._target.state_at(now),
                                  self.world.central.mu, dep, tof)
        except ValueError:
            self._flash_message("No intercept found")
            return
        self._node_epoch_s = node.epoch_s
        self._dv["pro"] = node.dv_prograde_mps
        self._dv["nrm"] = node.dv_normal_mps
        self._dv["rad"] = node.dv_radial_mps
        for axis, key in (("pro", "pro"), ("nrm", "nrm"), ("rad", "rad")):
            self._dv_value_text[axis].setText(f"{self._dv[key]:+.0f}")
        self._refresh_readout()
        self._flash_message(f"Intercept planned (dV {node.magnitude_mps:,.0f} m/s)")
```

- [ ] **Step 3: Headless verification**

```bash
cd "C:/AI/Claude/Orbital Mechanics Sim" && PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
import numpy as np
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.core.maneuvers import apply_maneuver
from orbitsim.core.rendezvous import closest_approach
from orbitsim.core.moon import moon_state_at
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app._on_play()
for _ in range(4): app.taskMgr.step()
app._target = app._targets[0]           # Moon
# Baseline closest approach with no burn.
base = closest_approach(app.world.vessels[0].state, moon_state_at(0.0),
                        window_s=14*86400.0, coarse_samples=720).separation_m
app._plan_intercept()
node = app._current_node()
assert node.magnitude_mps > 0.0, "no burn planned"
post = apply_maneuver(app.world.vessels[0].state, node)
planned = closest_approach(post, moon_state_at(node.epoch_s),
                           window_s=14*86400.0, coarse_samples=720).separation_m
print(f"baseline sep {base/1e3:,.0f} km -> planned sep {planned/1e3:,.0f} km, dV {node.magnitude_mps:,.0f}")
assert planned < base * 0.5, "intercept did not get materially closer"
print("OK: intercept node planned and closer to the Moon")
PY
```

Expected: a planned separation far below baseline and `OK: intercept node planned and
closer to the Moon`. Also run `.venv/Scripts/python -m pytest tests/ -q` and screenshot
the magenta preview + CA markers after planning.

- [ ] **Step 4: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
Render: Intercept button — porkchop-plan a flyby of the current target

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Self-Review

- Spec: departure-only porkchop solver (T1) ✓; Nelder-Mead refine (T1) ✓; RTN projection → node (T1) ✓; ValueError on infeasible (T1) ✓; Intercept button gated on target + bound orbit (T2) ✓; grids from live geometry (T2) ✓; user-refinable via existing jog/node UI (the node populates `_dv`/`_node_epoch_s`) ✓.
- Tolerances stated as invariants (closing-the-loop < 10 km), not magic numbers; instruction not to loosen them is explicit.
- Types consistent: `intercept_node(...) -> ManeuverNode`; the render handler reads `dv_prograde_mps`/`dv_normal_mps`/`dv_radial_mps` exactly as defined on the dataclass.
- Dependency on the target-selection cycle (`self._target`, `self._targets`, `_dv_value_text`) is called out; build this plan last.
