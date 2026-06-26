# Moon Intercept / Target (Phase 6.2 A3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** Tasks 1–2 are pure physics (TDD). Task 3 is render-layer, executed **inline by the controller** with headless verification per project convention.

**Goal:** Add a Keplerian Moon to the Earth sandbox, let the player target it, and show a closest-approach (intercept) prediction computed on the planned post-node trajectory.

**Architecture:** Two pure core modules — `moon.py` (a fixed Keplerian Moon + `moon_state_at`) and `rendezvous.py` (`closest_approach`, coarse-sample + refine) — plus render changes that draw the Moon + orbit, add a Target Moon toggle, and draw closest-approach markers + a readout. No vessel-physics or save/load changes.

**Tech Stack:** Python 3, numpy, Panda3D. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- Core (`orbitsim/core/`) imports no render/panda3d; SI/float64. (verbatim from project layering)
- Moon is Keplerian around Earth (`mu = MU_EARTH`): `a=3.844e8`, `e=0.0549`, `i=0.0898` rad, `raan=argp=nu=0`, `epoch_s=0`. (verbatim from spec)
- `closest_approach` raises `ValueError` for `window_s ≤ 0` or `coarse_samples < 2`. (verbatim from spec)
- Closest-approach is computed on the predicted post-burn orbit when a node is scheduled, else the current vessel state; window = `min(vessel_period, 14 days)`. (verbatim from spec)
- No change to vessel physics, maneuver-execution, or save/load. Moon is derived (not persisted); target flag is render-only. (verbatim from spec)
- Run tests with `.venv/Scripts/python -m pytest`. (verbatim)
- Commits: explicit paths only; end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. (verbatim from repo git discipline)

---

## File Structure

- `orbitsim/core/moon.py` (new) — `MOON_ORBIT` + `moon_state_at`.
- `orbitsim/core/rendezvous.py` (new) — `ClosestApproach` + `closest_approach`.
- `tests/core/test_moon.py`, `tests/core/test_rendezvous.py` (new).
- `orbitsim/render/app.py` — Moon marker + orbit ring, Target Moon toggle, CA markers + readout, recompute throttle.

Reference shapes (existing):
- `KeplerianElements(a, e, i, raan, argp, nu, mu, epoch_s=0.0)`, property `.period_s`.
- `elements_to_state(elements) -> StateVector`; `propagate_kepler(state, dt_s) -> StateVector` (`.r`, `.v` arrays).
- `state_to_elements(state) -> KeplerianElements`.
- `MU_EARTH` in `core/constants.py`.
- Render: `make_uv_sphere`, `sample_orbit_points(elements, n)`, `build_orbit_node(points, color)`, `self.transform.to_render(r)`, sandbox update loop (~657–705), `_start_sim` sandbox branch (~163+).

---

## Task 1: Core Keplerian Moon (`core/moon.py`)

**Files:**
- Create: `orbitsim/core/moon.py`
- Test: `tests/core/test_moon.py`

**Interfaces:**
- Produces: `MOON_ORBIT: KeplerianElements`; `moon_state_at(t_s: float) -> StateVector` (geocentric Moon state at sim time `t_s`).

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_moon.py`:

```python
"""Tests for the idealized Keplerian Moon."""
import numpy as np
from orbitsim.core.moon import MOON_ORBIT, moon_state_at


def test_moon_distance_in_apsis_range():
    for t in (0.0, 1.0e5, 5.0e5, 1.0e6):
        r = np.linalg.norm(moon_state_at(t).r)
        assert 3.6e8 < r < 4.05e8, (t, r)


def test_moon_is_periodic():
    T = MOON_ORBIT.period_s
    a = moon_state_at(12345.0).r
    b = moon_state_at(12345.0 + T).r
    assert np.linalg.norm(a - b) < 1.0e3  # < 1 km after one period


def test_moon_state_geocentric_mu():
    assert moon_state_at(0.0).mu == MOON_ORBIT.mu
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_moon.py -q`
Expected: FAIL — `ModuleNotFoundError: orbitsim.core.moon`.

- [ ] **Step 3: Implement**

Create `orbitsim/core/moon.py`:

```python
"""An idealized Keplerian Moon orbiting Earth (geocentric, no perturbations).

Used by the sandbox as an intercept target. Real lunar elements vary; these are
fixed mean values, accurate enough for approach planning, not ephemeris-grade.
"""
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.elements import KeplerianElements, elements_to_state
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.state import StateVector

MOON_ORBIT = KeplerianElements(
    a=3.844e8, e=0.0549, i=0.0898, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH, epoch_s=0.0,
)
_MOON_EPOCH_STATE = elements_to_state(MOON_ORBIT)


def moon_state_at(t_s: float) -> StateVector:
    """Geocentric Moon state at sim time ``t_s`` [s past J2000], by two-body propagation."""
    return propagate_kepler(_MOON_EPOCH_STATE, t_s - MOON_ORBIT.epoch_s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_moon.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/moon.py tests/core/test_moon.py
git commit -m "$(cat <<'EOF'
Moon: idealized geocentric Keplerian Moon + moon_state_at

Fixed mean lunar elements (a=384,400 km, e=0.055), propagated two-body
around Earth; the sandbox intercept target.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Core closest-approach (`core/rendezvous.py`)

**Files:**
- Create: `orbitsim/core/rendezvous.py`
- Test: `tests/core/test_rendezvous.py`

**Interfaces:**
- Consumes: `propagate_kepler`, `StateVector`.
- Produces: `@dataclass(frozen=True) ClosestApproach(t_ca_s: float, separation_m: float, rel_speed_mps: float)`; `closest_approach(state_a: StateVector, state_b: StateVector, window_s: float, coarse_samples: int = 720) -> ClosestApproach`.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_rendezvous.py`:

```python
"""Tests for closest-approach between two Keplerian trajectories."""
import numpy as np
import pytest
from orbitsim.core.rendezvous import ClosestApproach, closest_approach
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH


def _circular(r, mu=MU_EARTH, plane="xy"):
    v = np.sqrt(mu / r)
    if plane == "xy":
        return StateVector(r=np.array([r, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=mu)
    return StateVector(r=np.array([r, 0.0, 0.0]), v=np.array([0.0, 0.0, v]), mu=mu)


def test_identical_states_zero_separation_now():
    s = _circular(7.0e6)
    ca = closest_approach(s, s, window_s=6000.0)
    assert ca.separation_m < 1.0
    assert ca.t_ca_s < 60.0
    assert ca.rel_speed_mps < 1e-6


def test_concentric_circles_min_is_radius_difference():
    r1, r2 = 7.0e6, 2.0e7
    a, b = _circular(r1), _circular(r2)
    n1 = np.sqrt(MU_EARTH / r1**3)
    n2 = np.sqrt(MU_EARTH / r2**3)
    synodic = 2.0 * np.pi / (n1 - n2)
    ca = closest_approach(a, b, window_s=1.3 * synodic, coarse_samples=2000)
    # Coplanar concentric circles: closest possible separation is |r2 - r1|.
    assert abs(ca.separation_m - (r2 - r1)) < 0.02 * (r2 - r1)


def test_refine_not_worse_than_coarse():
    r1, r2 = 7.0e6, 1.1e7
    a, b = _circular(r1), _circular(r2)
    n1 = np.sqrt(MU_EARTH / r1**3)
    n2 = np.sqrt(MU_EARTH / r2**3)
    synodic = 2.0 * np.pi / (n1 - n2)
    window = 1.3 * synodic
    ca = closest_approach(a, b, window_s=window, coarse_samples=500)
    # Manual coarse min over the same grid; refine must not be worse.
    times = np.linspace(0.0, window, 501)
    seps = [np.linalg.norm(propagate_a(a, t) - propagate_a(b, t)) for t in times]
    assert ca.separation_m <= min(seps) + 1.0


def propagate_a(state, t):
    from orbitsim.core.propagate import propagate_kepler
    return propagate_kepler(state, t).r


def test_rejects_bad_window():
    s = _circular(7.0e6)
    with pytest.raises(ValueError):
        closest_approach(s, s, window_s=0.0)
    with pytest.raises(ValueError):
        closest_approach(s, s, window_s=100.0, coarse_samples=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_rendezvous.py -q`
Expected: FAIL — `ModuleNotFoundError: orbitsim.core.rendezvous`.

- [ ] **Step 3: Implement**

Create `orbitsim/core/rendezvous.py`:

```python
"""Closest approach between two Keplerian trajectories (coarse scan + refine)."""
from dataclasses import dataclass

import numpy as np

from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.state import StateVector


@dataclass(frozen=True)
class ClosestApproach:
    """Result of a closest-approach search.

    Attributes
    ----------
    t_ca_s : float
        Time of closest approach, seconds from now.
    separation_m : float
        Distance between the two bodies at closest approach [m].
    rel_speed_mps : float
        Relative speed |v_a - v_b| at closest approach [m/s].
    """

    t_ca_s: float
    separation_m: float
    rel_speed_mps: float


def _sep(state_a: StateVector, state_b: StateVector, t: float) -> float:
    ra = propagate_kepler(state_a, t).r
    rb = propagate_kepler(state_b, t).r
    return float(np.linalg.norm(ra - rb))


def closest_approach(
    state_a: StateVector, state_b: StateVector, window_s: float, coarse_samples: int = 720
) -> ClosestApproach:
    """Minimum separation of two trajectories over ``[0, window_s]``.

    Coarse-scans ``coarse_samples+1`` uniform times, then refines the best one with a
    ternary search over its bracketing interval. Raises ValueError on bad inputs.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if coarse_samples < 2:
        raise ValueError(f"coarse_samples must be >= 2, got {coarse_samples}")

    times = np.linspace(0.0, window_s, coarse_samples + 1)
    seps = np.array([_sep(state_a, state_b, float(t)) for t in times])
    k = int(np.argmin(seps))

    # Ternary-search refine within [t_{k-1}, t_{k+1}].
    lo = times[max(0, k - 1)]
    hi = times[min(len(times) - 1, k + 1)]
    for _ in range(60):
        if hi - lo < 1e-3:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _sep(state_a, state_b, m1) < _sep(state_a, state_b, m2):
            hi = m2
        else:
            lo = m1
    t_ca = 0.5 * (lo + hi)

    sa = propagate_kepler(state_a, t_ca)
    sb = propagate_kepler(state_b, t_ca)
    sep = float(np.linalg.norm(sa.r - sb.r))
    rel = float(np.linalg.norm(sa.v - sb.v))
    return ClosestApproach(t_ca_s=t_ca, separation_m=sep, rel_speed_mps=rel)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_rendezvous.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/rendezvous.py tests/core/test_rendezvous.py
git commit -m "$(cat <<'EOF'
Rendezvous: closest_approach between two Keplerian trajectories

Coarse scan over a window + ternary-search refine; returns time,
separation, and relative speed at closest approach.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Render Moon + target + closest-approach markers

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `moon_state_at` (Task 1), `closest_approach`/`ClosestApproach` (Task 2); existing `apply_maneuver`, `state_to_elements`, `propagate_kepler`, `make_uv_sphere`, `sample_orbit_points`, `build_orbit_node`.
- Produces (app state/methods): `self._target_moon: bool`, `self._toggle_target()`, Moon marker/orbit nodes, CA marker nodes, a target readout; recompute throttle.

- [ ] **Step 1: Imports + sandbox scene (Moon marker + orbit ring) in `_start_sim`**

At the top of `app.py`, add: `from orbitsim.core.moon import MOON_ORBIT, moon_state_at` and `from orbitsim.core.rendezvous import closest_approach`.

In `_start_sim`, inside the sandbox (`else:` non-solar) branch where the maneuver UI is built (~200), add the Moon scene + target state:

```python
            self._target_moon = False
            self._ca_recompute_t = 0.0
            self._ca = None
            # Moon marker + its orbit ring (gray), Earth-centered.
            self._moon_np = make_uv_sphere(1.0, 12, 16)
            self._moon_np.reparent_to(self.render)
            self._moon_np.set_color(0.7, 0.7, 0.72, 1.0)
            self._moon_np.set_light_off()
            self._moon_np.set_scale(7.0)
            moon_pts = [self.transform.to_render(p) for p in sample_orbit_points(MOON_ORBIT, n=256)]
            self._moon_orbit_np = build_orbit_node(moon_pts, color=(0.5, 0.5, 0.55, 1.0))
            self._moon_orbit_np.reparent_to(self.render)
            self._ca_marker_ship = None
            self._ca_marker_moon = None
            self._target_text = OnscreenText(
                text="", pos=(0.08, -0.48), scale=0.045, fg=(1.0, 0.7, 0.4, 1),
                shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True, parent=self.a2dTopLeft,
            )
```

Add a "Target Moon" button to the maneuver button row in `_build_maneuver_ui` (append one more entry to the `node_btns` list built in A2):

```python
            ("Target", self._toggle_target),
```

- [ ] **Step 2: Target toggle method + a CA-marker helper**

```python
    def _toggle_target(self):
        self._target_moon = not self._target_moon
        if not self._target_moon:
            for attr in ("_ca_marker_ship", "_ca_marker_moon"):
                np_ = getattr(self, attr, None)
                if np_ is not None:
                    np_.remove_node()
                    setattr(self, attr, None)
            self._ca = None
            self._target_text.setText("")

    def _ca_marker(self, attr, color):
        np_ = getattr(self, attr, None)
        if np_ is None:
            np_ = make_uv_sphere(1.0, 8, 12)
            np_.reparent_to(self.render)
            np_.set_color(*color)
            np_.set_light_off()
            np_.set_scale(5.0)
            setattr(self, attr, np_)
        return np_
```

- [ ] **Step 3: Update-loop block — place Moon, compute & draw closest approach**

In the sandbox update loop, after the maneuver-node block (after the `_node_ttn_text` update), add:

```python
        # Moon position this frame.
        moon_now = moon_state_at(self.clock.sim_time_s)
        self._moon_np.set_pos(*self.transform.to_render(moon_now.r))
        # Closest approach to the Moon when targeted (throttled recompute).
        if self._target_moon:
            import time as _time
            now_real = _time.monotonic()
            if now_real - self._ca_recompute_t > 0.5 or self._ca is None:
                self._ca_recompute_t = now_real
                # Predict on the post-node orbit if a node is scheduled, else current state.
                traj = apply_maneuver(v0.state, node) if (self._node_epoch_s is not None
                                                          and node.magnitude_mps > 0.0) else v0.state
                try:
                    period = state_to_elements(traj).period_s
                except ValueError:
                    period = 14.0 * 86400.0
                window = min(period, 14.0 * 86400.0)
                self._ca = closest_approach(traj, moon_now, window_s=window, coarse_samples=720)
            ca = self._ca
            traj = apply_maneuver(v0.state, node) if (self._node_epoch_s is not None
                                                      and node.magnitude_mps > 0.0) else v0.state
            ship_at = propagate_kepler(traj, ca.t_ca_s).r
            moon_at = moon_state_at(self.clock.sim_time_s + ca.t_ca_s).r
            self._ca_marker("_ca_marker_ship", (1.0, 0.5, 0.2, 1.0)).set_pos(*self.transform.to_render(ship_at))
            self._ca_marker("_ca_marker_moon", (1.0, 0.8, 0.3, 1.0)).set_pos(*self.transform.to_render(moon_at))
            mm, ss = divmod(int(max(0.0, ca.t_ca_s)), 60)
            self._target_text.setText(
                f"Target: Moon   CA T-{mm:02d}:{ss:02d}   sep {ca.separation_m/1000:,.0f} km"
                f"   rel {ca.rel_speed_mps:,.0f} m/s")
```

- [ ] **Step 4: Headless smoke — target shows CA; node shrinks separation**

```bash
PYTHONPATH=. .venv/Scripts/python - <<'PY'
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
from orbitsim.render.app import OrbitApp
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
app = OrbitApp(_default_world(), SimClock(0.0, 100.0), solar_system=False)
app._start_sim(); app.taskMgr.step()
app._toggle_target()
for _ in range(3): app.taskMgr.step()
assert app._ca is not None
leo_sep = app._ca.separation_m
assert leo_sep > 3.0e8, leo_sep  # LEO vessel: ~lunar distance away
# Plan a big prograde burn at periapsis to raise apoapsis toward the Moon.
app._node_to_pe()
app._dv["pro"] = 3000.0
app._ca_recompute_t = 0.0  # force recompute
for _ in range(3): app.taskMgr.step()
assert app._ca.separation_m < leo_sep, (app._ca.separation_m, leo_sep)
app._toggle_target()
assert app._ca is None and app._ca_marker_ship is None
print("OK: target Moon; node lowers predicted separation; clear removes markers")
PY
```

Expected: `OK: target Moon; node lowers predicted separation; clear removes markers`.

- [ ] **Step 5: Visual screenshot + full suite**

Capture a sandbox screenshot zoomed out to show the Moon + its orbit ring + (with a node) the CA markers; eyeball it. Then:

```bash
.venv/Scripts/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
Moon: render Moon + orbit, Target toggle, closest-approach markers

Sandbox draws the Keplerian Moon + its orbit ring; "Target" computes the
closest approach (on the predicted post-node orbit when a node is set) and
draws ship/Moon CA markers + a sep/rel-speed/T-CA readout.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Keplerian Moon `MOON_ORBIT` + `moon_state_at` → Task 1. ✓
- `ClosestApproach` + `closest_approach` (coarse + refine, ValueError) → Task 2. ✓
- Moon marker + orbit ring in sandbox → Task 3 Step 1. ✓
- Target toggle → Task 3 Steps 1–2. ✓
- CA markers (ship + Moon) + readout (sep, T-CA, rel speed) → Task 3 Step 3. ✓
- Computed on predicted post-node orbit when a node scheduled, else current; window min(period,14d) → Task 3 Step 3. ✓
- Recompute throttle (~0.5 s) → Task 3 Step 3. ✓
- No vessel-physics/save-load change; Moon derived, target render-only → all tasks. ✓
- Tests: moon distance/periodicity; closest-approach known answers + refine + ValueError; render headless → Tasks 1–3. ✓

**Placeholder scan:** No TBD/TODO; all code shown; headless check has a concrete script + expected output.

**Type consistency:** `moon_state_at(t_s) -> StateVector` defined Task 1, used Task 3. `closest_approach(state_a, state_b, window_s, coarse_samples=720) -> ClosestApproach` defined Task 2, used Task 3 with `.t_ca_s/.separation_m/.rel_speed_mps`. `_target_moon`, `_toggle_target`, `_ca`, `_ca_recompute_t`, `_ca_marker` consistent across Task 3 steps. The Task 3 code reuses A2's `node`/`v0`/`self._node_epoch_s` already in scope in the maneuver block.
