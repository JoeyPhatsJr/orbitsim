# Phase 2 — Rendering + Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw Earth and one vessel with its orbit line in a Panda3D window, fly a focus/zoom camera across the full solar-system dynamic range without floating-point jitter, and run a time-warpable clock.

**Architecture:** Physics stays in `core/` (float64 SI). A new `sim/` layer owns mutable world state and a sim clock. A new `render/` layer (Panda3D) reads `sim/`+`core/` and never the reverse. The jitter problem is solved by a **floating-origin** transform: subtract a float64 origin BEFORE casting to float32, so local detail near the camera focus keeps millimeter precision regardless of absolute coordinates.

**Tech Stack:** Python 3.10, numpy, Panda3D 1.10.14 (installed), pytest, hypothesis.

## Global Constraints

- SI units everywhere in `core/` and `sim/`: meters, seconds, radians, kg. Convert to km/degrees/UTC ONLY at the render/HUD boundary.
- All physics arrays are `numpy.ndarray`, shape `(3,)`, dtype `float64`.
- `core/` must NEVER import `panda3d`, `sim/`, or `render/`. `sim/` may import `core/` only. `render/` may import `sim/` and `core/`.
- `StateVector`, `KeplerianElements`, `CelestialBody` are frozen dataclasses — return new instances, never mutate.
- `black` line length 100. Type hints on every function. NumPy-style docstrings stating units and frame.
- Run `pytest tests/ -q` after every task; it must stay green. Render tasks that cannot be unit-tested end with a HUMAN VISUAL CHECKPOINT instead.

## Phase 1 API available to you (already implemented, do not change)

```python
# orbitsim/core/constants.py  (module-level float64)
MU_SUN, MU_EARTH, MU_MOON, R_EARTH, R_SUN, R_MOON, J2_EARTH, G

# orbitsim/core/bodies.py
@dataclass(frozen=True)
class CelestialBody:
    name: str; mu: float; radius_m: float
    j2: float = 0.0; rotation_period_s: float = float("inf")
    parent: "CelestialBody | None" = None
    def soi_radius_m(self, semi_major_axis_m: float) -> float: ...
# NOTE: no pre-built EARTH/SUN/MOON instances exist yet — Task 2 adds them.

# orbitsim/core/state.py
@dataclass(frozen=True)
class StateVector:
    r: np.ndarray; v: np.ndarray; mu: float; epoch_s: float = 0.0
    # properties: r_mag, v_mag, specific_energy, angular_momentum
    # arrays are validated shape (3,), float64, finite, and made read-only

# orbitsim/core/elements.py
@dataclass(frozen=True)
class KeplerianElements:
    a: float; e: float; i: float; raan: float; argp: float; nu: float
    mu: float; epoch_s: float = 0.0
    # properties: period_s (ValueError if a<=0), semi_latus_rectum
def state_to_elements(state: StateVector) -> KeplerianElements: ...
def elements_to_state(elements: KeplerianElements) -> StateVector: ...

# orbitsim/core/propagate.py
def propagate_kepler(state: StateVector, dt: float) -> StateVector: ...
def propagate_numeric(state, dt, *, j2=False, third_bodies=()) -> StateVector: ...
```

---

## File Structure

- `orbitsim/core/bodies.py` — MODIFY: append pre-built `SUN`, `EARTH`, `MOON` instances.
- `orbitsim/sim/clock.py` — CREATE: `SimClock` (sim time + time-warp).
- `orbitsim/sim/world.py` — CREATE: `Vessel`, `World`.
- `orbitsim/render/floating_origin.py` — CREATE: `RenderTransform` (the precision fix).
- `orbitsim/render/orbit_lines.py` — CREATE: `sample_orbit_points()` (pure math) + `build_orbit_node()` (Panda3D).
- `orbitsim/render/geometry.py` — CREATE: `make_uv_sphere()` procedural sphere (avoids missing-asset issues).
- `orbitsim/render/camera_rig.py` — CREATE: `CameraRig` (focus/zoom; zoom→scale mapping is unit-tested).
- `orbitsim/render/hud/__init__.py` — CREATE: `Hud` DirectGUI overlay.
- `orbitsim/render/app.py` — CREATE: `OrbitApp(ShowBase)`.
- `orbitsim/__main__.py` — MODIFY: wire a default scenario and run the app.
- Tests: `tests/sim/test_clock.py`, `tests/sim/test_world.py`, `tests/render/test_floating_origin.py`, `tests/render/test_orbit_lines.py`, `tests/render/test_camera_rig.py`, plus `tests/sim/__init__.py`, `tests/render/__init__.py`.

---

## Task 1: SimClock (sim time + time-warp)

**Files:**
- Create: `orbitsim/sim/clock.py`
- Test: `tests/sim/test_clock.py`, `tests/sim/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```python
  class SimClock:
      WARP_STEPS = [1, 5, 10, 50, 100, 1_000, 10_000, 100_000, 1_000_000]
      def __init__(self, sim_time_s: float = 0.0, warp: float = 1.0): ...
      sim_time_s: float   # float64
      warp: float
      def advance(self, real_dt_s: float) -> float:  # returns sim_dt, also adds it to sim_time_s
      def warp_up(self) -> None:    # step to next WARP_STEPS value
      def warp_down(self) -> None:  # step to previous WARP_STEPS value
  ```

- [ ] **Step 1: Create test package init**

Create `tests/sim/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing tests**

Create `tests/sim/test_clock.py`:
```python
"""Tests for SimClock (sim time + time-warp)."""
import pytest
from orbitsim.sim.clock import SimClock


def test_advance_returns_scaled_dt():
    clock = SimClock(sim_time_s=0.0, warp=10.0)
    sim_dt = clock.advance(2.0)
    assert sim_dt == 20.0


def test_advance_accumulates_sim_time():
    clock = SimClock(sim_time_s=100.0, warp=5.0)
    clock.advance(2.0)
    assert clock.sim_time_s == 110.0


def test_warp_up_steps_through_table():
    clock = SimClock(warp=1.0)
    clock.warp_up()
    assert clock.warp == 5.0
    clock.warp_up()
    assert clock.warp == 10.0


def test_warp_down_steps_back():
    clock = SimClock(warp=10.0)
    clock.warp_down()
    assert clock.warp == 5.0


def test_warp_up_clamps_at_max():
    clock = SimClock(warp=1_000_000.0)
    clock.warp_up()
    assert clock.warp == 1_000_000.0


def test_warp_down_clamps_at_min():
    clock = SimClock(warp=1.0)
    clock.warp_down()
    assert clock.warp == 1.0


def test_warp_value_must_be_in_table():
    # An off-table warp snaps to the nearest table value on the next step.
    clock = SimClock(warp=7.0)
    clock.warp_up()
    assert clock.warp in SimClock.WARP_STEPS
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/sim/test_clock.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.sim.clock).

- [ ] **Step 4: Implement SimClock**

Create `orbitsim/sim/clock.py`:
```python
"""Simulation time and time-warp."""
import bisect


class SimClock:
    """Owns simulation time (seconds past J2000, TDB) and the time-warp rate.

    Parameters
    ----------
    sim_time_s : float
        Initial simulation time [s past J2000 TDB].
    warp : float
        Sim seconds advanced per real second. Must be one of WARP_STEPS.
    """

    WARP_STEPS = [1, 5, 10, 50, 100, 1_000, 10_000, 100_000, 1_000_000]

    def __init__(self, sim_time_s: float = 0.0, warp: float = 1.0) -> None:
        self.sim_time_s = float(sim_time_s)
        self.warp = float(warp)

    def advance(self, real_dt_s: float) -> float:
        """Advance sim time by real_dt_s * warp; return the sim_dt applied [s]."""
        sim_dt = real_dt_s * self.warp
        self.sim_time_s += sim_dt
        return sim_dt

    def _current_index(self) -> int:
        # Snap an arbitrary warp to the nearest table index.
        steps = self.WARP_STEPS
        pos = bisect.bisect_left(steps, self.warp)
        if pos >= len(steps):
            return len(steps) - 1
        if pos > 0 and abs(steps[pos - 1] - self.warp) <= abs(steps[pos] - self.warp):
            return pos - 1
        return pos

    def warp_up(self) -> None:
        """Increase warp to the next allowed step (clamped at max)."""
        idx = min(self._current_index() + 1, len(self.WARP_STEPS) - 1)
        self.warp = float(self.WARP_STEPS[idx])

    def warp_down(self) -> None:
        """Decrease warp to the previous allowed step (clamped at min)."""
        idx = max(self._current_index() - 1, 0)
        self.warp = float(self.WARP_STEPS[idx])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/sim/test_clock.py -q`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add orbitsim/sim/clock.py tests/sim/
git commit -m "Phase 2 Task 1: SimClock with time-warp stepping"
```

---

## Task 2: Pre-built celestial bodies

**Files:**
- Modify: `orbitsim/core/bodies.py` (append instances at end of file)
- Test: `tests/core/test_bodies.py` (append tests)

**Interfaces:**
- Consumes: `CelestialBody`, constants `MU_SUN, MU_EARTH, MU_MOON, R_SUN, R_EARTH, R_MOON, J2_EARTH`.
- Produces: module-level `SUN`, `EARTH`, `MOON` of type `CelestialBody`, with `EARTH.parent is SUN`, `MOON.parent is EARTH`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_bodies.py`:
```python
def test_prebuilt_bodies_exist():
    from orbitsim.core.bodies import SUN, EARTH, MOON
    from orbitsim.core.constants import MU_EARTH, R_EARTH
    assert EARTH.mu == MU_EARTH
    assert EARTH.radius_m == R_EARTH
    assert EARTH.parent is SUN
    assert MOON.parent is EARTH
    assert SUN.parent is None


def test_prebuilt_earth_j2():
    from orbitsim.core.bodies import EARTH
    from orbitsim.core.constants import J2_EARTH
    assert EARTH.j2 == J2_EARTH
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_bodies.py -q`
Expected: FAIL (ImportError: cannot import name 'SUN').

- [ ] **Step 3: Append instances to bodies.py**

Append to `orbitsim/core/bodies.py`:
```python
from orbitsim.core.constants import (
    MU_SUN,
    MU_EARTH,
    MU_MOON,
    R_SUN,
    R_EARTH,
    R_MOON,
    J2_EARTH,
)

# Sidereal rotation periods [s] (IAU): Earth 86164.0905 s, Sun ~25.05 d, Moon ~27.32 d.
SUN = CelestialBody(
    name="Sun",
    mu=MU_SUN,
    radius_m=R_SUN,
    rotation_period_s=25.05 * 86400.0,
    parent=None,
)
EARTH = CelestialBody(
    name="Earth",
    mu=MU_EARTH,
    radius_m=R_EARTH,
    j2=J2_EARTH,
    rotation_period_s=86164.0905,
    parent=SUN,
)
MOON = CelestialBody(
    name="Moon",
    mu=MU_MOON,
    radius_m=R_MOON,
    rotation_period_s=27.321661 * 86400.0,
    parent=EARTH,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_bodies.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/bodies.py tests/core/test_bodies.py
git commit -m "Phase 2 Task 2: pre-built SUN/EARTH/MOON bodies"
```

---

## Task 3: Vessel + World

**Files:**
- Create: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py`

**Interfaces:**
- Consumes: `CelestialBody`, `StateVector`, `propagate_kepler`, `EARTH`.
- Produces:
  ```python
  @dataclass
  class Vessel:
      name: str
      state: StateVector            # mutable in the sim layer
      delta_v_budget_mps: float = 0.0
  class World:
      def __init__(self, central: CelestialBody, vessels: list[Vessel]): ...
      central: CelestialBody
      vessels: list[Vessel]
      def step(self, sim_dt_s: float) -> None:  # propagate every vessel by sim_dt_s
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/sim/test_world.py`:
```python
"""Tests for the sim-layer World and Vessel."""
import numpy as np
from orbitsim.sim.world import Vessel, World
from orbitsim.core.state import StateVector
from orbitsim.core.elements import state_to_elements
from orbitsim.core.bodies import EARTH
from orbitsim.core.constants import MU_EARTH


def _circular_vessel(r_m: float = 7.0e6) -> Vessel:
    v = np.sqrt(MU_EARTH / r_m)
    state = StateVector(r=np.array([r_m, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)
    return Vessel(name="test", state=state, delta_v_budget_mps=1000.0)


def test_world_step_period_closure():
    vessel = _circular_vessel()
    world = World(central=EARTH, vessels=[vessel])
    period = state_to_elements(vessel.state).period_s
    world.step(period)
    pos_error = np.linalg.norm(world.vessels[0].state.r - np.array([7.0e6, 0.0, 0.0]))
    assert pos_error < 1e-3  # 1 mm closure (analytic)


def test_world_step_updates_state_object():
    vessel = _circular_vessel()
    world = World(central=EARTH, vessels=[vessel])
    before = world.vessels[0].state
    world.step(100.0)
    assert world.vessels[0].state is not before  # new immutable instance


def test_vessel_carries_budget():
    vessel = _circular_vessel()
    assert vessel.delta_v_budget_mps == 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.sim.world).

- [ ] **Step 3: Implement world.py**

Create `orbitsim/sim/world.py`:
```python
"""Body registry + vessels; per-tick propagation."""
from dataclasses import dataclass

from orbitsim.core.bodies import CelestialBody
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler


@dataclass
class Vessel:
    """A point-mass vessel with a current state and a delta-V budget.

    Attributes
    ----------
    name : str
    state : StateVector
        Current inertial state (mutable here in the sim layer; updated each tick).
    delta_v_budget_mps : float
        Remaining delta-V budget [m/s].
    """

    name: str
    state: StateVector
    delta_v_budget_mps: float = 0.0


class World:
    """Holds the central body and all vessels; advances them analytically.

    Parameters
    ----------
    central : CelestialBody
        The body all vessel states are referenced to.
    vessels : list[Vessel]
    """

    def __init__(self, central: CelestialBody, vessels: list[Vessel]) -> None:
        self.central = central
        self.vessels = vessels

    def step(self, sim_dt_s: float) -> None:
        """Propagate every vessel forward by sim_dt_s seconds (on-rails)."""
        for vessel in self.vessels:
            vessel.state = propagate_kepler(vessel.state, sim_dt_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "Phase 2 Task 3: Vessel + World sim layer"
```

---

## Task 4: RenderTransform (floating origin) — the precision fix

**Files:**
- Create: `orbitsim/render/floating_origin.py`
- Test: `tests/render/test_floating_origin.py`, `tests/render/__init__.py`

**Interfaces:**
- Consumes: numpy only (no Panda3D — keep this file pure so it is unit-testable).
- Produces:
  ```python
  class RenderTransform:
      def __init__(self, origin_m: np.ndarray, scale_m_per_unit: float): ...
      origin_m: np.ndarray       # float64 (3,)
      scale_m_per_unit: float
      def to_render(self, physics_pos_m: np.ndarray) -> tuple[float, float, float]:
          # (physics_pos_m - origin_m) / scale, subtract in float64 BEFORE float32 cast
      def set_origin(self, origin_m: np.ndarray) -> None: ...
  ```

- [ ] **Step 1: Create render test package init**

Create `tests/render/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing tests**

Create `tests/render/test_floating_origin.py`:
```python
"""Tests for the floating-origin RenderTransform precision guarantee."""
import numpy as np
from orbitsim.render.floating_origin import RenderTransform


def test_origin_maps_to_zero():
    origin = np.array([1.0e11, 2.0e11, -3.0e11])
    rt = RenderTransform(origin_m=origin, scale_m_per_unit=1.0e6)
    out = rt.to_render(origin)
    assert out == (0.0, 0.0, 0.0)


def test_precision_preserved_across_float32_cast():
    """A 1 mm offset from a point 1e11 m away must survive the float32 cast.

    This is the whole reason the renderer works: subtract the float64 origin
    BEFORE casting to float32, so 1e-3 m local detail is not lost in 1e11.
    """
    far = np.array([1.0e11, 0.0, 0.0])
    offset = np.array([1.0e-3, 0.0, 0.0])
    point = far + offset
    scale = 1.0e-3  # 1 render unit == 1 mm, so the offset maps to ~1.0 render units
    rt = RenderTransform(origin_m=far, scale_m_per_unit=scale)
    rx, ry, rz = rt.to_render(point)
    implied_distance_m = rx * scale
    assert abs(implied_distance_m - 1.0e-3) < 1e-6


def test_scale_divides():
    rt = RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=1000.0)
    out = rt.to_render(np.array([2000.0, 0.0, 0.0]))
    assert abs(out[0] - 2.0) < 1e-9


def test_set_origin_updates_mapping():
    rt = RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=1.0)
    rt.set_origin(np.array([5.0, 0.0, 0.0]))
    out = rt.to_render(np.array([5.0, 0.0, 0.0]))
    assert out == (0.0, 0.0, 0.0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/render/test_floating_origin.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 4: Implement floating_origin.py**

Create `orbitsim/render/floating_origin.py`:
```python
"""Floating-origin transform: physics-space float64 -> render-space float32.

render_pos = (physics_pos_m - origin_m) / scale_m_per_unit

The subtraction happens in float64 BEFORE the float32 cast, which preserves
millimeter precision near the focus even at solar-system absolute coordinates.
"""
import numpy as np


class RenderTransform:
    """Maps physics-space SI float64 positions to render-space float32 positions.

    Parameters
    ----------
    origin_m : np.ndarray
        The physics point (float64, shape (3,)) currently mapped to render (0,0,0).
    scale_m_per_unit : float
        Meters per render unit (set from camera zoom).
    """

    def __init__(self, origin_m: np.ndarray, scale_m_per_unit: float) -> None:
        self.origin_m = np.asarray(origin_m, dtype=np.float64).copy()
        self.scale_m_per_unit = float(scale_m_per_unit)

    def set_origin(self, origin_m: np.ndarray) -> None:
        """Re-center the render space on a new physics point."""
        self.origin_m = np.asarray(origin_m, dtype=np.float64).copy()

    def to_render(self, physics_pos_m: np.ndarray) -> tuple[float, float, float]:
        """Convert a physics-space position to a render-space (x, y, z) tuple."""
        local = np.asarray(physics_pos_m, dtype=np.float64) - self.origin_m  # float64 first
        scaled = (local / self.scale_m_per_unit).astype(np.float32)          # then cast
        return (float(scaled[0]), float(scaled[1]), float(scaled[2]))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/render/test_floating_origin.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/floating_origin.py tests/render/
git commit -m "Phase 2 Task 4: floating-origin RenderTransform (precision-preserving)"
```

---

## Task 5: Orbit-line sampling (pure-math part)

**Files:**
- Create: `orbitsim/render/orbit_lines.py` (sampling function now; Panda3D node builder added in Task 8)
- Test: `tests/render/test_orbit_lines.py`

**Interfaces:**
- Consumes: `KeplerianElements`, `elements_to_state`.
- Produces:
  ```python
  def sample_orbit_points(elements: KeplerianElements, n: int = 256) -> np.ndarray:
      # returns (n, 3) float64 physics-space positions sampled in true anomaly.
      # Ellipse (e<1): nu over [0, 2pi). Hyperbola (e>=1): nu over the open
      # interval (-nu_max, nu_max) where cos(nu_max) = -1/e (asymptotes).
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/render/test_orbit_lines.py`:
```python
"""Tests for orbit-line sampling (pure math)."""
import numpy as np
from orbitsim.render.orbit_lines import sample_orbit_points
from orbitsim.core.elements import KeplerianElements
from orbitsim.core.constants import MU_EARTH


def test_ellipse_sample_shape_and_radius():
    elem = KeplerianElements(a=8.0e6, e=0.1, i=0.3, raan=1.0, argp=0.5, nu=0.0, mu=MU_EARTH)
    pts = sample_orbit_points(elem, n=128)
    assert pts.shape == (128, 3)
    radii = np.linalg.norm(pts, axis=1)
    rp = elem.a * (1 - elem.e)  # periapsis
    ra = elem.a * (1 + elem.e)  # apoapsis
    assert radii.min() >= rp - 1.0
    assert radii.max() <= ra + 1.0


def test_ellipse_is_closed():
    elem = KeplerianElements(a=8.0e6, e=0.2, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH)
    pts = sample_orbit_points(elem, n=64)
    # First and last sampled true anomalies are adjacent around the loop;
    # distance between them is one segment, far smaller than the orbit size.
    seg = np.linalg.norm(pts[0] - pts[-1])
    assert seg < 0.5 * elem.a


def test_hyperbola_sample_finite():
    a = -1.0e7  # negative for hyperbola
    elem = KeplerianElements(a=a, e=1.4, i=0.2, raan=0.5, argp=0.3, nu=0.0, mu=MU_EARTH)
    pts = sample_orbit_points(elem, n=100)
    assert pts.shape == (100, 3)
    assert np.isfinite(pts).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/render/test_orbit_lines.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement sample_orbit_points**

Create `orbitsim/render/orbit_lines.py`:
```python
"""Sample an orbit into a polyline and (Task 8) build a Panda3D LineSegs node."""
import numpy as np

from orbitsim.core.elements import KeplerianElements, elements_to_state


def sample_orbit_points(elements: KeplerianElements, n: int = 256) -> np.ndarray:
    """Sample physics-space positions along an orbit.

    Parameters
    ----------
    elements : KeplerianElements
    n : int
        Number of samples.

    Returns
    -------
    np.ndarray
        (n, 3) float64 positions [m] in the inertial frame.

    Notes
    -----
    Ellipse (e < 1): true anomaly spans [0, 2pi).
    Hyperbola (e >= 1): true anomaly spans the open interval bounded by the
    asymptote angle nu_max = arccos(-1/e); we sample (1 - margin)*nu_max so the
    radius stays finite.
    """
    e = elements.e
    if e < 1.0:
        nus = np.linspace(0.0, 2.0 * np.pi, n)
    else:
        nu_max = np.arccos(-1.0 / e)
        limit = 0.99 * nu_max
        nus = np.linspace(-limit, limit, n)

    pts = np.empty((n, 3), dtype=np.float64)
    for idx, nu in enumerate(nus):
        sampled = KeplerianElements(
            a=elements.a, e=elements.e, i=elements.i, raan=elements.raan,
            argp=elements.argp, nu=float(nu), mu=elements.mu, epoch_s=elements.epoch_s,
        )
        pts[idx] = elements_to_state(sampled).r
    return pts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/render/test_orbit_lines.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/orbit_lines.py tests/render/test_orbit_lines.py
git commit -m "Phase 2 Task 5: orbit-line sampling (ellipse + hyperbola)"
```

---

## Task 6: Camera zoom → scale mapping (unit-testable part)

**Files:**
- Create: `orbitsim/render/camera_rig.py` (the pure zoom→scale function now; Panda3D wiring in Task 8)
- Test: `tests/render/test_camera_rig.py`

**Interfaces:**
- Consumes: numpy.
- Produces:
  ```python
  def zoom_to_scale(distance_m: float) -> float:
      # log-scaled: meters-per-render-unit grows with camera distance so the
      # focused scene fits in roughly [-1000, 1000] render units.
      # Defined as distance_m / 1000.0 (1000 render units span the view radius).
  MIN_DISTANCE_M = 10.0
  MAX_DISTANCE_M = 1.0e12
  def clamp_distance(distance_m: float) -> float: ...
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/render/test_camera_rig.py`:
```python
"""Tests for camera zoom->scale mapping (pure math)."""
from orbitsim.render.camera_rig import zoom_to_scale, clamp_distance, MIN_DISTANCE_M, MAX_DISTANCE_M


def test_scale_proportional_to_distance():
    assert zoom_to_scale(1.0e6) == 1.0e6 / 1000.0
    assert zoom_to_scale(1.0e9) == 1.0e9 / 1000.0


def test_scale_monotonic():
    assert zoom_to_scale(10.0) < zoom_to_scale(1.0e12)


def test_clamp_distance_bounds():
    assert clamp_distance(1.0) == MIN_DISTANCE_M
    assert clamp_distance(1.0e15) == MAX_DISTANCE_M
    assert clamp_distance(1.0e6) == 1.0e6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/render/test_camera_rig.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement the pure mapping (Panda3D rig added in Task 8)**

Create `orbitsim/render/camera_rig.py`:
```python
"""Orbit-style camera: focus/zoom across a huge dynamic range.

This module's pure functions (zoom_to_scale, clamp_distance) are unit-tested.
The CameraRig class that drives Panda3D is added in Task 8 and exercised by the
visual checkpoint, not unit tests.
"""

MIN_DISTANCE_M = 10.0
MAX_DISTANCE_M = 1.0e12

# 1000 render units span the camera-to-focus distance, keeping the visible
# scene comfortably inside float32-friendly coordinates.
RENDER_UNITS_ACROSS_VIEW = 1000.0


def clamp_distance(distance_m: float) -> float:
    """Clamp a camera distance to the supported zoom range [10 m, 1e12 m]."""
    return max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance_m))


def zoom_to_scale(distance_m: float) -> float:
    """Meters per render unit for a given camera-to-focus distance [m]."""
    return distance_m / RENDER_UNITS_ACROSS_VIEW
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/render/test_camera_rig.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/camera_rig.py tests/render/test_camera_rig.py
git commit -m "Phase 2 Task 6: camera zoom->scale mapping"
```

---

## Task 7: Procedural sphere geometry (Panda3D)

This avoids depending on bundled Panda3D model assets, which vary by install. Not
unit-tested (it builds GPU geometry); verified visually in Task 9.

**Files:**
- Create: `orbitsim/render/geometry.py`

**Interfaces:**
- Produces:
  ```python
  def make_uv_sphere(radius: float = 1.0, num_lat: int = 24, num_lon: int = 48) -> NodePath:
      # returns a Panda3D NodePath holding a unit-ish sphere of the given radius.
  ```

- [ ] **Step 1: Implement geometry.py**

Create `orbitsim/render/geometry.py`:
```python
"""Procedural geometry helpers (no external model assets)."""
import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)


def make_uv_sphere(radius: float = 1.0, num_lat: int = 24, num_lon: int = 48) -> NodePath:
    """Build a UV sphere NodePath of the given radius (render units).

    Parameters
    ----------
    radius : float
        Sphere radius in render units.
    num_lat, num_lon : int
        Latitude/longitude subdivisions.

    Returns
    -------
    NodePath
        A NodePath wrapping the sphere geometry, centered at the origin.
    """
    fmt = GeomVertexFormat.get_v3n3()
    vdata = GeomVertexData("sphere", fmt, Geom.UHStatic)
    vdata.set_num_rows((num_lat + 1) * (num_lon + 1))
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")

    for i in range(num_lat + 1):
        theta = math.pi * i / num_lat
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        for j in range(num_lon + 1):
            phi = 2.0 * math.pi * j / num_lon
            x = sin_t * math.cos(phi)
            y = sin_t * math.sin(phi)
            z = cos_t
            vertex.add_data3(x * radius, y * radius, z * radius)
            normal.add_data3(x, y, z)

    tris = GeomTriangles(Geom.UHStatic)
    row = num_lon + 1
    for i in range(num_lat):
        for j in range(num_lon):
            a = i * row + j
            b = a + 1
            c = a + row
            d = c + 1
            tris.add_vertices(a, c, b)
            tris.add_vertices(b, c, d)

    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("sphere")
    node.add_geom(geom)
    return NodePath(node)
```

- [ ] **Step 2: Smoke-check it imports and builds (no window)**

Run:
```bash
.venv/Scripts/python -c "from orbitsim.render.geometry import make_uv_sphere; n = make_uv_sphere(2.0); print('sphere bounds ok:', not n.is_empty())"
```
Expected: prints `sphere bounds ok: True`.

- [ ] **Step 3: Commit**

```bash
git add orbitsim/render/geometry.py
git commit -m "Phase 2 Task 7: procedural UV-sphere geometry"
```

---

## Task 8: Panda3D app, orbit-line node, camera rig, HUD

This wires everything into a running window. These steps build GPU/UI objects and
are verified in Task 9's visual checkpoint, not by unit tests.

**Files:**
- Modify: `orbitsim/render/orbit_lines.py` (add `build_orbit_node`)
- Modify: `orbitsim/render/camera_rig.py` (add `CameraRig`)
- Create: `orbitsim/render/hud/__init__.py`
- Create: `orbitsim/render/app.py`

**Interfaces:**
- Produces:
  ```python
  # orbit_lines.py
  def build_orbit_node(points_render: list[tuple[float,float,float]],
                       color: tuple[float,float,float,float]) -> NodePath: ...
  # camera_rig.py
  class CameraRig:
      def __init__(self, base, transform): ...
      def set_distance(self, distance_m: float) -> None: ...  # updates transform.scale_m_per_unit
      def orbit(self, d_azimuth: float, d_elevation: float) -> None: ...
      def apply(self) -> None: ...  # positions base.camera each frame
  # hud/__init__.py
  class Hud:
      def __init__(self, base): ...
      def update(self, *, sim_time_s, warp, altitude_m, speed_mps,
                 periapsis_m, apoapsis_m, period_s) -> None: ...
  # app.py
  class OrbitApp(ShowBase):
      def __init__(self, world, clock): ...
      def run_app(self) -> None: ...
  ```

- [ ] **Step 1: Add `build_orbit_node` to orbit_lines.py**

Append to `orbitsim/render/orbit_lines.py`:
```python
from panda3d.core import LineSegs, NodePath


def build_orbit_node(
    points_render: list[tuple[float, float, float]],
    color: tuple[float, float, float, float] = (0.3, 0.7, 1.0, 1.0),
) -> NodePath:
    """Build a Panda3D LineSegs polyline from render-space points.

    Parameters
    ----------
    points_render : list of (x, y, z)
        Render-space points (already passed through RenderTransform.to_render).
    color : (r, g, b, a)

    Returns
    -------
    NodePath
        A NodePath holding the line strip.
    """
    segs = LineSegs()
    segs.set_color(*color)
    segs.set_thickness(1.5)
    for idx, (x, y, z) in enumerate(points_render):
        if idx == 0:
            segs.move_to(x, y, z)
        else:
            segs.draw_to(x, y, z)
    return NodePath(segs.create())
```

- [ ] **Step 2: Add `CameraRig` to camera_rig.py**

Append to `orbitsim/render/camera_rig.py`:
```python
import math

from panda3d.core import Vec3


class CameraRig:
    """Orbit camera: azimuth/elevation around a focus, log zoom drives scale.

    Parameters
    ----------
    base : ShowBase
    transform : RenderTransform
        Its scale_m_per_unit is updated as the camera zooms.
    """

    def __init__(self, base, transform) -> None:
        self.base = base
        self.transform = transform
        self.distance_m = 2.0e7
        self.azimuth = 0.0
        self.elevation = 0.3
        self._apply_scale()

    def _apply_scale(self) -> None:
        self.transform.scale_m_per_unit = zoom_to_scale(self.distance_m)

    def set_distance(self, distance_m: float) -> None:
        self.distance_m = clamp_distance(distance_m)
        self._apply_scale()

    def zoom(self, factor: float) -> None:
        """Multiply camera distance (factor < 1 zooms in, > 1 zooms out)."""
        self.set_distance(self.distance_m * factor)

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth += d_azimuth
        self.elevation = max(-1.5, min(1.5, self.elevation + d_elevation))

    def apply(self) -> None:
        """Place the camera in render space (focus is always render origin)."""
        # In render units the focus sits at (0,0,0); camera distance is fixed
        # at RENDER_UNITS_ACROSS_VIEW because scale already encodes zoom.
        d = RENDER_UNITS_ACROSS_VIEW
        ce = math.cos(self.elevation)
        x = d * ce * math.cos(self.azimuth)
        y = d * ce * math.sin(self.azimuth)
        z = d * math.sin(self.elevation)
        self.base.camera.set_pos(x, y, z)
        self.base.camera.look_at(0, 0, 0)
        lens = self.base.camLens
        lens.set_near_far(0.01, 1.0e6)
```

- [ ] **Step 3: Implement the HUD**

Create `orbitsim/render/hud/__init__.py`:
```python
"""Minimal DirectGUI overlay. Converts SI -> km/UTC at this boundary only."""
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class Hud:
    """On-screen text panel showing time, warp, and focused-vessel orbit info."""

    def __init__(self, base) -> None:
        self.text = OnscreenText(
            text="",
            pos=(-1.3, 0.9),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
            parent=base.a2dTopLeft if hasattr(base, "a2dTopLeft") else None,
        )

    def update(
        self,
        *,
        sim_time_s: float,
        warp: float,
        altitude_m: float,
        speed_mps: float,
        periapsis_m: float,
        apoapsis_m: float,
        period_s: float,
    ) -> None:
        lines = [
            f"Sim time: {sim_time_s:,.0f} s past J2000",
            f"Warp: x{warp:,.0f}",
            f"Altitude: {altitude_m / 1000.0:,.1f} km",
            f"Speed: {speed_mps / 1000.0:,.3f} km/s",
            f"Periapsis: {periapsis_m / 1000.0:,.1f} km",
            f"Apoapsis: {apoapsis_m / 1000.0:,.1f} km",
            f"Period: {period_s / 60.0:,.1f} min",
        ]
        self.text.setText("\n".join(lines))
```

- [ ] **Step 4: Implement the app**

Create `orbitsim/render/app.py`:
```python
"""Panda3D ShowBase bootstrap and per-frame loop."""
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import ClockObject, AmbientLight, DirectionalLight, Vec4

from orbitsim.core.elements import state_to_elements
from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.orbit_lines import sample_orbit_points, build_orbit_node
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.hud import Hud

_global_clock = ClockObject.get_global_clock()


class OrbitApp(ShowBase):
    """Renders one central body + vessels with orbit lines; time-warpable."""

    def __init__(self, world, clock) -> None:
        super().__init__()
        self.world = world
        self.clock = clock

        self.transform = RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=2.0e4)
        self.rig = CameraRig(self, self.transform)
        self.disable_mouse()
        self.hud = Hud(self)

        # Central body sphere, sized in render units via the current scale.
        self.central_np = make_uv_sphere(1.0, 24, 48)
        self.central_np.reparent_to(self.render)
        self.central_np.set_color(0.2, 0.4, 0.9, 1.0)

        # Lighting so the sphere is visible.
        amb = AmbientLight("amb"); amb.set_color(Vec4(0.3, 0.3, 0.3, 1))
        self.render.set_light(self.render.attach_new_node(amb))
        dirl = DirectionalLight("dir"); dirl.set_color(Vec4(0.9, 0.9, 0.9, 1))
        dnp = self.render.attach_new_node(dirl); dnp.set_hpr(45, -45, 0)
        self.render.set_light(dnp)

        # Vessel markers + orbit lines.
        self.vessel_nps = []
        self.orbit_nps = []
        for _ in world.vessels:
            m = make_uv_sphere(0.03, 8, 12)
            m.reparent_to(self.render)
            m.set_color(1.0, 0.9, 0.2, 1.0)
            self.vessel_nps.append(m)
            self.orbit_nps.append(None)

        self._setup_input()
        self.task_mgr.add(self._update, "update")

    def _setup_input(self) -> None:
        self.accept("wheel_up", lambda: self.rig.zoom(0.8))
        self.accept("wheel_down", lambda: self.rig.zoom(1.25))
        self.accept("arrow_left", lambda: self.rig.orbit(-0.1, 0.0))
        self.accept("arrow_right", lambda: self.rig.orbit(0.1, 0.0))
        self.accept("arrow_up", lambda: self.rig.orbit(0.0, 0.1))
        self.accept("arrow_down", lambda: self.rig.orbit(0.0, -0.1))
        self.accept("period", self.clock.warp_up)     # ">" key
        self.accept("comma", self.clock.warp_down)    # "<" key

    def _rebuild_orbit(self, idx, vessel) -> None:
        elem = state_to_elements(vessel.state)
        pts = sample_orbit_points(elem, n=256)
        pts_render = [self.transform.to_render(p) for p in pts]
        if self.orbit_nps[idx] is not None:
            self.orbit_nps[idx].remove_node()
        node = build_orbit_node(pts_render)
        node.reparent_to(self.render)
        self.orbit_nps[idx] = node

    def _update(self, task):
        real_dt = _global_clock.get_dt()
        sim_dt = self.clock.advance(real_dt)
        self.world.step(sim_dt)

        # Focus origin on the first vessel; central body sits relative to it.
        focus = self.world.vessels[0].state.r if self.world.vessels else np.zeros(3)
        self.transform.set_origin(focus)

        # Re-scale + place the central body (at physics origin).
        cx, cy, cz = self.transform.to_render(np.zeros(3))
        self.central_np.set_pos(cx, cy, cz)
        body_render_radius = self.world.central.radius_m / self.transform.scale_m_per_unit
        self.central_np.set_scale(max(body_render_radius, 1e-3))

        for idx, vessel in enumerate(self.world.vessels):
            vx, vy, vz = self.transform.to_render(vessel.state.r)
            self.vessel_nps[idx].set_pos(vx, vy, vz)
            self._rebuild_orbit(idx, vessel)

        self.rig.apply()

        v0 = self.world.vessels[0]
        elem = state_to_elements(v0.state)
        rp = elem.a * (1 - elem.e)
        ra = elem.a * (1 + elem.e)
        try:
            period = elem.period_s
        except ValueError:
            period = float("nan")
        self.hud.update(
            sim_time_s=self.clock.sim_time_s,
            warp=self.clock.warp,
            altitude_m=v0.state.r_mag - self.world.central.radius_m,
            speed_mps=v0.state.v_mag,
            periapsis_m=rp - self.world.central.radius_m,
            apoapsis_m=ra - self.world.central.radius_m,
            period_s=period,
        )
        return task.cont

    def run_app(self) -> None:
        self.run()
```

- [ ] **Step 5: Smoke-check imports (no window opened)**

Run:
```bash
.venv/Scripts/python -c "import orbitsim.render.app; import orbitsim.render.hud; print('render imports ok')"
```
Expected: prints `render imports ok` (and does NOT open a window — only imports).

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/orbit_lines.py orbitsim/render/camera_rig.py orbitsim/render/hud/ orbitsim/render/app.py
git commit -m "Phase 2 Task 8: Panda3D app, orbit-line node, camera rig, HUD"
```

---

## Task 9: Wire `__main__` + HUMAN VISUAL CHECKPOINT

**Files:**
- Modify: `orbitsim/__main__.py`

**Interfaces:**
- Consumes: `SimClock`, `World`, `Vessel`, `EARTH`, `StateVector`, `OrbitApp`.
- Produces: `main()` entry point that builds a default LEO scenario and runs the app.

- [ ] **Step 1: Implement __main__.py**

Replace `orbitsim/__main__.py` with:
```python
"""Entry point for `python -m orbitsim`: launches the render app."""
import numpy as np

from orbitsim.core.bodies import EARTH
from orbitsim.core.constants import MU_EARTH, R_EARTH
from orbitsim.core.state import StateVector
from orbitsim.sim.clock import SimClock
from orbitsim.sim.world import Vessel, World


def _default_world() -> World:
    # A slightly eccentric, inclined LEO so the orbit line is visibly non-circular.
    r0 = R_EARTH + 500e3
    v_circ = np.sqrt(MU_EARTH / r0)
    state = StateVector(
        r=np.array([r0, 0.0, 0.0]),
        v=np.array([0.0, v_circ * 1.05, v_circ * 0.15]),
        mu=MU_EARTH,
    )
    vessel = Vessel(name="Sandbox-1", state=state, delta_v_budget_mps=2000.0)
    return World(central=EARTH, vessels=[vessel])


def main() -> None:
    from orbitsim.render.app import OrbitApp  # imported here so tests can skip graphics

    world = _default_world()
    clock = SimClock(sim_time_s=0.0, warp=100.0)
    app = OrbitApp(world, clock)
    app.run_app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Full suite stays green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS (all Phase 1 + Phase 2 unit tests). No graphics opened.

- [ ] **Step 3: HUMAN VISUAL CHECKPOINT — launch the app**

Run: `.venv/Scripts/python -m orbitsim`

A reviewer must confirm ALL of the following, then report back:
1. A window opens showing a blue Earth sphere and a yellow vessel marker on a visible orbit line.
2. The vessel moves along its orbit (time warp is x100). The HUD shows sim time, warp, altitude, speed, Pe/Ap, period — all updating and physically sensible (LEO altitude a few hundred km, speed ~7.6 km/s, period ~90+ min).
3. Mouse wheel zooms in/out across a wide range. **Zoom all the way in to the vessel marker — it must be rock-steady, no jitter/shaking** (this verifies the floating origin).
4. Arrow keys orbit the camera; `,` and `.` step warp down/up and the orbit speed changes smoothly.

If jitter appears at deep zoom, the bug is in `RenderTransform.to_render` or in `_update` not re-centering `origin_m` on the focus each frame — fix before proceeding.

- [ ] **Step 4: Commit**

```bash
git add orbitsim/__main__.py
git commit -m "Phase 2 Task 9: wire default scenario + launch entry point"
```

---

## Phase 2 Exit Criteria

- `python -m orbitsim` shows Earth + a vessel on a visible orbit line.
- Camera focuses/zooms across the full range without jitter (verified at deep zoom).
- Time warp speeds the orbit smoothly; HUD shows correct live altitude/speed/period.
- `pytest tests/ -q` is fully green (all `to_render`, clock, world, sampling, camera-mapping tests pass).

Then proceed to `docs/superpowers/plans/2026-06-24-phase3-maneuvers.md`.

## Self-Review Notes

- Spec coverage: clock (2.1), world (2.2), floating origin (2.3), orbit lines (2.4), camera (2.5), app+main (2.6), HUD (2.7) — all mapped. Pre-built bodies gap from Phase 1 closed in Task 2.
- Non-testable graphics isolated to Tasks 7–9 with a single explicit visual checkpoint; all math (clock, world, transform, sampling, zoom mapping) is red-green TDD.
