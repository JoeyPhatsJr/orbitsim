# Continuous-Thrust Flight Model + Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the player fly a real rocket in real time — point it, throttle up, watch fuel drain and the trajectory bend under gravity — with a full 3D navball, while the analytic engine (Phases 1–5) still handles coasting, time-warp, and planning.

**Architecture:** Hybrid coast/burn. When coasting, the vessel stays on analytic Kepler rails (time-warp + planning intact). While thrusting, it integrates numerically (RK4) under two-body gravity + thrust with a draining mass (Tsiolkovsky), and time-warp is forced to 1×. All flight math is pure functions in `core/attitude.py` and `core/flight.py` (TDD with known answers); the sim layer adds propulsion/attitude state to `Vessel`; the renderer adds keybinds, a navball, and HUD. Design spec: `docs/superpowers/specs/2026-06-24-continuous-thrust-flight-design.md`.

**Tech Stack:** Python 3.10, numpy, Panda3D, pytest.

## Global Constraints

- SI units everywhere in `core/`/`sim/`: meters, seconds, radians, m/s, kg, newtons. Convert to km/% only at the HUD boundary.
- `core/` must NEVER import `panda3d`/`sim`/`render`. Pure float64, numpy arrays shape `(3,)` (or `(4,)` for quaternions).
- Frozen dataclasses stay frozen; `Vessel` (sim layer) is the mutable exception — reassign fields, never mutate `StateVector`.
- Quaternion convention: numpy array `[w, x, y, z]`, unit norm. Ship **nose** = body `+Z` axis rotated by `orientation`.
- Angles normalized; clamp `arccos` arguments to `[-1, 1]`.
- Raise `ValueError` on invalid physics input; never silently clamp.
- `black` is NOT installed in this venv — write clean code at line length ≤ 100, skip formatting.
- Always use `.venv/Scripts/python`. Run the full suite with `.venv/Scripts/python -m pytest tests/ -q`.
- Commit after each task with the exact message given. Use ONLY `git add <specific files>` — NEVER `git add -A`/`git add .`. Never stage: `data/`, `debug_curtis.py`, `kickbacks.vsix`, `.hypothesis/`, `CLAUDE.md`, `porkchop.png`.
- Render tasks end with a HUMAN VISUAL CHECKPOINT; verify headlessly first with `loadPrcFileData("", "window-type offscreen")` + `app.taskMgr.step()` + offscreen screenshots.

## Gate

Phase 5 complete (94 tests green, `--solar` viewer works). This plan is sub-project #1 of the "playable game" effort; missions/save/packaging are separate later cycles.

## Phase 1–5 API available

```python
from orbitsim.core.state import StateVector            # frozen; .r, .v, .mu, .epoch_s, .r_mag, .v_mag, .angular_momentum, .specific_energy
from orbitsim.core.elements import state_to_elements, elements_to_state
from orbitsim.core.propagate import propagate_kepler   # (state, dt) -> StateVector, advances epoch_s by dt
from orbitsim.core.constants import MU_EARTH
from orbitsim.sim.world import Vessel, World           # Vessel mutable; World.step(sim_dt)
```

---

## Task 1: Quaternion helpers (core/attitude.py)

**Files:**
- Create: `orbitsim/core/attitude.py`
- Test: `tests/core/test_attitude.py`

**Interfaces:**
- Produces:
  ```python
  def quat_identity() -> np.ndarray                       # [1,0,0,0]
  def quat_normalize(q: np.ndarray) -> np.ndarray
  def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray
  def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray
  def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray
  def angle_between(u: np.ndarray, v: np.ndarray) -> float   # radians, [0, pi]
  def nose_direction(q: np.ndarray) -> np.ndarray            # quat_rotate_vector(q, [0,0,1])
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_attitude.py`:
```python
"""Tests for quaternion attitude helpers."""
import numpy as np
from orbitsim.core.attitude import (
    quat_identity, quat_normalize, quat_from_axis_angle, quat_multiply,
    quat_rotate_vector, angle_between, nose_direction,
)


def test_identity_rotates_nothing():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(quat_rotate_vector(quat_identity(), v), v)


def test_nose_of_identity_is_plus_z():
    assert np.allclose(nose_direction(quat_identity()), [0.0, 0.0, 1.0])


def test_90deg_about_x_maps_z_to_minus_y():
    q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    out = quat_rotate_vector(q, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(out, [0.0, -1.0, 0.0], atol=1e-9)


def test_90deg_about_z_maps_x_to_y():
    q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    out = quat_rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_multiply_composes_rotations():
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    # Applying qx twice == 180 deg about x: z -> -z.
    q2 = quat_multiply(qx, qx)
    assert np.allclose(quat_rotate_vector(q2, np.array([0.0, 0.0, 1.0])), [0.0, 0.0, -1.0], atol=1e-9)


def test_rotation_preserves_length():
    q = quat_from_axis_angle(np.array([1.0, 1.0, 1.0]), 1.234)
    v = np.array([3.0, -2.0, 0.5])
    assert abs(np.linalg.norm(quat_rotate_vector(q, v)) - np.linalg.norm(v)) < 1e-12


def test_angle_between_orthogonal_and_clamped():
    assert abs(angle_between(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) - np.pi / 2) < 1e-12
    # Identical directions -> 0 even with float error (arccos clamp).
    u = np.array([1.0, 1.0, 1.0])
    assert angle_between(u, u) < 1e-7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.core.attitude).

- [ ] **Step 3: Implement attitude.py**

Create `orbitsim/core/attitude.py`:
```python
"""Pure quaternion attitude helpers (float64, SI). Convention: q = [w, x, y, z],
unit norm. The ship's nose is the body +Z axis rotated by the orientation quaternion."""
import numpy as np

_NOSE_BODY = np.array([0.0, 0.0, 1.0])


def quat_identity() -> np.ndarray:
    """Identity rotation [1, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Quaternion for a rotation of angle_rad about `axis` (axis need not be unit)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return quat_identity()
    axis = axis / n
    half = angle_rad / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b (apply b first, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by quaternion q."""
    w = q[0]
    u = np.asarray(q[1:], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle [0, pi] between two non-zero vectors (arccos argument clamped)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        raise ValueError("angle_between requires non-zero vectors")
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(c))


def nose_direction(q: np.ndarray) -> np.ndarray:
    """Unit nose (thrust) direction = body +Z rotated by q."""
    return quat_rotate_vector(q, _NOSE_BODY)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/attitude.py tests/core/test_attitude.py
git commit -m "Flight Task 1: quaternion attitude helpers"
```

---

## Task 2: Attitude slew (rotate toward a target, no overshoot)

**Files:**
- Modify: `orbitsim/core/attitude.py`
- Test: `tests/core/test_attitude.py` (append)

**Interfaces:**
- Consumes: `quat_from_axis_angle`, `quat_multiply`, `quat_normalize`, `quat_rotate_vector`, `nose_direction`, `angle_between`.
- Produces:
  ```python
  def slew_toward(q: np.ndarray, target_dir: np.ndarray, max_rate_radps: float, dt_s: float) -> np.ndarray
      # rotate q so its nose moves toward target_dir by at most max_rate*dt; never overshoot
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_attitude.py`:
```python
from orbitsim.core.attitude import slew_toward


def test_slew_reaches_target_in_expected_time():
    q = quat_identity()                      # nose +Z
    target = np.array([1.0, 0.0, 0.0])       # +X, 90 deg away
    rate = 0.5                               # rad/s
    dt = 0.1
    # pi/2 / 0.5 = ~3.14 s -> ~32 steps; run 40 to be safe.
    for _ in range(40):
        q = slew_toward(q, target, rate, dt)
    assert angle_between(nose_direction(q), target) < 1e-6


def test_slew_single_step_never_overshoots():
    q = quat_identity()
    target = np.array([1.0, 0.0, 0.0])
    before = angle_between(nose_direction(q), target)
    q2 = slew_toward(q, target, max_rate_radps=100.0, dt_s=1.0)  # huge step
    after = angle_between(nose_direction(q2), target)
    assert after <= before + 1e-9            # clamped to the target, no overshoot
    assert after < 1e-6                       # lands exactly on target


def test_slew_handles_antiparallel_target():
    q = quat_identity()                      # nose +Z
    target = np.array([0.0, 0.0, -1.0])      # exactly opposite
    for _ in range(400):
        q = slew_toward(q, target, max_rate_radps=0.5, dt_s=0.1)
    assert angle_between(nose_direction(q), target) < 1e-3


def test_slew_rate_limited_per_step():
    q = quat_identity()
    target = np.array([1.0, 0.0, 0.0])       # 90 deg away
    q2 = slew_toward(q, target, max_rate_radps=0.1, dt_s=1.0)  # only 0.1 rad allowed
    moved = angle_between(nose_direction(q), nose_direction(q2))
    assert abs(moved - 0.1) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -k slew -q`
Expected: FAIL (ImportError: cannot import name 'slew_toward').

- [ ] **Step 3: Implement slew_toward**

Append to `orbitsim/core/attitude.py`:
```python
def slew_toward(
    q: np.ndarray,
    target_dir: np.ndarray,
    max_rate_radps: float,
    dt_s: float,
) -> np.ndarray:
    """Rotate q so its nose turns toward target_dir, by at most max_rate*dt (no overshoot).

    Parameters
    ----------
    q : np.ndarray
        Current orientation quaternion [w, x, y, z].
    target_dir : np.ndarray
        Desired nose direction (need not be unit).
    max_rate_radps : float
        Maximum slew rate [rad/s].
    dt_s : float
        Time step [s].
    """
    nose = nose_direction(q)
    target = np.asarray(target_dir, dtype=np.float64)
    if np.linalg.norm(target) == 0.0:
        return q
    gap = angle_between(nose, target)
    if gap < 1e-12:
        return q
    step = min(gap, max_rate_radps * dt_s)
    if step <= 0.0:
        return q
    axis = np.cross(nose, target)
    if np.linalg.norm(axis) < 1e-12:
        # Parallel or antiparallel: pick any axis perpendicular to the nose.
        seed = np.array([1.0, 0.0, 0.0]) if abs(nose[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(nose, seed)
    dq = quat_from_axis_angle(axis, step)
    return quat_normalize(quat_multiply(dq, q))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/attitude.py tests/core/test_attitude.py
git commit -m "Flight Task 2: rate-limited attitude slew"
```

---

## Task 3: SAS target directions from orbital state

**Files:**
- Modify: `orbitsim/core/attitude.py`
- Test: `tests/core/test_attitude.py` (append)

**Interfaces:**
- Consumes: `StateVector`.
- Produces:
  ```python
  SAS_MODES = ("PROGRADE","RETROGRADE","NORMAL","ANTINORMAL","RADIAL_IN","RADIAL_OUT","TARGET","ANTITARGET")
  def sas_target_dir(mode: str, state: StateVector, target_pos: np.ndarray | None = None) -> np.ndarray
      # unit direction for the given SAS hold mode; raises ValueError if TARGET without target_pos
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_attitude.py`:
```python
import pytest
from orbitsim.core.attitude import sas_target_dir
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH


def _leo_state() -> StateVector:
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(MU_EARTH / 7.0e6), 0.0])  # +y prograde, h along +z
    return StateVector(r=r, v=v, mu=MU_EARTH)


def test_prograde_is_velocity_direction():
    s = _leo_state()
    assert np.allclose(sas_target_dir("PROGRADE", s), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(sas_target_dir("RETROGRADE", s), [0.0, -1.0, 0.0], atol=1e-12)


def test_normal_is_angular_momentum_direction():
    s = _leo_state()
    assert np.allclose(sas_target_dir("NORMAL", s), [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(sas_target_dir("ANTINORMAL", s), [0.0, 0.0, -1.0], atol=1e-12)


def test_radial_out_points_away_from_central_body():
    s = _leo_state()
    # RTN radial-out = h_hat x v_hat = +x here.
    assert np.allclose(sas_target_dir("RADIAL_OUT", s), [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(sas_target_dir("RADIAL_IN", s), [-1.0, 0.0, 0.0], atol=1e-12)


def test_target_points_at_target():
    s = _leo_state()
    tgt = np.array([7.0e6, 1.0e6, 0.0])
    d = sas_target_dir("TARGET", s, target_pos=tgt)
    expected = (tgt - s.r) / np.linalg.norm(tgt - s.r)
    assert np.allclose(d, expected, atol=1e-12)


def test_target_without_position_raises():
    with pytest.raises(ValueError):
        sas_target_dir("TARGET", _leo_state())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -k sas -q`
Expected: FAIL (ImportError: cannot import name 'sas_target_dir').

- [ ] **Step 3: Implement sas_target_dir**

Append to `orbitsim/core/attitude.py`:
```python
from orbitsim.core.state import StateVector

SAS_MODES = (
    "PROGRADE", "RETROGRADE", "NORMAL", "ANTINORMAL",
    "RADIAL_IN", "RADIAL_OUT", "TARGET", "ANTITARGET",
)


def sas_target_dir(mode, state: StateVector, target_pos=None) -> np.ndarray:
    """Unit nose direction for an SAS hold mode, from the vessel's orbital state.

    Radial-out uses the orthonormal RTN axis h_hat x v_hat (consistent with
    core.maneuvers), not r/|r|.
    """
    v_hat = state.v / np.linalg.norm(state.v)
    h = np.cross(state.r, state.v)
    h_hat = h / np.linalg.norm(h)
    radial_out = np.cross(h_hat, v_hat)
    mode = mode.upper()
    if mode == "PROGRADE":
        return v_hat
    if mode == "RETROGRADE":
        return -v_hat
    if mode == "NORMAL":
        return h_hat
    if mode == "ANTINORMAL":
        return -h_hat
    if mode == "RADIAL_OUT":
        return radial_out
    if mode == "RADIAL_IN":
        return -radial_out
    if mode in ("TARGET", "ANTITARGET"):
        if target_pos is None:
            raise ValueError(f"{mode} requires target_pos")
        d = np.asarray(target_pos, dtype=np.float64) - state.r
        n = np.linalg.norm(d)
        if n == 0.0:
            raise ValueError("target coincides with vessel")
        d = d / n
        return d if mode == "TARGET" else -d
    raise ValueError(f"unknown SAS mode: {mode}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -q`
Expected: PASS (16 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/attitude.py tests/core/test_attitude.py
git commit -m "Flight Task 3: SAS target directions"
```

---

## Task 4: Rocket-equation functions (core/flight.py)

**Files:**
- Create: `orbitsim/core/flight.py`
- Test: `tests/core/test_flight.py`

**Interfaces:**
- Produces:
  ```python
  def tsiolkovsky_dv(ve_mps: float, m0_kg: float, mf_kg: float) -> float       # ve*ln(m0/mf)
  def mass_flow_rate(throttle: float, max_thrust_n: float, ve_mps: float) -> float  # throttle*thrust/ve
  def thrust_accel_mps2(throttle: float, max_thrust_n: float, mass_kg: float) -> float  # throttle*thrust/mass
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_flight.py`:
```python
"""Tests for rocket-equation flight physics."""
import numpy as np
import pytest
from orbitsim.core.flight import tsiolkovsky_dv, mass_flow_rate, thrust_accel_mps2


def test_tsiolkovsky_known_answer():
    # ve=3000, m0=2000, mf=1000 -> 3000*ln(2) = 2079.44 m/s.
    assert abs(tsiolkovsky_dv(3000.0, 2000.0, 1000.0) - 3000.0 * np.log(2.0)) < 1e-9


def test_tsiolkovsky_zero_fuel_is_zero_dv():
    assert tsiolkovsky_dv(3000.0, 1000.0, 1000.0) == 0.0


def test_tsiolkovsky_rejects_bad_masses():
    with pytest.raises(ValueError):
        tsiolkovsky_dv(3000.0, 1000.0, 2000.0)   # mf > m0


def test_mass_flow_rate():
    # ṁ = throttle*thrust/ve = 1.0 * 30000 / 3000 = 10 kg/s.
    assert abs(mass_flow_rate(1.0, 30000.0, 3000.0) - 10.0) < 1e-12
    assert mass_flow_rate(0.0, 30000.0, 3000.0) == 0.0


def test_thrust_accel():
    # a = throttle*thrust/mass = 0.5 * 30000 / 1500 = 10 m/s^2.
    assert abs(thrust_accel_mps2(0.5, 30000.0, 1500.0) - 10.0) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_flight.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.core.flight).

- [ ] **Step 3: Implement the rocket-equation functions**

Create `orbitsim/core/flight.py`:
```python
"""Continuous-thrust flight physics (pure, float64, SI). Real rocket equation +
two-body powered RK4 integrator."""
import numpy as np

from orbitsim.core.state import StateVector


def tsiolkovsky_dv(ve_mps: float, m0_kg: float, mf_kg: float) -> float:
    """Ideal delta-V = ve * ln(m0/mf) [m/s]. Requires 0 < mf <= m0, ve > 0."""
    if ve_mps <= 0.0:
        raise ValueError(f"ve must be positive, got {ve_mps}")
    if mf_kg <= 0.0 or m0_kg < mf_kg:
        raise ValueError(f"need 0 < mf <= m0, got m0={m0_kg}, mf={mf_kg}")
    return float(ve_mps * np.log(m0_kg / mf_kg))


def mass_flow_rate(throttle: float, max_thrust_n: float, ve_mps: float) -> float:
    """Propellant mass flow ṁ = throttle * thrust / ve [kg/s]."""
    if ve_mps <= 0.0:
        raise ValueError(f"ve must be positive, got {ve_mps}")
    return float(throttle * max_thrust_n / ve_mps)


def thrust_accel_mps2(throttle: float, max_thrust_n: float, mass_kg: float) -> float:
    """Thrust acceleration magnitude = throttle * thrust / mass [m/s^2]."""
    if mass_kg <= 0.0:
        raise ValueError(f"mass must be positive, got {mass_kg}")
    return float(throttle * max_thrust_n / mass_kg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_flight.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/flight.py tests/core/test_flight.py
git commit -m "Flight Task 4: rocket-equation functions"
```

---

## Task 5: Powered RK4 integrator

**Files:**
- Modify: `orbitsim/core/flight.py`
- Test: `tests/core/test_flight.py` (append)

**Interfaces:**
- Consumes: `StateVector`, `mass_flow_rate`.
- Produces:
  ```python
  def integrate_powered(
      state: StateVector, dry_mass_kg: float, fuel_kg: float, thrust_dir_unit: np.ndarray,
      throttle: float, max_thrust_n: float, ve_mps: float, dt_s: float, substeps: int = 50,
  ) -> tuple[StateVector, float]
      # RK4 under two-body gravity + thrust with draining mass; returns (new_state, new_fuel_kg)
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_flight.py`:
```python
from orbitsim.core.flight import integrate_powered
from orbitsim.core.state import StateVector
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.constants import MU_EARTH


def test_zero_throttle_matches_kepler():
    """With no thrust the integrator is pure two-body gravity -> must track Kepler."""
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(MU_EARTH / 7.0e6), 0.0])
    s = StateVector(r=r, v=v, mu=MU_EARTH)
    dt = 120.0
    out, fuel = integrate_powered(
        s, dry_mass_kg=1000.0, fuel_kg=500.0, thrust_dir_unit=np.array([0.0, 1.0, 0.0]),
        throttle=0.0, max_thrust_n=30000.0, ve_mps=3000.0, dt_s=dt, substeps=200,
    )
    kep = propagate_kepler(s, dt)
    assert fuel == 500.0                                   # no fuel burned
    assert np.linalg.norm(out.r - kep.r) < 1.0            # within 1 m of analytic
    # Energy conserved (no thrust): epsilon unchanged.
    assert abs(out.specific_energy - s.specific_energy) / abs(s.specific_energy) < 1e-7


def test_free_space_burn_matches_rocket_equation():
    """In free space (mu=0), a full burn to depletion reaches ve*ln(m0/mf)."""
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]), mu=0.0)
    dry, fuel0, thrust, ve = 1000.0, 1000.0, 30000.0, 3000.0
    mdot = thrust / ve                                     # 10 kg/s
    burn_time = fuel0 / mdot                               # 100 s
    out, fuel = integrate_powered(
        s, dry_mass_kg=dry, fuel_kg=fuel0, thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=thrust, ve_mps=ve, dt_s=burn_time, substeps=2000,
    )
    expected_dv = ve * np.log((dry + fuel0) / dry)         # 3000*ln(2)=2079.44
    assert abs(out.v[0] - expected_dv) / expected_dv < 1e-3
    assert abs(fuel) < 1e-6                                 # fuel depleted


def test_burn_stops_when_fuel_exhausted():
    """Asking for more burn than fuel allows: speed caps at the rocket-equation dv."""
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]), mu=0.0)
    out, fuel = integrate_powered(
        s, dry_mass_kg=1000.0, fuel_kg=100.0, thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=30000.0, ve_mps=3000.0, dt_s=1000.0, substeps=2000,
    )
    expected_dv = 3000.0 * np.log(1100.0 / 1000.0)
    assert fuel == 0.0
    assert abs(out.v[0] - expected_dv) / expected_dv < 2e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_flight.py -k "kepler or rocket or exhaust" -q`
Expected: FAIL (ImportError: cannot import name 'integrate_powered').

- [ ] **Step 3: Implement integrate_powered**

Append to `orbitsim/core/flight.py`:
```python
def _accel(r, v, fuel, dry_mass_kg, thrust_dir_unit, throttle, max_thrust_n, ve_mps, mu):
    """Acceleration [m/s^2] = two-body gravity + thrust (thrust off when fuel <= 0)."""
    a = np.zeros(3)
    rn = np.linalg.norm(r)
    if mu != 0.0 and rn > 0.0:
        a = a - mu * r / rn**3
    if fuel > 0.0 and throttle > 0.0:
        mass = dry_mass_kg + fuel
        a = a + (throttle * max_thrust_n / mass) * thrust_dir_unit
    return a


def integrate_powered(
    state: StateVector,
    dry_mass_kg: float,
    fuel_kg: float,
    thrust_dir_unit: np.ndarray,
    throttle: float,
    max_thrust_n: float,
    ve_mps: float,
    dt_s: float,
    substeps: int = 50,
) -> tuple:
    """Integrate r, v, fuel over dt_s under two-body gravity + thrust (fixed-step RK4).

    Mass decreases as fuel burns (real rocket equation). Thrust direction is held
    constant over the interval (the sim layer slews attitude separately). When fuel
    reaches zero, thrust stops mid-interval.

    Returns
    -------
    (StateVector, float)
        New state (same mu/epoch_s+dt) and remaining fuel [kg].
    """
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    thrust_dir_unit = np.asarray(thrust_dir_unit, dtype=np.float64)
    r = np.asarray(state.r, dtype=np.float64).copy()
    v = np.asarray(state.v, dtype=np.float64).copy()
    fuel = float(fuel_kg)
    mu = state.mu
    h = dt_s / substeps
    mdot = mass_flow_rate(throttle, max_thrust_n, ve_mps) if ve_mps > 0 else 0.0

    def deriv(r_, v_, fuel_):
        a = _accel(r_, v_, fuel_, dry_mass_kg, thrust_dir_unit, throttle,
                   max_thrust_n, ve_mps, mu)
        df = -mdot if fuel_ > 0.0 else 0.0
        return v_, a, df

    for _ in range(substeps):
        k1r, k1v, k1f = deriv(r, v, fuel)
        k2r, k2v, k2f = deriv(r + 0.5 * h * k1r, v + 0.5 * h * k1v, fuel + 0.5 * h * k1f)
        k3r, k3v, k3f = deriv(r + 0.5 * h * k2r, v + 0.5 * h * k2v, fuel + 0.5 * h * k2f)
        k4r, k4v, k4f = deriv(r + h * k3r, v + h * k3v, fuel + h * k3f)
        r = r + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        fuel = fuel + (h / 6.0) * (k1f + 2 * k2f + 2 * k3f + k4f)
        if fuel < 0.0:
            fuel = 0.0

    return (
        StateVector(r=r, v=v, mu=mu, epoch_s=state.epoch_s + dt_s),
        float(fuel),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_flight.py -q`
Expected: PASS (8 passed). If `test_zero_throttle_matches_kepler` is marginally over 1 m, raise `substeps` in the test call (do NOT loosen the assertion); RK4 at 200 substeps over 120 s on LEO is well under 1 m.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/flight.py tests/core/test_flight.py
git commit -m "Flight Task 5: powered RK4 integrator (two-body + thrust, draining mass)"
```

---

## Task 6: Vessel propulsion + attitude state (sim layer)

**Files:**
- Modify: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py` (append)

**Interfaces:**
- Consumes: `quat_identity`, `tsiolkovsky_dv`.
- Produces: new `Vessel` fields + derived helpers:
  ```python
  Vessel(..., dry_mass_kg=1000.0, fuel_mass_kg=0.0, max_thrust_n=0.0, exhaust_velocity_mps=3000.0,
         max_turn_rate_radps=0.6, throttle=0.0, sas_mode="OFF", orientation=<quat_identity>)
  vessel.mass_kg -> float          # dry + fuel
  vessel.delta_v_remaining -> float  # tsiolkovsky(ve, mass, dry) ; 0 if no fuel
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/sim/test_world.py`:
```python
def test_vessel_propulsion_defaults_and_derived():
    import numpy as np
    from orbitsim.core.attitude import quat_identity
    v = _circular_vessel()
    # New fields exist with defaults.
    assert v.throttle == 0.0
    assert v.sas_mode == "OFF"
    assert np.allclose(v.orientation, quat_identity())
    # Configure a rocket and check derived quantities.
    v.dry_mass_kg = 1000.0
    v.fuel_mass_kg = 1000.0
    v.exhaust_velocity_mps = 3000.0
    assert v.mass_kg == 2000.0
    assert abs(v.delta_v_remaining - 3000.0 * np.log(2.0)) < 1e-6


def test_vessel_delta_v_zero_without_fuel():
    v = _circular_vessel()
    v.dry_mass_kg = 1000.0
    v.fuel_mass_kg = 0.0
    assert v.delta_v_remaining == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -k propulsion -q`
Expected: FAIL (AttributeError: 'Vessel' object has no attribute 'throttle').

- [ ] **Step 3: Add the fields + derived properties**

In `orbitsim/sim/world.py`, update the imports and the `Vessel` dataclass. Add near the top:
```python
import numpy as np
from orbitsim.core.attitude import quat_identity
from orbitsim.core.flight import tsiolkovsky_dv
```
Replace the `Vessel` dataclass with:
```python
@dataclass
class Vessel:
    """A point-mass vessel: orbital state, propulsion, and attitude.

    delta_v_budget_mps is retained for display/back-compat; the authoritative
    delta-V is the derived `delta_v_remaining` (rocket equation).
    """

    name: str
    state: StateVector
    delta_v_budget_mps: float = 0.0
    nodes: list = field(default_factory=list)
    # Propulsion (SI).
    dry_mass_kg: float = 1000.0
    fuel_mass_kg: float = 0.0
    max_thrust_n: float = 0.0
    exhaust_velocity_mps: float = 3000.0
    # Attitude / control.
    max_turn_rate_radps: float = 0.6
    throttle: float = 0.0
    sas_mode: str = "OFF"
    orientation: np.ndarray = field(default_factory=quat_identity)

    @property
    def mass_kg(self) -> float:
        """Current total mass = dry + fuel [kg]."""
        return self.dry_mass_kg + self.fuel_mass_kg

    @property
    def delta_v_remaining(self) -> float:
        """Remaining delta-V from the rocket equation [m/s]; 0 if no fuel."""
        if self.fuel_mass_kg <= 0.0:
            return 0.0
        return tsiolkovsky_dv(self.exhaust_velocity_mps, self.mass_kg, self.dry_mass_kg)
```
(Keep the existing `nodes` import of `ManeuverNode` typing if present; `list` annotation is fine since the type isn't enforced.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -q`
Expected: PASS (existing world tests still green + 2 new).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "Flight Task 6: Vessel propulsion + attitude state"
```

---

## Task 7: World.step coast/burn branch + attitude slew

**Files:**
- Modify: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py` (append)

**Interfaces:**
- Consumes: `integrate_powered`, `slew_toward`, `sas_target_dir`, `nose_direction`, `propagate_kepler`.
- Produces: `World.step(sim_dt_s)` now slews attitude every tick and integrates powered flight when `throttle>0 and fuel>0`, else analytic Kepler. Adds `World.any_thrusting() -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/sim/test_world.py`:
```python
def test_world_step_burn_drains_fuel_and_adds_speed():
    import numpy as np
    from orbitsim.core.attitude import quat_identity
    v = _circular_vessel()
    v.dry_mass_kg = 1000.0
    v.fuel_mass_kg = 500.0
    v.max_thrust_n = 30000.0
    v.exhaust_velocity_mps = 3000.0
    v.throttle = 1.0
    v.sas_mode = "PROGRADE"
    # Point the nose prograde so the burn adds energy.
    world = World(central=EARTH, vessels=[v])
    speed0 = v.state.v_mag
    fuel0 = v.fuel_mass_kg
    # Slew first so the nose is prograde, then a short burn.
    for _ in range(60):
        world.step(0.1)
    assert v.fuel_mass_kg < fuel0          # fuel burned
    assert v.state.v_mag > speed0          # prograde burn sped us up
    assert world.any_thrusting() is True


def test_world_step_coast_is_on_rails():
    v = _circular_vessel()
    v.throttle = 0.0
    world = World(central=EARTH, vessels=[v])
    import numpy as np
    period = state_to_elements(v.state).period_s
    world.step(period)
    pos_err = np.linalg.norm(world.vessels[0].state.r - np.array([7.0e6, 0.0, 0.0]))
    assert pos_err < 1e-3                   # analytic period closure preserved
    assert world.any_thrusting() is False


def test_world_step_slews_attitude_toward_prograde():
    import numpy as np
    from orbitsim.core.attitude import nose_direction
    v = _circular_vessel()
    v.sas_mode = "PROGRADE"
    v.throttle = 0.0                        # slewing works while coasting
    world = World(central=EARTH, vessels=[v])
    for _ in range(100):
        world.step(0.1)
    prograde = v.state.v / v.state.v_mag
    # Nose should have turned to (near) prograde.
    assert np.dot(nose_direction(v.orientation), prograde) > 0.999
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/sim/test_world.py -k "burn or coast_is_on_rails or slews" -q`
Expected: FAIL (AttributeError: 'World' object has no attribute 'any_thrusting').

- [ ] **Step 3: Implement the step branch**

In `orbitsim/sim/world.py`, add imports:
```python
from orbitsim.core.flight import integrate_powered
from orbitsim.core.attitude import slew_toward, sas_target_dir, nose_direction
```
Replace `World.step` and add `any_thrusting`:
```python
    def step(self, sim_dt_s: float) -> None:
        """Advance every vessel by sim_dt_s: slew attitude, then translate.

        Coasting vessels propagate analytically (on rails); thrusting vessels
        (throttle>0 and fuel) integrate numerically under gravity + thrust.
        """
        for vessel in self.vessels:
            # 1) Attitude: slew toward the SAS hold direction (if any) each tick.
            if vessel.sas_mode not in ("OFF", "STABILITY"):
                try:
                    target = sas_target_dir(vessel.sas_mode, vessel.state)
                except ValueError:
                    target = None
                if target is not None:
                    vessel.orientation = slew_toward(
                        vessel.orientation, target, vessel.max_turn_rate_radps, sim_dt_s
                    )
            # 2) Translation.
            if vessel.throttle > 0.0 and vessel.fuel_mass_kg > 0.0:
                new_state, new_fuel = integrate_powered(
                    vessel.state,
                    dry_mass_kg=vessel.dry_mass_kg,
                    fuel_kg=vessel.fuel_mass_kg,
                    thrust_dir_unit=nose_direction(vessel.orientation),
                    throttle=vessel.throttle,
                    max_thrust_n=vessel.max_thrust_n,
                    ve_mps=vessel.exhaust_velocity_mps,
                    dt_s=sim_dt_s,
                )
                vessel.state = new_state
                vessel.fuel_mass_kg = new_fuel
            else:
                vessel.state = propagate_kepler(vessel.state, sim_dt_s)

    def any_thrusting(self) -> bool:
        """True if any vessel is currently producing thrust (throttle>0 and fuel)."""
        return any(v.throttle > 0.0 and v.fuel_mass_kg > 0.0 for v in self.vessels)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS (all prior tests + new; ~110+ tests).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "Flight Task 7: World.step coast/burn branch + attitude slew"
```

---

## Task 8: Flight controls + warp lock (render) — HUMAN VISUAL CHECKPOINT

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `World.any_thrusting`, `Vessel` propulsion fields, `quat_*`, `sas_target_dir`.
- Produces: throttle/attitude/SAS keybinds that mutate vessel 0; warp forced to 1× while thrusting. Sandbox mode only (skip when `self.solar_system`).

- [ ] **Step 1: Add a per-frame control-input pass + keybinds**

In `orbitsim/render/app.py`, add imports near the top:
```python
from orbitsim.core.attitude import (
    quat_from_axis_angle, quat_multiply, quat_normalize, quat_rotate_vector, nose_direction,
)
```
In `_setup_input`, after the existing bindings, add (guarded to sandbox flight):
```python
        if not self.solar_system and self.world.vessels:
            self._keys = {k: False for k in
                          ("w", "s", "a", "d", "q", "e", "shift", "control")}
            for k in list(self._keys):
                self.accept(k, self._set_key, [k, True])
                self.accept(f"{k}-up", self._set_key, [k, False])
            self.accept("z", self._throttle_full)
            self.accept("x", self._throttle_cut)
            self.accept("t", self._toggle_sas)
            sas_keys = ["PROGRADE", "RETROGRADE", "NORMAL", "ANTINORMAL",
                        "RADIAL_IN", "RADIAL_OUT", "TARGET"]
            for i, mode in enumerate(sas_keys, start=1):
                self.accept(str(i), self._set_sas, [mode])
```
Add these methods to `OrbitApp`:
```python
    ROTATE_RATE_RADPS = 0.8       # manual pitch/yaw/roll rate
    THROTTLE_STEP = 0.5           # per second for shift/ctrl

    def _set_key(self, key, down):
        self._keys[key] = down

    def _throttle_full(self):
        self.world.vessels[0].throttle = 1.0

    def _throttle_cut(self):
        self.world.vessels[0].throttle = 0.0

    def _toggle_sas(self):
        v = self.world.vessels[0]
        v.sas_mode = "STABILITY" if v.sas_mode == "OFF" else "OFF"

    def _set_sas(self, mode):
        self.world.vessels[0].sas_mode = mode

    def _apply_flight_input(self, dt):
        """Manual throttle + rotation from held keys (sandbox flight)."""
        if self.solar_system or not self.world.vessels:
            return
        v = self.world.vessels[0]
        k = self._keys
        # Throttle trim.
        if k["shift"]:
            v.throttle = min(1.0, v.throttle + self.THROTTLE_STEP * dt)
        if k["control"]:
            v.throttle = max(0.0, v.throttle - self.THROTTLE_STEP * dt)
        # Manual rotation overrides SAS hold when any rotate key is held.
        ax = np.zeros(3)
        if k["w"]:
            ax = ax + np.array([1.0, 0.0, 0.0])   # pitch
        if k["s"]:
            ax = ax + np.array([-1.0, 0.0, 0.0])
        if k["a"]:
            ax = ax + np.array([0.0, 1.0, 0.0])   # yaw
        if k["d"]:
            ax = ax + np.array([0.0, -1.0, 0.0])
        if k["q"]:
            ax = ax + np.array([0.0, 0.0, 1.0])   # roll
        if k["e"]:
            ax = ax + np.array([0.0, 0.0, -1.0])
        if np.linalg.norm(ax) > 0.0:
            v.sas_mode = "OFF"                     # taking manual control
            # Rotate about the chosen body-frame axis, expressed in world coordinates.
            world_axis = quat_rotate_vector(v.orientation, ax)
            dq = quat_from_axis_angle(world_axis, self.ROTATE_RATE_RADPS * dt)
            v.orientation = quat_normalize(quat_multiply(dq, v.orientation))
```

- [ ] **Step 2: Enforce warp lock + call the input pass in `_update`**

In `OrbitApp._update`, replace the top (sandbox path) so it applies input and locks warp:
```python
    def _update(self, task):
        real_dt = _global_clock.get_dt()

        if self.solar_system:
            self.clock.advance(real_dt)
            self._update_solar_system()
            return task.cont

        # Flight input + warp lock while thrusting.
        self._apply_flight_input(real_dt)
        if self.world.any_thrusting() and self.clock.warp != 1.0:
            self.clock.warp = 1.0
        sim_dt = self.clock.advance(real_dt)
        self.world.step(sim_dt)
        self._apply_jogs(real_dt)
        ...
```
Also guard warp-up so the player can't warp while thrusting; change the `period` binding:
```python
        self.accept("period", self._warp_up_guarded)
```
and add:
```python
    def _warp_up_guarded(self):
        if not self.world.any_thrusting():
            self.clock.warp_up()
```

- [ ] **Step 3: Smoke-check imports**

Run: `.venv/Scripts/python -c "import orbitsim.render.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Headless control test**

Create `tmp_flight_check.py` at repo root (delete after):
```python
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
import numpy as np
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp
w = _default_world()
w.vessels[0].fuel_mass_kg = 500.0
w.vessels[0].max_thrust_n = 30000.0
app = OrbitApp(w, SimClock(0.0, 100.0))
app._on_play()
app._throttle_full()
app._set_sas("PROGRADE")
for _ in range(30):
    app.taskMgr.step()
print("throttle:", w.vessels[0].throttle, "warp locked to 1:", app.clock.warp == 1.0)
print("fuel burned:", 500.0 - w.vessels[0].fuel_mass_kg > 0)
app.destroy()
print("OK")
```
Run: `PYTHONPATH=. .venv/Scripts/python tmp_flight_check.py`
Expected: `warp locked to 1: True`, `fuel burned: True`, `OK`. Then delete the file.

- [ ] **Step 5: Full suite stays green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: HUMAN VISUAL CHECKPOINT**

Run: `.venv/Scripts/python -m orbitsim`, Play, then press `1` (SAS prograde), `Z` (full throttle). Reviewer confirms: the orbit line grows as you burn prograde, warp can't be increased while thrusting, `X` cuts throttle and the orbit settles.

- [ ] **Step 7: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "Flight Task 8: flight controls (throttle/attitude/SAS) + warp lock"
```

---

## Task 9: Right-click-drag camera orbit (render)

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `CameraRig.orbit`, Panda `mouseWatcherNode`.
- Produces: holding RMB and dragging orbits the camera (both sandbox and solar modes).

- [ ] **Step 1: Track RMB + feed pointer delta to the rig**

In `_setup_input` (unconditional, both modes), add:
```python
        self._rmb_down = False
        self._last_mouse = None
        self.accept("mouse3", self._rmb, [True])
        self.accept("mouse3-up", self._rmb, [False])
```
Add methods:
```python
    MOUSE_ORBIT_SENS = 3.0     # radians per unit of normalized mouse travel

    def _rmb(self, down):
        self._rmb_down = down
        self._last_mouse = None

    def _apply_mouse_orbit(self):
        mw = self.mouseWatcherNode
        if not (self._rmb_down and mw.has_mouse()):
            self._last_mouse = None
            return
        x, y = mw.get_mouse_x(), mw.get_mouse_y()
        if self._last_mouse is not None:
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self.rig.orbit(dx * self.MOUSE_ORBIT_SENS, dy * self.MOUSE_ORBIT_SENS)
        self._last_mouse = (x, y)
```
Call `self._apply_mouse_orbit()` once per frame in `_update` (both branches), just before `self.rig.apply()`. For the solar branch, add the call inside `_update_solar_system` before `self.rig.apply()`; for the sandbox branch, before its `self.rig.apply()`.

- [ ] **Step 2: Smoke-check imports**

Run: `.venv/Scripts/python -c "import orbitsim.render.app; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Headless test (simulate RMB + mouse delta)**

Create `tmp_mouse_check.py`:
```python
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp
app = OrbitApp(_default_world(), SimClock(0.0, 1.0))
app._on_play()
az0 = app.rig.azimuth
# Simulate a drag: rmb down, then two frames with injected mouse positions.
app._rmb(True)
app._last_mouse = (0.0, 0.0)
app.rig.orbit(0.3, 0.0)   # equivalent to a horizontal drag step
print("azimuth changed:", app.rig.azimuth != az0)
app.destroy()
print("OK")
```
Run: `PYTHONPATH=. .venv/Scripts/python tmp_mouse_check.py` → `azimuth changed: True`, `OK`. Delete the file.

- [ ] **Step 4: HUMAN VISUAL CHECKPOINT**

Run the app; hold right mouse button and drag — the camera orbits the focus smoothly; scroll still zooms; left-click still works on GUI.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "Flight Task 9: right-click-drag camera orbit"
```

---

## Task 10: Flight HUD (throttle/fuel/mass/TWR/ΔV/warp-lock)

**Files:**
- Modify: `orbitsim/render/hud/__init__.py`
- Modify: `orbitsim/render/app.py` (feed the new fields)

**Interfaces:**
- Consumes: `Vessel` propulsion fields, `World.any_thrusting`.
- Produces: `Hud.update_flight(throttle, fuel_kg, fuel_frac, mass_kg, thrust_n, twr, dv_remaining, warp_locked)`.

- [ ] **Step 1: Add a flight readout to the HUD**

In `orbitsim/render/hud/__init__.py`, add a second `OnscreenText` (top-right, `base.a2dTopRight`) in `Hud.__init__`:
```python
        self.flight = OnscreenText(
            text="", pos=(-0.05, -0.12), scale=0.05, fg=(0.8, 1.0, 0.8, 1),
            shadow=(0, 0, 0, 1), align=TextNode.ARight, mayChange=True,
            parent=base.a2dTopRight,
        )
```
Add the method:
```python
    def update_flight(self, *, throttle, fuel_kg, fuel_frac, mass_kg, thrust_n,
                      twr, dv_remaining, warp_locked) -> None:
        lines = [
            f"Throttle: {throttle*100:,.0f}%",
            f"Fuel: {fuel_frac*100:,.0f}%  ({fuel_kg:,.0f} kg)",
            f"Mass: {mass_kg:,.0f} kg",
            f"Thrust: {thrust_n/1000:,.1f} kN   TWR: {twr:,.2f}",
            f"dV left: {dv_remaining:,.0f} m/s",
        ]
        if warp_locked:
            lines.append("WARP LOCKED - thrusting")
        self.flight.setText("\n".join(lines))
```

- [ ] **Step 2: Feed it each frame from the sandbox `_update`**

In `OrbitApp._update` (sandbox path), after `self.hud.update(...)`, add:
```python
        g_local = self.world.central.mu / max(v0.state.r_mag, 1.0) ** 2
        twr = (v0.max_thrust_n / (v0.mass_kg * g_local)) if v0.mass_kg > 0 else 0.0
        cap = getattr(self, "_fuel_capacity", 0.0)
        fuel_frac = v0.fuel_mass_kg / cap if cap > 0 else 0.0
        self.hud.update_flight(
            throttle=v0.throttle,
            fuel_kg=v0.fuel_mass_kg,
            fuel_frac=fuel_frac,
            mass_kg=v0.mass_kg,
            thrust_n=v0.max_thrust_n,
            twr=twr,
            dv_remaining=v0.delta_v_remaining,
            warp_locked=self.world.any_thrusting(),
        )
```
`self._fuel_capacity` is the initial fuel load; set it in `_on_play` (Task 12) or default to the vessel's fuel at sim start. For now, in `_start_sim` (sandbox branch) add: `self._fuel_capacity = self.world.vessels[0].fuel_mass_kg if self.world.vessels else 0.0`.

- [ ] **Step 3: Headless check**

Reuse `tmp_flight_check.py` logic: after stepping, assert `app.hud.flight.getText()` contains `"Throttle"` and `"TWR"`. Run a quick inline `-c` check or extend the temp script; expected substrings present.

- [ ] **Step 4: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: HUMAN VISUAL CHECKPOINT**

Launch, Play, throttle up: the top-right readout shows throttle %, fuel draining, mass dropping, TWR, ΔV left, and the WARP LOCKED line while burning.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/hud/__init__.py orbitsim/render/app.py
git commit -m "Flight Task 10: flight HUD (throttle/fuel/mass/TWR/dV/warp-lock)"
```

---

## Task 11: 3D navball instrument (render/navball.py) — HUMAN VISUAL CHECKPOINT

**Files:**
- Create: `orbitsim/render/navball.py`
- Modify: `orbitsim/render/app.py` (instantiate + update in sandbox mode)

**Interfaces:**
- Consumes: `make_uv_sphere`, `nose_direction`, `sas_target_dir`, Panda3D `NodePath`, `DisplayRegion`.
- Produces:
  ```python
  class Navball:
      def __init__(self, base) -> None
      def update(self, *, orientation_q, state, target_pos=None) -> None
  ```

- [ ] **Step 1: Implement the navball**

Create `orbitsim/render/navball.py`:
```python
"""A 3D navball: a sphere oriented in the orbital frame, with velocity/normal/radial
markers and a fixed nose reticle. Rendered in a small bottom-center display region."""
import numpy as np
from panda3d.core import NodePath, OrthographicLens, TextNode, Vec3

from orbitsim.render.geometry import make_uv_sphere
from orbitsim.core.attitude import nose_direction
from orbitsim.core.attitude import sas_target_dir

_MARKER_COLORS = {
    "PROGRADE": (0.2, 1.0, 0.3, 1), "RETROGRADE": (0.2, 1.0, 0.3, 1),
    "NORMAL": (0.7, 0.3, 1.0, 1), "ANTINORMAL": (0.7, 0.3, 1.0, 1),
    "RADIAL_OUT": (0.2, 0.8, 1.0, 1), "RADIAL_IN": (0.2, 0.8, 1.0, 1),
}


class Navball:
    """Bottom-center attitude sphere. Its own camera + display region so it overlays
    the scene without being affected by the world camera."""

    def __init__(self, base) -> None:
        self.base = base
        self.root = NodePath("navball_root")
        # Dedicated display region, lower-center of the window.
        dr = base.win.make_display_region(0.38, 0.62, 0.0, 0.28)
        dr.set_sort(20)
        self.cam = base.make_camera(base.win, displayRegion=(0.38, 0.62, 0.0, 0.28))
        lens = OrthographicLens()
        lens.set_film_size(2.6, 2.6)
        self.cam.node().set_lens(lens)
        self.cam.reparent_to(self.root)
        self.cam.set_pos(0, -10, 0)
        self.cam.look_at(0, 0, 0)

        self.ball = make_uv_sphere(1.0, 18, 36)
        self.ball.reparent_to(self.root)
        self.ball.set_light_off()
        self.ball.set_color(0.25, 0.45, 0.75, 1.0)   # ocean-blue ball

        # A pole cap so rotation is visible.
        self.pole = make_uv_sphere(0.12, 8, 12)
        self.pole.reparent_to(self.ball)
        self.pole.set_pos(0, 0, 1.0)
        self.pole.set_color(0.9, 0.9, 0.4, 1)
        self.pole.set_light_off()

        # Velocity/normal/radial markers (small spheres parented to root, not the ball,
        # because they live in the orbital frame, not the body frame).
        self._markers = {}
        for mode, col in _MARKER_COLORS.items():
            m = make_uv_sphere(0.09, 6, 10)
            m.reparent_to(self.root)
            m.set_color(*col)
            m.set_light_off()
            self._markers[mode] = m

        # Fixed nose reticle in screen space (always center).
        tn = TextNode("reticle")
        tn.set_text("+")
        tn.set_text_color(1, 1, 1, 1)
        self.reticle = self.root.attach_new_node(tn)
        self.reticle.set_scale(0.5)
        self.reticle.set_pos(-0.12, -2.0, -0.18)

    def update(self, *, orientation_q, state, target_pos=None) -> None:
        """Rotate the ball to the ship attitude and place orbital-frame markers."""
        # Orient the ball so the ship's nose points toward the camera (+ -Y here).
        nose = nose_direction(orientation_q)
        self.ball.look_at(Vec3(*nose))           # cheap visual: pole follows nose
        for mode, m in self._markers.items():
            try:
                d = sas_target_dir(mode, state, target_pos)
            except ValueError:
                m.hide()
                continue
            m.show()
            m.set_pos(Vec3(*(d)))                 # on the unit sphere in orbital frame
```

- [ ] **Step 2: Wire it into the sandbox app**

In `OrbitApp._start_sim` (sandbox branch, after the maneuver UI), add:
```python
            from orbitsim.render.navball import Navball
            self.navball = Navball(self)
```
In `_update` (sandbox path, near the HUD update), add:
```python
        self.navball.update(orientation_q=v0.orientation, state=v0.state)
```

- [ ] **Step 3: Smoke-check + headless screenshot**

Create `tmp_navball_check.py`:
```python
from panda3d.core import loadPrcFileData, Filename
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 900 700")
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp
app = OrbitApp(_default_world(), SimClock(0.0, 1.0))
app._on_play()
for _ in range(5):
    app.taskMgr.step()
app.win.save_screenshot(Filename.from_os_specific(r"tmp_navball.png"))
print("OK")
```
Run: `PYTHONPATH=. .venv/Scripts/python tmp_navball_check.py`, open `tmp_navball.png`, confirm a sphere with markers appears bottom-center. Delete temp files after.

- [ ] **Step 4: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: HUMAN VISUAL CHECKPOINT**

Launch, Play: a navball sits bottom-center; pressing SAS modes / rotating the ship visibly moves the ball and the green prograde / purple normal / cyan radial markers track the orbit. Refine colors/markers as needed.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/navball.py orbitsim/render/app.py
git commit -m "Flight Task 11: 3D navball instrument"
```

---

## Task 12: Title screen fuel-load + default propulsion stats

**Files:**
- Modify: `orbitsim/render/app.py` (title slider label + `_on_play`)
- Modify: `orbitsim/__main__.py` (default propulsion on the sandbox vessel)

**Interfaces:**
- Consumes: `Vessel` propulsion fields, `tsiolkovsky_dv`.
- Produces: title slider sets `fuel_mass_kg`; the ΔV readout is derived; defaults make the ship flyable.

- [ ] **Step 1: Give the default sandbox vessel real propulsion**

In `orbitsim/__main__.py::_default_world`, set propulsion on the vessel:
```python
    vessel = Vessel(
        name="Sandbox-1", state=state, delta_v_budget_mps=2000.0,
        dry_mass_kg=1000.0, fuel_mass_kg=800.0, max_thrust_n=30000.0,
        exhaust_velocity_mps=3000.0, max_turn_rate_radps=0.8,
    )
```

- [ ] **Step 2: Repurpose the title slider to fuel load**

In `orbitsim/render/app.py::_build_title_screen`, change the slider range/label to fuel and show derived ΔV:
```python
        default_fuel = (self.world.vessels[0].fuel_mass_kg if self.world.vessels else 800.0)
        self._budget_slider = DirectSlider(
            pos=(0.0, 0.0, -0.08), scale=0.6, range=(0.0, 4000.0),
            value=default_fuel, pageSize=100.0,
            command=self._refresh_budget_label, parent=self.aspect2d,
        )
```
Replace `_refresh_budget_label`:
```python
    def _refresh_budget_label(self) -> None:
        from orbitsim.core.flight import tsiolkovsky_dv
        fuel = float(self._budget_slider["value"])
        if self.world.vessels:
            v = self.world.vessels[0]
            dry, ve = v.dry_mass_kg, v.exhaust_velocity_mps
        else:
            dry, ve = 1000.0, 3000.0
        dv = tsiolkovsky_dv(ve, dry + fuel, dry) if fuel > 0 else 0.0
        self._budget_label.setText(f"Fuel: {fuel:,.0f} kg   (dV {dv:,.0f} m/s)")
```
Update the hint text to `"fuel load  (drag to set)"`.

- [ ] **Step 3: Apply fuel on Play**

Replace `_on_play`'s budget loop with fuel assignment + capacity capture:
```python
    def _on_play(self) -> None:
        fuel = float(self._budget_slider["value"])
        for vessel in self.world.vessels:
            vessel.fuel_mass_kg = fuel
        self._fuel_capacity = fuel if self.world.vessels else 0.0
        for node in self._title_nodes:
            node.destroy() if hasattr(node, "destroy") else node.remove_node()
        self._title_nodes = []
        self._start_sim()
```

- [ ] **Step 4: Headless check**

Create `tmp_title_fuel.py`:
```python
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
from orbitsim.__main__ import _default_world
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp
app = OrbitApp(_default_world(), SimClock(0.0, 100.0))
app.taskMgr.step()
app._budget_slider["value"] = 1200.0
app._refresh_budget_label()
print("label:", app._budget_label.getText())
app._on_play()
print("fuel applied:", app.world.vessels[0].fuel_mass_kg == 1200.0)
print("capacity:", app._fuel_capacity == 1200.0)
app.destroy()
print("OK")
```
Run: `PYTHONPATH=. .venv/Scripts/python tmp_title_fuel.py` → label shows `Fuel: 1,200 kg (dV ...)`, both bools True, `OK`. Delete temp file.

- [ ] **Step 5: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: HUMAN VISUAL CHECKPOINT**

Launch: title slider reads "Fuel: N kg (dV …)" and updates live; Play with a chosen fuel load; the flight HUD's fuel/ΔV reflect it.

- [ ] **Step 7: Commit**

```bash
git add orbitsim/render/app.py orbitsim/__main__.py
git commit -m "Flight Task 12: title-screen fuel load + default propulsion"
```

---

## Exit Criteria

- `core/attitude.py` and `core/flight.py` fully TDD-green: quaternion ops, slew (no overshoot), SAS vectors, rocket equation, powered RK4 (zero-thrust == Kepler within 1 m; free-space burn == ve·ln(m0/mf) within 0.1%).
- A vessel can throttle up, drain fuel, gain/lose orbital energy, and return to analytic rails on cutoff; time-warp is locked to 1× while thrusting.
- Right-click-drag orbits the camera; the navball and flight HUD reflect attitude/throttle/fuel/TWR/ΔV.
- Title screen sets fuel load with a live derived-ΔV readout.
- `pytest tests/ -q` fully green.

## Self-Review Notes

- Spec coverage: rocket equation (§2/Task 4), powered RK4 (Task 5), attitude quaternion + slew (Tasks 1–2), SAS vectors (Task 3), Vessel state (Task 6), coast/burn World.step + warp lock (Tasks 7–8), keybinds (Task 8), right-drag camera (Task 9), HUD (Task 10), navball (Task 11), title fuel + defaults (Task 12). All mapped.
- Pure-physics (Tasks 1–7) are red-green TDD with known answers; render (Tasks 8–12) verified headlessly + visual checkpoints, per project convention.
- Type consistency: `slew_toward`, `sas_target_dir`, `nose_direction`, `integrate_powered`, `tsiolkovsky_dv`, `World.any_thrusting`, `Vessel.mass_kg`/`delta_v_remaining` used identically across tasks.
- The navball (Task 11) is the roughest visual unit; its first version prioritizes a visible, correctly-tracking instrument over polish — expect iteration at its checkpoint.
```
