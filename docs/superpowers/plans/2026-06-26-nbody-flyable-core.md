# N-Body Flyable — Core Physics (Cycle 1b, Part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The pure, unit-tested physics for flying under N-body gravity in the Earth-centered sandbox: a circular Moon, the Earth+Moon third-body force model (with the indirect term), a forward propagator, osculating elements, and a time-warp-safety policy.

**Architecture:** Keep Earth fixed at the origin; circularize the geocentric Moon and orbit it at the Earth-Moon two-body rate. The ship feels `earth_moon_accel` (central Earth + Moon direct + indirect term), integrated by 1a's velocity-Verlet core generalized to an acceleration function. Two helpers — `osculating_elements` and `max_safe_warp` — support the (later) render layer.

**Tech Stack:** Python 3, numpy, scipy. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- SI, float64 `(3,)` arrays; frame = **Earth-centered inertial** (Earth fixed at origin). (spec)
- `core/` never imports render/sim/panda3d. (project rule)
- Constants from `core/constants.py` (`MU_EARTH`, `MU_MOON` exist). (project rule)
- TDD; **never loosen a tolerance to pass** — fix the implementation. (project rule)
- Reuse 1a's `orbitsim/core/nbody.py` integrator; do not duplicate it. (spec)
- Run tests with `.venv/Scripts/python -m pytest`. Commits: explicit paths; end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; then `git push`. (repo discipline)
- This plan is **Part 1 (pure physics)**; render integration (World.step, trajectory line, warp cap wiring, HUD) is Part 2, a later plan.

## File Structure

- `orbitsim/core/moon.py` — circularize `MOON_ORBIT` (e→0) and orbit at the Earth-Moon rate.
- `orbitsim/core/nbody.py` — add `earth_moon_accel`, generalize the Verlet core to an accel fn, add `propagate_earth_moon`, `MOON_SOI_M`, `osculating_elements`, `max_safe_warp`.
- `tests/core/test_moon.py`, `tests/core/test_nbody.py` — new tests.

---

## Task 1: Circularize the Moon at the Earth-Moon rate

**Files:**
- Modify: `orbitsim/core/moon.py`
- Test: `tests/core/test_moon.py`

**Interfaces:**
- Produces: `MOON_ORBIT` with `e == 0.0` and `mu == MU_EARTH + MU_MOON`; `moon_state_at(t)` returns a constant-distance (circular) geocentric Moon orbiting at the Earth-Moon two-body rate.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_moon.py`:

```python
def test_moon_orbit_is_circular_at_earth_moon_rate():
    from orbitsim.core.constants import MU_EARTH, MU_MOON
    import numpy as np
    from orbitsim.core.moon import MOON_ORBIT, moon_state_at
    assert MOON_ORBIT.e == 0.0
    assert MOON_ORBIT.mu == MU_EARTH + MU_MOON
    # Circular => distance is constant across the orbit.
    dists = [np.linalg.norm(moon_state_at(t).r) for t in (0.0, 5.0e5, 1.0e6, 1.5e6)]
    assert max(dists) - min(dists) < 1.0e3   # < 1 km variation
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_moon.py -q -k circular_at_earth_moon`
Expected: FAIL — `e` is 0.0549 / `mu` is `MU_EARTH`.

- [ ] **Step 3: Implement**

In `orbitsim/core/moon.py`, change the `MOON_ORBIT` definition (import `MU_MOON` alongside `MU_EARTH`):

```python
from orbitsim.core.constants import MU_EARTH, MU_MOON

MOON_ORBIT = KeplerianElements(
    a=3.844e8, e=0.0, i=0.0898, raan=0.0, argp=0.0, nu=0.0,
    mu=MU_EARTH + MU_MOON, epoch_s=0.0,
)
```

(Circular so Lagrange points are steady; the Earth-Moon two-body `mu` gives the rate at
which the indirect-term L4 balance is exact — see Task 2.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_moon.py -q`
Expected: PASS (all — the existing range/period tests still hold at e=0).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/moon.py tests/core/test_moon.py
git commit -m "$(cat <<'EOF'
Moon: circularize orbit (e=0) at the Earth-Moon two-body rate

Steady Lagrange points; mu = MU_EARTH+MU_MOON gives the rate at which the
third-body L4 balance is exact. Gravity Moon == target Moon (one body).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: `earth_moon_accel` (third-body force model) + Lagrange balance

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `core.moon.moon_state_at` (Task 1), `MU_EARTH`, `MU_MOON`.
- Produces: `earth_moon_accel(r_m, t_s) -> np.ndarray (3,)` — central Earth + Moon direct + indirect.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_nbody.py`:

```python
from orbitsim.core.moon import moon_state_at


def test_earth_moon_accel_has_indirect_term():
    r = np.array([2.0e7, 1.0e7, 0.0])
    a = nb.earth_moon_accel(r, 0.0)
    rM = moon_state_at(0.0).r
    direct = (-MU_EARTH * r / np.linalg.norm(r)**3
              - MU_MOON * (r - rM) / np.linalg.norm(r - rM)**3)
    indirect = -MU_MOON * rM / np.linalg.norm(rM)**3
    assert np.allclose(a, direct + indirect, rtol=1e-12)


def test_L4_balances_in_the_earth_fixed_model():
    # L4: 60 deg ahead of the Moon in its orbital plane, distance d from Earth and Moon.
    t = 1.0e5
    m = moon_state_at(t)
    d = np.linalg.norm(m.r)
    w = np.cross(m.r, m.v) / d**2                 # Moon's angular velocity vector
    omega = np.linalg.norm(w)
    axis = w / omega
    # Rodrigues rotation of r_M by +60 deg about the orbit normal.
    c, s = np.cos(np.radians(60)), np.sin(np.radians(60))
    L4 = m.r * c + np.cross(axis, m.r) * s + axis * np.dot(axis, m.r) * (1 - c)
    # Net rotating-frame acceleration (gravity + centrifugal) must vanish.
    centrifugal = -np.cross(w, np.cross(w, L4))
    net = nb.earth_moon_accel(L4, t) + centrifugal
    assert np.linalg.norm(net) < 1e-7, np.linalg.norm(net)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k "indirect or L4_balances"`
Expected: FAIL — `AttributeError: ... 'earth_moon_accel'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py` (import `moon_state_at`):

```python
from orbitsim.core.moon import moon_state_at


def earth_moon_accel(r_m, t_s):
    """Ship acceleration in the EARTH-CENTERED frame [m/s^2]: central Earth plus the
    Moon's third-body perturbation. The indirect term (+mu_M * r_M/|r_M|^3) is required
    because Earth is held fixed at the origin (non-inertial frame); it is also what makes
    the Lagrange points balance."""
    r = np.asarray(r_m, dtype=np.float64)
    rM = moon_state_at(t_s).r
    a = -MU_EARTH * r / np.linalg.norm(r)**3
    a += -MU_MOON * ((r - rM) / np.linalg.norm(r - rM)**3 + rM / np.linalg.norm(rM)**3)
    return a
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k "indirect or L4_balances"`
Expected: PASS. The L4 balance passing is the proof that the indirect term + Earth-Moon
rate are correct; if it fails, the bug is a missing indirect term or the wrong Moon rate
(Task 1) — do NOT loosen the 1e-7 bound.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: earth_moon_accel third-body model (indirect term; L4 balances)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: Generalize the Verlet core + `propagate_earth_moon`

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `earth_moon_accel` (Task 2); existing `propagate_nbody`, `gravity_accel`, `_substep_count` (1a).
- Produces:
  - `_verlet(r, v, t, dt_s, accel_fn, n_sub) -> (r, v, t)` — pure kick-drift-kick core taking an acceleration function `accel_fn(r, t)`.
  - `propagate_earth_moon(state, dt_s, max_step_s=3600.0) -> StateVector` — propagate the ship under `earth_moon_accel`, proximity-substepped over Earth (origin) and the Moon.
  - `propagate_nbody` unchanged externally (still attractor-list based; refactored to call `_verlet`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_nbody.py`:

```python
def test_propagate_earth_moon_reduces_to_two_body_near_earth():
    # A LEO orbit: the Moon's perturbation is tiny, so it tracks Kepler closely.
    r = 7.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)
    out = nb.propagate_earth_moon(st, period / 4, max_step_s=0.5)
    # Within ~1 km of two-body over a quarter LEO orbit (Moon tug is sub-km here).
    assert np.linalg.norm(out.r - propagate_kepler(st, period / 4).r) < 1.0e3


def test_propagate_earth_moon_reversible():
    st = StateVector(r=np.array([5.0e7, 0.0, 0.0]),
                     v=np.array([0.0, 1500.0, 100.0]), mu=MU_EARTH, epoch_s=0.0)
    T = 3600.0 * 6
    fwd = nb.propagate_earth_moon(st, T, max_step_s=20.0)
    back = nb.propagate_earth_moon(fwd, -T, max_step_s=20.0)
    assert np.linalg.norm(back.r - st.r) < 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k earth_moon_reduces`
Expected: FAIL — `AttributeError: ... 'propagate_earth_moon'`.

- [ ] **Step 3: Implement**

Refactor `orbitsim/core/nbody.py`: extract the Verlet loop into `_verlet`, have
`propagate_nbody` call it, and add `propagate_earth_moon`. Replace the body of
`propagate_nbody` and add the new function:

```python
def _verlet(r, v, t, dt_s, accel_fn, n_sub):
    """Velocity-Verlet (kick-drift-kick) for n_sub uniform steps. accel_fn(r, t)->a."""
    r = np.asarray(r, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    h = dt_s / n_sub
    a = accel_fn(r, t)
    for _ in range(n_sub):
        v_half = v + 0.5 * a * h
        r = r + v_half * h
        t = t + h
        a = accel_fn(r, t)
        v = v_half + 0.5 * a * h
    return r, v, t


def propagate_nbody(state, dt_s, attractors=EARTH_MOON, max_step_s=3600.0):
    """Advance the ship by dt_s with velocity Verlet under summed attractors. Reversible."""
    n = _substep_count(state, dt_s, attractors, max_step_s)
    r, v, t = _verlet(state.r, state.v, state.epoch_s, dt_s,
                      lambda rr, tt: gravity_accel(rr, tt, attractors), n)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=t)


def _earth_moon_substeps(state, dt_s, max_step_s):
    """Sub-steps for propagate_earth_moon: cap by 1/200 of the local orbital timescale
    at Earth (origin) and at the Moon."""
    r = np.asarray(state.r, dtype=np.float64)
    cap = max_step_s
    rE = np.linalg.norm(r)
    cap = min(cap, (2 * np.pi * np.sqrt(rE**3 / MU_EARTH)) / 200.0)
    rM = np.linalg.norm(r - moon_state_at(state.epoch_s).r)
    cap = min(cap, (2 * np.pi * np.sqrt(rM**3 / MU_MOON)) / 200.0)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


def propagate_earth_moon(state, dt_s, max_step_s=3600.0):
    """Advance the ship by dt_s under earth_moon_accel (central Earth + Moon + indirect)."""
    n = _earth_moon_substeps(state, dt_s, max_step_s)
    r, v, t = _verlet(state.r, state.v, state.epoch_s, dt_s, earth_moon_accel, n)
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=t)
```

- [ ] **Step 4: Run to verify they pass + no 1a regressions**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: PASS (all — 1a's `propagate_nbody` tests still green after the `_verlet` refactor).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: extract _verlet core; add propagate_earth_moon (Earth+Moon+indirect)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 4: `osculating_elements`

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `core.elements.state_to_elements`, `core.moon.moon_state_at`, `MU_EARTH`, `MU_MOON`.
- Produces:
  - `MOON_SOI_M` — Moon sphere-of-influence radius `= a_moon * (MU_MOON/MU_EARTH)**0.4`.
  - `osculating_elements(state, t_s) -> KeplerianElements` — instantaneous Keplerian
    elements about the dominant body (Moon if within `MOON_SOI_M`, else Earth).

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_nbody.py`:

```python
from orbitsim.core.elements import state_to_elements


def test_osculating_elements_earth_dominant_matches_two_body():
    r = 8.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    osc = nb.osculating_elements(st, 0.0)
    ref = state_to_elements(StateVector(st.r, st.v, MU_EARTH, 0.0))
    assert abs(osc.a - ref.a) < 1.0 and abs(osc.e - ref.e) < 1e-9


def test_osculating_elements_switches_to_moon_inside_soi():
    t = 0.0
    m = moon_state_at(t)
    r_lo = 3.0e6                                   # 3000 km lunar orbit (inside SOI)
    # Circular about the Moon, in the Moon's frame.
    st = StateVector(r=m.r + np.array([r_lo, 0.0, 0.0]),
                     v=m.v + np.array([0.0, np.sqrt(MU_MOON / r_lo), 0.0]),
                     mu=MU_EARTH, epoch_s=t)
    osc = nb.osculating_elements(st, t)
    assert osc.mu == MU_MOON                       # dominant body is the Moon
    assert abs(osc.a - r_lo) < 1.0e4 and osc.e < 0.01   # ~circular lunar orbit
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k osculating`
Expected: FAIL — `AttributeError: ... 'osculating_elements'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py` (import `state_to_elements`):

```python
from orbitsim.core.elements import state_to_elements

MOON_SOI_M = 3.844e8 * (MU_MOON / MU_EARTH)**0.4   # Moon sphere of influence [m]


def osculating_elements(state, t_s):
    """Instantaneous Keplerian elements about the dominant body (Moon if the ship is
    within MOON_SOI_M of it, else Earth). Used for the HUD; drifts under perturbation."""
    rM = moon_state_at(t_s)
    if np.linalg.norm(state.r - rM.r) < MOON_SOI_M:
        rel = StateVector(state.r - rM.r, state.v - rM.v, MU_MOON, state.epoch_s)
    else:
        rel = StateVector(state.r, state.v, MU_EARTH, state.epoch_s)
    return state_to_elements(rel)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: osculating_elements about the dominant body (Earth/Moon SOI)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 5: `max_safe_warp` policy

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `MU_EARTH`, `MU_MOON`, `moon_state_at`.
- Produces: `max_safe_warp(state, t_s, warp_steps, real_dt_s=1/60, budget_substeps=200) -> float`
  — the largest value in `warp_steps` whose frame `sim_dt = real_dt_s * warp` integrates
  within `budget_substeps` Verlet sub-steps at the current proximity; at least `min(warp_steps)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_nbody.py`:

```python
WARP_STEPS = (1.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 10000.0, 100000.0)


def test_max_safe_warp_is_low_near_earth_high_in_deep_space():
    leo = StateVector(r=np.array([7.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 7546.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    deep = StateVector(r=np.array([3.0e8, 1.0e8, 0.0]),
                       v=np.array([0.0, 200.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    w_leo = nb.max_safe_warp(leo, 0.0, WARP_STEPS)
    w_deep = nb.max_safe_warp(deep, 0.0, WARP_STEPS)
    assert w_leo in WARP_STEPS and w_deep in WARP_STEPS
    assert w_deep > w_leo                     # smoother far out => faster warp allowed
    assert w_leo >= 1.0                        # never below the floor


def test_max_safe_warp_respects_substep_budget():
    leo = StateVector(r=np.array([7.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 7546.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    w = nb.max_safe_warp(leo, 0.0, WARP_STEPS, real_dt_s=1 / 60, budget_substeps=200)
    n = nb._earth_moon_substeps(leo, (1 / 60) * w, max_step_s=3600.0)
    assert n <= 200
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k max_safe_warp`
Expected: FAIL — `AttributeError: ... 'max_safe_warp'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py`:

```python
def max_safe_warp(state, t_s, warp_steps, real_dt_s=1 / 60, budget_substeps=200):
    """Largest warp in warp_steps whose frame integrates within budget_substeps Verlet
    sub-steps at the current proximity (so time-warp stays accurate near bodies)."""
    allowed = [w for w in warp_steps
               if _earth_moon_substeps(state, real_dt_s * w, 3600.0) <= budget_substeps]
    return max(allowed) if allowed else min(warp_steps)
```

- [ ] **Step 4: Run to verify they pass + full core suite**

Run: `.venv/Scripts/python -m pytest tests/core -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: max_safe_warp proximity policy (caps warp near bodies)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Self-Review

- **Spec coverage (Part 1):** circular Moon (T1) ✓; `earth_moon_accel` + indirect term + L-point balance proof (T2) ✓; generalized integrator + `propagate_earth_moon` (T3) ✓; `osculating_elements` dominant-body (T4) ✓; `max_safe_warp` policy (T5) ✓. Part 2 (World.step wiring, trajectory line, warp-cap wiring, HUD) is a later plan — explicitly out of scope here.
- **Placeholder scan:** none — every step has complete code and concrete tolerances.
- **Type consistency:** `earth_moon_accel(r,t)`, `_verlet(r,v,t,dt,accel_fn,n)`, `propagate_earth_moon(state,dt,max_step_s)`, `_earth_moon_substeps(state,dt,max_step_s)`, `osculating_elements(state,t)`, `MOON_SOI_M`, `max_safe_warp(state,t,warp_steps,...)` are used consistently across tasks (T5 reuses T3's `_earth_moon_substeps`).
- **Refinement noted:** T1 sets `MOON_ORBIT.mu = MU_EARTH+MU_MOON` (beyond the spec's "e=0") because the exact L4 balance in T2 requires the Moon to orbit at the Earth-Moon two-body rate; the algebra (`earth_moon_accel(L4) = -(mu_E+mu_M)/d^3 * L4`, cancelled by centrifugal `omega^2*L4` with `omega^2=(mu_E+mu_M)/d^3`) confirms it.
