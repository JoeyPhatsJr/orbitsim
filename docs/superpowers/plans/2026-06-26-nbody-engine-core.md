# Restricted N-Body Engine Core (Cycle 1a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure, unit-tested restricted-3-body (CR3BP) propagation core for the ship as a massless test particle under an idealized circular Earth+Moon, with Jacobi-constant conservation and Lagrange-point computation.

**Architecture:** New `orbitsim/core/nbody.py`. Earth and Moon are point masses on circular barycentric orbits (on rails); a velocity-Verlet symplectic integrator advances the ship under the summed gravity of an extensible attractor list. Diagnostic helpers give the rotating-frame Jacobi constant and the five Lagrange points. Two-body `core/` is untouched.

**Tech Stack:** Python 3, numpy, scipy (`brentq`). Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- SI everywhere; float64 numpy arrays shape `(3,)`; frame = **barycentric inertial**. (spec)
- `core/` never imports render/sim/panda3d. (project rule)
- Constants come from `core/constants.py` — `MU_EARTH`, `MU_MOON` already exist; never hard-type GMs. (project rule)
- TDD mandatory; **never loosen a tolerance to force a pass** — fix the implementation. (project rule)
- Force model sums over an **attractor list** (each attractor has `.mu` and `.state_at(t) -> StateVector`) so Sun/planets append later. (spec)
- Run tests with `.venv/Scripts/python -m pytest`. Commits: explicit paths; end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; then `git push`. (repo discipline)
- All tasks are pure physics (Haiku/Sonnet TDD).

## Shared constants (defined in Task 1, used throughout)

```
D_EM      = 3.844e8                      # Earth-Moon separation [m]
MU_TOTAL  = MU_EARTH + MU_MOON           # [m^3/s^2]
OMEGA_EM  = sqrt(MU_TOTAL / D_EM**3)     # mean motion [rad/s]
MASS_RATIO= MU_MOON / MU_TOTAL           # ~0.01215
EARTH_X   = -MASS_RATIO * D_EM           # Earth rotating-frame x [m]
MOON_X    = (1 - MASS_RATIO) * D_EM      # Moon  rotating-frame x [m]
```

## File Structure

- `orbitsim/core/nbody.py` (new) — bodies, force model, propagator, rotating frame, Jacobi, Lagrange points.
- `tests/core/test_nbody.py` (new) — all tests.

---

## Task 1: Idealized barycentric Earth + Moon bodies

**Files:**
- Create: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Produces: module constants `D_EM, MU_TOTAL, OMEGA_EM, MASS_RATIO, EARTH_X, MOON_X`; class `_CircularBody` with `.mu` and `.state_at(t_s) -> StateVector` (barycentric); module list `EARTH_MOON = [EARTH, MOON]` where `MOON.mu == MU_MOON`, `EARTH.mu == MU_EARTH`.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_nbody.py`:

```python
import numpy as np
from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core import nbody as nb


def test_bodies_sit_on_the_barycenter_axis_at_t0():
    e = nb.EARTH.state_at(0.0)
    m = nb.MOON.state_at(0.0)
    # Moon on +x at (1-ratio)d, Earth on -x at -ratio*d.
    assert np.allclose(m.r, [nb.MOON_X, 0.0, 0.0])
    assert np.allclose(e.r, [nb.EARTH_X, 0.0, 0.0])
    # Mass-weighted positions cancel at the barycenter.
    bary = MU_EARTH * e.r + MU_MOON * m.r
    assert np.linalg.norm(bary) < 1e-3 * MU_TOTAL_FOR_TEST


def test_bodies_are_circular_and_rotate_with_omega():
    # Quarter period later, the Moon has rotated 90 degrees (+x -> +y).
    t = (np.pi / 2) / nb.OMEGA_EM
    m = nb.MOON.state_at(t)
    assert np.allclose(m.r, [0.0, nb.MOON_X, 0.0], atol=1.0)
    # Circular speed = omega * radius; velocity perpendicular to radius.
    assert abs(np.linalg.norm(m.r) - nb.MOON_X) < 1.0
    assert abs(np.linalg.norm(m.v) - nb.OMEGA_EM * nb.MOON_X) < 1e-6
    assert abs(np.dot(m.r, m.v)) < 1.0


MU_TOTAL_FOR_TEST = MU_EARTH + MU_MOON
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: FAIL — `ModuleNotFoundError: orbitsim.core.nbody`.

- [ ] **Step 3: Implement**

Create `orbitsim/core/nbody.py`:

```python
"""Restricted N-body (CR3BP) core: idealized circular Earth+Moon, a velocity-Verlet
ship propagator, the Jacobi constant, and the Lagrange points. Barycentric inertial,
SI, float64. The ship is a massless test particle; the bodies are on rails."""
import numpy as np

from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.state import StateVector

D_EM = 3.844e8                          # Earth-Moon separation [m]
MU_TOTAL = MU_EARTH + MU_MOON
OMEGA_EM = np.sqrt(MU_TOTAL / D_EM**3)  # mean motion [rad/s]
MASS_RATIO = MU_MOON / MU_TOTAL         # ~0.01215
EARTH_X = -MASS_RATIO * D_EM            # Earth rotating-frame x [m]
MOON_X = (1.0 - MASS_RATIO) * D_EM      # Moon rotating-frame x [m]


class _CircularBody:
    """A point mass on a circular barycentric orbit (signed radius along the
    rotating x-axis at t=0; rotates at OMEGA_EM)."""

    def __init__(self, mu: float, signed_radius_m: float):
        self.mu = mu
        self._R = signed_radius_m

    def state_at(self, t_s: float) -> StateVector:
        th = OMEGA_EM * t_s
        u = np.array([np.cos(th), np.sin(th), 0.0])
        n = np.array([-np.sin(th), np.cos(th), 0.0])
        return StateVector(r=self._R * u, v=self._R * OMEGA_EM * n,
                           mu=self.mu, epoch_s=t_s)


EARTH = _CircularBody(MU_EARTH, EARTH_X)
MOON = _CircularBody(MU_MOON, MOON_X)
EARTH_MOON = [EARTH, MOON]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: idealized circular barycentric Earth+Moon bodies

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: `gravity_accel` — summed attractor force model

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `EARTH_MOON`, `_CircularBody` (Task 1).
- Produces: `gravity_accel(r_m, t_s, attractors=EARTH_MOON) -> np.ndarray (3,)` — `Σ −μ_i (r − r_i)/|r − r_i|³`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_nbody.py`:

```python
def test_single_attractor_matches_point_mass_gravity():
    r = np.array([2.0e7, 0.0, 0.0])
    a = nb.gravity_accel(r, 0.0, attractors=[nb.EARTH])
    e = nb.EARTH.state_at(0.0).r
    d = r - e
    expected = -MU_EARTH * d / np.linalg.norm(d)**3
    assert np.allclose(a, expected, rtol=1e-12)


def test_two_attractors_sum():
    r = np.array([1.0e8, 5.0e7, 0.0])
    a_both = nb.gravity_accel(r, 0.0, attractors=nb.EARTH_MOON)
    a_e = nb.gravity_accel(r, 0.0, attractors=[nb.EARTH])
    a_m = nb.gravity_accel(r, 0.0, attractors=[nb.MOON])
    assert np.allclose(a_both, a_e + a_m, rtol=1e-12)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k attractor`
Expected: FAIL — `AttributeError: module ... has no attribute 'gravity_accel'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py`:

```python
def gravity_accel(r_m, t_s, attractors=EARTH_MOON):
    """Summed gravitational acceleration on a test particle at r_m, time t_s [m/s^2]."""
    r = np.asarray(r_m, dtype=np.float64)
    a = np.zeros(3)
    for body in attractors:
        d = r - body.state_at(t_s).r
        a += -body.mu * d / np.linalg.norm(d)**3
    return a
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: PASS (4).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: gravity_accel summed-attractor force model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: `propagate_nbody` — velocity-Verlet propagator

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `gravity_accel`, `EARTH_MOON`, `OMEGA_EM` (Tasks 1-2).
- Produces: `propagate_nbody(state, dt_s, attractors=EARTH_MOON, max_step_s=3600.0) -> StateVector`
  — advances the ship `dt_s` with velocity Verlet, sub-stepped so each internal step is
  ≤ `max_step_s` and ≤ 1/200 of the smallest local orbital timescale `2π√(r³/μ)` over
  the attractors at the start state. Time-reversible (negative `dt_s` allowed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_nbody.py`:

```python
from orbitsim.core.propagate import propagate_kepler


def _leo_state():
    r = 7.0e6
    # State referenced to Earth's *barycentric* position so it's a clean Earth orbit.
    e = nb.EARTH.state_at(0.0)
    return StateVector(r=e.r + np.array([r, 0.0, 0.0]),
                       v=e.v + np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                       mu=MU_EARTH, epoch_s=0.0)


def test_reduces_to_two_body_with_only_earth():
    st = _leo_state()
    period = 2 * np.pi * np.sqrt(7.0e6**3 / MU_EARTH)
    # Geocentric two-body reference (subtract Earth's fixed barycentric offset).
    e = nb.EARTH.state_at(0.0)
    geo = StateVector(r=st.r - e.r, v=st.v - e.v, mu=MU_EARTH, epoch_s=0.0)
    # Quarter orbit: fine step so 2nd-order Verlet error is sub-metre.
    out = nb.propagate_nbody(st, period / 4, attractors=[nb.EARTH], max_step_s=0.5)
    ref = propagate_kepler(geo, period / 4).r + nb.EARTH.state_at(period / 4).r
    assert np.linalg.norm(out.r - ref) < 1.0
    # One period closes.
    out2 = nb.propagate_nbody(st, period, attractors=[nb.EARTH], max_step_s=0.5)
    ref2 = propagate_kepler(geo, period).r + nb.EARTH.state_at(period).r
    assert np.linalg.norm(out2.r - ref2) < 10.0


def test_propagation_is_reversible():
    st = _leo_state()
    T = 3600.0 * 3
    fwd = nb.propagate_nbody(st, T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    back = nb.propagate_nbody(fwd, -T, attractors=nb.EARTH_MOON, max_step_s=10.0)
    assert np.linalg.norm(back.r - st.r) < 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k "two_body or reversible"`
Expected: FAIL — `AttributeError: ... 'propagate_nbody'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py`:

```python
def _substep_count(state, dt_s, attractors, max_step_s):
    """Number of uniform Verlet sub-steps for |dt_s|: small enough to resolve the
    closest body's local orbital timescale (1/200 of 2*pi*sqrt(r^3/mu))."""
    r = np.asarray(state.r, dtype=np.float64)
    cap = max_step_s
    for body in attractors:
        rb = np.linalg.norm(r - body.state_at(state.epoch_s).r)
        cap = min(cap, (2 * np.pi * np.sqrt(rb**3 / body.mu)) / 200.0)
    return max(1, int(np.ceil(abs(dt_s) / cap)))


def propagate_nbody(state, dt_s, attractors=EARTH_MOON, max_step_s=3600.0):
    """Advance the ship by dt_s using velocity Verlet (kick-drift-kick). Reversible."""
    n = _substep_count(state, dt_s, attractors, max_step_s)
    h = dt_s / n
    r = np.asarray(state.r, dtype=np.float64).copy()
    v = np.asarray(state.v, dtype=np.float64).copy()
    t = state.epoch_s
    a = gravity_accel(r, t, attractors)
    for _ in range(n):
        v_half = v + 0.5 * a * h
        r = r + v_half * h
        t = t + h
        a = gravity_accel(r, t, attractors)
        v = v_half + 0.5 * a * h
    return StateVector(r=r, v=v, mu=state.mu, epoch_s=t)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: PASS (6). If the two-body match exceeds tolerance, the bug is the
integrator/force model (or the geocentric reference shift) — do NOT loosen the 1 m /
10 m bounds.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: velocity-Verlet ship propagator (reduces to two-body, reversible)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 4: rotating frame + Jacobi constant

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `OMEGA_EM, EARTH_X, MOON_X, MU_EARTH, MU_MOON, propagate_nbody` (Tasks 1-3).
- Produces:
  - `rotating_frame(r_m, v_mps, t_s) -> (r_rot, v_rot)` — inertial → frame co-rotating at `OMEGA_EM` (Moon fixed on +x at `MOON_X`, Earth at `EARTH_X`).
  - `jacobi_constant(state, t_s) -> float` — `2Ω(r_rot) − |v_rot|²`, `Ω = ½ω²(x²+y²) + μ_E/r1 + μ_M/r2`. Conserved along a coast.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_nbody.py`:

```python
def test_jacobi_constant_conserved_over_seven_days():
    # A ship out between Earth and Moon where both attractors matter.
    e = nb.EARTH.state_at(0.0)
    st = StateVector(r=np.array([1.2e8, 0.0, 0.0]),
                     v=np.array([0.0, 900.0, 50.0]), mu=MU_EARTH, epoch_s=0.0)
    c0 = nb.jacobi_constant(st, 0.0)
    far = nb.propagate_nbody(st, 7 * 86400.0, attractors=nb.EARTH_MOON, max_step_s=600.0)
    c1 = nb.jacobi_constant(far, far.epoch_s)
    assert abs(c1 - c0) / abs(c0) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k jacobi`
Expected: FAIL — `AttributeError: ... 'jacobi_constant'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py`:

```python
def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # inertial->rotating


def rotating_frame(r_m, v_mps, t_s):
    """Map an inertial state into the frame co-rotating at OMEGA_EM (Moon fixed +x)."""
    R = _rot_z(OMEGA_EM * t_s)
    r_rot = R @ np.asarray(r_m, dtype=np.float64)
    w = np.array([0.0, 0.0, OMEGA_EM])
    v_rot = R @ np.asarray(v_mps, dtype=np.float64) - np.cross(w, r_rot)
    return r_rot, v_rot


def jacobi_constant(state, t_s):
    """Jacobi constant C = 2*Omega - |v_rot|^2 (conserved along a coast)."""
    r_rot, v_rot = rotating_frame(state.r, state.v, t_s)
    x, y = r_rot[0], r_rot[1]
    r1 = np.linalg.norm(r_rot - np.array([EARTH_X, 0.0, 0.0]))
    r2 = np.linalg.norm(r_rot - np.array([MOON_X, 0.0, 0.0]))
    omega2 = 0.5 * OMEGA_EM**2 * (x**2 + y**2) + MU_EARTH / r1 + MU_MOON / r2
    return float(2.0 * omega2 - np.dot(v_rot, v_rot))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: PASS (7). If Jacobi drifts past 1e-6, the integrator step or the rotating-frame
math is wrong — do NOT loosen the tolerance; prefer a smaller `max_step_s` in the test
only if a fixed-step symplectic genuinely needs it (it should not at 600 s here).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: rotating frame + Jacobi constant (conserved over 7 days)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 5: `lagrange_points`

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `D_EM, MASS_RATIO, EARTH_X, MOON_X, OMEGA_EM, MU_EARTH, MU_MOON, propagate_nbody, _rot_z` (Tasks 1-4).
- Produces: `lagrange_points(t_s) -> dict[str, np.ndarray]` with keys `"L1".."L5"`, each an inertial position `(3,)` at `t_s`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_nbody.py`:

```python
def test_L4_L5_exact_equilateral_geometry():
    L = nb.lagrange_points(0.0)
    e = np.array([nb.EARTH_X, 0.0, 0.0])
    m = np.array([nb.MOON_X, 0.0, 0.0])
    for key, sign in (("L4", +1), ("L5", -1)):
        p = L[key]
        assert abs(np.linalg.norm(p - e) - nb.D_EM) < 1e-3   # distance d from Earth
        assert abs(np.linalg.norm(p - m) - nb.D_EM) < 1e-3   # distance d from Moon
        assert np.sign(p[1]) == sign                          # L4 leads (+y), L5 trails


def test_collinear_points_match_known_positions_and_balance():
    L = nb.lagrange_points(0.0)
    # Published Earth-Moon CR3BP rotating-frame x (units of d), barycenter origin.
    for key, x_over_d in (("L1", 0.8369), ("L2", 1.1557), ("L3", -1.0051)):
        x = L[key][0]                              # at t=0 rotating == inertial x
        assert abs(x / nb.D_EM - x_over_d) < 1e-3
        # Net effective (gravity + centrifugal) acceleration ~ 0 at the point.
        p = L[key]
        g = nb.gravity_accel(p, 0.0, attractors=nb.EARTH_MOON)
        centrifugal = nb.OMEGA_EM**2 * np.array([p[0], p[1], 0.0])
        assert np.linalg.norm(g + centrifugal) < 1e-6


def test_L4_stays_bounded_over_a_day():
    L = nb.lagrange_points(0.0)
    p = L["L4"]
    v = np.cross([0.0, 0.0, nb.OMEGA_EM], p)       # co-rotating: stationary in rot frame
    st = StateVector(r=p, v=v, mu=MU_EARTH, epoch_s=0.0)
    out = nb.propagate_nbody(st, 86400.0, attractors=nb.EARTH_MOON, max_step_s=60.0)
    moved_L4 = nb.lagrange_points(86400.0)["L4"]
    assert np.linalg.norm(out.r - moved_L4) < 0.1 * nb.D_EM   # stable: doesn't escape
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k "L4 or collinear"`
Expected: FAIL — `AttributeError: ... 'lagrange_points'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py` (add `from scipy.optimize import brentq` to the imports):

```python
def _collinear_accel_x(x):
    """Net rotating-frame x-acceleration for a point on the Earth-Moon axis at x."""
    ax_g = (-MU_EARTH * (x - EARTH_X) / abs(x - EARTH_X)**3
            - MU_MOON * (x - MOON_X) / abs(x - MOON_X)**3)
    return OMEGA_EM**2 * x + ax_g


def lagrange_points(t_s):
    """Inertial positions of L1..L5 at t_s [m]."""
    eps = 1e-3 * D_EM
    x1 = brentq(_collinear_accel_x, EARTH_X + eps, MOON_X - eps)      # between bodies
    x2 = brentq(_collinear_accel_x, MOON_X + eps, MOON_X + 0.4 * D_EM)  # beyond Moon
    x3 = brentq(_collinear_accel_x, -1.6 * D_EM, EARTH_X - eps)       # beyond Earth
    h = np.sqrt(3.0) / 2.0 * D_EM
    xtri = (0.5 - MASS_RATIO) * D_EM
    rot = {
        "L1": np.array([x1, 0.0, 0.0]),
        "L2": np.array([x2, 0.0, 0.0]),
        "L3": np.array([x3, 0.0, 0.0]),
        "L4": np.array([xtri, h, 0.0]),
        "L5": np.array([xtri, -h, 0.0]),
    }
    Rinv = _rot_z(OMEGA_EM * t_s).T   # rotating -> inertial
    return {k: Rinv @ v for k, v in rot.items()}
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
Expected: PASS (all 10). If a collinear root bracket fails (`f(a)` and `f(b)` same sign),
fix the bracket — the known roots are L1≈0.837d, L2≈1.156d, L3≈−1.005d; do NOT widen the
position tolerance to mask a wrong root.

- [ ] **Step 5: Run the full core suite + commit**

Run: `.venv/Scripts/python -m pytest tests/core -q` → all pass.

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: Lagrange points (L4/L5 exact, L1/L2/L3 collinear solve)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Self-Review

- **Spec coverage:** barycentric circular Earth+Moon (T1) ✓; attractor-list `gravity_accel` (T2) ✓; velocity-Verlet `propagate_nbody` with proximity sub-stepping (T3) ✓; `rotating_frame` + `jacobi_constant` (T4) ✓; `lagrange_points` (T5) ✓; `MU_MOON` already in constants (used, not re-added) ✓. Tests cover all spec invariants: two-body reduction (T3), Jacobi conservation (T4), reversibility (T3), L4/L5 geometry + L1/L2/L3 known positions & balance + L4 bounded (T5).
- **Placeholder scan:** none — every step has complete code and concrete tolerances.
- **Type consistency:** `gravity_accel(r, t, attractors)`, `propagate_nbody(state, dt, attractors, max_step_s)`, `rotating_frame(r, v, t) -> (r_rot, v_rot)`, `jacobi_constant(state, t)`, `lagrange_points(t) -> dict` are used identically wherever referenced. `_rot_z` (T4) is reused in T5. `_CircularBody.state_at` returns `StateVector` throughout.
- **Tolerance honesty:** the two-body match uses a fine `max_step_s=0.5` so 2nd-order Verlet genuinely reaches < 1 m / < 10 m (not a loosened bound); Jacobi 1e-6 over 7 days is within symplectic reach at 600 s steps.
