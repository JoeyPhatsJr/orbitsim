# N-Body Lagrange-Point Visualization (Cycle 1c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** Task 1 and Task 2 are pure (TDD) — dispatch to **Haiku subagents**. Task 3 is render integration — executed **inline by the controller** with headless verification. Review each task before the next. (Independent review subagents may be unavailable under a monthly spend cap; if so, the controller reviews inline and flags the gap.)

**Goal:** Show the Earth-Moon Lagrange points (L1–L5) in the live sandbox and let the player select them as lightweight navigation targets.

**Architecture:** A new pure core function computes the five L-points consistent with the live `earth_moon_accel` force model and the Moon's real inclined geometry. A `LagrangePointTarget` value object turns each point into a moving target. The render layer draws 5 constant-size markers + labels that track the rotating Earth-Moon line, adds the points to the existing target list, and shows a live distance/relative-speed readout when one is selected.

**Tech Stack:** Python 3, numpy, scipy (`brentq`), Panda3D. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- SI, float64 `(3,)` arrays; frame = Earth-centered inertial. (project rule)
- `core/` never imports render/sim/panda3d; `render/targets.py` is pure (imports core only). (project rule)
- Constants from `core/constants.py`; never hard-type μ values. (project rule)
- TDD for Tasks 1–2; **never loosen a tolerance to pass** — fix the implementation. (project rule)
- The existing barycentric `lagrange_points` stays untouched (1a tested reference). The new function is `earth_fixed_lagrange_points`. (spec)
- L-points computed relative to the Moon's **actual instantaneous geometry** (`moon_state_at`, real orbit normal), not a z=0 plane. (spec)
- Markers use a constant render-space scale (camera sits a fixed render distance from focus), not a tiny world radius. (project gotcha)
- Run tests with `.venv/Scripts/python -m pytest`. Commits: explicit paths; end with the `Co-Authored-By` line; then `git push`. (repo discipline)
- All existing tests stay green after each task.

## File Structure

- `orbitsim/core/nbody.py` — add `_earth_fixed_collinear_accel_along` (helper) and `earth_fixed_lagrange_points(t_s)`.
- `orbitsim/render/targets.py` — add `supports_closest_approach` flag to `MoonTarget`; add `LagrangePointTarget`.
- `orbitsim/render/app.py` — 5 markers + labels, append L-targets to `self._targets`, branch the target readout.
- `tests/core/test_nbody.py` — `earth_fixed_lagrange_points` tests.
- `tests/render/test_targets.py` — `LagrangePointTarget` tests.

---

## Task 1: `earth_fixed_lagrange_points` (Haiku subagent)

**Files:**
- Modify: `orbitsim/core/nbody.py`
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Consumes: `earth_moon_accel(r, t) -> (3,)`, `moon_state_at(t)`, `OMEGA_EM`, `D_EM`, `MU_EARTH`, `MU_MOON`, `brentq` — all already in `nbody.py`.
- Produces: `earth_fixed_lagrange_points(t_s) -> dict[str, np.ndarray]` with keys `"L1".."L5"`, each a float64 `(3,)` inertial position [m].

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_nbody.py` (the file already imports `numpy as np`, `nb` for the module, `moon_state_at`, and `MU_EARTH`/`MU_MOON`; if any are missing add them):

```python
def test_earth_fixed_lagrange_points_are_equilibria():
    # Each L-point nulls the rotating-frame acceleration (gravity+indirect + centrifugal),
    # with the rotation about the Moon's ACTUAL orbit normal. Holds at t=0 and t!=0.
    for t in (0.0, 1.0e5):
        m = moon_state_at(t)
        n = np.cross(m.r, m.v)
        omega = nb.OMEGA_EM * (n / np.linalg.norm(n))
        lps = nb.earth_fixed_lagrange_points(t)
        assert set(lps) == {"L1", "L2", "L3", "L4", "L5"}
        for name, r in lps.items():
            centrifugal = -np.cross(omega, np.cross(omega, r))
            net = nb.earth_moon_accel(r, t) + centrifugal
            assert np.linalg.norm(net) < 1e-6, (name, t, float(np.linalg.norm(net)))


def test_earth_fixed_L4_L5_equilateral():
    t = 0.0
    m = moon_state_at(t)
    d = np.linalg.norm(m.r)
    lps = nb.earth_fixed_lagrange_points(t)
    for name in ("L4", "L5"):
        L = lps[name]
        assert abs(np.linalg.norm(L) - d) < 1.0          # distance d from Earth
        assert abs(np.linalg.norm(L - m.r) - d) < 1.0    # distance d from the Moon
        cosang = np.dot(L, m.r) / (np.linalg.norm(L) * d)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        assert abs(ang - 60.0) < 1e-3                     # 60 deg from the Moon
    # L4 leads the Moon, L5 trails (opposite sides of the Earth-Moon line).
    nrm = np.cross(m.r, m.v)
    assert np.dot(np.cross(m.r, lps["L4"]), nrm) > 0
    assert np.dot(np.cross(m.r, lps["L5"]), nrm) < 0


def test_earth_fixed_collinear_placement():
    t = 0.0
    m = moon_state_at(t)
    d = np.linalg.norm(m.r)
    u = m.r / d
    lps = nb.earth_fixed_lagrange_points(t)
    s = {k: float(np.dot(lps[k], u)) for k in ("L1", "L2", "L3")}
    assert 0.0 < s["L1"] < d < s["L2"]      # L1 between bodies, L2 beyond the Moon
    assert s["L3"] < 0.0                     # L3 beyond Earth
    for k in ("L1", "L2", "L3"):
        assert np.linalg.norm(np.cross(lps[k], u)) < 1.0   # on the Earth-Moon line


def test_earth_fixed_lagrange_distance_invariant_under_rotation():
    names = ("L1", "L2", "L3", "L4", "L5")
    dist = {n: [] for n in names}
    for t in (0.0, 3.0e5, 6.0e5, 9.0e5):
        lps = nb.earth_fixed_lagrange_points(t)
        for n in names:
            dist[n].append(np.linalg.norm(lps[n]))
    for n in names:
        assert max(dist[n]) - min(dist[n]) < 1.0   # rigid rotation: |L| constant
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q -k "earth_fixed"`
Expected: FAIL — `AttributeError: module ... has no attribute 'earth_fixed_lagrange_points'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/nbody.py` (after the existing `lagrange_points` function). `earth_moon_accel`, `moon_state_at`, `OMEGA_EM`, `brentq` are already imported/defined in the module:

```python
def _earth_fixed_collinear_accel_along(s, u, t_s):
    """Net rotating-frame acceleration at the point p = s*u (signed distance s along the
    Earth-Moon unit vector u), projected onto u: live gravity (with the indirect term) plus
    the centrifugal term OMEGA_EM**2 * p about the origin."""
    p = s * np.asarray(u, dtype=np.float64)
    a = earth_moon_accel(p, t_s) + OMEGA_EM**2 * p
    return float(np.dot(a, u))


def earth_fixed_lagrange_points(t_s):
    """Inertial positions of L1..L5 [m] in the live Earth-fixed circular-Moon frame, consistent
    with earth_moon_accel (indirect term) and the Moon's actual inclined geometry at t_s.

    Collinear points solve net_along(s)=0 along the Earth-Moon line; the equilateral points are
    the Moon position rotated +/-60 deg about the orbit normal."""
    m = moon_state_at(t_s)
    rM = np.asarray(m.r, dtype=np.float64)
    d = np.linalg.norm(rM)
    u = rM / d
    n_hat = np.cross(rM, m.v)
    n_hat = n_hat / np.linalg.norm(n_hat)
    eps = 1e-3 * d
    s1 = brentq(_earth_fixed_collinear_accel_along, eps, d - eps, args=(u, t_s))           # L1
    s2 = brentq(_earth_fixed_collinear_accel_along, d + eps, d + 0.4 * d, args=(u, t_s))   # L2
    s3 = brentq(_earth_fixed_collinear_accel_along, -1.6 * d, -eps, args=(u, t_s))         # L3

    def _rot(vec, ang):   # Rodrigues rotation of vec by ang about n_hat
        c, sn = np.cos(ang), np.sin(ang)
        return vec * c + np.cross(n_hat, vec) * sn + n_hat * np.dot(n_hat, vec) * (1.0 - c)

    return {
        "L1": s1 * u,
        "L2": s2 * u,
        "L3": s3 * u,
        "L4": _rot(rM, np.radians(60.0)),
        "L5": _rot(rM, np.radians(-60.0)),
    }
```

- [ ] **Step 4: Run to verify they pass + no regressions**

Run: `.venv/Scripts/python -m pytest tests/core/test_nbody.py -q`
then `.venv/Scripts/python -m pytest tests/ -q`
Expected: all PASS. If `test_earth_fixed_lagrange_points_are_equilibria` fails with a residual just over `1e-6`, the bug is the collinear projection or the orbit-normal handling — **do not loosen the bound**. (`brentq` raising "f(a) and f(b) must have different signs" means a bracket is wrong — re-check the three intervals, not the tolerance.)

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/nbody.py tests/core/test_nbody.py
git commit -m "$(cat <<'EOF'
N-body: earth_fixed_lagrange_points (live Earth-fixed frame, real Moon geometry)

L1-L5 consistent with earth_moon_accel (indirect term) and the Moon's actual
inclined orbit normal: collinear points solve the projected rotating-frame
acceleration along the Earth-Moon line, equilateral points rotate the Moon
+/-60 deg about its orbit normal. Barycentric lagrange_points untouched.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: `LagrangePointTarget` (Haiku subagent)

**Files:**
- Modify: `orbitsim/render/targets.py`
- Test: `tests/render/test_targets.py`

**Interfaces:**
- Consumes: `earth_fixed_lagrange_points(t_s)` (Task 1), `OMEGA_EM`, `moon_state_at`, `StateVector`, `MU_EARTH`.
- Produces:
  - `MoonTarget.supports_closest_approach == True`.
  - `LagrangePointTarget(name: str, point_id: str)` with `.name`, `.supports_closest_approach == False`, and `.state_at(t_s) -> StateVector` (r = the L-point position, v = its rigid-rotation velocity about the Moon's orbit normal).

- [ ] **Step 1: Write the failing tests**

Append to `tests/render/test_targets.py` (add `import numpy as np` if absent):

```python
def test_moon_target_supports_closest_approach():
    from orbitsim.render.targets import MoonTarget
    assert MoonTarget.supports_closest_approach is True


def test_lagrange_target_position_matches_core():
    import numpy as np
    from orbitsim.render.targets import LagrangePointTarget
    from orbitsim.core.nbody import earth_fixed_lagrange_points
    t = 1.0e5
    tgt = LagrangePointTarget("L1", "L1")
    assert tgt.name == "L1"
    assert tgt.supports_closest_approach is False
    st = tgt.state_at(t)
    assert np.allclose(st.r, earth_fixed_lagrange_points(t)["L1"])


def test_lagrange_target_velocity_matches_finite_difference():
    # The point's velocity must equal a central finite-difference of its position
    # (proves v is the true rigid-rotation velocity, not a z-axis approximation).
    import numpy as np
    from orbitsim.render.targets import LagrangePointTarget
    from orbitsim.core.nbody import earth_fixed_lagrange_points
    t, dt = 1.0e5, 1.0
    st = LagrangePointTarget("L4", "L4").state_at(t)
    r0 = earth_fixed_lagrange_points(t - dt)["L4"]
    r1 = earth_fixed_lagrange_points(t + dt)["L4"]
    v_fd = (r1 - r0) / (2.0 * dt)
    assert np.linalg.norm(st.v - v_fd) < 1.0   # within 1 m/s
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/render/test_targets.py -q -k "lagrange or supports"`
Expected: FAIL — `ImportError`/`AttributeError` for `LagrangePointTarget` / `supports_closest_approach`.

- [ ] **Step 3: Implement**

Replace the contents of `orbitsim/render/targets.py` with:

```python
"""Targetable bodies for maneuver planning (pure; no Panda3D).

A Target answers 'where is it at time t' in the same inertial, Earth-centered
frame as the vessel. Ships become Targets in a later cycle.
"""
import numpy as np

from orbitsim.core.moon import moon_state_at
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.nbody import earth_fixed_lagrange_points, OMEGA_EM


class MoonTarget:
    name = "Moon"
    supports_closest_approach = True   # the Moon is ~Keplerian -> CA prediction is meaningful

    def state_at(self, t_s: float) -> StateVector:
        return moon_state_at(t_s)


class LagrangePointTarget:
    """An Earth-Moon Lagrange point as a navigation target. It rotates rigidly with the Moon,
    so it is NOT on a Keplerian orbit; closest-approach prediction is not applicable (the render
    layer shows a live distance/relative-speed readout instead)."""

    supports_closest_approach = False

    def __init__(self, name: str, point_id: str) -> None:
        self.name = name
        self.point_id = point_id

    def state_at(self, t_s: float) -> StateVector:
        r = earth_fixed_lagrange_points(t_s)[self.point_id]
        m = moon_state_at(t_s)
        n_hat = np.cross(m.r, m.v)
        n_hat = n_hat / np.linalg.norm(n_hat)
        v = np.cross(OMEGA_EM * n_hat, r)   # rigid-rotation velocity about the Moon's normal
        return StateVector(r=np.asarray(r, dtype=np.float64), v=v, mu=MU_EARTH, epoch_s=t_s)
```

- [ ] **Step 4: Run to verify they pass + no regressions**

Run: `.venv/Scripts/python -m pytest tests/render/test_targets.py -q`
then `.venv/Scripts/python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/targets.py tests/render/test_targets.py
git commit -m "$(cat <<'EOF'
Targets: LagrangePointTarget + supports_closest_approach flag

LagrangePointTarget.state_at returns the L-point position (from
earth_fixed_lagrange_points) and its rigid-rotation velocity about the Moon's
orbit normal. supports_closest_approach distinguishes the Keplerian Moon
(CA prediction) from L-points (live distance/rel-vel readout).

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: Render markers, labels, and target readout (controller inline)

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `earth_fixed_lagrange_points` (Task 1), `MoonTarget`/`LagrangePointTarget` (Task 2), existing `make_uv_sphere`, `TextNode`, `self.transform.to_render`, `self._targets`, `self._target`, `self._target_text`, `self._ca`.
- Produces: 5 L-markers + labels updated each frame; L-points selectable; live distance/rel-speed readout for L-point targets.

### Step 3a: Create markers, labels, and targets in `_start_sim`

- [ ] **Step 3a-1: Add L-targets and marker/label nodes**

In `_start_sim`, the sandbox branch currently has:

```python
            from orbitsim.render.targets import MoonTarget
            self._targets = [MoonTarget()]
            self._target = None     # current Target or None
```

Replace with:

```python
            from orbitsim.render.targets import MoonTarget, LagrangePointTarget
            self._targets = [MoonTarget()] + [
                LagrangePointTarget(n, n) for n in ("L1", "L2", "L3", "L4", "L5")
            ]
            self._target = None     # current Target or None
```

Then, just after the Moon marker block (the lines that create `self._moon_np` ... `self._moon_orbit_scale = None`), add the Lagrange marker + label creation:

```python
            # Lagrange-point markers (constant on-screen size) + billboard labels.
            self._lagrange_nps = []
            self._lagrange_labels = []
            for name in ("L1", "L2", "L3", "L4", "L5"):
                mk = make_uv_sphere(1.0, 8, 12)
                mk.reparent_to(self.render)
                mk.set_color(0.3, 0.9, 0.8, 1.0)   # teal — distinct from Moon/CA/node
                mk.set_light_off()
                mk.set_scale(4.0)
                self._lagrange_nps.append(mk)
                tn = TextNode(f"label_{name}")
                tn.set_text(name)
                tn.set_text_color(0.5, 1.0, 0.9, 1.0)
                lbl = self.render.attach_new_node(tn)
                lbl.set_scale(12.0)
                lbl.set_billboard_point_eye()
                lbl.set_light_off()
                self._lagrange_labels.append(lbl)
```

- [ ] **Step 3a-2: Verify import + suite**

`TextNode` and `make_uv_sphere` are already imported in `app.py`. Run:
`.venv/Scripts/python -c "import orbitsim.render.app"` then `.venv/Scripts/python -m pytest tests/ -q`
Expected: import OK, all tests PASS.

### Step 3b: Place markers + labels each frame

- [ ] **Step 3b-1: Update marker/label positions in `_update`**

In `_update`, find the Moon-position lines:

```python
        # Moon position this frame.
        moon_now = moon_state_at(self.clock.sim_time_s)
        self._moon_np.set_pos(*self.transform.to_render(moon_now.r))
```

Immediately after them add:

```python
        # Lagrange points this frame (rotate with the Moon).
        from orbitsim.core.nbody import earth_fixed_lagrange_points
        lps = earth_fixed_lagrange_points(self.clock.sim_time_s)
        for name, mk, lbl in zip(("L1", "L2", "L3", "L4", "L5"),
                                 self._lagrange_nps, self._lagrange_labels):
            rx, ry, rz = self.transform.to_render(lps[name])
            mk.set_pos(rx, ry, rz)
            lbl.set_pos(rx, ry, rz + 6.0)
```

- [ ] **Step 3b-2: Verify suite**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all PASS.

### Step 3c: Branch the target readout for L-points

- [ ] **Step 3c-1: Add the L-point readout branch**

In `_update`, the target block currently begins:

```python
        if self._target is not None:
            import time as _time
            now_real = _time.monotonic()
```

Replace that line `if self._target is not None:` (and only that line) with a branch that handles
L-point targets first, keeping the existing Moon CA block as the `elif`:

```python
        if self._target is not None and not self._target.supports_closest_approach:
            # Lagrange-point target: live distance + relative speed, no closest-approach
            # prediction (an L-point is not Keplerian — you fly to it and null relative velocity).
            self._ca = None
            L = self._target.state_at(self.clock.sim_time_s)
            dist = float(np.linalg.norm(v0.state.r - L.r))
            relsp = float(np.linalg.norm(v0.state.v - L.v))
            self._target_text.setText(
                f"Target: {self._target.name}   dist {dist / 1000:,.0f} km"
                f"   rel {relsp:,.0f} m/s")
        elif self._target is not None:
            import time as _time
            now_real = _time.monotonic()
```

(`v0` is already defined earlier in `_update`, in the maneuver-node block. The rest of the
existing CA block — the `if self._ca is None ...` recompute, the markers, and the
`Target: ... CA T-...` readout — stays exactly as-is under the new `elif`, just indented one
level deeper. Confirm indentation after editing.)

- [ ] **Step 3c-2: Verify suite**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all PASS.

### Step 3d: Headless verification

- [ ] **Step 3d-1: Markers/labels build; L-point select gives a live readout, no CA**

Save to the scratchpad and run (PYTHONPATH = repo root):

```python
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
import numpy as np
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.bodies import EARTH
from orbitsim.sim.world import Vessel, World
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp
from orbitsim.core.nbody import earth_fixed_lagrange_points

r = 7.0e6
st = StateVector(r=np.array([r, 0.0, 0.0]),
                 v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]), mu=MU_EARTH, epoch_s=0.0)
app = OrbitApp(World(central=EARTH, vessels=[Vessel(name="ship", state=st)]), SimClock(warp=1.0))
app._on_play()
app.taskMgr.step()

# Markers + labels exist and sit at the core-computed L-point positions.
assert len(app._lagrange_nps) == 5 and len(app._lagrange_labels) == 5
lps = earth_fixed_lagrange_points(app.clock.sim_time_s)
exp = app.transform.to_render(lps["L4"])
got = app._lagrange_nps[3].get_pos()
assert max(abs(got[i] - exp[i]) for i in range(3)) < 1e-3
print("markers placed at core L-point positions: PASS")

# Select an L-point target -> live readout, CA path not entered (_ca stays None).
app._target = app._targets[4]   # L4 (index 0 is the Moon)
app.taskMgr.step()
txt = app._target_text.getText()
print("L4 readout:", txt)
assert "L4" in txt and "dist" in txt and "rel" in txt
assert app._ca is None, "CA path must not run for an L-point target"
print("L-point live readout + no CA: PASS")
app.destroy()
print("ALL TASK 3 HEADLESS CHECKS PASSED")
```

Expected: both PASS lines and `ALL TASK 3 HEADLESS CHECKS PASSED`.

- [ ] **Step 3d-2: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
Render: Lagrange-point markers, labels, and target selection

Five teal L1-L5 markers + billboard labels track the rotating Earth-Moon
line each frame (positions from earth_fixed_lagrange_points). The points are
appended to the target list so the existing click-to-pick selects them; a
selected L-point shows a live distance + relative-speed readout (no
closest-approach prediction) while the Moon keeps its CA path.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Self-Review

**Spec coverage:**
- Earth-fixed L1–L5 consistent with `earth_moon_accel` + real Moon geometry → Task 1 ✓
- Barycentric `lagrange_points` left as reference → Task 1 (new function only) ✓
- `LagrangePointTarget` + `supports_closest_approach` → Task 2 ✓
- 5 markers + labels tracking the rotating line → Task 3a/3b ✓
- L-points appended to `self._targets`, reuse existing picking → Task 3a ✓
- navball TARGET/ANTITARGET via existing `sas_target_pos` → no change needed (existing wiring) ✓
- Live distance + rel-speed readout, skip CA for L-points → Task 3c ✓
- Tests: equilibrium <1e-6, L4/L5 equilateral, collinear ordering/on-line, distance invariance → Task 1 ✓; position match + finite-difference velocity → Task 2 ✓; markers build + select gives readout, `_ca` None → Task 3d ✓

**Placeholder scan:** none — every step has complete code and concrete tolerances/commands.

**Type consistency:** `earth_fixed_lagrange_points(t_s) -> dict[str, (3,)]` is produced in Task 1 and consumed identically in Task 2 (`[point_id]`) and Task 3 (`[name]`). `LagrangePointTarget(name, point_id)` constructor matches its use in Task 3a (`LagrangePointTarget(n, n)`). `supports_closest_approach` set in Task 2, read in Task 3c. `self._lagrange_nps`/`self._lagrange_labels` created in Task 3a, used in Task 3b/3d. `OMEGA_EM` used consistently (module constant).
