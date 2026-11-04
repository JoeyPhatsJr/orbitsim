# Map Readability, Plan 2 of 5: Live, Accurate Maneuver Preview — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The planned-burn line follows the player's edits every frame, then within about a second settles onto an exact N-body path that matches what the ship will really do.

**Architecture:**
1. **Faster, accurate exact prediction (core).** The drawn predictions (live trajectory and maneuver preview) get their own stepping in `core/nbody.py`: step size from the **current** distance to each body at 1/1000 of the local timescale, plus a bit-identical per-time memo of the prediction ephemeris. The flown simulation's propagator is untouched.
2. **Live estimate and settle (render).** `maneuver_editor` draws an instant two-body estimate while the burn is changing, and submits the exact job only after the burn has been still for `PREVIEW_SETTLE_S`.

**Tech Stack:** Python 3.11, numpy, Panda3D, pytest (+xdist); the app-golden harness in `tests/render/app_harness/`.

**Spec:** `docs/superpowers/specs/2026-11-04-map-and-planning-readability-design.md`, section "Live line while editing". Plan 1 (PR #11) is merged; this builds on its `render/line_style.py`.

## What planning measured (2026-11-04, M1 Pro)

Pinned state: the post-burn state the app previews for the `I` (intercept) Moon plan.

| Method | Time | Moon periapsis radius | Epoch |
|---|---|---|---|
| Converged truth: current-distance steps at divisor 10,000 give 45,677 km / 296.258 h; periapsis steps at divisor 3,200 give 45,664 km / 296.26 h (agree to 0.03%) | 5–9.9 s | **45,670 km** | **296.259 h** |
| Flight integrator (periapsis steps, divisor 200) | 21 s | 43,100 km | 296.66 h |
| **Today's preview** (periapsis steps, 120 s floor) | **6.1 s** | **6,057 km**; position error up to 420,000 km | 301.2 h |
| Current-distance steps, divisor 1000, 1 ms floor | 1.1 s | 46,008 km (0.7%) | 296.205 h (3 min) |

What this shows:
- Today's preview is **wrong, not just slow**. Its 120 s floor is far too coarse near LEO, and a translunar flyby amplifies the departure error.
- A memo of the prediction ephemeris (same planet state reused for the same time) is bit-identical and takes 6.2 s → 4.3 s.
- The flight integrator itself is about 5% off truth on this flyby. That's out of scope here; flag it to the user and leave it.

## Global Constraints

- **The flown physics doesn't change.** `propagate_solar_system`, `propagate_earth_moon`, `_solar_cap_s` and `_timescale_cap_s` must give bit-identical results with their default arguments; `tests/core/test_nbody.py`'s `_PINNED_LEO_COAST_R/V` is the check. App-golden fields `vessel`, `sim_time_s`, `warp`, `target`, `events` must stay identical. `texts` (encounter readouts become accurate), `world_labels` and `rig` may be re-recorded, with the reason in the commit.
- **Preview accuracy:** Moon-periapsis **altitude** within **1%**, and encounter (periapsis) time within **10 minutes**, of the converged reference on the pinned state.
- **Preview speed:** a Moon-transfer exact preview in **about a second or less** (goal ≤ 1.0 s; a verification step, not a pytest assertion).
- **While the burn is changing:** an instant two-body estimate, drawn every frame and **fainter** than the exact line. A stale exact result (for an older burn) is discarded, as today via `_preview_future_key`.
- `core/` never imports Panda3D; pool jobs stay module-level and picklable (the golden harness's pool stand-in enforces it).
- TDD; commit + push after each task; stage by explicit path; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Escape / hyperbolic burns:**
   - the estimate must still draw (`sample_orbit_points` handles e ≥ 1);
   - the exact preview's 400-day horizon must stay fast with the new stepping;
   - Pinned by `test_estimate_points_hyperbolic` (Task 3) and `test_escape_preview_is_finite_and_fast_enough` (Task 2).
2. **Burn cleared mid-computation** (Δv back to 0 while a job is in flight): the estimate and exact lines both disappear, and the late result is discarded. Pinned by the app probe in Task 3.
3. **A burn that never changes:** no more endless resubmits. Today the code resubmits the same preview every 0.2 s while a burn exists. Pinned by `test_no_resubmit_when_exact_is_current` (Task 3).
4. **Offline (no DE440):** the accuracy test skips; the preview still runs on the circular fallback. The existing preview tests already run either way.
5. **The live trajectory line shares the sampler,** so it gets the new stepping too and must not slow down. Pinned by the profiler gate in Task 4 (`idle`, `high_warp`).

---

### Task 1: Bit-identical prediction-ephemeris memo

**Files:**
- Modify: `orbitsim/core/nbody.py` (`stable_prediction_ephemeris`, `_prediction_ephemeris_state`)
- Test: `tests/core/test_nbody.py`

**Interfaces:**
- Produces: inside `stable_prediction_ephemeris()`, `_prediction_ephemeris_state(name, fallback_fn, t_s)` returns the **same object** for repeated calls with the same `name` and the identical float `t_s`. It does this through a per-context `_ephemeris_context.prediction_last` dict, `{name: (t_s, state)}`, created and restored alongside `prediction_cache`. Outside the context, behaviour is unchanged.

- [ ] **Step 1: Write the failing test**

```python
def test_prediction_ephemeris_reuses_the_state_for_a_repeated_time(monkeypatch):
    calls = []
    real = nb.hermite_state
    monkeypatch.setattr(nb, "hermite_state", lambda *a, **k: calls.append(1) or real(*a, **k))
    with nb.stable_prediction_ephemeris():
        a = nb._csun(12345.678)
        b = nb._csun(12345.678)      # the accel and the step cap ask for the same t
        c = nb._csun(12346.0)
    assert a is b and len(calls) == 2 and c is not a
```

- [ ] **Step 2: Run it, expect FAIL** (`len(calls) == 3`).
- [ ] **Step 3: Implement the memo.**
- [ ] **Step 4: Run `tests/core/test_nbody.py` (pinned LEO coast included) and `tests/render/test_trajectory_sampling.py tests/render/test_maneuver_preview.py`.** Expected: all pass.
- [ ] **Step 5: Commit** `perf(core): memoize the prediction ephemeris per time (bit-identical)`.

### Task 2: Prediction stepping that matches reality, fast

**Files:**
- Modify: `orbitsim/core/nbody.py`
  - `_timescale_cap_s(radius_m, mu, divisor=SUBSTEP_DIVISOR)`
  - `_solar_cap_s(r, v, t_s, max_step_s, periapsis_fn=_conic_periapsis_m, divisor=SUBSTEP_DIVISOR)`
  - new `propagate_solar_prediction`
- Modify: `orbitsim/render/trajectory_sampling.py` (the solar branch calls `propagate_solar_prediction`)
- Modify: `orbitsim/render/app.py:36` (`PREDICTION_MIN_SUBSTEP_S = 1.0`; it's only a stall guard now)
- Test: `tests/render/test_preview_accuracy.py` (new); `tests/core/test_nbody.py`

**Interfaces:**
- Produces:
  - `PREDICTION_SUBSTEP_DIVISOR = 1000.0`
  - `def propagate_solar_prediction(state, dt_s, max_step_s=24 * 3600.0, min_substep_s=1.0) -> StateVector`: the same Verlet and gravity as `propagate_solar_system`, but each body's cap uses its **current** distance (`periapsis_fn = lambda r_rel, v_rel, mu: _norm3(r_rel)`) and `divisor=PREDICTION_SUBSTEP_DIVISOR`. For drawn predictions only; never called by `World.step`.
- Consumes: Task 1's memo, which is active through `stable_prediction_ephemeris` in `sample_trajectory`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/render/test_preview_accuracy.py
"""The drawn Moon-transfer preview must predict the converged path (see plan 2's table)."""
import time
import numpy as np, pytest
from orbitsim.core.constants import R_MOON
from orbitsim.core.state import StateVector
from orbitsim.core import nbody as nb
from orbitsim.render.trajectory_sampling import sample_preview

# Post-burn state the app previews for the "I" Moon intercept (captured 2026-11-04).
PINNED = StateVector(
    r=np.array([-8739999.637530766, -1247379.9275047905, -178197.13320472895]),
    v=np.array([1496.0148216363118, -9205.571968293121, -1324.9368000308782]),
    mu=398600400000000.0, epoch_s=3669.367180937791)
# Converged truth (two fine methods agree to 0.03%; see the plan's table).
REF_PERIAPSIS_RADIUS_M = 45_670.0e3
REF_PERIAPSIS_EPOCH_S = 296.259 * 3600.0
PREVIEW_FLOOR_S = 1.0          # OrbitApp.PREDICTION_MIN_SUBSTEP_S after this task

def _moon(encounters):
    return next(e for e in encounters if e.body_name == "Moon")

@pytest.fixture
def de440():
    if not nb.refresh_ephemeris_cache(PINNED.epoch_s) and not nb.ephemeris_available():
        pytest.skip("DE440 ephemeris not available offline")

def test_moon_transfer_preview_matches_converged_truth(de440):
    _pts, encounters = sample_preview(PINNED, True, PREVIEW_FLOOR_S)
    moon = _moon(encounters)
    ref_alt = REF_PERIAPSIS_RADIUS_M - R_MOON
    assert abs((moon.periapsis_radius_m - R_MOON) - ref_alt) <= 0.01 * ref_alt
    assert abs(moon.periapsis_epoch_s - REF_PERIAPSIS_EPOCH_S) <= 600.0

def test_escape_preview_is_finite_and_fast_enough(de440):
    esc = StateVector(r=PINNED.r, v=PINNED.v * 1.25, mu=PINNED.mu, epoch_s=PINNED.epoch_s)
    t0 = time.perf_counter()
    pts, _enc = sample_preview(esc, True, PREVIEW_FLOOR_S)
    assert np.isfinite(pts).all() and time.perf_counter() - t0 < 5.0   # generous CI bound
```

```python
# tests/core/test_nbody.py: the prediction stepper never feeds the flown sim
def test_world_step_never_uses_the_prediction_stepper(monkeypatch):
    monkeypatch.setattr(nb, "propagate_solar_prediction",
                        lambda *a, **k: pytest.fail("flight must not use prediction stepping"))
    ... build a World(solar_system=True) with one LEO vessel as the existing World tests do; world.step(600.0)
```

- [ ] **Step 2: Run them, expect FAIL.** The accuracy test fails today on periapsis (about 6,057 km vs 45,677 km); the World test fails on a missing attribute.
- [ ] **Step 3: Implement.** Parameterise `_timescale_cap_s` and `_solar_cap_s` with defaults equal to today's values, so flight is bit-identical. Add `propagate_solar_prediction`. Switch `sample_trajectory`'s solar branch to it, keeping `max_step_s=24 h`. Set `PREDICTION_MIN_SUBSTEP_S = 1.0`.
- [ ] **Step 4: Run** `tests/render/test_preview_accuracy.py tests/core/test_nbody.py tests/render/test_trajectory_sampling.py tests/render/test_maneuver_preview.py`. Expected: all pass, `_PINNED_LEO_COAST` untouched.
- [ ] **Step 5: Timing check (verification, not a test).** Time `sample_preview(PINNED, True, 1.0)` warm. Expected **≤ 1.0 s** (spike: about 1.1 s before the memo). If it's over, note it in the ledger and continue; don't trade accuracy for it.
- [ ] **Step 6: Commit** `fix(core): drawn predictions step by current distance (accurate Moon flybys, ~6x faster)`.

### Task 3: Live estimate while editing, exact after it settles

**Files:**
- Modify: `orbitsim/render/line_style.py` (add `ESTIMATE_STYLE`)
- Modify: `orbitsim/render/trajectory_helpers.py` (pure helpers)
- Modify: `orbitsim/render/controllers/maneuver_editor.py` (`_update_maneuver_preview`; drop `PREVIEW_THROTTLE_S`, add `PREVIEW_SETTLE_S = 0.25`)
- Modify: `orbitsim/render/app.py:130-135` (new state: `_preview_estimate_np`, `_preview_exact_key`, `_preview_last_key`, `_preview_key_changed_t`; drop `_preview_submit_t`)
- Test: `tests/render/test_trajectory_helpers.py` (create if missing), `tests/render/test_line_style.py`, `tests/render/test_live_preview_app.py` (new, subprocess probe like `test_opening_camera_app.py`)

**Interfaces:**
- Produces:
  - `ESTIMATE_STYLE = LineStyle((1.0, 0.62, 0.18, 0.45), 3.0, 0.0, 0.75, dash=(3, 4))`: the plan colour, fainter and finer-dashed.
  - `def kepler_estimate_points(state, n: int = 256) -> np.ndarray | None`: geocentric (n, 3) points of the post-burn two-body orbit via `sample_orbit_points(state_to_elements(state), n)`, or `None` for a degenerate state (ValueError). Lives in `trajectory_helpers.py`.
  - `def should_submit_exact(current_key, exact_key, inflight_key, last_change_t, now, settle_s) -> bool`: True only when `current_key != exact_key`, `current_key != inflight_key`, no job is in flight (`inflight_key is None`), and `now - last_change_t >= settle_s`.
- Behaviour, each frame in `_update_maneuver_preview`:
  1. Poll the finished job. If its key == the current key: draw the exact line (`PLAN_STYLE`, as today), set `_preview_exact_key`, remove the estimate. Otherwise discard it.
  2. If Δv is 0: remove both lines, forget the keys, return (as today).
  3. If the key changed since the last frame: record `_preview_key_changed_t`, remove the stale exact line and its encounters, and rebuild the estimate (`ESTIMATE_STYLE`, parented at `to_render(0)`, Earth-centred points / scale).
  4. If `should_submit_exact(...)`: submit `sample_preview` as today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/render/test_trajectory_helpers.py
def test_estimate_points_match_the_two_body_orbit():
    st = StateVector(r=np.array([6.878e6, 0, 0]), v=np.array([0, 9000.0, 0]), mu=MU_EARTH)
    np.testing.assert_array_equal(kepler_estimate_points(st, 64),
                                  sample_orbit_points(state_to_elements(st), 64))

def test_estimate_points_hyperbolic():
    st = StateVector(r=np.array([6.878e6, 0, 0]), v=np.array([0, 12500.0, 0]), mu=MU_EARTH)
    pts = kepler_estimate_points(st, 64)
    assert pts.shape == (64, 3) and np.isfinite(pts).all()

def test_estimate_points_none_for_degenerate_state():
    st = StateVector(r=np.array([6.878e6, 0, 0]), v=np.zeros(3), mu=MU_EARTH)
    assert kepler_estimate_points(st, 64) is None

K1, K2 = (100.0, 0.0, 0.0, 50.0), (110.0, 0.0, 0.0, 50.0)
def test_no_submit_while_still_changing():
    assert not should_submit_exact(K2, K1, None, last_change_t=10.0, now=10.1, settle_s=0.25)
def test_submit_once_settled():
    assert should_submit_exact(K2, K1, None, last_change_t=10.0, now=10.3, settle_s=0.25)
def test_no_resubmit_when_exact_is_current():
    assert not should_submit_exact(K2, K2, None, last_change_t=0.0, now=99.0, settle_s=0.25)
def test_no_submit_while_a_job_is_in_flight():
    assert not should_submit_exact(K2, K1, K1, last_change_t=0.0, now=99.0, settle_s=0.25)
```

```python
# tests/render/test_line_style.py
def test_estimate_is_a_fainter_plan_line():
    assert ESTIMATE_STYLE.color[:3] == PLAN_STYLE.color[:3]
    assert ESTIMATE_STYLE.color[3] < PLAN_STYLE.color[3] and ESTIMATE_STYLE.dash is not None
```

`tests/render/test_live_preview_app.py`: a subprocess probe on the real app with inline executors (`install_determinism()`). It launches, adds a prograde burn with hooks (`hooks.set_jog` or the existing node hooks), then checks:
- **frame 1:** `app._preview_estimate_np` is not None and `app._preview_np` is None;
- **after 30 frames** (0.5 s at the fixed 60 fps): `_preview_np` is not None and `_preview_estimate_np` is None;
- **after clearing the burn** (Clear button): both are None.

It prints one RESULT line and the test asserts on it. Private attributes are read through new `hooks.preview_lines(app) -> (estimate_shown: bool, exact_shown: bool)`, since hooks.py is the one test file allowed to touch them.

- [ ] **Step 2: Run them, expect FAIL** (ImportError for the helpers and style; the probe fails on the missing hook).
- [ ] **Step 3: Implement** the style, the helpers, the hook and the new `_update_maneuver_preview`.
- [ ] **Step 4: Run** `tests/render/test_trajectory_helpers.py tests/render/test_line_style.py tests/render/test_live_preview_app.py tests/render/test_maneuver_preview.py`. Expected: all pass.
- [ ] **Step 5: Eyeball it.** Screenshot right after an `I` plan with a nudge (estimate, faint dashed), then 1 s later (exact, bold dashed, encounter patch).
- [ ] **Step 6: Commit** `feat(render): maneuver line follows edits instantly; exact path settles after`.

### Task 4: Goldens, gates, docs, PR

- [ ] **Step 1:** Full suite with `-n auto`. Sandbox goldens whose `texts` show encounter readouts (`target_intercept`, `zoom_with_target_node`, maybe `node_edit_execute`) will change; **only** `texts`, `world_labels` and `rig` may differ.
- [ ] **Step 2:** Re-record each failing golden with `python -m tests.render.app_harness.record <name>`. Prove `vessel`, `sim_time_s`, `warp`, `target` and `events` are identical vs `HEAD`, using the same comparison as plan 1 (git-show the old JSON and diff those five keys).
- [ ] **Step 3: Profiler gate vs `main`** (export with `git archive` into the scratchpad, as in plan 1): `idle`, `zoom`, `zoom_node_planned`, `high_warp` mean within 10%. `node_planned` with `--inline` should drop sharply (the preview job ran inline in the profile).
- [ ] **Step 4:** CLAUDE.md, under the "Trajectory lines are the main per-frame cost" gotcha: drawn predictions use `propagate_solar_prediction`, which steps by current distance at 1/1000 timescale because a periapsis-sized step with a 120 s floor mispredicted Moon flybys by 7×. Never point `World.step` at it. The preview is debounced (`PREVIEW_SETTLE_S`) with a two-body estimate while editing.
- [ ] **Step 5:** Final whole-branch review, fix pass, PR, merge.
