# Flight Accuracy: 4th-Order Symplectic Coast Integrator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A flown coast follows the true N-body path closely enough that a planned 200 km Moon flyby is flown as a ~200 km flyby, not an impact, and the result no longer depends on time warp.

**Architecture:** Replace the velocity-Verlet step in `core/nbody.py::_verlet_adaptive` with Yoshida's 4th-order symplectic composition: three Verlet sub-steps with weights w1, w0, w1. It keeps the same block-quantized adaptive step selection and the same bounded-energy (symplectic) behaviour, and the error drops from O(h²) to O(h⁴). Flight steps grow (divisor 200 → 100), and the warp budget counts **gravity evaluations** instead of steps, so frame cost at max warp stays what it is today. Powered flight (`core/flight.py`, always at 1× warp) is unchanged.

**Tech Stack:** Python 3.11, numpy, pytest (+xdist), the app-golden harness.

**Spec:** The user's decision of 2026-11-05: "Fix flight accuracy first". Plan 4 found that the planned 200 km Moon pass impacts when flown, at −372 to −1,140 km depending on chunk size. This plan comes before Plan 4a. Plan 3 is independent and may run first.

## What planning measured (2026-11-05, M1 Pro, pinned Moon transfer from `tests/render/test_preview_accuracy.py::PINNED`, converged truth 45,670 km)

Each method flies the post-burn state with `propagate_solar_system` in fixed chunks, like frames at warp, with the closest approach sampled at chunk ends:

| Integrator, divisor | Chunk 3,600 s | Chunk 60 s | Gravity evals (3,600 s) |
|---|---|---|---|
| **Verlet 200 (today)** | **−2,547 km** | **−2,354 km** | 36k |
| Verlet 800 | −141 km | — | 143k (4×) |
| Yoshida 67 | +124 km | — | 38k (1×) |
| **Yoshida 100** | **+24 km** | — | **54k (1.5×)** |
| Yoshida 200 | +17 km | **+5 km** | 108k (3×) |

What this shows:
- Yoshida at divisor 100 is about **100× more accurate than today for 1.5× the evaluations**, and it agrees across chunk sizes.
- A trap: `_timescale_cap_s` / `_solar_cap_s` take `divisor=SUBSTEP_DIVISOR` as a **default argument**, bound at definition time (Plan 2). Changing the module constant at runtime does nothing. The spike's first run fell into this.
- Spike script: `scratchpad/spike_flight.py`, outside the repo.

## Global Constraints

- **This is a deliberate physics change.** Every sandbox golden's `vessel` fields and the pinned LEO coast (`tests/core/test_nbody.py::_PINNED_LEO_COAST_R/V`) will change. Re-record or re-pin them only after the new values are shown to be **closer to a high-accuracy reference** than the old ones (method in Task 3). State the reason in the commit. The solar goldens have no vessel and must not change.
- **Accuracy targets:**
  - Pinned Moon transfer flown in 3,600 s and 60 s chunks: closest approach within **0.2% (91 km)** of 45,670 km.
  - Chunk-size spread **≤ 50 km**.
  - The integrator's convergence order measures as **4** (error ratio 10–22 when the step halves).
- **Symplectic:** steps stay uniform along a coast (periapsis caps, block quantization unchanged). Energy over 100 LEO orbits stays bounded, with no secular drift. Don't switch to RK4 (CLAUDE.md).
- **Warp:** `max_safe_warp` / `max_safe_warp_solar` budget **gravity evaluations** (today's 200 per frame), so a Yoshida step costs 3. Report max warp before/after for LEO, HEO, a translunar coast and an interplanetary cruise. If any regime loses more than **one** warp level, stop and tell the user before merging.
- **Frame time:** `high_warp`, `idle`, `zoom`, `zoom_node_planned`, `node_planned` within 10% of `main`.
- The drawn-prediction stepper (`propagate_solar_prediction`) also uses `_verlet_adaptive`. It gets the new step automatically; its divisor (1000) stays. Re-run `tests/render/test_preview_accuracy.py`: it must still pass and should get faster or more accurate.
- TDD; commit + push per task; explicit-path staging; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; final review, then PR, then merge.

## Review Focus

1. **Warp near bodies:** a low orbit at the highest allowed warp must not take more gravity evaluations per frame than today (budget in evals). Pinned by `test_warp_budget_counts_gravity_evaluations` (Task 2).
2. **Long coasts:** energy over 100 LEO orbits stays bounded. Pinned by `test_yoshida_energy_bounded_over_100_orbits` (Task 1).
3. **Landing / surface contact:** `World.step`'s clamp still works with larger steps. The existing landing tests must pass unchanged.
4. **Backward propagation** (negative dt is supported today): the 4th-order composition must also be time-reversible. Pinned by `test_yoshida_forward_then_back_returns_home` (Task 1).
5. **`MAX_SUBSTEPS_PER_CALL` exhaustion path** (finish coarsely): it must still terminate.

---

### Task 1: Yoshida 4th-order step inside `_verlet_adaptive`

**Files:** Modify `orbitsim/core/nbody.py` (`_verlet_adaptive`, renamed in place to `_symplectic_adaptive`; keep `_verlet_adaptive = _symplectic_adaptive` as an alias so callers and tests don't churn). Test: `tests/core/test_nbody.py`.

**Interfaces:**
- Produces: `YOSHIDA_W1 = 1 / (2 - 2**(1/3))`, `YOSHIDA_W0 = -(2**(1/3)) * YOSHIDA_W1`, `EVALS_PER_STEP = 3`.
- Each adaptive step of size `h` runs three kick-drift-kick sub-steps with sizes `w1·h, w0·h, w1·h`, reusing the end-acceleration (FSAL), so a step costs 3 gravity evaluations.
- Step-size selection (cap, block quantization, `MAX_SUBSTEPS_PER_CALL`) is unchanged.

- [ ] **Step 1: Write the failing tests** (two-body only, via `propagate_nbody` with a single Earth attractor; compare with `propagate_kepler`):

```python
def _kepler_err(divisor_scale):
    # one LEO period, error vs analytic Kepler; base step scaled to test convergence order
    ...
def test_symplectic_step_is_fourth_order():
    e1, e2 = _kepler_err(1.0), _kepler_err(0.5)
    assert 10.0 < e1 / e2 < 22.0          # Verlet gives ~4; Yoshida ~16

def test_yoshida_energy_bounded_over_100_orbits():
    # specific energy relative spread over 100 LEO orbits < 1e-9, no monotonic trend
    ...

def test_yoshida_forward_then_back_returns_home():
    # propagate +T then -T: back to the start within 1e-6 m
    ...
```

- [ ] **Step 2: Run them, expect FAIL** (the order ratio is about 4 with Verlet).
- [ ] **Step 3: Implement.** The `_PINNED_LEO_COAST` test **will now fail**. That's expected and handled in Task 3; don't re-pin yet.
- [ ] **Step 4: Run** `tests/core/test_nbody.py` excluding the pinned coast (`-k "not pinned"`). Expected: pass.
- [ ] **Step 5: Commit** `feat(core): 4th-order symplectic (Yoshida) coast step`.

### Task 2: Flight divisor 100 and an evaluation-based warp budget

**Files:** Modify `orbitsim/core/nbody.py`:
- `SUBSTEP_DIVISOR = 100.0`. Remember the default-argument trap: the defaults of `_timescale_cap_s` / `_solar_cap_s` must pick up the new value, since they're bound when the module loads.
- `max_safe_warp` / `max_safe_warp_solar`: allowed if `substeps * EVALS_PER_STEP <= budget_evals`, with the parameter renamed `budget_substeps` → `budget_evals=200`.

Test: `tests/core/test_nbody.py`.

- [ ] **Step 1: Write the failing tests**:
  - `test_warp_budget_counts_gravity_evaluations`: for a LEO state at the returned max warp, `_solar_system_substeps(...) * EVALS_PER_STEP <= 200`.
  - `test_flown_moon_transfer_matches_converged_truth`: fly `PINNED` in 3,600 s chunks and in 60 s chunks. Closest approach within 91 km of 45,670 km; spread ≤ 50 km. Skip if DE440 is unavailable. It's slow (~10 s), so mark it `@pytest.mark.slow` if the suite has the marker, otherwise keep it.
- [ ] **Step 2: Run them, expect FAIL** (budget semantics, and the divisor is still 200).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and the preview accuracy test.** Expected: pass.
- [ ] **Step 5: Warp report (verification).** Max warp before/after for LEO (500 km), HEO (Ap 40,000 km), mid-translunar coast, and an interplanetary cruise (from the porkchop scenario). If any regime drops more than one `WARP_STEPS` level: stop and ask the user.
- [ ] **Step 6: Commit** `feat(core): flight steps 2x larger, warp budget in gravity evaluations`.

### Task 3: Re-pin the LEO coast and re-record the sandbox goldens (deliberate physics change)

- [ ] **Step 1: Reference.** For `_PINNED_LEO_COAST`'s case and for each sandbox scenario's final vessel state, compute a high-accuracy reference: the same scenario's coast with Yoshida at divisor 1000. Scenarios that thrust use the same thrust code, which is unchanged. Use a scratch script that replays the scenario with the divisor patched through the functions' `__defaults__`.
- [ ] **Step 2:** Re-pin `_PINNED_LEO_COAST_R/V` to the new values. Assert in the commit message that |new − ref| < |old − ref|, with the numbers.
- [ ] **Step 3:** Re-record every failing sandbox golden (`python -m tests.render.app_harness.record <name>`). For each, record |Δr| old→new and confirm new is closer to its reference. Expected: `rig`, `texts`, `world_labels` and `events` are unchanged unless a readout shows a physics number. The solar goldens are unchanged.
- [ ] **Step 4:** Full suite with `-n auto`, green.
- [ ] **Step 5: Commit** with a table of scenario, old error, new error.

### Task 4: Gates, docs, review, PR

- [ ] **Step 1:** Profiler gate vs `main` (git-archive export, as in plans 1–2). `high_warp` matters most.
- [ ] **Step 2:** CLAUDE.md:
  - the "N-body substeps…" gotcha now says Yoshida 4th order, divisor 100, warp budget in evaluations;
  - the "Drawn predictions have their own stepper" note: flight is no longer ~5% off; quote the new number;
  - the default-argument trap.
- [ ] **Step 3:** Final whole-branch review, fix pass, PR (include the warp report and the accuracy table), merge.
