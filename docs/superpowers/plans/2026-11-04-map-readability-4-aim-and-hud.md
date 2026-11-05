# Map Readability, Plan 4 of 5: Aiming Help, Auto-Plan, and a Clean Screen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With the Moon (or a planet) targeted, the player sees where the target will be when they arrive and a trustworthy green / amber / red status. One button plans a safe low Moon pass (about 200 km). The screen is reduced to small corner panels that never overlap each other or the map labels.

**Architecture:** This is two PRs' worth of work, so the plan has two parts with a hard boundary.
- **Part 4a, aim and auto-plan (Tasks 1–6, PR A).**
  - The exact N-body preview's encounter periapsis becomes accurate for low passes (a Moon-relative conic from the nearest sample, not a parabola through samples 2.8 h apart).
  - The preview also reports the closest approach to the selected target.
  - A pure `aim_status` turns that into the status box and a "Moon when you arrive" ghost.
  - A new core planner, `core/flyby_targeting.py`, corrects the Lambert intercept onto a 200 km periapsis. It uses B-plane Newton/Broyden steps on the same prediction propagator the preview uses.
- **Part 4b, clean screen (Tasks 7–13, PR B).**
  - A pure `render/ui/hud_layout.py` owns every panel rectangle and is tested for no overlaps.
  - The panels are rebuilt on it: time+warp top-left, target top-right, orbit+fuel bottom-left, navball cluster bottom-centre, burn panel right.
  - The jog-slider panel and the ORBIT/VESSEL/MANEUVER dock are deleted.
  - World labels hide when they would land on a panel.
  - Map markers get their short spec labels.

**Tech Stack:** Python 3.11, numpy, scipy, Panda3D (DirectGUI, LineSegs), pytest (+xdist); the app-golden harness in `tests/render/app_harness/`.

**Spec:** `docs/superpowers/specs/2026-11-04-map-and-planning-readability-design.md`. This plan covers "The look" (Markers, Labels, Screen panels) and "Planning a burn" items 4–6. Plans 1 and 2 are merged. **Plan 3** (click-orbit menu, drag handles on the node, keyboard nudges; `docs/superpowers/plans/2026-11-04-map-readability-3-click-and-drag-burns.md`) lands first. Part 4b assumes a burn can be created and shaped without the slider panel.

## What planning measured (2026-11-04/05, M1 Pro, DE440 present)

Default sandbox LEO; Moon targeted at epoch 0; today's `I` plan is `intercept_node` with a 24×48 grid. It returns node epoch 3669.37 s with dV (pro, nrm, rad) = (3129.32, −9.77, −38.14) m/s. It is a **12-day** transfer, because a single-orbit departure window forces it.

| What | Result |
|---|---|
| Lambert grid + Nelder-Mead (`intercept_node`, serial) | 0.38–0.46 s |
| Exact preview (`sample_preview`) of that plan | 0.58 s; Moon Pe altitude **44,271 km** (the "aim at the centre" plan misses by 44,000 km in N-body; its Kepler readout says 229 km) |
| One N-body evaluation to the first Moon periapsis (`propagate_solar_prediction`, 1 h chunks) | 0.26–0.29 s |
| Prototype B-plane Broyden correction (FD Jacobian h = 0.02 m/s, halving line search) | **8 evaluations, 1.8 s serial**; dV (3131.43, −27.14, −87.12) m/s, 52 m/s from the seed; predicted Pe **188 km** |
| Same corrected node, independent check at different chunking (3600 s) | Pe **190 km** (chunking changes the answer by ~2 km, so the planner need not copy the preview's sampling) |
| Same node through today's `sample_preview` encounter readout | **3,258 km**. The parabola through samples 2.8 h apart is useless for low passes, which is why Task 2 exists |
| Same node **flown** with the flight integrator (`propagate_solar_system`, 600 s / 3600 s / 60 s / 10 s chunks) | **Impact** (conic Pe −1,134 / −1,138 / −1,052 / −372 km) |
| Flight integrator for only the first hour after the burn, prediction after | Impact (−1,033 km): the flight error is in the departure leg |

**The last two rows are a pre-existing physics problem this plan must not fix.** Plan 2 noted "the flight integrator is about 5% off truth on this flyby". At a 200 km aim, that 5% is the difference between a pass and a crash: the preview (≈ converged truth) says 190 km, but the flown ship hits the Moon at any warp above about 10×. Within an hour of the burn, the live line (also exact) shows the real path and the status box turns red. The player can correct it, but the auto-plan alone does not deliver a flyable 200 km pass until the flight integrator is fixed. This is raised to the user as an open question. Task 6 re-measures it and reports it in the PR.

## Global Constraints

- **The flown physics doesn't change.** `World.step`, `propagate_solar_system`, `propagate_earth_moon` and `integrate_powered_solar` stay bit-identical. `tests/core/test_nbody.py`'s pinned LEO coast is the check. The planner and the preview never call the flight propagators.
- **Golden fields** `vessel`, `sim_time_s`, `warp`, `target` stay identical, with two exceptions:
  - `target_intercept` and `zoom_with_target_node` may change only `vessel.nodes`, because the planner now chooses a different node. Each re-record states the old and new node and the preview's Pe in its commit.
  - `events` stay identical **after removing per-widget entries ending in `-pg*`** (`click-mouse1-pg*`, `adjust-pg*`, …). Adding and removing DirectGUI widgets changes only those counts. Every key/mouse binding entry must match.
- **Presentation fields** `texts`, `world_labels` and `rig` may be re-recorded, with the reason in the commit.
- **Golden comparison method:** git-show the old JSON at the parent commit and diff the protected keys in a scratch script, as in plans 1–2. Paste the result in the commit body.
- **Scenario edits** follow the spec's one-time exception: a script may change *how* it drives the UI (a hook that sets the same dV the jog produced, a renamed button). Its recorded physics fields must still match. Retire a scenario only if its behaviour is gone, and record a replacement then.
- **Status box copy** (spec, exact): green "✓ Moon encounter · Pe 4,317 km", amber "No encounter · closest 120,000 km", red "✗ Impact". "Pe" is the altitude above the target's **surface**; "closest" is the distance from the target's **centre**; the encounter test is "the exact preview path enters the target's SOI".
  - The bundled Rajdhani font has no ✓ or ✗ glyphs (checked with fontTools: `·`, `…`, `×` exist; `✓`, `✗`, `→` do not). The tick and cross are therefore drawn as a coloured LineSegs icon in front of the text, and the text strings omit them.
- **"checking…":** while the exact result for the *current* burn (and current target) is not in, the box and ghost are grey and show the Earth-only estimate. They take a colour only from an exact result whose key matches.
- **Planner:** aim periapsis altitude `FLYBY_TARGET_ALT_M = 200e3` above the Moon's surface, converged within ±25 km of the prediction model; node epoch unchanged from the Lambert seed. Planets and Lagrange points keep today's planners.
- **Performance:** no frame-time regression > 10% on `idle`, `zoom`, `zoom_node_planned`, `high_warp`, `node_planned` (`python -m tests.render.app_harness.profile <w>`, vs `main` on the same machine and session).
- **Layering:** `core/` never imports Panda3D. Pool jobs are module-level and picklable (the harness's pool stand-in pickles every job). Controllers stay stateless; state lives on `OrbitApp`.
- TDD; commit + push after each task; stage by explicit path; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Each part ends with a whole-branch review, then a PR and merge.

## Review Focus

1. **Target switched or cleared while an exact preview is in flight:** the box must never show the old target's result. The preview key includes the target name. Pinned by `test_preview_key_changes_with_target` (Task 2) and `test_status_checking_when_key_stale` (Task 3).
2. **Burn edited (Plan 3 drag or nudge) while the exact preview is in flight:** grey "checking…", never the previous colour, and the encounter markers must not fall back to the un-burned path (Plan 2's deferred minor). Pinned by `test_encounter_source_*` (Task 3) and the app probe in Task 3.
3. **A seed that never gets near the Moon, or offline (circular Moon):** the planner raises `ValueError`, the toast reads "No intercept found", and there's no crash. Offline it still converges on the circular Moon. Pinned by `test_moon_flyby_node_raises_when_no_pass` and the offline branch of `test_moon_flyby_node_reaches_200_km` (Task 4).
4. **Extreme times and warps** (T+ 969,100 d, ×100,000,000) at the smallest supported window (1280×720) and at 960×600: the text stays inside its cell. Pinned by `test_warp_cell_fits_worst_case` (Task 8).
5. **Window aspect 3:2 … 21:9 and UI scale 0.84 … 1.08:** no two panels overlap, and none leaves the screen. Pinned by `test_panels_never_overlap` (Task 7).

---
# Part 4a — Aim helpers and auto-plan (PR A)

### Task 1: Conic flyby geometry (pure core)

**Files:**
- Modify: `orbitsim/core/flyby.py`
- Test: `tests/core/test_flyby.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ConicPeriapsis: radius_m: float; time_to_periapsis_s: float; point_rel_m: np.ndarray`. `time_to_periapsis_s` is signed: negative once periapsis has passed. `point_rel_m` is the periapsis position relative to the body.
  - `def conic_periapsis(r_rel_m, v_rel_mps, mu: float) -> ConicPeriapsis`. Works for elliptic and hyperbolic relative states. Raises `ValueError` for zero angular momentum (radial line, no periapsis direction).
  - `def b_plane(r_rel_m, v_rel_mps, mu: float) -> tuple[np.ndarray, float]`. Returns `(B, v_inf)`, where `B = [B·T, B·R]` [m]. `S` is the incoming asymptote, `T = unit(S × ẑ)`, `R = S × T`, and `B = (S × h) / v_inf`. Raises `ValueError` if the state is not hyperbolic relative to `mu`.

- [ ] **Step 1: Write the failing tests**

```python
from orbitsim.core.constants import MU_MOON, R_MOON
from orbitsim.core.elements import KeplerianElements, elements_to_state
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.flyby import conic_periapsis, b_plane

def _hyperbola(rp, v_inf, i=0.4, raan=0.3, argp=1.1, nu=-1.2):
    a = -MU_MOON / v_inf**2
    e = 1.0 - rp / a
    return elements_to_state(KeplerianElements(a=a, e=e, i=i, raan=raan, argp=argp, nu=nu,
                                               mu=MU_MOON, epoch_s=0.0))

def test_conic_periapsis_recovers_rp_and_time():
    st = _hyperbola(R_MOON + 200e3, 1050.0)
    pe = conic_periapsis(st.r, st.v, MU_MOON)
    assert abs(pe.radius_m - (R_MOON + 200e3)) < 1e-3
    at_pe = propagate_kepler(st, pe.time_to_periapsis_s)
    assert abs(np.linalg.norm(at_pe.r) - pe.radius_m) < 1e-2
    np.testing.assert_allclose(at_pe.r, pe.point_rel_m, atol=1.0)

def test_conic_periapsis_time_is_negative_after_passing():
    st = _hyperbola(R_MOON + 200e3, 1050.0, nu=+0.8)
    assert conic_periapsis(st.r, st.v, MU_MOON).time_to_periapsis_s < 0.0

def test_conic_periapsis_elliptic():
    st = elements_to_state(KeplerianElements(a=5e6, e=0.3, i=0.2, raan=0.0, argp=0.5, nu=2.0,
                                             mu=MU_MOON, epoch_s=0.0))
    assert abs(conic_periapsis(st.r, st.v, MU_MOON).radius_m - 3.5e6) < 1e-3

def test_b_plane_magnitude_matches_impact_parameter():
    rp, v_inf = R_MOON + 200e3, 1050.0
    st = _hyperbola(rp, v_inf)
    B, vinf = b_plane(st.r, st.v, MU_MOON)
    assert abs(vinf - v_inf) < 1e-6
    assert abs(np.linalg.norm(B) - rp * np.sqrt(1 + 2 * MU_MOON / (rp * v_inf**2))) < 1e-3

def test_b_plane_raises_for_bound_state():
    st = elements_to_state(KeplerianElements(a=5e6, e=0.3, i=0.2, raan=0.0, argp=0.5, nu=2.0,
                                             mu=MU_MOON, epoch_s=0.0))
    with pytest.raises(ValueError):
        b_plane(st.r, st.v, MU_MOON)
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/core/test_flyby.py -q`.** Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** both functions with the eccentricity vector from CLAUDE.md's formula (with the `r`-magnitude factor). Clamp `arccos` arguments. Get the time to periapsis from the hyperbolic or elliptic anomaly. The hyperbolic incoming asymptote is `S = (sqrt(e²−1)/e)·P̂ + (e − 1/e)·Q̂` (normalised), with `P̂ = ê` and `Q̂ = ĥ × P̂`.
- [ ] **Step 4: Run the file.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): conic periapsis and B-plane for flyby targeting`.

### Task 2: Accurate encounter periapsis and target approach on the drawn path

**Files:**
- Modify: `orbitsim/core/encounters.py` (`find_encounters`, `_build_encounter`; new `TargetApproach`, `closest_approach_on_path`)
- Modify: `orbitsim/render/trajectory_sampling.py` (collect velocities; `approach_body`; `sample_preview` returns a 3-tuple)
- Modify: `orbitsim/render/trajectory_helpers.py` (`maneuver_preview_key` gains the target name)
- Modify: `orbitsim/render/controllers/maneuver_editor.py` (submit/poll the 3-tuple; key with target; store `app._preview_approach`)
- Modify: `orbitsim/render/controllers/trajectory_view.py` (live job passes `approach_body`; store `app._traj_approach`)
- Modify: `orbitsim/render/app.py` (`_build_background_work`: `_preview_approach = None`, `_traj_approach = [None]`)
- Test: `tests/core/test_encounters.py`, `tests/render/test_trajectory_sampling.py`, `tests/render/test_preview_accuracy.py`, `tests/render/test_trajectory_helpers.py`

**Interfaces:**
- Consumes: Task 1 `conic_periapsis`.
- Produces:
  - `find_encounters(points, epochs, dominant_fn, primary_name="Earth", velocities=None, body_state_fn=None)`. When both optional arguments are given, each encounter's periapsis comes from `conic_periapsis` of the Moon/planet-relative state at its nearest sample (`periapsis_index`), using the body velocity from `body_state_fn(name, t).v`:
    - `periapsis_radius_m` is the conic radius;
    - `periapsis_epoch_s` is the sample epoch plus `time_to_periapsis_s`;
    - `periapsis_point_m` is `body_state_fn(name, pe_epoch).r + point_rel_m`;
    - `impact` is `radius < body_radius_m`.

    Without them, behaviour is unchanged.
  - `@dataclass(frozen=True) class TargetApproach: body_name: str; distance_m: float; epoch_s: float; ship_point_m: np.ndarray; body_point_m: np.ndarray` (distance from the body's **centre**).
  - `def closest_approach_on_path(points, epochs, body_name, body_state_fn) -> TargetApproach`: the minimum over samples.
  - `sample_trajectory(..., approach_body: str | None = None)`: velocities are collected per sample and passed with `body_state_fn=geocentric_body_state` (from `core.nbody`) to `find_encounters`. When `approach_body` is set, the return tuple gains a trailing `TargetApproach`, computed **inside the same `stable_prediction_ephemeris` context**.
  - `sample_preview(state, solar_system, min_substep_s, target_name=None) -> (points, encounters, approach_or_None)`. Always a 3-tuple; update both existing test call sites.
  - `maneuver_preview_key(node, scheduled_epoch_s, target_name=None)` appends `target_name`. A target switch makes the drawn exact preview stale and resubmits it (≈0.6 s).
  - Lagrange-point targets and the Sun pass `target_name=None`: no approach, no status. Today's live distance readout stays.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_encounters.py — coarse samples of an exact hyperbola about a fixed body
def test_conic_refinement_beats_parabola_on_coarse_samples():
    st0 = _moon_hyperbola(rp=R_MOON + 200e3, v_inf=1050.0)        # helper: Kepler state 6 h before Pe
    epochs = np.arange(0.0, 12 * 3600.0, 2.8 * 3600.0)             # preview-like 2.8 h spacing
    states = [propagate_kepler(st0, t) for t in epochs]
    pts = np.array([s.r for s in states]); vels = np.array([s.v for s in states])
    body = lambda name, t: StateVector(np.zeros(3), np.zeros(3), MU_MOON, t)
    dom = lambda r, t: ("Moon", np.zeros(3), MU_MOON, R_MOON) if np.linalg.norm(r) < 6.6e7 \
        else ("Earth", np.zeros(3), 0.0, 0.0)
    old = find_encounters(pts, epochs, dom)[0]
    new = find_encounters(pts, epochs, dom, velocities=vels, body_state_fn=body)[0]
    assert abs(new.periapsis_radius_m - (R_MOON + 200e3)) < 1.0
    assert abs(new.periapsis_epoch_s - 6 * 3600.0) < 1.0
    assert abs(old.periapsis_radius_m - (R_MOON + 200e3)) > 1e6       # documents the old failure

def test_closest_approach_on_path_reports_centre_distance():
    pts, epochs = _straight_line(v_mps=1000.0, b_m=1.2e8, t0=-1e5, t1=1e5, n=201)
    body = lambda name, t: StateVector(np.zeros(3), np.zeros(3), MU_MOON, t)
    ap = closest_approach_on_path(pts, epochs, "Moon", body)
    assert ap.body_name == "Moon" and abs(ap.distance_m - 1.2e8) < 1e3 and abs(ap.epoch_s) < 1e3
```

```python
# tests/render/test_preview_accuracy.py (append; reuse the de440 fixture)
LOW_PASS_NODE = ManeuverNode(3669.3721983198875, 3131.43288823, -27.139359, -87.11806646)
REF_LOW_PASS_ALT_M = 190.1e3     # prediction propagator, 3600 s chunks, conic Pe (plan 4 table)

def test_low_moon_pass_reads_its_true_periapsis(de440):
    post = apply_maneuver(_default_world().vessels[0].state, LOW_PASS_NODE)
    _pts, encounters, _ap = sample_preview(post, True, 1.0)
    moon = next(e for e in encounters if e.body_name == "Moon")
    assert abs((moon.periapsis_radius_m - R_MOON) - REF_LOW_PASS_ALT_M) < 25e3   # was 3,258 km

def test_preview_reports_target_approach(de440):
    _pts, _enc, ap = sample_preview(PINNED, True, 1.0, target_name="Moon")
    assert ap.body_name == "Moon" and ap.distance_m < 1.0e8
```

```python
# tests/render/test_trajectory_helpers.py
def test_preview_key_changes_with_target():
    n = ManeuverNode(10.0, 1.0, 0.0, 0.0)
    assert maneuver_preview_key(n, 10.0, "Moon") != maneuver_preview_key(n, 10.0, None)
```

The existing `test_moon_transfer_preview_matches_converged_truth` (1% / 10 min) must still pass. The conic Pe at the 45,670 km pass is within ~10 km of the parabola's.

- [ ] **Step 2: Run** `tests/core/test_encounters.py tests/render/test_preview_accuracy.py tests/render/test_trajectory_helpers.py tests/render/test_trajectory_sampling.py`. Expected: the new tests FAIL; update the two 2-tuple unpackings to 3-tuples.
- [ ] **Step 3: Implement** as in Interfaces. In `maneuver_editor`:
  - pass `target_name` (the target's name if it is a `MoonTarget`/`PlanetTarget` other than the Sun, else None) into the key and into `sample_preview`;
  - store `app._preview_approach` with the encounters;
  - clear it wherever `_preview_encounters` is cleared.

  In `trajectory_view.rebuild_trajectory`, pass `approach_body` the same way. On poll, unpack the 4-tuple into `_traj_approach[idx]` when an approach was requested.
- [ ] **Step 4: Run the four files plus** `tests/render/test_maneuver_preview.py tests/render/test_live_preview_app.py`. Expected: PASS.
- [ ] **Step 5: Commit** `fix(core): exact preview reads low flyby periapses from the conic; reports target approach`.

### Task 3: Status box, "when you arrive" ghost, and "checking…"

**Files:**
- Create: `orbitsim/render/aim_status.py` (pure, no Panda3D)
- Modify: `orbitsim/render/trajectory_helpers.py` (`encounter_source`)
- Modify: `orbitsim/render/controllers/targeting.py`:
  - `build_targeting` builds the target panel: a DirectFrame on `a2dTopRight` with a title row, a status row (icon + text) and a buttons row that Task 5 fills. Until Task 9 moves it to the top, it sits at frame `(-0.98, -0.04, -0.90, -0.60)`, below today's VESSEL panel. It is hidden with no target;
  - new `frame_aim(node)` replaces the CA-marker half of `frame_closest_approach`;
  - remove `_ca_marker_ship`, `_ca_labels` and the `_target_line` readout for Moon/planet targets.
- Modify: `orbitsim/render/controllers/trajectory_view.py` (`update_encounter_cues` uses `encounter_source`)
- Modify: `orbitsim/render/controllers/readouts.py` (`update_marker_readability`: drop the CA candidates, add the ghost label at priority 85)
- Modify: `orbitsim/render/app.py` (`_update`: call `targeting.frame_aim(node)` where `frame_closest_approach` was)
- Modify: `tests/render/app_harness/hooks.py` (`aim_status(app) -> (kind, text)`; `encounter_marker_count(app) -> int`, the number of shown `_encounter_marker_nps`)
- Test: `tests/render/test_aim_status.py` (new), `tests/render/test_trajectory_helpers.py`, `tests/render/test_aim_app.py` (new subprocess probe, like `test_live_preview_app.py`)

**Interfaces:**
- Consumes: Task 2 `TargetApproach`, `Encounter`, `app._preview_approach`, `app._traj_approach[0]`, the key with target.
- Produces:
  - `@dataclass(frozen=True) class AimStatus: kind: str; text: str; icon: str | None; color: tuple[float, float, float, float]`. `kind` is one of `"encounter"`, `"miss"`, `"impact"`, `"checking"`. `icon` is `"check"`, `"cross"` or None.
  - Colours come from `THEME`: encounter `THEME.green`, miss `THEME.amber`, impact `THEME.danger`, checking `THEME.text_muted`.
  - `def aim_status(target_name: str, encounters, approach, exact: bool, estimate_closest_m: float | None) -> AimStatus`, with these rules:
    - not `exact` → checking. Text: `"checking…"`, plus `" · closest ~{fmt_dist(estimate)}"` when an estimate exists.
    - an encounter with `body_name == target_name` and `impact` → impact, text `"Impact"`, icon `"cross"`.
    - such an encounter without impact → encounter. Text: `f"{target} encounter · Pe {alt_km:,.0f} km"`, alt = `periapsis_radius_m − body_radius_m`; icon `"check"`.
    - otherwise → miss. Text: `f"No encounter · closest {fmt_dist(approach.distance_m)}"`, or `"No encounter"` when `approach` is None; no icon.
  - `def encounter_source(burn_exists: bool, exact_current: bool, preview_encounters, live_encounters) -> list`. It returns the preview's encounters if a burn exists and the exact result is current, `[]` if a burn exists but it isn't, and the live trajectory's otherwise. This fixes Plan 2's deferred minor: no fallback to the un-burned path while editing.
  - "Exact" is defined as:
    - with a burn: `app._preview_exact_key == maneuver_preview_key(node, app._node_epoch_s, target_name)`;
    - with no burn: `app._traj_approach[0]` exists and has `body_name == target_name`.
  - **Ghost:** a translucent copy of the target sphere. Its alpha is 0.35 when coloured and 0.18 when checking. It sits at `approach.body_point_m` when exact, and at the Earth-only Kepler closest-approach target position (today's throttled `closest_approach` computation, kept as the estimate) while checking. Its world label reads `"{Target} when you arrive"`. It is hidden for Lagrange/Sun targets and with no target.
  - The status box shows only for Moon/planet targets. Lagrange targets keep today's `Target: L1 dist … rel …` readout.

- [ ] **Step 1: Write the failing tests**

```python
# tests/render/test_aim_status.py
ENC = lambda pe_alt, impact=False: Encounter("Moon", 0, 1, 0.0, 1.0, 0, 0.0, R_MOON + pe_alt,
                                             np.zeros(3), 1000.0, MU_MOON, R_MOON, True, impact)
AP = TargetApproach("Moon", 1.2e8, 0.0, np.zeros(3), np.ones(3))

def test_green_encounter_text():
    s = aim_status("Moon", [ENC(4_317e3)], AP, exact=True, estimate_closest_m=None)
    assert (s.kind, s.text, s.icon) == ("encounter", "Moon encounter · Pe 4,317 km", "check")

def test_amber_miss_uses_centre_distance():
    s = aim_status("Moon", [], AP, exact=True, estimate_closest_m=None)
    assert (s.kind, s.text, s.icon) == ("miss", "No encounter · closest 120,000 km", None)

def test_red_impact():
    s = aim_status("Moon", [ENC(-500e3, impact=True)], AP, exact=True, estimate_closest_m=None)
    assert (s.kind, s.text, s.icon) == ("impact", "Impact", "cross")

def test_status_checking_when_key_stale():
    s = aim_status("Moon", [ENC(4_317e3)], AP, exact=False, estimate_closest_m=229e3)
    assert s.kind == "checking" and s.text == "checking… · closest ~229 km" and s.icon is None

def test_encounter_for_another_body_is_not_the_targets():
    other = dataclasses.replace(ENC(1e6), body_name="Mars")
    assert aim_status("Moon", [other], AP, exact=True, estimate_closest_m=None).kind == "miss"
```

```python
# tests/render/test_trajectory_helpers.py
def test_encounter_source_exact_preview():   assert encounter_source(True, True, ["p"], ["l"]) == ["p"]
def test_encounter_source_editing_shows_none(): assert encounter_source(True, False, ["p"], ["l"]) == []
def test_encounter_source_no_burn_uses_live():  assert encounter_source(False, False, [], ["l"]) == ["l"]
```

`tests/render/test_aim_app.py` probe (inline executors, a 60 fps fake clock):
1. Launch, `hooks.select_target(app, "Moon")`, step 150 (the live trajectory refreshes at 2 s).
2. `hooks.aim_status(app)[0] == "miss"` (LEO never reaches the Moon).
3. `h.send("i")`; step 2: the kind is in `("encounter", "miss", "impact")`, i.e. coloured and exact (inline executors finish at submit and the settle passes).
4. Then set a different burn through the Plan 3 hook (or `hooks.set_burn` from Task 10 if Plan 3 has none; add it now if missing) and step 1: the kind is `"checking"`, and `hooks.encounter_marker_count(app) == 0`.

- [ ] **Step 2: Run them; expect FAIL** (ImportError / missing hook).
- [ ] **Step 3: Implement.** The icon is a LineSegs glyph (check: 2 segments; cross: 2 segments) parented to the status row, coloured like the text, hidden when `icon is None`. Remove `CA: SHIP` / `CA: TARGET` labels and markers. Their `world_labels` entries disappear from goldens (presentation).
- [ ] **Step 4: Run** `tests/render/test_aim_status.py tests/render/test_trajectory_helpers.py tests/render/test_aim_app.py tests/render/test_controller_wiring.py`. Expected: PASS.
  - Then run the golden suite. Goldens differ only in `texts` / `world_labels` (CA labels gone, ghost and status added); `target_intercept` and `zoom_with_target_node` also differ in `-pg*` events.
  - Re-record them and run the protected-field diff (all protected fields identical, since the planner is unchanged until Task 5).
- [ ] **Step 5: Eyeball.** Take screenshots with the scratch `shoot.py` pattern (Moon targeted, no burn → amber; after `I` → coloured box + ghost). Check the ghost and status read clearly at map zoom `1e9` m.
- [ ] **Step 6: Commit** `feat(render): target status box and "when you arrive" ghost from the exact path`.

### Task 4: Flyby-targeting planner (core)

**Files:**
- Create: `orbitsim/core/flyby_targeting.py`
- Test: `tests/core/test_flyby_targeting.py` (new)

**Interfaces:**
- Consumes: Task 1 `conic_periapsis`, `b_plane`; `nbody.propagate_solar_prediction`, `nbody.stable_prediction_ephemeris`, `nbody.MOON_SOI_M`; `moon.moon_state_at`; `maneuvers.apply_maneuver`; `optimize._map_rows`.
- Produces:
  - Constants: `FLYBY_TARGET_ALT_M = 200e3`, `FLYBY_TOL_M = 25e3`, `PASS_SAMPLE_S = 3600.0`, `PASS_HORIZON_S = 20 * 86400.0` (the Lambert TOF grid tops out at 14 d), `FD_STEP_MPS = 0.02`, `MAX_ITER = 12`.
  - `@dataclass(frozen=True) class MoonPass: periapsis_radius_m: float; periapsis_epoch_s: float; b_plane_m: np.ndarray; v_inf_mps: float; in_soi: bool`.
  - `def first_moon_pass(post_burn: StateVector, horizon_s=PASS_HORIZON_S, sample_s=PASS_SAMPLE_S) -> MoonPass`:
    - inside `stable_prediction_ephemeris()`, step `propagate_solar_prediction(cur, sample_s, max_step_s=86400.0, min_substep_s=1.0)` and track the Moon distance;
    - stop at the first local minimum inside `MOON_SOI_M`, otherwise use the global minimum over the horizon;
    - build the result from `conic_periapsis` and `b_plane` of the Moon-relative state at the minimum sample (`MU_MOON`);
    - raise `ValueError("no Moon pass")` if the minimum is beyond `3 * MOON_SOI_M` (B-plane meaningless) or the relative state is bound.
  - `def _pass_for_dv(ship_state, epoch_s, dv_rtn) -> MoonPass`: module-level and picklable (`apply_maneuver`, then `first_moon_pass`).
  - `@dataclass(frozen=True) class FlybyPlan: node: ManeuverNode; moon_pass: MoonPass; converged: bool`.
  - `def moon_flyby_node(ship_state, seed: ManeuverNode, target_alt_m=FLYBY_TARGET_ALT_M, executor=None) -> FlybyPlan`. Damped Broyden on the B-plane:
    - Controls: x = (pro, nrm, rad). The node epoch stays fixed at `seed.epoch_s`.
    - Target each iteration: `B* = b*·B̂`, with `b* = r_p*·sqrt(1 + 2μ/(r_p*·v∞²))` and `r_p* = R_MOON + target_alt_m`. B̂ is the current B direction (smallest correction; T axis if `|B| == 0`).
    - Jacobian (2×3) by forward differences with `FD_STEP_MPS`, once at the seed. The 3 perturbed evaluations go through `_map_rows(partial(_pass_for_dv, …), …)` (or an equivalent ordered `executor.map`), so serial vs pooled is bit-identical.
    - Step: `dx = pinv(J) @ (B* − B)`. Halve it up to 4 times until `|B_new − B*| < |B − B*|`. Then apply the Broyden rank-1 update `J += ((ΔB − J·Δx) Δxᵀ) / (Δx·Δx)`.
    - Stop when `|r_p − r_p*| ≤ FLYBY_TOL_M`, or after `MAX_ITER`. Return the best iterate seen; `converged` says whether it met the tolerance.
  - Never calls `propagate_solar_system`, `propagate_earth_moon` or `World.step`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_flyby_targeting.py
SEED = ManeuverNode(3669.3721983198875, 3129.318767185485, -9.7657275283089, -38.144927341423305)

def _leo():
    from orbitsim.__main__ import _default_world      # pure sim/core imports only
    return _default_world().vessels[0].state

def _independent_pe_alt(post):
    """Reference: prediction propagator at a different cadence (600 s) + conic Pe."""
    # loop propagate_solar_prediction(cur, 600.0) inside stable_prediction_ephemeris,
    # stop after the first in-SOI distance minimum, return conic_periapsis(...).radius_m - R_MOON

def test_first_moon_pass_matches_plan2_converged_truth(de440):   # copy the plan-2 fixture
    p = first_moon_pass(PINNED)       # copy the PINNED literal from tests/render/test_preview_accuracy.py
    assert abs(p.periapsis_radius_m - 45_670e3) <= 0.01 * (45_670e3 - R_MOON)
    assert abs(p.periapsis_epoch_s - 296.259 * 3600) <= 600.0 and p.in_soi

def test_moon_flyby_node_reaches_200_km():
    nb.refresh_ephemeris_cache(0.0)          # DE440 if present, circular Moon otherwise
    st = _leo()
    seed = intercept_node(st, moon_state_at(0.0), MU_EARTH,      # online: == SEED (measured)
                          np.linspace(0, state_to_elements(st).period_s, 24),
                          np.linspace(3.0e3, 14 * 86400.0, 48))
    plan = moon_flyby_node(st, seed)
    assert plan.converged and plan.node.epoch_s == seed.epoch_s
    assert abs(plan.moon_pass.periapsis_radius_m - R_MOON - 200e3) <= 25e3
    assert 150e3 <= _independent_pe_alt(apply_maneuver(st, plan.node)) <= 250e3
    assert abs(plan.node.magnitude_mps - seed.magnitude_mps) < 150.0

def test_moon_flyby_node_executor_matches_serial():
    st = _leo()
    with ThreadPoolExecutor(3) as ex:
        assert moon_flyby_node(st, SEED, executor=ex) == moon_flyby_node(st, SEED)

def test_moon_flyby_node_raises_when_no_pass():
    with pytest.raises(ValueError):
        moon_flyby_node(_leo(), ManeuverNode(3669.37, 10.0, 0.0, 0.0))   # stays in LEO

def test_planner_never_uses_flight_propagators(monkeypatch):
    for name in ("propagate_solar_system", "propagate_earth_moon"):
        monkeypatch.setattr(nb, name, lambda *a, **k: pytest.fail("planner used flight physics"))
    moon_flyby_node(_leo(), SEED)
```

The executor test and the flight-propagator test must use the same computed seed as `test_moon_flyby_node_reaches_200_km` (a module-scoped fixture), not the literal `SEED`, so they also pass offline on the circular Moon. `SEED` documents the measured online value. `MoonPass` holds a numpy array, so dataclass `==` would compare arrays ambiguously. Compare `plan.node` (floats) and `np.array_equal(plan.moon_pass.b_plane_m, …)` in the executor test instead of the whole plan.

- [ ] **Step 2: Run `tests/core/test_flyby_targeting.py`.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it plus `tests/core/test_flyby.py tests/core/test_optimize.py tests/core/test_nbody.py`.** Expected: PASS.
- [ ] **Step 5: Timing check (verification, not a test).** Time `moon_flyby_node(_leo(), SEED)` warm, serial. The prototype took 8 evaluations / 1.8 s. Record the count and time in the commit body.
- [ ] **Step 6: Commit** `feat(core): flyby targeting — B-plane correction of the Lambert seed to a 200 km pass`.

### Task 5: "Plan Moon transfer" button and planner wiring

**Files:**
- Modify: `orbitsim/render/controllers/targeting.py` (`plan_intercept`, `_apply_planned_node`, `poll_planning`, target panel buttons)
- Modify: `orbitsim/render/keybind_overlay.py` (`("I", "Plan transfer to target")`)
- Modify: `orbitsim/render/controllers/maneuver_editor.py` (remove the `Intercept` and `Clear Tgt` buttons from the slider panel; they now live in the target panel)
- Modify: `tests/render/app_harness/golden/target_intercept.json`, `zoom_with_target_node.json` (re-record)
- Test: `tests/render/test_aim_app.py` (extend)

**Interfaces:**
- Consumes: Task 4 `moon_flyby_node`, `FlybyPlan`; Task 3 target panel.
- Produces:
  - Target panel buttons, shown while a target is selected:
    - `"Plan {name} transfer"` (e.g. "Plan Moon transfer") → `targeting.plan_intercept`. Hidden for the Sun.
    - `"Clear target"` → `targeting.clear_target`.
  - In `plan_intercept`'s non-planet branch, when the target is a `MoonTarget`, the job becomes `moon_flyby_node(ship_state, intercept_node(...), executor=grid_executor)` and returns the `FlybyPlan`. Lagrange targets keep `intercept_node` alone; planets keep `interplanetary_departure_node_by_name`. `poll_planning` accepts a `ManeuverNode` or a `FlybyPlan`.
  - Toasts:
    - converged: `f"Moon transfer planned · Pe {alt_km:,.0f} km · dV {dv:,.0f} m/s"`;
    - not converged: `f"Moon transfer planned · closest Pe {alt_km:,.0f} km (couldn't reach 200 km)"`;
    - `ValueError`: "No intercept found" (as today).
  - `_apply_planned_node` stops writing `_dv_value_text`; the slider readouts are refreshed by `refresh_readout` until Task 10 deletes them.

- [ ] **Step 1: Extend the probe.** After `h.send("i")`, step 2, assert:
  - `hooks.aim_status(app)[0] == "encounter"`;
  - the Pe in its text is within 150–250 km (parse the number);
  - the button exists: `h.press("Plan Moon transfer")` (it re-plans; `driver.press` raises `LookupError` if the label is missing). Step 2 more and assert the status is still `"encounter"` (re-planning the same state is idempotent).
- [ ] **Step 2: Run it; expect FAIL** (the status is still the 44,000 km pass).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/render/test_aim_app.py tests/render/test_controllers_pool_safety.py` (the pickled-job rule), then the golden suite: `.venv/bin/python -m pytest tests/render/app_harness -q -n auto`. Expected:
  - `target_intercept` and `zoom_with_target_node` FAIL on `vessel.nodes`, `texts`, `world_labels` and `events` (`-pg*` counts) only;
  - the other sandbox goldens fail only on `texts`/`world_labels`/`-pg*` events from Task 3's panel;
  - solar goldens pass unchanged.
- [ ] **Step 5: Re-record** the failing goldens (`python -m tests.render.app_harness.record <name>`). Run the protected-field diff from Global Constraints. In the commit body, give for each golden:
  - the old node `(3129.32, −9.78, −38.14) m/s @ 3669.37 s`;
  - the new node;
  - the preview's Moon Pe (expected ≈ 190 km);
  - the line "physics change: the planner now aims a 200 km pass, not the centre".

  `vessel.r`, `vessel.v`, `sim_time_s`, `warp` and `target` must be identical.
- [ ] **Step 6: New golden (spec: "record new scenarios … encounter status turning green").** Add `moon_transfer_status` to `scenarios.py`:
  1. launch;
  2. `hooks.select_target(h.app, "Moon")`; step 150 (the amber status appears);
  3. `h.press("Plan Moon transfer")`; step 10 (green status, ghost, encounter dot).

  Record it with `python -m tests.render.app_harness.record moon_transfer_status`. It runs twice and refuses non-determinism. Check that its `texts` contain `"Moon encounter · Pe …"` with a value between 150 and 250 km.
- [ ] **Step 7: Commit** `feat(render): Plan Moon transfer aims a 200 km pass; goldens re-recorded for the new aim`.

### Task 6: Part 4a gate — suite, flown check, profiler, docs, PR

- [ ] **Step 1:** Full suite `-n auto`. Expected green.
- [ ] **Step 2: Flown-vs-predicted report (verification, not a test).** In a scratch script, apply the Task 5 plan on the default world. Fly 20 days with `propagate_solar_system` in 600 s and 3600 s chunks, as `World.step` would at high warp, and compute the conic Moon Pe. Put the numbers in the PR description next to the predicted Pe. Planning measured an **impact** (≈ −1,100 km) against a predicted +190 km. If it still impacts, the PR says so plainly and links the open question; don't tune the aim to hide it.
- [ ] **Step 3: Profiler gate vs `main`** (export `main` with `git archive` into the scratchpad, as in plan 1): `idle`, `zoom`, `zoom_node_planned`, `high_warp`, `node_planned`, mean within 10%.
- [ ] **Step 4: CLAUDE.md.** Add to the gotchas: encounter periapsis comes from the body-relative conic at the nearest sample, because samples 2.8 h apart made a 190 km pass read 3,258 km. `core/flyby_targeting.py` corrects Lambert seeds on the prediction propagator (B-plane Broyden), and the flight integrator disagrees with it on departure legs (see the open question).
- [ ] **Step 5:** Final whole-branch review, fix pass, PR "Map readability 4a: aim helpers and auto-plan", merge.

---

# Part 4b — A clean screen (PR B, after 4a and Plan 3 are merged)

**Pre-flight (no commit):**
- Read Plan 3 as merged. Note:
  - the hook or controller call it uses to set a burn's Δv (Task 10 needs one);
  - what now owns the `mouse1-up` → `try_pick_target` path (today it lives in `ManeuverEditor._release_jogs`);
  - whether Plan 3 replaced the node marker sphere.
- Use Plan 3's names wherever this part says "if Plan 3 has none".
- Take before screenshots at 1600×900, 1280×720 and 960×600 for: boot, Moon targeted + planned burn, and long time at high warp (`shoot.py` pattern from planning).

### Task 7: One layout module that owns every panel rectangle

**Files:**
- Create: `orbitsim/render/ui/hud_layout.py` (pure)
- Test: `tests/render/test_hud_layout.py` (new)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PanelRect: name: str; anchor: str; x0: float; y0: float; x1: float; y1: float`. Coordinates are aspect2d units relative to the anchor. `anchor` is one of `"top_left"`, `"top_right"`, `"bottom_left"`, `"bottom_center"`, `"bottom_right"`.
  - `PANELS: dict[str, PanelRect]` with these values (decided):

    | name | anchor | x0 | y0 | x1 | y1 | holds |
    |---|---|---|---|---|---|---|
    | `time_warp` | top_left | 0.04 | −0.21 | 0.84 | −0.03 | time row; `<<` · warp cell · `>>` · `MAX` row |
    | `target` | top_right | −0.98 | −0.31 | −0.04 | −0.03 | target name, status (icon + text), Plan / Clear target buttons |
    | `orbit_fuel` | bottom_left | 0.04 | 0.04 | 0.84 | 0.66 | orbit rows + fuel / Δv left / throttle·TWR |
    | `navball` | bottom_center | −0.31 | 0.00 | 0.31 | 0.62 | the ball's display region (30% of window height) |
    | `speed_chip` | bottom_center | −0.30 | 0.63 | 0.30 | 0.71 | "Orbital 8.053 km/s" |
    | `sas` | bottom_center | 0.36 | 0.04 | 0.92 | 0.52 | SAS readout + 3×3 buttons |
    | `burn` | bottom_right | −1.10 | 0.58 | −0.04 | 0.92 | burn readout, Next Pe / Next Ap / Delete burn, Execute |

  - `def to_absolute(rect: PanelRect, aspect: float, ui_scale: float = 1.0) -> tuple[float, float, float, float]`: absolute aspect2d `(x0, y0, x1, y1)`. Each rect scales by `ui_scale` about its anchor.
  - `def to_pixels(rect, width_px, height_px, ui_scale=1.0) -> tuple[float, float, float, float]`: pixel rect, origin bottom-left (same convention as `Targeting.marker_px`).
  - `FOCUS_SCREEN_Y = 0.20`: where the camera focus sits in normalised screen y. It is the middle of the free band between the bottom cluster and the top panels. Task 11 uses it.
  - `MIN_ASPECT = 1.5`, `MAX_ASPECT = 21 / 9`.

- [ ] **Step 1: Write the failing tests**

```python
ASPECTS = (1.5, 16 / 10, 16 / 9, 21 / 9)
SCALES = (0.84, 1.0, 1.08)        # ResponsiveLayout range, capped at 1.08 as app.py does

def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

@pytest.mark.parametrize("aspect", ASPECTS)
@pytest.mark.parametrize("scale", SCALES)
def test_panels_never_overlap(aspect, scale):
    rects = {n: to_absolute(r, aspect, scale) for n, r in PANELS.items()}
    names = sorted(rects)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not _overlap(rects[a], rects[b]), (a, b, aspect, scale)

@pytest.mark.parametrize("aspect", ASPECTS)
def test_panels_stay_on_screen(aspect):
    for n, r in PANELS.items():
        x0, y0, x1, y1 = to_absolute(r, aspect, 1.08)
        assert -aspect <= x0 < x1 <= aspect and -1.0 <= y0 < y1 <= 1.0, n

def test_navball_rect_matches_display_region():
    from orbitsim.render.navball import _BALL_FRAC_H
    r = PANELS["navball"]; assert abs((r.y1 - r.y0) - 2.0 * _BALL_FRAC_H) < 0.03

def test_to_pixels_bottom_left_origin():
    x0, y0, x1, y1 = to_pixels(PANELS["orbit_fuel"], 1600, 900)
    assert x0 < 100 and y0 < 50 and x1 > x0 and y1 > y0
```

- [ ] **Step 2: Run; expect FAIL** (module missing).
- [ ] **Step 3: Implement.** If a value in the table makes a test fail, adjust the value. The table is a starting point, not a contract; the tests are the contract.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `feat(render): one screen-layout table with no-overlap tests`.

### Task 8: Time and warp panel, top-left

**Files:**
- Modify: `orbitsim/render/hud/format.py` (`fmt_mission_time`, `fmt_warp`)
- Modify: `orbitsim/render/controllers/warp_controls.py` (`build_warp_controls` on `PANELS["time_warp"]`; `update_warp_readout` writes two OnscreenTexts)
- Modify: `orbitsim/render/hud/__init__.py` (`orbit_panel_lines` drops the "Sim time: … past J2000" row; the solar viewer's free text `Hud.text` moves below the time panel, to `(-aspect + 0.08, 0.72)`)
- Test: `tests/render/test_hud_format.py`, `tests/render/test_warp_layout.py` (new; builds `TextNode`s offscreen, no ShowBase)

**Interfaces:**
- Produces:
  - `def fmt_mission_time(sim_time_s: float) -> str`:
    - under 1 day: `"T+ 03:20:15"`;
    - under 1,000 days: `"T+ 12d 03h 20m"`;
    - otherwise: `"T+ 969,100d 03h"`.

    Sim time is seconds past J2000, as today's "T+ … DAYS". Negative times get a `"T− "` prefix with the same fields.
  - `def fmt_warp(warp: float, locked: bool) -> str`: `"×100"`, `"×1,000,000"`, and `"×1 LOCKED"` while thrusting.
  - Layout inside `time_warp`:
    - row 1: the time text, left-aligned at x 0.07;
    - row 2: `<<` at x 0.12, the warp cell centred at x 0.36 (cell width `WARP_CELL_W = 0.30`), `>>` at x 0.60, `MAX` at x 0.74.

    The text never shares space with a button.
  - `MIN_WINDOW_PX = (1280, 720)`: the smallest supported window.

- [ ] **Step 1: Write the failing tests**

```python
def test_mission_time_formats():
    assert fmt_mission_time(12015.0) == "T+ 03:20:15"
    assert fmt_mission_time(12 * 86400 + 3 * 3600 + 20 * 60) == "T+ 12d 03h 20m"
    assert fmt_mission_time(969_100 * 86400 + 3 * 3600) == "T+ 969,100d 03h"

def test_warp_formats():
    assert fmt_warp(100.0, False) == "×100" and fmt_warp(1e6, False) == "×1,000,000"
    assert fmt_warp(1.0, True) == "×1 LOCKED"

def test_warp_cell_fits_worst_case():
    # Text width in aspect2d units = TextNode.calc_width(text) * scale (default font = Rajdhani)
    from orbitsim.render.controllers.warp_controls import WARP_CELL_W, WARP_TEXT_SCALE, TIME_ROW_W
    assert _width(fmt_warp(1e8, True), WARP_TEXT_SCALE) <= WARP_CELL_W
    assert _width(fmt_mission_time(9_999_999 * 86400.0), WARP_TEXT_SCALE) <= TIME_ROW_W
```

`_width` loads `font_path()` into a `TextNode`, sets the text, and returns `calc_width() * scale`. `TIME_ROW_W = 0.74` (the panel's inner width).

- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.** Delete the top-centre `_warp_frame`. The solar viewer uses the same panel: `build_warp_controls` runs in both modes.
- [ ] **Step 4: Run** `tests/render/test_hud_format.py tests/render/test_warp_layout.py tests/render/test_panel.py`; PASS.
- [ ] **Step 5: Eyeball** the 1280×720 long-time shot (`longtime` in `shoot.py`): `<<` and `>>` are clear of the text.
- [ ] **Step 6: Commit** `feat(render): time and warp panel top-left; warp text can no longer run into its buttons`.

### Task 9: Orbit and fuel panel bottom-left; target panel to the top; the dock goes

**Files:**
- Modify: `orbitsim/render/hud/__init__.py` (`Hud`: one panel at `PANELS["orbit_fuel"]`, laid out bottom-up; `update_flight` feeds its fuel rows; remove `_right`, `toggle_orbit`, `toggle_vessel` and the MANEUVER section)
- Modify: `orbitsim/render/hud/panel.py` (`HudPanel` gains `grow="down" | "up"`; "up" anchors the last row at `bottom`)
- Modify: `orbitsim/render/controllers/readouts.py` (no `PanelDock`; no `_maneuver_frame.hide()` dock default)
- Modify: `orbitsim/render/ui/widgets.py` (delete `PanelDock`)
- Modify: `orbitsim/render/controllers/targeting.py` (the target panel moves from Task 3's interim spot to `PANELS["target"]`)
- Test: `tests/render/test_panel.py`, `tests/render/test_hud_format.py`

**Interfaces:**
- Produces:
  - `def orbit_fuel_lines(*, altitude_m, speed_mps, periapsis_m, apoapsis_m, period_s, inclination_rad, units, dominant_body, fuel_frac, dv_remaining, throttle, twr) -> list[str]`, in this order:
    1. `"Altitude 520.9 km"`
    2. `"Speed 8.053 km/s"`
    3. `"Pe 500.0 km · Ap 2,465.2 km"`
    4. `"Inc 8.1° · Period 115.6 min"`
    5. `"Orbiting Moon"` (only when not Earth)
    6. `"Fuel 100% · dV left 5,633 m/s"` (`INF` when unlimited)
    7. `"Throttle 0% · TWR 0.60"`
    8. `"WARP LOCKED — thrusting"` (only when locked)

    Mass and thrust leave the HUD: they are constants of the one ship.
  - The maneuver lines (`set_maneuver`) leave the left panel. Task 10's burn panel shows the burn; Task 3's box shows the target; the "Approach …" line is replaced by the status box and the encounter label. `Hud.set_maneuver` is deleted, and `ManeuverEditor.sync_maneuver_hud` calls the burn panel instead.
  - The in-SOI flyby line (`_encounter_line`, "Flyby Mars: v-inf …") keeps a home: it becomes an extra row in the target panel while inside a planet's SOI.

- [ ] **Step 1: Write the failing tests**

```python
def test_orbit_fuel_lines_compact():
    lines = orbit_fuel_lines(altitude_m=520.9e3, speed_mps=8053.0, periapsis_m=500e3,
                             apoapsis_m=2465.2e3, period_s=115.6 * 60, inclination_rad=np.radians(8.1),
                             units="km", dominant_body="Earth", fuel_frac=1.0,
                             dv_remaining=5633.0, throttle=0.0, twr=0.6)
    assert lines == ["Altitude 520.9 km", "Speed 8.053 km/s", "Pe 500.0 km · Ap 2,465.2 km",
                     "Inc 8.1° · Period 115.6 min", "Fuel 100% · dV left 5,633 m/s",
                     "Throttle 0% · TWR 0.60"]

def test_panel_grows_up_from_bottom():
    lay = layout_panel([3], top=None, bottom=0.08, line_height=0.06, padding=0.02,
                       section_gap=0.03, grow="up")
    assert lay.line_ys[0][-1] == pytest.approx(0.08 + 0.06) and lay.frame_bottom == pytest.approx(0.06)
```

(`layout_panel` gains `bottom=` and `grow=`; `top=` stays for the default "down" behaviour, so existing panel tests pass unchanged.)

- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/render/test_panel.py tests/render/test_hud_format.py tests/render/test_controller_wiring.py`; PASS.
- [ ] **Step 5: Commit** `feat(render): orbit and fuel in one bottom-left panel; dock and vessel panel removed`.

### Task 10: The burn panel replaces the slider panel

**Files:**
- Modify: `orbitsim/core/flight.py` (`burn_duration_s`)
- Modify: `orbitsim/render/hud/format.py` (`fmt_until`, `fmt_burn_duration`, `burn_summary`)
- Modify: `orbitsim/render/controllers/maneuver_editor.py`:
  - `build_maneuver_ui` builds only the burn panel at `PANELS["burn"]`;
  - delete `_jog`, `_node_time_jog`, `_dv_value_text`, `_dv_total_text`, `_node_time_text`, `_action_frame`, `apply_jogs`, `_jog_rate_mps_per_s`, `_release_jogs`, `toggle_maneuver_drawer`, the `JOG_*`/`NODE_JOG_*` constants;
  - add `delete_burn`.
- Modify: `orbitsim/render/app.py` (drop the `apply_jogs` call from `_update`; `_update_responsive_ui` scales the new panels)
- Modify: `orbitsim/render/controllers/flight_input.py` (`mouse1-up` → whatever Plan 3 made the click handler, else `targeting.try_pick_target`)
- Modify: `tests/render/app_harness/hooks.py` (`set_burn`; delete `set_jog`)
- Modify: `tests/render/app_harness/scenarios.py` (`node_edit_execute`), `tests/render/test_live_preview_app.py` (four `set_jog` uses)
- Test: `tests/core/test_flight.py`, `tests/render/test_hud_format.py`

**Interfaces:**
- Produces:
  - `def burn_duration_s(mass_kg: float, exhaust_velocity_mps: float, thrust_n: float, dv_mps: float) -> float = m·vₑ/F·(1 − exp(−Δv/vₑ))`. Raises `ValueError` for thrust ≤ 0.
  - `def fmt_until(seconds) -> str`: `"now"` (≤ 2 s), `"in 45 s"`, `"in 23 min"`, `"in 2h 05m"`, `"in 3d 04h"`.
  - `def fmt_burn_duration(seconds) -> str`: `"35s burn"`, `"6m 54s burn"`, `"1h 02m burn"`.
  - `def burn_summary(dv_mps, ttn_s, duration_s) -> str`, e.g. `"Burn 3,130 m/s · in 23 min · 6m 54s burn"`. The `in …` part is dropped for a burn-now node (`ttn_s is None`). Δv is rounded to whole m/s and durations to the nearest second, so 413.8 s becomes "6m 54s".
  - Burn panel contents:
    - **Readout row:** `burn_summary` (or `"Cancel Auto Burn"` status when armed), shown only while a burn exists: `_node_epoch_s is not None or node.magnitude_mps > 0`.
    - **Row 2**, always shown: `"Next Pe"`, `"Next Ap"` (labels kept, because scenarios press them), plus `"Delete burn"` when a burn exists.
    - **Row 3**, only when a burn with Δv exists: the Execute button with today's labels and states (`"Execute Burn"` / `"Jump & Execute"` / `"Cancel Auto Burn"`).
  - `ManeuverEditor.delete_burn()` clears the node time **and** zeroes all three Δv components (today's `Clear` kept the Δv as a burn-now node). `_clear_node` stays internal for the expired-node path.
  - Unlimited Δv stays where it already is: the `U` key and the Esc settings panel ("Unlimited dV: on/off"). No button is added.
  - `hooks.set_burn(app, pro=0.0, nrm=0.0, rad=0.0)` writes `app._dv` and calls `maneuver_editor.refresh_readout()`. Use Plan 3's equivalent if it exists and delete this one.
  - `node_edit_execute` becomes:

    ```python
    def node_edit_execute(h):
        _launch_sandbox(h)
        # Was: jog "pro" at 0.6 for 30 frames. That accumulated exactly this dV (captured
        # from the slider code on 2026-11-05); the burn doesn't fire until Execute, so
        # setting it up front flies identically.
        hooks.set_burn(h.app, pro=37.401206667377515)
        h.step(30)
        h.send("mouse1-up")
        h.press("Next Ap")
        h.step(10)
        h.press("Execute Burn")
        h.step(900)
    ```

  - `test_live_preview_app.py`: each `set_jog(app, "pro", 0.6); h.step(k); set_jog(app, "pro", 0.0)` becomes a `set_burn(app, pro=<value>)` followed by the same steps. The values don't need to match, since these probes assert line visibility, not physics. **One exception:** the first probe asserts "estimate only" on two consecutive edited frames. Keep that by calling `set_burn` with a different value before each of those two checks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_flight.py
def test_burn_duration_known_answer():
    # 10 t ship, ve 3.5 km/s, 50 kN, 3,130 m/s: 700 s * (1 - e^-0.894286) = 413.8 s
    assert burn_duration_s(10_000.0, 3500.0, 50_000.0, 3130.0) == pytest.approx(413.78, abs=0.05)

def test_burn_duration_rejects_zero_thrust():
    with pytest.raises(ValueError):
        burn_duration_s(10_000.0, 3500.0, 0.0, 10.0)

# tests/render/test_hud_format.py
def test_burn_summary():
    assert burn_summary(3130.0, 23 * 60, 413.8) == "Burn 3,130 m/s · in 23 min · 6m 54s burn"
    assert burn_summary(37.4, None, 7.5) == "Burn 37 m/s · 7s burn"
    assert fmt_until(1.0) == "now" and fmt_until(2 * 3600 + 5 * 60) == "in 2h 05m"
```

- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement**, then run the golden suite. `node_edit_execute` must pass on `vessel`, `sim_time_s`, `warp` and `target` unchanged. Its `texts`, `world_labels` and `-pg*` events will differ; re-record and run the protected-field diff. If `vessel` differs at all, the captured Δv or the step order is wrong. Fix it; never re-record over a physics diff.
- [ ] **Step 4: Run** `tests/core/test_flight.py tests/render/test_hud_format.py tests/render/test_live_preview_app.py tests/render/test_controller_wiring.py tests/render/app_harness -n auto`; PASS after re-recording presentation-only diffs.
- [ ] **Step 5: Commit** `feat(render): burn panel replaces the jog sliders; node_edit_execute drives the same dV via a hook`. The body explains the scenario change and gives the protected-field diff.

### Task 11: Navball cluster, the view centred above it, and labels that never sit on panels

**Files:**
- Modify: `orbitsim/render/sas_panel.py` (`SasPanel`, `VelocityReadout` positions from `PANELS["sas"]` / `PANELS["speed_chip"]`)
- Modify: `orbitsim/render/camera_rig.py` (`apply`: lens film offset so the focus projects at `FOCUS_SCREEN_Y`)
- Modify: `orbitsim/render/world_markers.py` (`blocked_by_panels`)
- Modify: `orbitsim/render/controllers/readouts.py` (`update_marker_readability` hides labels inside visible panel rects)
- Modify: `tests/render/app_harness/hooks.py`. Add `labels_over_panels(app) -> list[str]`: the texts of shown `render` TextNode labels whose `marker_px` anchor is inside a shown panel's pixel rect. Add `orbit_points_under_navball(app) -> int`: the count of `_traj_world_cache[0]` points whose `marker_px` falls inside the navball rect.
- Test: `tests/render/test_world_markers.py`, `tests/render/test_camera_rig.py`, `tests/render/test_layout_app.py` (new subprocess probe)

**Interfaces:**
- Consumes: Task 7 `PANELS`, `to_pixels`, `FOCUS_SCREEN_Y`.
- Produces:
  - `def blocked_by_panels(point_px, rects_px, margin_px: float = 10.0) -> bool`: True when a label anchor lies inside any rect grown by the margin.
  - `update_marker_readability` takes the pixel rects of the panels currently shown, read from `app.win` size and the layout's `ui_scale`. Any label for which `blocked_by_panels` is true is hidden, at every priority, the target label included.
  - `CameraRig.apply` sets `camLens.set_film_offset(0, y)` so that `camLens.project(focus)` gives normalised y = `FOCUS_SCREEN_Y`. Picking (`marker_px`) already uses `camLens.project`, so it stays consistent. The navball has its own camera and region, so it is unaffected.

- [ ] **Step 1: Write the failing tests**

```python
# tests/render/test_world_markers.py
def test_blocked_by_panels():
    rects = [(100.0, 0.0, 400.0, 300.0)]
    assert blocked_by_panels((250.0, 150.0), rects)
    assert blocked_by_panels((405.0, 150.0), rects)           # inside the 10 px margin
    assert not blocked_by_panels((450.0, 150.0), rects)
```

```python
# tests/render/test_camera_rig.py (offscreen ShowBase like the existing rig tests)
def test_focus_projects_above_the_navball():
    rig.apply()
    p = Point2(); base.camLens.project(base.cam.get_relative_point(base.render, (0, 0, 0)), p)
    assert abs(p.y - FOCUS_SCREEN_Y) < 0.01
```

`tests/render/test_layout_app.py` probe, at both 1600×900 and 1280×720 (`install_determinism(win_size=...)`):
1. Boot, then step 120.
2. Assert `hooks.labels_over_panels(app) == []`. This means every shown `render` TextNode label's projected anchor is outside every shown panel rect.
3. In the opening view, assert every sampled point of the live orbit line projects above the top of the `navball` rect or outside its x-range: `hooks.orbit_points_under_navball(app) == 0`.
4. Select the Moon, press `I`, set the rig to `1e9` m, step 60, and assert `labels_over_panels` is still `[]`. This is the planning screenshot's case, where the "TARGET: MOON" label sat under the orbit panel.

- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.** If the opening view still puts orbit points under the navball, adjust `vessel_view.apply_opening_camera`'s distance (builder's call within spec: "Earth lit and your orbit clearly"). Don't move the navball.
- [ ] **Step 4: Run** the three tests plus `tests/render/test_opening_camera_app.py tests/render/test_picking.py`; PASS.
- [ ] **Step 5: Eyeball** 1600×900 and 1280×720 boot and planned-burn shots against the pre-flight "before" set.
- [ ] **Step 6: Commit** `feat(render): navball cluster, view centred above it, labels never on panels`.

### Task 12: Map markers say what they mean

**Files:**
- Modify: `orbitsim/render/hud/format.py` (`apsis_label`, `encounter_label`)
- Modify: `orbitsim/render/controllers/readouts.py` (`place_apsis_markers`: tick geometry + `apsis_label` text)
- Modify: `orbitsim/render/world_labels.py` (`build_tick_marker`: a short constant-screen-size bar, billboarded)
- Modify: `orbitsim/render/controllers/maneuver_editor.py` (node marker: a ring + one `burn_summary` label, unless Plan 3 already replaced the sphere, in which case only the label text changes)
- Modify: `orbitsim/render/controllers/trajectory_view.py` (`update_encounter_cues` label text = `encounter_label`)
- Modify: `orbitsim/render/controllers/bodies_view.py` (`frame_lagrange`: fade unless targeted)
- Test: `tests/render/test_hud_format.py`, `tests/render/test_world_labels.py`

**Interfaces:**
- Produces:
  - `def apsis_label(kind: str, altitude_m: float) -> str`: `"Pe 500 km"` / `"Ap 2,465 km"` (whole km).
  - `def encounter_label(body_name: str, periapsis_alt_m: float, impact: bool) -> str`: `"Moon encounter · Pe 4,317 km"`, or `"Moon impact"`.
  - Apsis altitudes are measured above the dominant body's surface. `_install_trajectory` already caches the Pe/Ap points; it also caches `_apsis_altitudes = {"PE": m, "AP": m}` from the same points.
  - The burn label is `burn_summary(...)` from Task 10, the same string as the burn panel's readout.
  - Lagrange points: marker alpha 0.25 and label hidden unless that point is the target (alpha 1, label shown). Applied after `update_marker_readability`'s fade, as a multiplier. `LAGRANGE_IDLE_ALPHA = 0.25`.

- [ ] **Step 1: Write the failing tests**

```python
def test_apsis_label():
    assert apsis_label("PE", 500.0e3) == "Pe 500 km" and apsis_label("AP", 2_465.2e3) == "Ap 2,465 km"

def test_encounter_label():
    assert encounter_label("Moon", 4_317.4e3, False) == "Moon encounter · Pe 4,317 km"
    assert encounter_label("Moon", -10.0, True) == "Moon impact"

def test_tick_marker_is_constant_size_unlit():   # offscreen NodePath, like test_world_labels
    tick = build_tick_marker(render, color=(1, 1, 1, 1), length=6.0)
    assert tick.get_light_off() is not None and tick.get_billboard() is not None
```

(Adapt the last test's attribute checks to whatever `build_labeled_marker`'s existing test asserts; mirror it.)

- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the two files plus `tests/render/test_aim_app.py tests/render/test_layout_app.py`; PASS.
- [ ] **Step 5: Eyeball** a planned Moon transfer at 1e9 m. You should see: the Pe/Ap ticks with short labels, the burn ring with its single line, the yellow encounter dot, faded L-points, and the ghost.
- [ ] **Step 6: Commit** `feat(render): short Pe/Ap/burn/encounter labels; L-points fade unless targeted`.

### Task 13: Part 4b gate — goldens, profiler, docs, PR

- [ ] **Step 1:** Full suite `-n auto`.
- [ ] **Step 2: Re-record** the sandbox and solar goldens whose `texts`, `world_labels` or `-pg*` events changed. Run the protected-field diff: `vessel`, `sim_time_s`, `warp`, `target` and the non-`-pg*` events must be identical for **every** scenario. Paste it in the commit body with the reason ("HUD rebuilt into corner panels; slider panel and dock removed; marker labels shortened").
- [ ] **Step 3: Profiler gate vs `main`** (same method as Task 6), all five workloads within 10%. The per-frame label/panel hit-test adds O(labels × panels) ≈ 20 × 7 work. If `idle` regresses, compute the panel pixel rects once per window-size change (cache keyed on `(w, h, ui_scale)` on `OrbitApp`).
- [ ] **Step 4: Screens.** Record `python -m tests.render.app_harness.screens --record` on `main`, then `--compare` here. Expected: both modes differ (intended); eyeball both PNGs.
- [ ] **Step 5: CLAUDE.md.**
  - Add `hud_layout.PANELS` as the one place panel positions live.
  - Note that world labels hide when they would land on a panel.
  - Note the camera's film offset (the focus sits at `FOCUS_SCREEN_Y`, not screen centre).
  - Update the controller list (the slider panel is gone).
  - Update the "Verifying graphics headlessly" note if helpful.
- [ ] **Step 6:** Final whole-branch review, fix pass, PR "Map readability 4b: a clean screen", merge.
