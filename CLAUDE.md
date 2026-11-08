# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A desktop 3D orbital mechanics simulator/game in Python — "KSP but the physics are real." Three
pillars: transfer & ΔV planning (Hohmann/bi-elliptic/Lambert + porkchop optimizer), the real solar
system (JPL/Skyfield ephemerides + full N-body gravity), and a sandbox (maneuver nodes with live
orbit prediction, gravity assists). Vessels are point masses with a ΔV budget; **no**
aerodynamics/atmosphere (out of scope). The sandbox runs a **full solar-system N-body** model
(Earth + Moon + Sun + all 7 planets as perturbers, with optional real JPL ephemeris positions via
a per-frame cache; falls back to circular approximations when offline). Buildable/launchable ships
(mass + 3D models, stations, landers) are a longer-term goal.

## Commands

Use the venv interpreter. Bare `python` resolves to the global install, which lacks dependencies.
On Linux the venv lives at `.venv/bin/`, on Windows at `.venv/Scripts/`.

```bash
# Linux
.venv/bin/python -m pytest tests/ -q -n auto             # full suite in parallel (~460 tests, ~2.5 min)
.venv/bin/python -m pytest tests/ -q                     # same, serial (~4 min: the app goldens dominate)
.venv/bin/python -m pytest tests/core -q                 # physics core only (no graphics needed)
.venv/bin/python -m tests.render.app_harness.profile zoom --project   # frame-time profile of a workload
.venv/bin/python -m pytest tests/core/test_flight.py -q                          # one file
.venv/bin/python -m pytest "tests/core/test_kepler.py::TestEllipticAnomalies"    # one class/test
.venv/bin/python -m orbitsim                             # launch the sandbox (LEO, flyable)
.venv/bin/python -m orbitsim --solar                     # launch the solar-system viewer

# Windows (same commands, just Scripts instead of bin)
.venv/Scripts/python -m pytest tests/ -q
```

Dependencies are in `pyproject.toml` extras: `dev` (pytest, black) and `render` (panda3d,
skyfield, lamberthub, matplotlib). `pytest` is preconfigured (`testpaths=tests`, `-v`).

First launch downloads + caches large assets into `data/` (gitignored): the DE440 ephemeris
kernel (`de440s.bsp`, ~32 MB) and texture maps (`data/textures/`). All download code MUST degrade
gracefully offline — see the gotchas.

Verifying graphics headlessly: prepend `loadPrcFileData("", "window-type offscreen")` before
constructing `OrbitApp`, drive it with `app.taskMgr.step()`, and capture
`app.win.save_screenshot(Filename.from_os_specific(path))`. The easy way: the golden harness's
`install_determinism(win_size="1600 900")` + `build_app("sandbox")` + `Harness(app).step(n)`;
check layouts at 1280×720 (the smallest supported window) and a 21:9 size too.

## Architecture — the one rule that governs everything

Strict one-directional layering. Violating it breaks the project's core guarantee (that the
physics is unit-testable to textbook accuracy with zero graphics installed):

```
orbitsim/core/   pure physics. float64, SI units. NEVER imports panda3d, sim, or render.
orbitsim/sim/    stateful world: World, Vessel, SimClock. imports core only.
orbitsim/render/ Panda3D. imports sim + core. ALL graphics live here.
```

If you are tempted to `import panda3d` inside `core/`, stop — the design is wrong at that point.

Data flow each frame (`render/app.py::OrbitApp._update`, an ordered list of steps):
`_frame_flight` (input, warp caps, `clock.advance(dt)`, `refresh_ephemeris_cache(t)`,
`world.step(sim_dt)`) → `_frame_origin` (recentre the floating origin on the focused vessel) →
controller `frame_*` steps remap everything to render space (bodies, vessels, maneuver node,
targeting) → `readouts.frame_readouts` (HUD/navball). **Order matters** — keep it.

`OrbitApp` (`render/app.py`, ~360 lines) only wires things up: `_start_sim` is an ordered list of
controller `build_*` calls and `_update` an ordered list of `frame_*` calls. Controller methods
other objects call are public; a leading underscore means "internal to this controller". The behavior lives in
**stateless controllers** in `render/controllers/` (`title_screen`, `solar_view`, `warp_controls`,
`flight_input`, `quicksave`, `targeting`, `porkchop_modal`, `trajectory_view`, `maneuver_editor`,
`bodies_view`, `vessel_view`, `readouts`, `node_placement`, `node_handles`). Each holds only `self.app`; **all state stays on
`OrbitApp`** (`self.app._node_epoch_s`, `self.app._target`, …). Cross-feature calls go through the
owning controller (`self.app.maneuver_editor.current_node()`), never `self.app._current_node()`.

`OrbitApp` has **two modes**: the **sandbox** (default — Earth-centered, one flyable vessel,
maneuver editor) and the **solar viewer** (`--solar` — Sun-centered, planets from the ephemeris,
no vessel).

**Solar viewer** (`render/controllers/solar_view.py`): planets come from
`core.planets.heliocentric_state` (DE440, circular fallback offline), one call per planet per
frame. Orbit lines are baked in `SOLAR_ORBIT_UNIT_M` (1e9 m) under `solar_orbit_frame`, which is
rescaled each frame, so zoom never rebuilds them; they refresh every 30 sim-days from that frame's
states. Dots and names are sized in pixels each frame via `label_layout.units_per_pixel` (×
`line_px_scale()`). Names are placed by the pure `label_layout.layout_labels`: priority
`SOLAR_LABEL_PRIORITY`, the hovered body pinned, other dots and the HUD (panels + HUD text) as
obstacles, last frame's slot kept, hidden when nothing fits. The solar names are *owned* world
labels (solar_view shows/hides them; the panel pass only hides more).

**App golden tests** (`tests/render/app_harness/`): each scenario drives a real `OrbitApp`
headlessly in its own process (inline executors, fake wall clock, fixed 60 fps) through key
events and button clicks, and must reproduce its recorded JSON (physics state, on-screen text,
bound events, camera). A failure means behavior changed — fix the code, never the golden; record
a *new* scenario with `python -m tests.render.app_harness.record <name>` (it runs twice and
refuses non-deterministic output). `hooks.py` is the only test code allowed to touch OrbitApp
private attributes. `python -m tests.render.app_harness.screens --record/--compare` gives
pixel-diffed screenshots against a local baseline.

### N-body gravity model

**Full solar-system N-body** (`core/nbody.py`): the ship is a massless test particle in a
geocentric non-inertial frame. Gravity comes from:
- Earth (central, at origin)
- Moon (real DE440 position via `moon_state_at`, evaluated per substep; circular fallback offline)
- Sun + 7 planets (third-body perturbations + indirect terms for the non-inertial frame)

**Ephemeris cache** (`refresh_ephemeris_cache`): snapshots all planet geocentric positions from
JPL/DE440 via Skyfield once per frame; substeps use these constant cached positions (planet
motion within one frame is negligible). When the DE440 kernel is unavailable, cached wrapper
functions (`_csun`, `_cmercury`, etc.) transparently fall back to circular approximations from
`core/planets.py`. The render layer also uses these cached functions for planet placement,
targeting, and encounter calculations — ensuring consistent positions everywhere.

**Hybrid coast/burn flight** (`sim/world.py::World.step`): coasting vessels propagate under
`propagate_solar_system` (or `propagate_earth_moon` without the solar system flag); thrusting
vessels use `integrate_powered_solar` (same gravity + real rocket equation with operator
splitting). The integrator uses exact rocket-equation velocity impulses per substep so ΔV
telescopes to `vₑ·ln(m₀/m_f)` and fuel hits exactly 0. The render layer forces warp to 1×
while thrusting and caps warp via `max_safe_warp_solar` otherwise.

The two-body `core/` (`propagate_kepler`, `integrate_powered`) is **not** deleted — it stays
the analysis layer (transfers, intercept seeds) and on-rails body motion.

### Flyby / gravity assist physics

`core/flyby.py`: pure hyperbolic encounter math — `v_infinity`, `flyby_deflection`,
`flyby_periapsis`, `rotate_v_infinity`, `flyby_exit_velocity`, `encounter_parameters`. The
render layer shows encounter info (periapsis, deflection, ΔV equivalent) in the HUD when
inside a planet's SOI.

### The scale/precision problem (why `render/floating_origin.py` exists)

The solar system spans ~4.5e12 m but a docking maneuver needs ~1e-3 m precision, and GPUs render
in float32 (~7 digits). The fix: physics stays float64 SI; `RenderTransform.to_render` subtracts a
float64 origin (set to the focused body each frame) **before** casting to float32.

## Conventions that pervade `core/` and `sim/`

- **SI everywhere** (meters, seconds, radians, kg). Convert to km/degrees/UTC only at the
  render/HUD boundary. Suffix ambiguous vars `_m`, `_s`, `_rad`, `_mps`.
- **float64 numpy arrays, shape `(3,)`.** Never float32 in core.
- **Frame:** J2000 / ICRF, origin at the central body. Sim time is **seconds past J2000 TDB**.
- **Frozen dataclasses:** `StateVector`, `KeplerianElements`, `CelestialBody`, `ManeuverNode`.
  Operations return new instances — never mutate. (`Vessel` in the sim layer is the mutable
  exception: its `.state` is reassigned each tick.)
- **Constants come from `core/constants.py`** (sourced from `astropy.constants`). Never hard-type
  `398600` or `6.674e-11`.
- **Angles** normalized `[0, 2π)` for anomalies, `[0, π]` for inclination. Clamp `arccos`
  arguments to `[-1, 1]` (float error produces 1.0000000002).
- Raise `ValueError` on invalid physics input; do not silently clamp or return `None`.

## How work is structured

**TDD is mandatory** because this is math a model can get wrong silently. For every `core`
function: write the known-answer test first, implement until green. Never loosen a tolerance to
force a pass — if an invariant drifts, the most recently added function is wrong.

Universal invariants (encode as property tests): elements↔state round-trip (1e-7 rel), energy
`ε = v²/2 − μ/r = −μ/(2a)` conserved, angular momentum `h = r × v` conserved, period closure
(< 1 mm analytic), vis-viva `v² = μ(2/r − 1/a)` at every point.

### Commit & push after every change

Remote is `origin` → https://github.com/JoeyPhatsJr/orbitsim (default branch `main`). After each
change is made and verified (tests green / screenshot checked), **commit and push to GitHub** —
don't batch many changes into one late commit. Stage specific files by **explicit path** (never
`git add -A`/`.`), keep `data/`, scratch files, generated assets (`porkchop.png`, `saves/`), and
`.claude/settings.local.json` out of commits.

## Domain gotchas (learned the hard way)

- **Reference textbook:** Curtis, *Orbital Mechanics for Engineering Students*.
- **Perifocal→inertial rotation** uses the **active** convention `Q = R3(Ω) R1(i) R3(ω)` (positive
  angles), not `R3(−Ω) R1(−i) R3(−ω)`. Getting this backwards passes some tests and fails the
  hyperbolic anchor (Curtis Ex 4.7).
- **Eccentricity vector:** `e = (1/μ)[(v² − μ/r)·r − r·v_r·v]` where `v_r = (r·v)/r` — note the
  `r` (magnitude) factor on the second term.
- **Degenerate orbits:** circular (`e≈0`) and equatorial (`i≈0`) make argp/raan/ν individually
  undefined; round-trip element tests must exclude or special-case them (only the angle *sums*
  are recoverable).
- **lamberthub** signature is `izzo2015(mu, r1, r2, tof, M=0, prograde=True)`, all SI. Lambert is
  singular at exactly 180° transfer angle — perturb test geometry slightly off π.
- **Skyfield loader** is `from skyfield.api import Loader; Loader(dir)` — NOT `load.Loader(dir)`.
- **Render markers** must use constant on-screen size, not tiny world-space radii. The
  **`CameraRig` owns scale** (`scale_m_per_unit = distance/1000`); set the rig's `distance_m`,
  not the transform scale, to change zoom. The ship's map icon is the real model
  (`ship_model.build_ship_icon`, `SHIP_ICON_RENDER_UNITS` long), oriented by the vessel's
  attitude every frame.
- **Lines and markers must be unlit (`set_light_off`).** `LineSegs` have no normals, so the
  scene's ambient + sun lights rendered every orbit line at ~10% of its colour. That was the
  "can't see the lines" bug. Line looks come from `render/line_style.py` named styles
  (`ORBIT_STYLE`, `PLAN_STYLE`, `reference_style(targeted)`, …). Widths are **framebuffer
  pixels** × `OrbitApp.line_px_scale()` (Retina-aware), which GL honours on the dev Mac (M1,
  GL 2.1), so they're constant on screen at any zoom.
- **Panda3D path resolution:** load shaders/textures via `Filename.from_os_specific(p)`. Raw
  Windows backslash paths silently fail to resolve.
- **Quaternions** (`core/attitude.py`) are `[w, x, y, z]`, unit norm; the ship **nose** is body
  `+Z` rotated by the orientation. Roll is controllable but doesn't affect thrust.
- **Download-and-cache** (DE440 kernel, texture maps) lives under `data/` (gitignored). Every path
  MUST return `None`/fall back and never crash offline (validate magic numbers — a CAPTCHA HTML
  page is not a JPEG). Texture source that works: three.js GitHub-raw maps.
- **N-body substeps are sized by the osculating periapsis of the dominant body** (constant
  along a coast → uniform steps → symplectic, no secular energy drift), block-quantized
  to `max_step/2^k`, at 1/`SUBSTEP_DIVISOR` (100) of the periapsis timescale. The coast step
  (`nbody._symplectic_adaptive`, alias `_verlet_adaptive`) is **Yoshida 4th-order**: three
  kick-drift-kick sub-steps weighted w1, w0, w1, so a step costs `EVALS_PER_STEP` (3) gravity
  evaluations. `max_safe_warp[_solar]` budget **gravity evaluations** (`budget_evals=200` per
  frame), not steps; in deep space the 12 h max step (`nbody.FLIGHT_MAX_STEP_S`) binds, so
  the top warp there is 100,000,000×. 12 h / 2^(k+1) = 6 h / 2^k, so raising it from 6 h
  changed no step whose cap is under 12 h (goldens bit-identical). A single giant `world.step(period)` is accurate even across periapsis
  passages, but flight-cadence stepping remains the truest exercise of the sandbox loop. Don't
  switch the coast integrator to RK4 — it isn't symplectic and drifts on long coasts.
- **The cap divisor is a default argument, bound at import.** `_timescale_cap_s` and
  `_solar_cap_s` take `divisor=SUBSTEP_DIVISOR`; assigning `nbody.SUBSTEP_DIVISOR` at runtime
  changes nothing. To experiment, patch their `__defaults__` (or pass `divisor=`). Powered
  flight (`core/flight.py`) and the warp caps size substeps through the same caps.
- **Vessels land, not sink — or tunnel.** Before a coast tick, `core.impact.surface_crossing_s`
  gives the conic time to the inbound surface crossing; inside the tick, the vessel coasts to
  it, lands, and rides the surface for the rest (at 1e5× a LEO frame is ~1,700 s, enough to dive
  under and climb back out). `World.step` also resolves surface contact after each tick: below the
  dominant body's surface a vessel is clamped to it with the body's velocity and `landed_on`
  set; TWR > 1 lifts off again. This (plus the surface-radius floor in the substep caps) is
  what keeps the singular r→0 gravity region unreachable — don't remove one without the other.
- **Zero-velocity states are legal** (landed vessels). Anything dividing by `|v|` or `|r×v|`
  must guard: orbital SAS directions raise `ValueError`; the navball/HUD horizon comes from
  `local_horizon_basis`, which falls back to an r-only vertical.
- **Hyperbolic Kepler must converge on the Newton step, not |f|.** Converge on
  `abs(dF) <= tol*(1+abs(F))` instead of `abs(f) < tol`. The elliptic solver is immune.
- **Profile before optimizing; most hot spots were per-point Python calls.** Use
  `tests.render.app_harness.profile` (fixed-step clock, so runs are comparable). Past wins: orbit
  lines sampled in one numpy pass (not a StateVector per point); closest approach via
  `core.propagate.KeplerOrbit` (elements + rotation derived once; `state_at` is bit-identical to
  `propagate_kepler`); `_norm3`/`_cross3` in the substep hot loop (bit-identical to numpy's, minus
  the n-d overhead). Prefer bit-identical speedups — then the physics goldens stay exact.
- **Stutter that isn't in the render thread's profile is GIL contention.** A background
  Python thread doing CPU work steals the render thread's time in 5 ms slices, so a cheap
  call (0.06 ms) shows up as 50–200 ms. Sample `sys._current_frames()` from a side thread
  to see who holds it. The big one was lamberthub's numba JIT: plain `@jit` with no cache,
  so every process re-compiled the Lambert solver on first use (~2.4 s per process) on the
  first intercept/porkchop plan. `core/jit_cache.py` turns numba's disk cache on at
  `core/transfers.py` import. That call must stay *before* the first Lambert call.
- **Drawn predictions have their own stepper; never point `World.step` at it.** The live line
  and the maneuver preview integrate with `nbody.propagate_solar_prediction`: each body's
  substep is sized by its *current* distance at 1/`PREDICTION_SUBSTEP_DIVISOR` (300, with the 4th-order step) of its
  timescale. Flight's periapsis-sized steps with the old 120 s preview floor mispredicted a
  Moon flyby by 7× (6,057 km vs ~45,670 km converged), taking 6 s. The prediction stepper
  (now on the same Yoshida step) gets it within 0.01% in ~1.9 s
  (`tests/render/test_preview_accuracy.py`). Flight itself now flies that transfer within
  ~25 km (+24 km in 3,600 s chunks, +9 km in 60 s chunks; velocity Verlet at divisor 200 was
  −2,500 km), pinned by `test_flown_moon_transfer_matches_converged_truth`. Moon flybys
  amplify departure errors, so be careful with tolerances.
- **Maneuver preview: estimate while editing, exact once settled.** While the burn changes,
  a faint two-body estimate (`kepler_estimate_points`, `ESTIMATE_STYLE`) redraws every frame.
  The exact N-body job is submitted once the burn has been still `PREVIEW_SETTLE_S`, once per
  burn (`should_submit_exact`). Both lines rebuild from cached points on zoom; don't resubmit
  to rescale. Keep the live line cheap (1 orbit, 256 pts).
- **Encounter periapsis comes from the conic, not the samples.** Drawn paths are sampled
  hours apart (2.8 h on a Moon transfer). A parabola through the samples read a grazing
  (-26 km) Moon pass as 3,298 km. `find_encounters(..., velocities=, body_state_fn=)` takes
  the body-relative conic at the nearest sample (`core.flyby.conic_periapsis`) for the
  radius, epoch, Pe point and impact test. Without those arguments it falls back to the
  parabola (synthetic tests).
- **Where to aim: status box + "when you arrive" ghost (`render/aim_status.py`,
  `targeting.frame_aim`).** The box takes a colour only from an exact N-body result for
  the *current* burn and target: `maneuver_editor.exact_preview_current(node)` (the preview
  key includes the target name), or with no burn the live line's `_traj_approach[0]` for
  that target. Otherwise it is grey "checking…" with the Earth-only Kepler estimate.
  While a burn is being edited the encounter cues show nothing (`encounter_source`), never
  the un-burned path. Rajdhani has no ✓/✗ glyphs, so the icons are LineSegs.
- **Moon auto-plan = Lambert seed + B-plane correction (`core/flyby_targeting.py`).**
  `intercept_node` aims at the Moon's centre. In N-body that seed passes ~44,000 km from
  the Moon. `moon_flyby_node` keeps the burn epoch and corrects the dV on the B-plane to a
  200 km periapsis: FD Jacobian through `_map_rows` (pooled == serial), damped Newton,
  Broyden updates. About 8 evaluations, ~2 s. With an executor *every* evaluation runs on
  it, not just the FD fan-out. On the planning thread the serial ones held the GIL, and
  `node_planned` frame time rose 47%. It runs on the *prediction* propagator, so
  the plan matches the drawn preview (198 km). The flight integrator flies that plan to a
  147–165 km pass (3600 / 600 / 60 s chunks), safe but about 40 km low. The pre-Yoshida
  flight crashed it.
- **Editing a burn: one state, two inputs.** The nudge keys (`node_edit.NUDGE_KEYS`) and the
  3D handles (`node_handles`) write `app._dv` / `app._node_epoch_s` through
  `maneuver_editor.set_dv` / `set_node_epoch` / `place_node`, so they always agree (the jog
  slider panel is gone; tests set a burn with `hooks.set_burn`). Clicking your orbit opens
  `node_placement`'s "Add burn here / Warp to here" menu. The burn panel shows only what
  exists: `burn_summary_text()` (also the map label), Next Pe / Next Ap / Delete burn, and an
  Execute button relabelled every frame (Execute Burn / Jump & Execute).
- **Screen layout lives in one table: `render/ui/hud_layout.PANELS`.** Every corner panel
  (time+warp, target, orbit+fuel, navball, speed chip, SAS, burn) is a `PanelRect` relative
  to its `a2d*` anchor; `tests/render/test_hud_layout.py` pins no overlap from 3:2 to 21:9
  and UI scale 0.84–1.08. Build a panel with `ui.widgets.layout_panel_frame` and register it
  in `app._hud_panels` (that's what `_update_responsive_ui` scales and what the label pass
  hit-tests). Don't place HUD widgets with ad-hoc coordinates.
- **World labels never sit on panels.** `readouts.hide_labels_on_panels` is the *last* step
  of `_update`: any world label whose drawn box (its card, laid along the camera's right/up
  axes, since labels are eye billboards) overlaps a shown panel (its current frame, +4 px)
  is hidden, whatever its priority. Testing only the anchor isn't enough: a centre-aligned
  label is ~240 px wide, so an anchor just off a panel left half the text under it. A new world label must be listed in
  `Readouts._world_labels` (owned = shown/hidden by its controller each frame; static = only
  the pass hides it). `hooks.labels_over_panels` checks it from the widgets themselves.
- **The camera focus is not at screen centre.** In the sandbox `CameraRig.focus_screen_y =
  FOCUS_SCREEN_Y` (0.20) applies a lens film offset so the ship sits above the navball
  cluster. Picking goes through `camLens.project`, so it follows. Any change to framing moves
  the pixels scripted clicks land on: `orbit_menu_warp_here`'s pixel-space pick moved its warp
  target by 7 µs when the offset landed (re-recorded, see its commit).
- **Input gotchas:**
  - Headless windows have no `mouseWatcherNode`. Read the mouse and keys only through
    `flight_input.mouse_px` / `mouse1_held` / `mouse_over_gui` / `key_held`; the harness
    patches them via `hooks.set_mouse_px` / `set_mouse1_held` / `set_key_held`.
  - Panda3D names punctuation keys by the character (`","`, `"."`, `"["`), not `"comma"`.
  - Modifiers prefix events (`shift-j`, `shift-j-up`), so a release can arrive under another
    name. Held-key and drag state is re-checked each frame against the real button state.
  - Every `accept()` changes the `events` field of every sandbox golden.
  - The navball has its own display region, drawn over all 2D GUI; popups must keep clear of
    it (`node_edit.menu_clear_of`).
- **Burn radial sign (open issue):** `apply_maneuver`'s radial axis is `h_hat x v_hat`, which
  points *toward* the central body, while SAS `RADIAL_OUT` is `v x h` (away). The planner and
  `rtn_basis` use the same `h x v`, so burns are self-consistent; only the "radial in/out"
  naming is inverted. Fixing it means flipping the sign in core and in stored burns.
- **Camera is smoothed: `CameraRig` holds current AND target state.** `set_distance` snaps both;
  `move_to_distance`/`zoom`/`orbit` set only the *target*; `CameraRig.update(dt)` eases each
  frame. Don't write `rig.azimuth`/`distance_m` directly from input handlers.
- **Ephemeris cache positions are constant within a frame — inside Earth's SOI.** The cached
  wrappers (`_csun`, etc.) ignore their `t_s` argument when the cache is populated; near Earth
  that's harmless (Earth and ship feel nearly the same Sun). Outside Earth's SOI the direct and
  indirect solar terms stop cancelling: a Sun frozen for a whole frame put a deep-space coast
  2,000,000 km off after 60 days at 1e7×. There `World.step` runs inside
  `nbody.time_correct_flight_ephemeris()` (daily DE440 knots, Hermite-interpolated per substep,
  kept across frames, fetched 64 days per vectorized `ephemeris.body_states` call, bit-identical
  to scalar calls); `test_deep_space_coast_at_warp_follows_the_moving_planets` pins it. The Moon
  is NOT frame-frozen: it moves
  too much at high warp, so `moon_state_at()` is evaluated per substep — from hourly DE440
  samples, cubic-Hermite interpolated (~1 cm error, ~8 µs/call; a raw Skyfield call is ~35×
  slower), fetched in aligned blocks of 256 knots (a 1e8× frame spans ~463). Offline or outside DE440's 1849–2150 range it falls back to
  `moon_circular_state_at`, a circle fitted to the real Moon's J2000 plane/phase (i = 20.9° to
  the **equator** — 5.1° is the inclination to the *ecliptic*; the frame here is ICRF).
- **The real Moon is eccentric, so Lagrange points pulsate.** `earth_fixed_lagrange_points`
  balances gravity against the Kepler rate for the *current* separation, `sqrt(μ/d³)` (the
  elliptic-R3BP pulsating frame), not the constant `OMEGA_EM`. L-point distances scale with
  d(t); their velocity is `ω_line × r + (ḋ/d)·r`. `OMEGA_EM`/`rotating_frame`/`jacobi_constant`
  remain the idealized circular-CR3BP analysis model only.
- **Multi-core work goes through the app's one persistent `ProcessPoolExecutor`**
  (`render/process_pool.py::build_process_pool`, `spawn` context, thread-executor fallback). Rules:
  work sent to it must be **module-level + picklable** — the grid fan-out uses top-level `_*_row`
  functions in `core/optimize.py`; the trajectory samplers live in `render/trajectory_sampling.py`
  (no `self`, no Panda3D). Never submit a bound `OrbitApp` **or controller** method (controllers
  hold `app`, so either drags Panda3D across the pickle); the app golden tests' pool stand-in
  pickles every job and fails if you do. The planning
  *search* runs on a 1-worker **thread** (`_planning_executor`) that fans cells across the pool and
  is polled in `_update` — so the render thread never blocks. A passed `executor=` must not change a
  result (serial vs pooled is bit-identical; tests pin it). Sequential ODEs (one trajectory sample,
  high-warp `world.step`) are **not** parallelisable — don't try. Keep `orbitsim/__main__.py`'s
  `if __name__ == "__main__"` guard: `spawn` re-imports it in every worker.
