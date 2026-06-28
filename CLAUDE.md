# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A desktop (Windows) 3D orbital mechanics simulator/game in Python — "KSP but the physics are
real." Three pillars: transfer & ΔV planning (Hohmann/bi-elliptic/Lambert + porkchop optimizer),
the real solar system (JPL/Skyfield ephemerides + patched conics with SOI handoffs), and a
sandbox (maneuver nodes with live orbit prediction). Vessels are point masses with a ΔV budget;
**no** aerodynamics/atmosphere (out of scope). The sandbox now runs a **restricted N-body** model
(real Moon gravity + Lagrange points — all three cycles landed; see "Direction shift" below), and
buildable/launchable ships (mass + 3D models, stations, landers) are a longer-term goal — so
rocket-part building is no longer permanently off the table, though not yet built.

## Commands

Always use the venv interpreter. Bare `python` resolves to the global install, which lacks the
dependencies and will fail with `ModuleNotFoundError: astropy` / `panda3d`.

```bash
.venv/Scripts/python -m pytest tests/ -q                 # full suite (~259 tests)
.venv/Scripts/python -m pytest tests/core -q             # physics core only (no graphics needed)
.venv/Scripts/python -m pytest tests/core/test_flight.py -q                          # one file
.venv/Scripts/python -m pytest "tests/core/test_kepler.py::TestEllipticAnomalies"    # one class/test
.venv/Scripts/python -m orbitsim                         # launch the sandbox (LEO, flyable)
.venv/Scripts/python -m orbitsim --solar                 # launch the solar-system viewer
```

Dependencies are in `pyproject.toml` extras: `dev` (pytest, black) and `render` (panda3d,
skyfield, lamberthub, matplotlib). **`black` is NOT installed in the venv** — `python -m black`
fails with `No module named black`; just hand-format to line length 100 (or `pip install black`
first). `pytest` is preconfigured (`testpaths=tests`, `-v`).

First launch downloads + caches large assets into `data/` (gitignored): the DE440 ephemeris
kernel (`de440s.bsp`, ~32 MB) and texture maps (`data/textures/`). All download code MUST degrade
gracefully offline — see the gotchas.

Verifying graphics headlessly (how render code is checked without a display): prepend
`loadPrcFileData("", "window-type offscreen")` before constructing `OrbitApp`, drive it with
`app.taskMgr.step()`, and capture `app.win.save_screenshot(Filename.from_os_specific(path))` to
inspect the result. Set `PYTHONPATH` to the repo root when running such scripts from elsewhere.

## Architecture — the one rule that governs everything

Strict one-directional layering. Violating it breaks the project's core guarantee (that the
physics is unit-testable to textbook accuracy with zero graphics installed):

```
orbitsim/core/   pure physics. float64, SI units. NEVER imports panda3d, sim, or render.
orbitsim/sim/    stateful world: World, Vessel, SimClock. imports core only.
orbitsim/render/ Panda3D. imports sim + core. ALL graphics live here.
```

If you are tempted to `import panda3d` inside `core/`, stop — the design is wrong at that point.

Data flow each frame (`render/app.py::OrbitApp._update`): `clock.advance(dt)` → `world.step(sim_dt)`
→ recentre the floating origin on the focused vessel → remap all positions to render space →
update HUD/navball.

`OrbitApp` has **two modes**, chosen at construction: the **sandbox** (default — Earth-centered,
one flyable vessel, maneuver editor) and the **solar viewer** (`--solar` — Sun-centered, planets
from the ephemeris, no vessel). The title screen defers all scene construction to `_start_sim()`,
called on Play; the update loop branches on `self.solar_system`.

**Hybrid coast/burn flight, now N-body** (`sim/world.py::World.step`): a vessel coasting (throttle 0)
propagates under `core.nbody.propagate_earth_moon` (Earth fixed + circular Moon + indirect term,
velocity-Verlet); a vessel thrusting uses `core.flight.integrate_powered_nbody` (same gravity model
+ real rocket equation). Both replaced the two-body `propagate_kepler` / `integrate_powered` when
N-body 1b Part 2 landed. The integrator still uses **operator splitting** — an exact rocket-equation
velocity impulse per substep, then a gravity drift — so Δv telescopes to `vₑ·ln(m₀/m_f)` and fuel
hits exactly 0 (the naive "integrate the fuel ODE with a switch" leaves residual fuel; that was a
real bug fixed at review). Attitude slews toward the SAS target every tick regardless. The render
layer **forces time-warp to 1×** while any vessel thrusts, and otherwise **caps warp each frame via
`core.nbody.max_safe_warp`** (numerical integration can't run at unbounded warp — the cap bounds
per-frame sub-steps to a budget near bodies). The two-body `core/` (`propagate_kepler`,
`integrate_powered`) is **not** deleted — it stays the analysis layer (transfers, intercept seeds)
and on-rails body motion.

> **N-body sandbox gotchas (all three cycles shipped):**
> - **Closest-approach stays Keplerian** (`render/app.py`, `core/rendezvous.py`). N-body CA is
>   *infeasible* as written: `closest_approach` propagates **both** trajectories with one propagator,
>   and the target is the on-rails Moon — propagating the Moon under `earth_moon_accel` is singular
>   (it sits at its own gravity source → NaN), and re-integrating the ship from base to each of 720
>   coarse samples over a multi-day window is O(samples×window), which freezes the loop. Don't naively
>   pass `propagate_earth_moon` here. The live N-body **trajectory line** is the visual source of truth;
>   CA is an approximate aid, like the (also Keplerian) intercept/porkchop seeds. True N-body CA needs
>   a trajectory-**sampling** implementation — a later cycle.
> - **High-warp coast drift is expected.** `max_safe_warp` bounds per-frame sub-steps (budget 200), but
>   in LEO that still permits ~1e5× with h≈29 s, so the coast drifts ~km/orbit at extreme warp. Accepted
>   tradeoff of the N-body pivot (two-body was exact-on-rails at any warp).
> - **Lagrange points:** `core.nbody.earth_fixed_lagrange_points(t)` computes L1–L5 in the **live**
>   Earth-fixed frame (real inclined Moon geometry, indirect term) — distinct from the 1a barycentric
>   `lagrange_points` (kept as a tested reference). Both coexist on purpose.

### The scale/precision problem (why `render/floating_origin.py` exists)

The solar system spans ~1.5e11 m but a docking maneuver needs ~1e-3 m precision, and GPUs render
in float32 (~7 digits). The fix: physics stays float64 SI; `RenderTransform.to_render` subtracts a
float64 origin (set to the focused body each frame) **before** casting to float32. Test this by
the math, not by eye: at 1e11 m, float64's own ULP is ~1.5e-5 m, so precision assertions tighter
than that are unsatisfiable by any implementation — the float32 cast itself adds nothing.

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

The original six phases (`docs/00-OVERVIEW.md`, read first) are all **complete** (Phases 1–5
built; Phase 6 polish/packaging never run as written). Work since then is a **"playable game"
effort** — each sub-project gets its own brainstorm → spec (`docs/superpowers/specs/`) → plan
(`docs/superpowers/plans/`) → build cycle. Status:

| Area | Status |
|---|---|
| 1–5 Physics core, render, maneuvers, transfers, solar system | **complete** (whole suite now **259 tests** green) |
| Continuous-thrust flight (rocket eq, navball, controls) | **complete** (`...plans/2026-06-24-continuous-thrust-flight.md`) |
| Realistic Earth (textured, day/night, atmosphere) | **complete** (`...plans/2026-06-25-realistic-earth.md`) |
| Starfield / skybox | **complete** (`render/skybox.py`; `...plans/2026-06-25-starfield.md`) |
| Save/load (Phase 6.1: sandbox JSON, F5/F9 quicksave) | **complete** (`sim/persistence.py`; `...plans/2026-06-25-save-load.md`) |
| HUD/UX polish (Phase 6.2 A1: F1 overlay, inclination, toast, Esc settings/units) | **complete** (`render/keybind_overlay.py`, `render/settings_panel.py`; `...plans/2026-06-25-hud-ux-polish.md`) |
| Scheduled maneuver nodes (Phase 6.2 A2: time-to-node, Pe/Ap presets, auto-warp-down) | **complete** (`core.maneuvers.time_to_periapsis/apoapsis`; `...plans/2026-06-25-scheduled-maneuver-nodes.md`) |
| Moon intercept/target (Phase 6.2 A3: Keplerian Moon, closest-approach markers) | **complete** (`core/moon.py`, `core/rendezvous.py`; `...plans/2026-06-25-moon-intercept.md`) |
| Orbit-line caching (Phase 6.3 B: render-unit orbit frame, rebuild on change) | **complete** (`...plans/2026-06-25-orbit-line-caching.md`) |
| Targeting + ΔV controls (click-to-target, working TARGET SAS, unlimited-ΔV cheat, porkchop intercept node) | **complete** (`render/targets.py`, `render/picking.py`, `core.optimize.intercept_node`; `...plans/2026-06-26-{deltav-controls,target-selection,intercept-node}.md`) |
| **Restricted N-body engine core (Cycle 1a)** | **complete** (`core/nbody.py`: CR3BP, velocity-Verlet, Jacobi, Lagrange points; `...plans/2026-06-26-nbody-engine-core.md`) |
| **N-body flyable — core physics (Cycle 1b Part 1)** | **complete** (`core/nbody.py` `earth_moon_accel`/`propagate_earth_moon`/`osculating_elements`/`max_safe_warp`, circular Moon; `...plans/2026-06-26-nbody-flyable-core.md`) |
| **N-body flyable — render integration (Cycle 1b Part 2)** | **complete** (`World.step` N-body, `integrate_powered_nbody`, forward-integrated trajectory line, warp cap, osculating HUD; CA stays Keplerian; `...plans/2026-06-26-nbody-render-integration.md`) |
| **N-body Lagrange-point visualization (Cycle 1c)** | **complete** (`core.nbody.earth_fixed_lagrange_points`, `render.targets.LagrangePointTarget`, 5 markers/labels + live distance/rel-vel readout; `...plans/2026-06-27-nbody-lagrange-visualization.md`) |
| **Graphical: ship view (3rd-person model + camera mode-switch)** | **complete** (`render/ship_model.py` procedural lit ship + plume; cross-fade marker↔model on zoom, `m` toggles map/ship framing; `...plans/2026-06-27-ship-view.md`) |
| **Graphical 2a: HUD overlay readability** | **complete** (`render/hud/panel.py` self-sizing grouped panels, `core.attitude.heading_pitch`, `render/sas_panel.py` SAS chip + clickable mode buttons + orbital/target velocity readout; fixed the colliding top-left readouts; `...plans/2026-06-27-hud-overlay-readability.md`) |
| **Graphical 2b: in-world readability + visual polish** | **complete** (`render/world_markers.py` pure apsis/fade/declutter math, `render/world_labels.py` carded billboard labels, Pe/Ap markers, camera easing in `render/camera_rig.py`, outlined/fading trajectory lines in `render/orbit_lines.py`; built directly with Codex, no separate spec) |
| **Graphical: Moon SOI sphere** | **complete** (faint true-scale **translucent tinted shell** — low-alpha unlit two-sided `make_uv_sphere` — at `core.nbody.MOON_SOI_M` centered on the Moon: blue tint outside, tints the scene green when the vessel is inside, distance-fades; sandbox-only. Started as a wireframe (`make_wireframe_sphere`, since removed); `...plans/2026-06-27-soi-sphere.md`) |
| Rest of Phase 6: docs, packaging | planned (next) |

**Direction shift (2026-06-26), now fully landed:** the project pivoted to a **restricted N-body** model
(ship as a massless test particle; Earth fixed + circular Moon + indirect/third-body term) so it became
an *alternative* to KSP with real Lagrange points — **reversing** the earlier "stick with patched conics"
decision. All three cycles are **complete**: 1a engine core ✅ → 1b flyable (Part 1 physics ✅, Part 2
render integration ✅) → 1c Lagrange-point visualization ✅. The sandbox now flies under real Moon gravity
with selectable Lagrange points. Specs: `docs/superpowers/specs/2026-06-26-nbody-flyable-design.md`,
`2026-06-26-nbody-render-integration-design.md`, `2026-06-27-nbody-lagrange-visualization-design.md`.
Longer-term vision (not yet started): buildable/launchable ships with mass + 3D models, stations,
landers — vessels stay point masses for *propagation* (mass only affects ΔV). Two-body `core/`
remains the analysis layer (transfers, intercept seeds) and on-rails body motion; it is NOT discarded.

Plans are executed via the `superpowers:subagent-driven-development` workflow: pure-physics tasks
go to Haiku implementer subagents (TDD), render/visual tasks are done by the controller with
headless screenshots. **Review subagent output independently** — see the gotcha below.

**TDD is mandatory** because this is math a model can get wrong silently. For every `core`
function: write the known-answer test first (concrete values are in the phase/plan docs), watch it
fail, implement until green, then add the invariants below as `hypothesis` property tests. Never
loosen a tolerance to force a pass — if an invariant drifts, the most recently added function is
wrong (or, occasionally, the plan's stated tolerance is — verify the physics before either).

Universal invariants (encode as property tests): elements↔state round-trip (1e-7 rel), energy
`ε = v²/2 − μ/r = −μ/(2a)` conserved, angular momentum `h = r × v` conserved, period closure
(< 1 mm analytic), vis-viva `v² = μ(2/r − 1/a)` at every point.

### Commit & push after every change

Remote is `origin` → https://github.com/JoeyPhatsJr/orbitsim (default branch `main`). After each
change is made and verified (tests green / screenshot checked), **commit and push to GitHub** —
don't batch many changes into one late commit. Staging rules still hold: stage the specific files
you changed by **explicit path** (never `git add -A`/`.`), keep `data/`, scratch files
(`debug_curtis.py`, `kickbacks.vsix`, `.hypothesis/`), generated assets (`porkchop.png`,
`saves/`), and `.claude/settings.local.json` out of commits, and `CLAUDE.md` stays untracked.
End commit messages with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` line, then
`git push`.

## Domain gotchas (learned the hard way)

- **Reference textbook:** Curtis, *Orbital Mechanics for Engineering Students*. Algorithm/Example
  numbers in the phase docs come from it.
- **Perifocal→inertial rotation** uses the **active** convention `Q = R3(Ω) R1(i) R3(ω)` (positive
  angles), not `R3(−Ω) R1(−i) R3(−ω)`. Getting this backwards passes some tests and fails the
  hyperbolic anchor (Curtis Ex 4.7).
- **Eccentricity vector:** `e = (1/μ)[(v² − μ/r)·r − r·v_r·v]` where `v_r = (r·v)/r` — note the
  `r` (magnitude) factor on the second term.
- **Degenerate orbits:** circular (`e≈0`) and equatorial (`i≈0`) make argp/raan/ν individually
  undefined; round-trip element tests must exclude or special-case them (only the angle *sums*
  are recoverable).
- **lamberthub** signature is `izzo2015(mu, r1, r2, tof, M=0, prograde=True)`, all SI. Lambert is
  singular at exactly 180° transfer angle (plane undefined) — perturb test geometry slightly off π.
- **Skyfield loader** is `from skyfield.api import Loader; Loader(dir)` — NOT `load.Loader(dir)`
  (that `AttributeError`s on Skyfield 1.49; the phase-5 plan had this wrong).
- **Render markers** (vessels, planets) must use a constant on-screen size, not tiny world-space
  radii — the camera sits a fixed render distance from the focus, so a world-space-small marker is
  a sub-pixel speck. The **`CameraRig` owns scale** (`scale_m_per_unit = distance/1000`); set the
  rig's `distance_m`, not the transform scale, to change zoom.
- **Panda3D path resolution:** load shaders/textures via `Shader.load(..., Filename.from_os_specific(p))`
  and `loader.load_texture(Filename.from_os_specific(p))`. Raw Windows backslash paths silently
  fail to resolve — `build_earth` quietly fell back to a flat sphere until this was fixed.
- **Quaternions** (`core/attitude.py`) are `[w, x, y, z]`, unit norm; the ship **nose** is body `+Z`
  rotated by the orientation. Roll is controllable but doesn't affect thrust.
- **Download-and-cache** (DE440 kernel, texture maps) lives under `data/` (gitignored). Every such
  path MUST return `None`/fall back and never crash when offline or the bytes aren't a valid image
  (validate magic numbers — a CAPTCHA HTML page is not a JPEG). Texture source that works:
  three.js GitHub-raw maps; solarsystemscope is CAPTCHA-walled, ESO fails SSL here.
- **N-body coast tests must step at flight cadence.** `World.step` under N-body is *numerically
  integrated*, not on-rails — a single giant `world.step(period)` is wildly inaccurate (≈14 km/orbit
  in LEO), so don't assert tight closure on one big leap. Drive it the way the render loop does: many
  small `dt` steps (a `_coast(world, total, dt)` helper). At flight cadence LEO coast tracks Kepler to
  metres; one-period closure lands within the ~130 m *physical* Moon perturbation (converges as dt→0).
  Exact two-body closure stays covered by `tests/core/test_propagate.py`.
- **Reviewing subagent work cuts both ways.** Implementer subagents quietly **loosen a test tolerance**
  to make a red test pass (`<1e-6` → `<1.0`) — re-derive the physics and restore the tight assertion;
  the bug is usually the implementation (or the plan's number), not the tolerance. They also do
  **out-of-scope edits** (e.g. adding guards to a shared, already-tested `core/` function to satisfy a
  degenerate test input) — revert and keep the fix in the task's own file. And review **subagents are
  sometimes too lenient** — one praised an out-of-scope change as a "strength." Always sanity-check the
  reviewer's verdict against the diff yourself.
- **Hyperbolic Kepler must converge on the Newton step, not |f|.** A ship deep in the Moon's well has
  an extreme hyperbolic orbit *about Earth* (e≈73); Keplerian closest-approach over a multi-day window
  then hits M≈2500, where `e·sinh(F)≈M` dwarfs 1.0 and the residual |f| floors at the float64 ULP
  (~1e-12) — never below an absolute tol, so Newton loops all 50 iters and raises (this crashed the
  render loop while shrinking a lunar orbit). Converge on `abs(dF) <= tol*(1+abs(F))` instead. The
  elliptic solver is immune (its M is bounded to `[0, 2π)`).
- **Render trajectory lines are the main per-frame cost — throttle, don't naively cache.** Each
  `_sample_trajectory` call (`render/app.py`) forward-integrates N-body (256-pt Verlet loop) *and*
  rebuilds a `LineSegs`; doing it every frame for the live orbit line **and** the post-burn preview
  was the maneuver-plotting lag. A change-detection cache can't save the preview: it re-applies the
  burn at the *current* state ("burn now"), which sweeps ~128 m/frame in LEO, so the orbit
  legitimately changes every frame. Fix is a **real-time throttle** (`PREVIEW_THROTTLE_S`, 5 Hz —
  same pattern as the 0.5 s closest-approach recompute), not a cache. Keep the live line cheap
  (1 orbit, 256 pts); the preview draws 2 orbits via `_sample_trajectory(..., n_orbits=2, n_pts=512)`.
  Don't reintroduce an unthrottled per-frame trajectory rebuild.
- **Subagent spend-cap fallback.** A monthly spend cap intermittently blocks subagent dispatch (returns
  ~0 tokens / "monthly spend limit"). When it hits, fall back to controller-inline implementation/review
  and **flag the missing independent review** in the progress ledger so it can be redone before merge.
- **Camera is smoothed: `CameraRig` holds current AND target state.** `set_distance` snaps both
  current+target (initial framing / load); `move_to_distance` / `zoom` / `orbit` set only the *target*,
  and `CameraRig.update(dt)` (called once at the top of `_update`) eases current→target each frame
  (log-space zoom, shortest-arc angle — pure helpers `smoothing_alpha`/`smooth_log_distance`/`smooth_angle`,
  unit-tested). Read `rig.target_distance_m` (not `distance_m`) for "where the user asked to be" — e.g.
  the `m` ship-view toggle classifies and remembers framings off the target, so easing in flight doesn't
  flip it. Don't write `rig.azimuth`/`distance_m` directly from input handlers; go through the targets.
- **In-world marker readability is split pure/impure on purpose.** `render/world_markers.py` is pure
  numpy (apsis sub-sample interpolation, smoothstep `distance_fade`, priority `declutter_indices`) and
  unit-tested; `render/world_labels.py` builds carded billboard labels (panda imports inside functions).
  `app._update_marker_readability` fades distant cues and declutters overlapping labels by navigation
  priority each frame; Pe/Ap world positions are cached in `_rebuild_trajectory` (idx 0, bound orbits
  only — hidden when `e≥1`) and re-placed via the floating origin in `_place_apsis_markers`.
- **Trajectory lines are now outlined + depth-faded** (`render/orbit_lines.py`): `build_orbit_node`
  draws a dark wide halo under-stroke + the colored stroke, both depth-tested with depth-write off, and
  `path_fade_alphas` (pure) ramps alpha by cumulative path length (present bright, future recedes). Pass
  `fade_minimum=1.0` to disable the fade (the reference Moon orbit does this). Shared colors live as
  module constants (`TRAJECTORY_COLOR`, `MANEUVER_COLOR`, `REFERENCE_ORBIT_COLOR`).

## Platform notes

Windows-only by design. The shell is PowerShell; `.venv/Scripts/` (not `bin/`). Line endings:
git will warn LF→CRLF on commit — harmless.
