# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A desktop (Windows) 3D orbital mechanics simulator/game in Python — "KSP but the physics are
real." Three pillars: transfer & ΔV planning (Hohmann/bi-elliptic/Lambert + porkchop optimizer),
the real solar system (JPL/Skyfield ephemerides + patched conics with SOI handoffs), and a
sandbox (maneuver nodes with live orbit prediction). Vessels are point masses with a ΔV budget;
**no** aerodynamics/atmosphere (out of scope). As of 2026-06-26 the sandbox is moving to a
**restricted N-body** model (real Moon gravity + Lagrange points — see "Direction shift" below),
and buildable/launchable ships (mass + 3D models, stations, landers) are a longer-term goal — so
rocket-part building is no longer permanently off the table, though not yet built.

## Commands

Always use the venv interpreter. Bare `python` resolves to the global install, which lacks the
dependencies and will fail with `ModuleNotFoundError: astropy` / `panda3d`.

```bash
.venv/Scripts/python -m pytest tests/ -q                 # full suite (~211 tests)
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

**Hybrid coast/burn flight** (`sim/world.py::World.step`): a vessel coasting (throttle 0) is
propagated **analytically** (`core.propagate.propagate_kepler`, on rails — time-warp works); a
vessel thrusting integrates **numerically** (`core.flight.integrate_powered`, RK4 + real rocket
equation) and the render layer **forces time-warp to 1×** while any vessel thrusts (you can't RK4
through 10⁶×). Attitude slews toward the SAS target every tick regardless. The integrator uses
**operator splitting** — an exact rocket-equation velocity impulse per substep, then RK4 gravity
drift — so Δv telescopes to `vₑ·ln(m₀/m_f)` and fuel hits exactly 0 (the naive "RK4 the fuel ODE
with a switch" leaves residual fuel; that was a real bug fixed at review).

> **In flux (N-body Part 2, not yet wired):** the tested N-body propagator (`core/nbody.py`:
> `propagate_earth_moon`, velocity-Verlet) exists but `World.step` still uses `propagate_kepler` /
> `integrate_powered`. Part 2 swaps the sandbox coast/powered to N-body, forward-integrates the
> trajectory line, and caps warp via `core.nbody.max_safe_warp` (warp can't be unlimited under
> numerical integration — it's bounded by a per-frame sub-step budget near bodies). Until then the
> sandbox is still two-body on rails.

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
| 1–5 Physics core, render, maneuvers, transfers, solar system | **complete** (whole suite now **211 tests** green) |
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
| Rest of Phase 6: docs, packaging | planned (later cycles) |

**Direction shift (2026-06-26):** the project is pivoting to a **restricted N-body** model (ship as a
massless test particle; Earth fixed + circular Moon + indirect/third-body term) so it becomes an
*alternative* to KSP with real Lagrange points — **reversing** the earlier "stick with patched conics"
decision. Built in cycles: 1a engine core ✅ → 1b flyable (Part 1 physics ✅, **Part 2 render
integration = next**) → 1c Lagrange-point visualization. Spec: `docs/superpowers/specs/2026-06-26-nbody-flyable-design.md`.
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
- **Reviewing subagent work:** implementer subagents will quietly **loosen a test tolerance** to
  make a red test pass (seen repeatedly: `<1e-6` → `<1.0`). Always re-derive the physics yourself
  and restore the tight assertion; the bug is usually the implementation (or the plan's number),
  not the tolerance.

## Platform notes

Windows-only by design. The shell is PowerShell; `.venv/Scripts/` (not `bin/`). Line endings:
git will warn LF→CRLF on commit — harmless.
