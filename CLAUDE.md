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
.venv/bin/python -m pytest tests/ -q                     # full suite (~266 tests)
.venv/bin/python -m pytest tests/core -q                 # physics core only (no graphics needed)
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
`app.win.save_screenshot(Filename.from_os_specific(path))`.

## Architecture — the one rule that governs everything

Strict one-directional layering. Violating it breaks the project's core guarantee (that the
physics is unit-testable to textbook accuracy with zero graphics installed):

```
orbitsim/core/   pure physics. float64, SI units. NEVER imports panda3d, sim, or render.
orbitsim/sim/    stateful world: World, Vessel, SimClock. imports core only.
orbitsim/render/ Panda3D. imports sim + core. ALL graphics live here.
```

If you are tempted to `import panda3d` inside `core/`, stop — the design is wrong at that point.

Data flow each frame (`render/app.py::OrbitApp._update`):
`clock.advance(dt)` → `refresh_ephemeris_cache(t)` → `world.step(sim_dt)` → recentre the floating
origin on the focused vessel → remap all positions to render space → update HUD/navball.

`OrbitApp` has **two modes**: the **sandbox** (default — Earth-centered, one flyable vessel,
maneuver editor) and the **solar viewer** (`--solar` — Sun-centered, planets from the ephemeris,
no vessel).

### N-body gravity model

**Full solar-system N-body** (`core/nbody.py`): the ship is a massless test particle in a
geocentric non-inertial frame. Gravity comes from:
- Earth (central, at origin)
- Moon (circular orbit, evaluated per substep via `moon_state_at`)
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
  not the transform scale, to change zoom.
- **Panda3D path resolution:** load shaders/textures via `Filename.from_os_specific(p)`. Raw
  Windows backslash paths silently fail to resolve.
- **Quaternions** (`core/attitude.py`) are `[w, x, y, z]`, unit norm; the ship **nose** is body
  `+Z` rotated by the orientation. Roll is controllable but doesn't affect thrust.
- **Download-and-cache** (DE440 kernel, texture maps) lives under `data/` (gitignored). Every path
  MUST return `None`/fall back and never crash offline (validate magic numbers — a CAPTCHA HTML
  page is not a JPEG). Texture source that works: three.js GitHub-raw maps.
- **N-body substeps are sized by the osculating periapsis of the dominant body** (constant
  along a coast → uniform steps → symplectic Verlet, no secular energy drift), block-quantized
  to `max_step/2^k`. A single giant `world.step(period)` is now accurate even across periapsis
  passages, but flight-cadence stepping remains the truest exercise of the sandbox loop. Don't
  switch the coast integrator to RK4 — it isn't symplectic and drifts on long coasts.
- **Vessels land, not sink.** `World.step` resolves surface contact after each tick: below the
  dominant body's surface a vessel is clamped to it with the body's velocity and `landed_on`
  set; TWR > 1 lifts off again. This (plus the surface-radius floor in the substep caps) is
  what keeps the singular r→0 gravity region unreachable — don't remove one without the other.
- **Zero-velocity states are legal** (landed vessels). Anything dividing by `|v|` or `|r×v|`
  must guard: orbital SAS directions raise `ValueError`; the navball/HUD horizon comes from
  `local_horizon_basis`, which falls back to an r-only vertical.
- **Hyperbolic Kepler must converge on the Newton step, not |f|.** Converge on
  `abs(dF) <= tol*(1+abs(F))` instead of `abs(f) < tol`. The elliptic solver is immune.
- **Trajectory lines are the main per-frame cost — throttle, don't naively cache.** Fix is a
  **real-time throttle** (`PREVIEW_THROTTLE_S`, 5 Hz), not a cache. Keep the live line cheap
  (1 orbit, 256 pts); the preview draws 2 orbits at 512 pts.
- **Camera is smoothed: `CameraRig` holds current AND target state.** `set_distance` snaps both;
  `move_to_distance`/`zoom`/`orbit` set only the *target*; `CameraRig.update(dt)` eases each
  frame. Don't write `rig.azimuth`/`distance_m` directly from input handlers.
- **Ephemeris cache positions are constant within a frame.** The cached wrappers (`_csun`, etc.)
  ignore their `t_s` argument when the cache is populated — this is by design (planet positions
  don't change meaningfully over one frame's substeps). The Moon is NOT cached; it uses
  `moon_state_at()` per substep since it moves significantly at high warp.
