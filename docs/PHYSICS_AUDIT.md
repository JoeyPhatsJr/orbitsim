# Physics & Architecture Audit — July 2026

A full-codebase audit of the N-body integration loop, floating-point/precision
handling, high-warp behavior, and hidden edge cases, followed by the upgrade
implemented on this branch. Every claim below was measured with runnable
scripts before the fix and re-measured after; the measurements are encoded as
regression tests.

## 1. What the audit found

### Integration scheme (before)

| Path | Scheme | Order | Notes |
|---|---|---|---|
| Coasting (`core/nbody.py`) | velocity-Verlet (kick-drift-kick) | 2, symplectic | substep count fixed **per call** from the starting radius |
| Burns (`core/flight.py`) | exact rocket-equation impulse, then RK4 (two-body) or Verlet (N-body) drift | splitting was **1st order** | impulse applied entirely before the drift |
| Analysis layer (`core/propagate.py`) | universal-variable Kepler | analytic | unchanged — correct |

The architecture itself is sound: strict `core → sim → render` layering, SI
float64 everywhere, a floating-origin transform that subtracts in float64
before the float32 GPU cast (no observable trajectory jitter — the claimed
design works), per-frame ephemeris caching, and prediction/preview integration
moved off the render thread. The problems were concentrated in step-size
policy and edge-case handling.

### Defects, ranked by severity

1. **Sandbox could not launch at HEAD.** `render/app.py` imported
   `sample_relative_orbit_points`, which did not exist in
   `render/orbit_lines.py` (introduced by the previous commit). ImportError on
   startup. *Fixed: function implemented + tests.*
2. **Per-call substep sizing destroyed eccentric orbits at warp.** The Verlet
   substep was sized once from the starting radius. A high-warp frame starting
   near apoapsis of an rp=6,600 km / ra=300,000 km orbit stepped through
   periapsis at ~3,500 s substeps: measured **3×10⁹ m** position error over
   one period (a spurious slingshot). A state falling toward a body's center
   made the substep count diverge — an unbounded frame hang.
3. **No surface collision.** Nothing stopped a vessel entering a body, where
   the point-mass field is singular; combined with (2) this is the classic
   "physics breaks near the planet center".
4. **Landed/zero-velocity states produced NaN.** `sas_target_dir`,
   `heading_pitch`, and the navball basis all divide by |v| and |r×v|. One
   tick of an orbital SAS hold at v=0 poisoned the orientation quaternion →
   thrust direction → entire state vector.
5. **Offline degradation was broken at the module level.**
   `core/ephemeris.py` downloaded the 32 MB DE440 kernel **at import time**,
   raising `OSError` offline — crashing test collection, breaking the solar
   viewer (which re-attempted the download every frame), and violating the
   project's own "must degrade gracefully offline" rule.
6. **MANEUVER SAS was dead code.** The UI stored `sas_maneuver_dir` on the
   vessel each frame, but `World.step` never read it; `sas_target_dir` raised
   on the unknown mode and the exception was swallowed. Pressing "9" did
   nothing, silently.
7. **Unlimited-ΔV with an empty tank silently coasted.** The powered
   integrators need propellant mass for the impulse math; with fuel = 0 the
   burn branch was entered but produced no thrust.
8. **Warp table inconsistency.** `SimClock.WARP_STEPS` stopped at 1e6 while
   `max_safe_warp_solar`, its test, and the UI promised 1e8 in deep space.
9. **Per-frame hot spots.** `earth_fixed_lagrange_points` ran three `brentq`
   root-solves every frame at any warp; the sandbox sun-direction did a
   try/except ephemeris call per frame (each one re-attempting a kernel
   download when offline).

## 2. Enhancement blueprint (what was chosen and why)

1. **Periapsis-based adaptive symplectic stepping** — the highest-impact
   physics change; fixes (2) and makes high warp safe on any orbit.
2. **Strang splitting for burns** — one-line-per-integrator upgrade from 1st
   to 2nd order finite-burn accuracy; preserves the exact ΔV-telescoping and
   fuel-hits-zero invariants.
3. **Surface contact resolution in the sim layer** — fixes (3), bounds the
   integrator, and adds landing/liftoff as gameplay for free.
4. **Offline-first ephemeris + degenerate-state guards** — fixes (4)(5)(6)(7),
   the reliability tier.
5. **Warp to 1e8 + render-loop throttling** — fixes (8)(9); deep-space coasts
   at 100M× while near-body warp stays capped by the integration budget.

Deliberately *not* done: aerodynamics/atmosphere (explicitly out of scope per
project charter), GPU-side line rendering (the 5 Hz throttle already bounds
it), and replacing Verlet with RK4 for coasts (RK4 is not symplectic; energy
drifts secularly on long coasts — the fix was better step-size policy, not a
different integrator).

## 3. What changed, with measurements

### Adaptive symplectic substepping (`core/nbody.py`)

The integrator re-evaluates its substep cap as it moves. The SOI-dominant
body is sized by 1/200 of the orbital timescale at the **osculating
periapsis** — constant along a coast, so steps stay uniform and Verlet keeps
its symplectic no-drift character — while other bodies cap by current
distance. Steps are block-quantized to `max_step/2^k` so SOI hand-offs switch
step size discretely. `MAX_SUBSTEPS_PER_CALL` plus a surface-radius floor on
rp bound the work per call. Warp budgeting uses the same cap function, so
`max_safe_warp` stays honest.

Measured on the rp=6,600 km / ra=300,000 km orbit, one full period in one
call (`max_step_s=3600`):

| | position error | energy error |
|---|---|---|
| before (fixed per-call substeps) | 3.0×10⁹ m | orbit destroyed |
| naive per-substep adaptive | 1.7×10⁵ m | 5×10⁻⁴/orbit, **linear drift** |
| **shipped (periapsis-sized, quantized)** | **2.1×10⁴ m** | **1.4×10⁻¹³** |

The 7-day Jacobi-constant guard (`test_jacobi_constant_conserved_over_seven_days`,
tolerance 1e-6, deliberately not relaxed) improved from 2.7×10⁻⁷ to 6.8×10⁻⁸.

### Strang splitting (`core/flight.py`)

Half the substep's exact rocket-equation impulse, gravity drift, then the
other half. The log increments telescope, so total ΔV is still exactly
`vₑ·ln(m₀/m_f)` and fuel still reaches exactly zero. Measured on a 60 s
full-throttle LEO burn vs a 20,000-substep reference:

| substeps | impulse-first error | Strang error |
|---|---|---|
| 10 | 3,672 m | 12.6 m |
| 50 (default) | 730 m | **0.50 m** |

### Surface contact (`sim/world.py`)

After each tick, a vessel below the dominant body's surface is placed on it
with the body's velocity and flagged `landed_on`. While landed with no
thrust it rides the surface point exactly (no integrate-then-clamp churn, and
it moves with the Moon if landed there); with thrust it can lift off — and a
TWR < 1 vessel correctly stays on the pad. Verified end-to-end: coast →
100,000× warp → retrograde deorbit → touchdown at rest → pad hold → liftoff
at TWR 2 → hand-off to RADIAL_OUT SAS, with every intermediate state finite.

### Reliability fixes

- DE440 kernel loads lazily; a failed attempt is remembered
  (`EphemerisUnavailableError`, a `ValueError` subclass so existing planning
  code treats it as "no solution"). `available()` gates the ephemeris tests.
- Solar viewer falls back to circular approximations offline and labels the
  data source in the HUD.
- `local_horizon_basis` (new) gives the navball and heading/pitch a finite
  horizon even at v=0; orbital SAS modes raise `ValueError` on degenerate
  states instead of emitting NaN.
- MANEUVER SAS wired through `World.step`; unlimited-ΔV floors the working
  propellant mass for the impulse math without touching the real tank.
- Lagrange-point solves throttled to a 10-sim-second bucket (~0 cost at low
  warp, automatic per-frame cadence at high warp).

## 4. Verification

- `python -m pytest tests/ -q`: **322 passed**, 14 skipped (DE440 kernel not
  downloadable in the build sandbox — the skips are the offline path working
  as designed). The 2 failures in `test_earth.py`/`test_skybox.py` on this
  machine are environmental (no X/EGL display available to Panda3D) and
  reproduce identically on the unmodified base commit.
- New regression tests: eccentric periapsis passage (position + symplectic
  energy), Strang convergence, landing/pad-hold/liftoff, MANEUVER SAS slew,
  unlimited-ΔV empty-tank thrust, degenerate-attitude NaN guards,
  relative-orbit sampling, warp-table reach.

## 5. Known limits (honest edges)

- Trajectory *prediction* for very eccentric orbits now costs more substeps
  (it resolves periapsis properly); it runs on a worker thread and is
  throttled, but the preview refresh can lag a beat behind on extreme orbits.
- The per-frame ephemeris cache still freezes planet positions within a
  frame; at 1e8 warp one frame spans ~19 days of Sun motion in the indirect
  term. Deep-space accuracy at extreme warp is game-grade, not
  ephemeris-grade. (Background predictions already use time-interpolated
  samples via `stable_prediction_ephemeris`.)
- Bodies do not rotate, so a landed vessel sits at a fixed inertial radial;
  there is no surface co-rotation velocity to exploit or fight at launch.
