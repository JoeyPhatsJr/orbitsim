# N-Body Render Integration (Cycle 1b Part 2) — Design

**Date:** 2026-06-26
**Status:** spec (awaiting build)
**Part of:** restricted N-body foundation. Cycle 1b Part 2 of 3: 1a core ✅ → 1b Part 1
physics ✅ → **1b Part 2 render integration (this spec)** → 1c Lagrange-point visualization.

## Why

Part 1 built and tested the Earth-centered N-body propagator (`earth_moon_accel`,
`propagate_earth_moon`, `osculating_elements`, `max_safe_warp`). Part 2 wires it into the live
sandbox so the ship actually flies under Moon gravity, the trajectory line shows perturbation
bending, warp caps near bodies, and the HUD reads osculating elements.

## Decisions

| Question | Answer |
|---|---|
| Trajectory horizon | Adaptive: osculating period capped at 7 days |
| Line updates during burns | Every frame (warp locked to 1× anyway) |
| Maneuver node system | Full N-body (preview + node marker + CA all forward-integrated) |
| Warp cap feedback | Silent (no HUD indicator when capped) |
| Closest-approach propagation | N-body (`propagate_earth_moon`) |
| Powered N-body code location | New `integrate_powered_nbody` in `core/flight.py` |

## Components

### 1. Core: Powered N-body flight (`core/flight.py`)

New `integrate_powered_nbody` alongside the existing `integrate_powered`. Same
operator-splitting pattern, different gravity model:

- Each substep: exact rocket-equation velocity impulse (unchanged), then **velocity-Verlet
  drift** under `earth_moon_accel(r, t)` instead of RK4 under two-body `_gravity_accel(r, mu)`.
- Same signature as `integrate_powered` plus a `t_s` parameter (epoch, needed for Moon
  position in `earth_moon_accel`). The epoch is the vessel's `state.epoch_s`.
- Returns `(StateVector, float)` — same contract: new state and remaining fuel.
- **Fuel-reaches-zero guarantee preserved:** the rocket-equation impulse step is identical
  to `integrate_powered`; only the gravity drift changes.
- **Substep count:** uses `_earth_moon_substeps` from `core/nbody.py` for proximity-aware
  stepping (replaces the fixed `substeps=50`). Near the Moon the step shrinks automatically.

### 2. Sim: World.step swap (`sim/world.py`)

Minimal change — swap two propagator calls:

- **Coast** (throttle 0 or no fuel): `propagate_earth_moon(state, sim_dt)` replaces
  `propagate_kepler(state, sim_dt)`.
- **Thrust** (throttle > 0 and fuel/unlimited): `integrate_powered_nbody(...)` replaces
  `integrate_powered(...)`.
- Attitude slew, throttle checks, `any_thrusting()` — all unchanged.

### 3. Render: Forward-integrated trajectory line (`render/app.py`)

The Keplerian `sample_orbit_points` → `build_orbit_node` pipeline for the **vessel orbit**
is replaced with forward integration. The Moon orbit ring stays Keplerian (Moon is on rails).

**`_rebuild_orbit` → `_rebuild_trajectory`:**

- Forward-propagates the vessel's current state via `propagate_earth_moon` in steps,
  collecting ~256 position samples.
- **Horizon:** the osculating orbital period (from `osculating_elements`), capped at 7 days
  (`7 * 86400` s). LEO shows ~1 orbit (~90 min); translunar trajectories show multi-day arcs.
- Points baked in render units (`p / scale`) under the existing orbit frame (translate-only
  anchor at `to_render(0)`). Orbit-frame machinery unchanged.
- **Cache invalidation:** keyed on `(r, v, epoch, scale)` with tolerance (position ~100 m,
  velocity ~0.1 m/s). Under N-body, state drifts continuously from Moon perturbation, so
  the old Keplerian-shape key no longer works. At high warp the line rebuilds more often;
  at 1× in LEO it's stable enough to skip most frames.
- **During burns:** rebuilds every frame. Warp is locked to 1× while thrusting, so the
  integration horizon is short and cost is bounded.

**Maneuver preview line (magenta):**

- Compute the post-burn state via `apply_maneuver` (existing — instantaneous dV).
- Forward-integrate that state with `propagate_earth_moon` for the same adaptive horizon.
- Rendered in magenta under the orbit frame, same as today.

**Node marker position:**

- `propagate_kepler(v0.state, ttn)` → `propagate_earth_moon(v0.state, ttn)` to find where
  the vessel will be at the scheduled node epoch.

### 4. Render: Time-warp cap (`render/app.py`)

Per-frame clamp in `_update`, before `clock.advance(real_dt)`:

- Call `max_safe_warp(vessel.state, clock.sim_time_s, SimClock.WARP_STEPS)`.
- Clamp `clock.warp` to the returned value.
- Runs alongside the existing thrust-locks-warp-to-1× rule. Thrust takes priority: if
  thrusting, warp is 1× regardless of the cap.
- `_warp_up_guarded` also checks the cap so `>>` / `.` won't exceed it.
- Silent — no HUD change when the cap is active.

### 5. Render: HUD osculating elements (`render/app.py`)

Replace `state_to_elements` with `osculating_elements` for the HUD readout:

- Currently: `elem = state_to_elements(v0.state)` computes two-body elements about Earth.
- New: `elem = osculating_elements(v0.state, clock.sim_time_s)` — picks Earth or Moon as
  the dominant body based on SOI proximity.
- Pe/Ap/period/inclination drift slowly under perturbation (correct N-body behavior).
- When inside the Moon's SOI, elements are Moon-relative. The altitude computation
  (`r - central.radius_m`) switches to subtract `bodies.MOON.radius_m` instead of
  `world.central.radius_m` when Moon-dominant. Detect this by checking whether
  `osculating_elements` returned Moon-relative elements (its `.mu == MU_MOON`).

### 6. Core: Closest-approach with N-body (`core/rendezvous.py` + `render/app.py`)

`closest_approach` currently propagates internally with `propagate_kepler`. Two changes:

- **`core/rendezvous.py`:** add an optional `propagator` parameter to `closest_approach`
  (default: `propagate_kepler` for backward compat). When called from the render layer,
  pass `propagate_earth_moon`.
- **`render/app.py`:** CA marker positions use `propagate_earth_moon` instead of
  `propagate_kepler` to find the ship at the CA epoch.

## Data flow (per frame, sandbox — after this spec)

```
_update:
  apply_flight_input(real_dt)
  if thrusting: warp = 1×
  else: clamp warp to max_safe_warp(state, t, WARP_STEPS)
  sim_dt = clock.advance(real_dt)
  world.step(sim_dt)          # coast → propagate_earth_moon
                               # thrust → integrate_powered_nbody
  set floating origin on vessel
  _rebuild_trajectory(vessel)  # forward-integrate ~256 pts, adaptive horizon
  rebuild preview if node dV > 0  # forward-integrate post-burn state
  node marker via propagate_earth_moon
  CA via closest_approach(..., propagator=propagate_earth_moon)
  HUD from osculating_elements(state, t)
```

## Files changed

| File | Change |
|---|---|
| `core/flight.py` | Add `integrate_powered_nbody` (operator-split, Verlet + rocket eq) |
| `core/rendezvous.py` | Add `propagator` param to `closest_approach` |
| `sim/world.py` | Swap coast → `propagate_earth_moon`, thrust → `integrate_powered_nbody` |
| `render/app.py` | Forward-integrated trajectory, preview, node marker, warp cap, osculating HUD, N-body CA |
| `tests/core/test_flight.py` | TDD: `integrate_powered_nbody` (dV telescopes, fuel=0, Moon perturbation) |
| `tests/core/test_rendezvous.py` | Test `closest_approach` with N-body propagator |
| `tests/sim/test_world.py` | Verify `World.step` uses N-body (Moon perturbation visible) |

## What stays unchanged

- Moon orbit ring (Keplerian `MOON_ORBIT` — Moon is on rails by design)
- `sample_orbit_points` / `orbit_shape_changed` (still used by Moon ring)
- Maneuver editor UI (jog sliders, execute burn, node time controls)
- Navball, targeting system, save/load, camera rig, floating origin
- Solar system viewer (separate mode, uses ephemeris)
- All existing two-body `core/` functions (analysis layer: transfers, intercept seeds)

## Testing

**Pure (TDD in `core/`):**

- **`integrate_powered_nbody`:** dV telescopes to `ve * ln(m0/mf)` and fuel reaches exactly
  0 (the operator-splitting invariant, now under `earth_moon_accel`). Verify Moon perturbation
  is present: a translunar burn under N-body diverges from the same burn under two-body.
- **`closest_approach` with N-body propagator:** CA result differs from Keplerian CA near
  the Moon (Moon bends the approach).
- **`World.step` N-body:** a vessel in LEO propagated for one period under N-body stays
  close to the Keplerian result (Moon perturbation is tiny at LEO distance); a vessel near
  the Moon diverges measurably.

**Render (headless screenshots):**

- Ship on a translunar trajectory: trajectory line visibly bends near the Moon (curvature
  that two-body would not show).
- Maneuver preview from a translunar-injection burn shows the Moon-bent path.
- Warp silently caps in LEO, frees up in deep space (verify via clock.warp after warp_up).

## Risks

- **Trajectory rebuild cost:** forward integration is heavier than conic sampling. Mitigated
  by the state-change cache (skip rebuild when position/velocity haven't moved beyond
  tolerance) and the 256-point cap. If performance is an issue, throttle rebuilds to N Hz
  (but start with every-frame and measure first).
- **Powered N-body + time-warp:** thrust forces warp to 1× (existing rule), so the powered
  step only ever runs at real-time. Cost is bounded.
- **Osculating element jumps at SOI boundary:** when the dominant body switches Earth↔Moon,
  the HUD values jump (Pe/Ap suddenly Moon-relative). This is correct behavior — KSP does
  the same at SOI transitions. No smoothing needed.

## Out of scope (later cycles)

- Lagrange-point markers/visualization → Cycle 1c.
- Reworking transfers/porkchop for N-body — they stay Keplerian (approximate seeds).
- Sun/planets/ephemeris in the sandbox; landing/surface; multiple vessels.
