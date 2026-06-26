# Restricted N-Body — Flyable Sandbox (Cycle 1b) — Design

**Date:** 2026-06-26
**Status:** spec (awaiting build)
**Part of:** restricted N-body foundation. **Cycle 1b of 3**: 1a core (done) → **1b flyable
(this spec)** → 1c Lagrange-point visualization.

## Why

Cycle 1a built the tested CR3BP integrator. 1b makes the ship actually **fly under
real Moon gravity** in the sandbox: the Moon's pull bends the trajectory, the predicted
path is forward-integrated (no longer a Keplerian ellipse), time-warp caps near bodies,
and the HUD reads osculating orbit elements.

## Key decision: no frame change; Earth fixed + circular Moon + indirect term

Rather than convert the sandbox to a barycentric frame, **keep Earth fixed at the
origin** and the Moon on its existing **geocentric** orbit, **circularized to e = 0**.
The ship feels Earth + Moon gravity in this Earth-centered frame, which requires the
**indirect (third-body) term** because the frame is non-inertial:

```
a(r,t) = −μ_E·r/|r|³  −  μ_M·[ (r − r_M)/|r − r_M|³  +  r_M/|r_M|³ ]
```

where `r_M = moon_state_at(t)` (geocentric, circular). The indirect term `+μ_M·r_M/|r_M|³`
is what makes the model physically correct **and makes the Lagrange points balance** —
without it 1c would be broken.

Consequences (all intended):
- The **gravity Moon and the target Moon are the same body** (one geocentric circular
  Moon used for both pull and targeting) — a consistency win over a barycentric split.
- Existing rendering, central body, maneuver editor, and the targeting system **stay
  put** — no barycentric conversion.
- `core/moon.py` `MOON_ORBIT` eccentricity changes `0.0549 → 0.0`; the Moon's distance
  stops varying (~5%), which slightly simplifies intercept/closest-approach (an
  acceptable, even cleaner, side effect). Tests touching the Moon's eccentric distance
  get updated expected values.
- 1a's *barycentric* `EARTH_MOON` bodies + exact `jacobi_constant`/`lagrange_points`
  become a tested **idealized reference**; the live model reuses 1a's **integrator and
  force-summation** (the hard part). 1c recomputes L-points for this Earth-fixed frame.

## Components

### Core (pure, `orbitsim/core/nbody.py` + `core/moon.py`)
- **Circularize the Moon:** `MOON_ORBIT` e → 0.0 in `core/moon.py`.
- **`earth_moon_accel(r_m, t_s) -> (3,)`** — the force model above (central Earth +
  Moon direct + indirect). Imports `moon_state_at` (core→core).
- **Generalize the integrator:** refactor 1a's `propagate_nbody` so the velocity-Verlet
  core accepts an **acceleration function** `accel_fn(r, t)`. Keep the existing
  `propagate_nbody(state, dt, attractors=EARTH_MOON, ...)` as a thin wrapper (1a tests
  unchanged). Add **`propagate_earth_moon(state, dt_s, max_step_s=...)`** using
  `earth_moon_accel`, with proximity sub-stepping over Earth (origin) and the Moon.
- **`osculating_elements(state, t_s) -> KeplerianElements`** — pick the dominant body
  (Earth vs Moon, by gravitational dominance / SOI), express the state relative to it,
  and return instantaneous Keplerian elements (reuse `state_to_elements`). Drives the
  HUD; values drift slowly under perturbation (correct).
- **`max_safe_warp(state, t_s, warp_steps) -> float`** — pure policy: the largest warp
  whose per-frame sub-step count stays within a budget given proximity to the nearest
  body; snapped to the existing `WARP_STEPS` table.

### Sim (`orbitsim/sim/world.py`)
- `World.step`: **coast** → `propagate_earth_moon`; **powered** → a powered N-body step
  (the project's operator-splitting: exact rocket-equation velocity impulse per sub-step
  + velocity-Verlet drift under `earth_moon_accel`, mass loss via the rocket equation),
  replacing `integrate_powered`'s two-body gravity. Attitude slew unchanged.

### Render (`orbitsim/render/app.py`, controller + headless screenshots)
- **Forward-integrated trajectory line:** replace the ship's Keplerian `sample_orbit_points`
  with a forward integration of `propagate_earth_moon` over a horizon (capped point
  count, e.g. ~256 points across a few days), rebuilt when the ship state changes beyond
  tolerance or a maneuver is edited; rendered through the existing orbit-frame. The
  **maneuver preview** becomes the same forward-integration of the post-burn state.
- **Moon as a gravitating body:** already drawn (target system); no change beyond it now
  being the gravity source.
- **Time-warp cap:** each frame, clamp the clock's warp to `max_safe_warp(...)`; extend
  the existing auto-warp-down to pull warp under the cap as the ship nears a body.
- **HUD:** feed `osculating_elements` into the existing Pe/Ap/period/inclination readout.

## Data flow (per frame, sandbox)

`clock.advance` → clamp warp to `max_safe_warp` → `world.step` (coast/powered under
`earth_moon_accel`) → recentre floating origin on ship → forward-integrate trajectory if
state changed → update Moon/markers → HUD from `osculating_elements`.

## Testing

**Pure (TDD):**
- **Lagrange balance proves the indirect term:** at the analytic L1 and L4 positions for
  the *current* Earth-fixed circular-Moon geometry, the net rotating-frame acceleration
  (`earth_moon_accel` + centrifugal) is ≈ 0 (< 1e-6 m/s²). This directly verifies the
  third-body model and that L-points will work in 1c. (Without the indirect term, this
  fails.)
- **Reduces to two-body when the Moon is far/removed:** with the Moon term suppressed,
  `propagate_earth_moon` matches `propagate_kepler` (< 1 m / quarter LEO orbit).
- **`osculating_elements`:** equals `state_to_elements` for a pure Earth orbit; switches
  dominant body to the Moon when the ship is deep in the Moon's SOI.
- **Powered N-body step:** Δv telescopes to `vₑ·ln(m₀/m_f)` and fuel reaches exactly 0
  (the project's operator-splitting invariant), now under `earth_moon_accel`.
- **`max_safe_warp`:** returns 1× when hugging a body, large in deep space; never exceeds
  the sub-step budget.

**Render (headless screenshots):**
- A ship on a translunar trajectory **visibly bends** near the Moon (path curvature that
  pure two-body would not show).
- The trajectory line is the forward-integrated path; the maneuver preview matches.
- Warp auto-caps near Earth, frees up in deep space.

## Out of scope (later cycles)
- **Lagrange-point markers/parking → 1c.**
- Reworking intercept/transfer/porkchop for N-body — they **stay Keplerian** (approximate
  seeds) this cycle.
- Sun/planets/ephemeris; landing/surface; multiple vessels.

## Risks
- **Powered N-body + time-warp:** thrust forces warp to 1× already (existing rule); the
  powered step only runs at 1×, bounding cost.
- **Trajectory-line cost:** forward integration is heavier than conic sampling; throttle
  rebuilds (only on state/maneuver change) and cap point count, mirroring the orbit-line
  caching already in place.
- **Circularizing the Moon** ripples into `tests/core/test_moon.py` / rendezvous tests —
  update their expected eccentric-distance values; do not loosen unrelated assertions.
