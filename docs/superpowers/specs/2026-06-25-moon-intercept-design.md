# Moon Intercept / Target (Phase 6.2, cycle A3) — Design

**Date:** 2026-06-25
**Phase:** 6.2 gameplay (third of the 6.2 cycles).
**Scope:** add a Keplerian Moon to the Earth sandbox, let the player target it, and show a
closest-approach (intercept) prediction — computed on the planned trajectory when an A2 maneuver
node is scheduled, so planning a trans-lunar injection visibly shrinks the predicted approach.

## Decisions (locked in brainstorming)

- **Target = the Moon**, on a fixed two-body **Keplerian** orbit around Earth (offline-safe, no
  DE440 dependency; idealized — no lunar perturbations).
- **Closest approach** between the vessel's trajectory and the Moon's trajectory over a window =
  the vessel's orbital period capped at 14 days; reports separation, time-to-CA, relative speed.
- **Integrates with A2:** closest approach is computed on the predicted post-burn orbit when a node
  is scheduled, else the current orbit.

## Components

### Core — `orbitsim/core/moon.py` (new), pure, TDD

- `MOON_ORBIT: KeplerianElements` — geocentric, idealized: `a = 3.844e8` m, `e = 0.0549`,
  `i = 0.0898` rad (~5.14°), `raan = 0`, `argp = 0`, `nu = 0`, `mu = MU_EARTH`, `epoch_s = 0`.
- `moon_state_at(t_s: float) -> StateVector` — `propagate_kepler(elements_to_state(MOON_ORBIT), t_s)`
  (the Moon orbits Earth, same `mu` as the vessel). `|r| ≈ 3.6e8–4.05e8` m across the orbit.

### Core — `orbitsim/core/rendezvous.py` (new), pure, TDD

- `@dataclass(frozen=True) ClosestApproach: t_ca_s: float; separation_m: float; rel_speed_mps: float`.
- `closest_approach(state_a: StateVector, state_b: StateVector, window_s: float, coarse_samples: int = 720) -> ClosestApproach`
  — propagate both states forward across `[0, window_s]` (both via `propagate_kepler`), evaluate
  separation `|r_a(t) − r_b(t)|` at `coarse_samples+1` uniformly spaced times, take the minimum,
  then refine with a local golden-section / ternary search in the bracketing interval. Return the
  CA time (seconds from now), the separation there, and the relative speed `|v_a − v_b|` at that
  time. Raises `ValueError` for `window_s ≤ 0` or `coarse_samples < 2`.

  Both bodies share `mu = MU_EARTH`; the function does not assume that, it just propagates each
  state with its own `propagate_kepler`.

### Render — `orbitsim/render/app.py` (sandbox)

- **Moon marker + orbit ring:** build a Moon marker (gray sphere, fixed render size like the vessel
  marker) and its orbit polyline (sampled from `MOON_ORBIT`) in `_start_sim` (sandbox branch). Each
  frame, place the marker at `moon_state_at(clock.sim_time_s)` via the floating-origin transform.
- **Target toggle:** a "Target Moon" button (and key, e.g. `g`) sets `self._target_moon = True`; a
  "Clear Target" toggle clears it. Off by default.
- **Closest-approach markers + readout:** when targeted, compute
  `closest_approach(traj_state, moon_state_at(now), window)` where `traj_state` is the predicted
  post-node state if a node is scheduled (`apply_maneuver` to the node), else the current vessel
  state; `window = min(vessel_period_or_14d)`. Draw two markers — one on the vessel's path at the CA
  time (`propagate_kepler(traj_state, t_ca).r`), one at the Moon's CA position
  (`moon_state_at(now + t_ca).r`). Show a target readout: `Target: Moon  CA T-MM:SS  sep N km
  rel N m/s`. Recompute on a throttle (every ~0.5 s of real time or on node/orbit change), not every
  frame, to keep it cheap.
- Camera already zooms across lunar distance; no rig change needed (the player pulls back).

## Architecture / boundaries

- Two new pure core modules (`moon.py`, `rendezvous.py`) — no graphics. Render changes confined to
  the sandbox scene build + update loop + a couple of buttons.
- No change to the vessel physics, the maneuver-execution model, or save/load (the Moon is derived
  from `MOON_ORBIT` + sim time, not persisted; the target flag is render-only UI state).

## Testing

- Core (TDD, `tests/core/`):
  - `moon_state_at`: `|r|` within the Moon's apsis range; periodicity `moon_state_at(t) ≈
    moon_state_at(t + period)` (< 1 km).
  - `closest_approach` known answers: two **concentric circular** coplanar orbits (radii r1, r2) →
    separation ≈ |r1 − r2| (constant), CA somewhere in window; two orbits that **cross** (e.g. a
    circular orbit and an ellipse through it) → separation ≈ 0 near the crossing; `ValueError` on
    `window_s ≤ 0`. A refine test: CA separation ≤ the best coarse-sample separation.
- Render (controller, headless): target the Moon → markers + readout appear and `|sep|` is sane
  (~lunar distance for a LEO vessel); schedule a node raising apoapsis toward the Moon → predicted
  separation drops vs the un-targeted LEO; clear target → markers/readout gone.

## Out of scope

- Real ephemeris Moon; lunar SOI capture / patched-conic handoff to the Moon (the vessel stays
  Earth-centered — this is an *approach* predictor, not a capture); targeting anything other than
  the Moon; rendezvous auto-pilot.

## Definition of done

- `moon.py` + `rendezvous.py` implemented with green TDD tests; full suite green.
- Headless run: Moon marker + orbit visible; targeting shows CA markers + readout; a planned
  apoapsis-raising node shrinks the predicted separation; clearing the target removes the markers.
  No vessel-physics or save/load changes.
