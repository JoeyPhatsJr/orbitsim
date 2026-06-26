# Restricted N-Body Engine — Core (Cycle 1a) — Design

**Date:** 2026-06-26
**Status:** spec (awaiting build)
**Part of:** the restricted N-body foundation. This is **Cycle 1a of 3**:
1a tested propagation core (**this spec**) → 1b sim/render integration (flyable) →
1c Lagrange-point visualization.

## Why

The game is pivoting from a two-body, on-rails sandbox to a **restricted N-body**
model so it becomes a genuine alternative to KSP: real lunar gravity, capture/flyby,
and — the differentiator — **Lagrange points** that exist because the physics is
3-body. The ship is a massless test particle; the massive bodies stay on rails. This
preserves the entire two-body `core/` as the analysis layer (transfers, intercept
seeds) and as the on-rails body motion, while adding a numerically-integrated ship.

Cycle 1a delivers **only the pure, unit-tested propagation core** — no sim/render
changes. It de-risks the hard physics first (the project's pure-core-first pattern).

## Model

Idealized **Circular Restricted 3-Body Problem (CR3BP)**, barycentric inertial frame:

- Earth and Moon are point masses on **circular** orbits about their common
  barycenter. Constants: `μ_E` (existing `MU_EARTH`), new `μ_M = MU_MOON` ≈
  4.9028×10¹² m³/s²; separation `d = 3.844×10⁸ m`; mean motion
  `ω = √((μ_E + μ_M) / d³)`; mass ratio `μ = μ_M / (μ_E + μ_M)` ≈ 0.01215.
- Barycenter at the origin. At `t`, the Moon is at angle `θ = ω·t` (from +X);
  Earth sits opposite. Earth radius from barycenter `= μ·d`, Moon radius `= (1−μ)·d`.
- The ship is a massless test particle integrated under the summed gravity of the
  attractors. **The force model sums over a list of attractors** so the Sun and
  planets append later without a rewrite.

The barycenter origin is invisible in-game (the camera floats on the ship); it is
what makes the Lagrange points exact and stationary in the rotating frame.

## Components — `orbitsim/core/nbody.py` (new)

All SI, float64, frame = barycentric inertial unless noted. No panda3d/sim imports.

- **`EARTH_MOON` body set** — an ordered list of attractor descriptors, each exposing
  `mu` and `state_at(t) -> StateVector` (barycentric). For 1a: idealized circular
  Earth + Moon. (`MoonTarget`/`core.moon` stay geocentric and untouched; this is a
  separate barycentric model.)
- **`gravity_accel(r_m, t_s, attractors=EARTH_MOON) -> np.ndarray (3,)`** — summed
  acceleration: `Σ −μ_i (r − r_i)/|r − r_i|³`, `r_i = attractors[i].state_at(t).r`.
- **`propagate_nbody(state, dt_s, attractors=EARTH_MOON, max_step_s=...) -> StateVector`**
  — advance the ship by `dt_s` using **velocity Verlet** (kick-drift-kick), sub-stepped
  so each internal step ≤ `max_step_s` and ≤ a proximity cap (a fraction of the
  free-fall/orbital timescale at the nearest body). Time-reversible. Returns a new
  `StateVector` at `state.epoch_s + dt_s`.
- **`rotating_frame(r_m, v_mps, t_s) -> (r_rot, v_rot)`** — map an inertial state into
  the frame co-rotating at `ω` (Moon fixed on +X). Used for Jacobi + L-points.
- **`jacobi_constant(state, t_s, attractors=EARTH_MOON) -> float`** —
  `C = 2Ω(r_rot) − |v_rot|²`, where the effective potential
  `Ω = ½ω²(x²+y²) + Σ μ_i / r_i` (rotating-frame distances). Conserved along a coast.
- **`lagrange_points(t_s) -> dict[str, np.ndarray]`** — inertial positions of L1…L5 at
  `t_s`. L4/L5 from exact equilateral geometry; L1/L2/L3 from a 1-D root solve of the
  collinear equilibrium (`scipy.optimize.brentq`), then rotated to inertial by `θ=ωt`.
- **`MU_MOON`** added to `core/constants.py` (value + source comment; astropy lacks a
  direct Moon GM, so a cited IAU/DE value with a comment, consistent with the
  "constants live in constants.py" rule).

## Data flow

`propagate_nbody` is the only stateful-ish entry point and it is pure (state in →
state out). Each internal Verlet step calls `gravity_accel`. `jacobi_constant` and
`lagrange_points` are diagnostic/analysis helpers built on `rotating_frame` and the
same constants. Nothing here mutates global state or touches the sim/render layers.

## Testing — `tests/core/test_nbody.py` (TDD, invariants + known-answers)

1. **Reduces to two-body.** With a single Earth attractor (Moon omitted), a LEO ship
   propagated by `propagate_nbody` matches `propagate_kepler` in position to < 1 m
   over a quarter orbit, and closes to < 10 m over one period. (Anchors the integrator
   to the trusted analytic two-body.)
2. **`gravity_accel` sanity.** Single Earth attractor → `−μ_E r/|r|³` exactly
   (rel < 1e-12). Two attractors → vector sum of the two single-body accelerations.
3. **Jacobi conservation.** A ship coasting in the full Earth–Moon field for 7 days:
   `|C(t) − C(0)| / |C(0)| < 1e-6` (velocity Verlet → bounded drift; do NOT loosen —
   a large drift means the integrator or the rotating-frame Jacobi is wrong).
4. **Reversibility.** `propagate_nbody` forward `+T` then `−T` returns to the start
   position within < 1 m for `T` = several hours.
5. **L4/L5 exact geometry.** `lagrange_points(t)["L4"]` is distance `d` from BOTH Earth
   and Moon (rel < 1e-9) and 60° ahead of the Moon; L5 60° behind.
6. **L1/L2/L3 known positions + equilibrium.** Their rotating-frame x-coordinates match
   the published normalized CR3BP values for μ≈0.01215 (L1 ≈ 0.8369 d, L2 ≈ 1.1557 d,
   L3 ≈ −1.0051 d from the barycenter; tol 1e-3 d), AND the net effective acceleration
   (gravity + centrifugal) in the rotating frame at each point is ≈ 0 (< 1e-6 m/s²).
7. **L4 co-rotation.** A ship placed at L4 with the matching rotating-frame-stationary
   velocity (inertial `v = ω × r`) stays within a small distance of the moving L4 point
   over 1 day (L4/L5 are linearly stable for the Earth–Moon mass ratio). L1/L2/L3 are
   unstable, so they are NOT asserted to stay — only the force-balance test (6) applies.

## Out of scope (this cycle)

- Any sim/render change — no `World.step` rewrite, no trajectory rendering, no HUD,
  no time-warp policy (those are Cycle 1b).
- Sun/planets/ephemeris (the attractor-list design accommodates them later).
- Elliptic Earth–Moon orbit, perturbations beyond the two primaries.
- Osculating-element extraction for the HUD (Cycle 1b).
- Lagrange-point markers/parking (Cycle 1c).

## Risks / notes

- **Step-size policy:** the proximity cap must keep accuracy through a close Earth
  pass without making deep-space steps tiny. Default `max_step_s` plus a cap of a
  small fraction of `2π√(r_near³/μ_near)`; tuned so test 1 (two-body match) and test 3
  (Jacobi) pass. If they fight, the integrator/policy is wrong — not the tolerance.
- **Velocity Verlet + variable step** is only near-symplectic; the step changes slowly
  (proximity-based), so Jacobi drift stays bounded in practice. If test 3 fails at the
  stated tolerance, prefer a fixed small sub-step over loosening the tolerance.
