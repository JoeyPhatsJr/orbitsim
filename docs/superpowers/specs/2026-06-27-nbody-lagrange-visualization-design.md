# N-Body — Lagrange-Point Visualization (Cycle 1c) — Design

**Date:** 2026-06-27
**Status:** spec (awaiting build)
**Part of:** restricted N-body foundation. **Cycle 1c of 3** (final): 1a engine core ✅ →
1b flyable render integration ✅ → **1c Lagrange-point visualization (this spec)**.

## Why

1b made the ship fly under `earth_moon_accel` (Earth fixed + circular Moon + indirect term).
1c surfaces the payoff that motivated the whole N-body pivot: the **Earth-Moon Lagrange points**
become visible in the live sandbox and selectable as navigation targets, so you can fly to and
park near L1/L2/L4/L5 — something patched conics could never show.

## Key constraint: recompute the L-points for the Earth-fixed frame

`core/nbody.py` already has `lagrange_points(t_s)`, but it is built on the **1a barycentric**
model (`EARTH_X`, `MOON_X`, centrifugal about the barycenter). The live sandbox uses the
**Earth-fixed** model: Earth at the origin, the Moon on a circular geocentric orbit at `D_EM`,
the ship feeling `earth_moon_accel` (which carries the indirect term `−μ_M·r_M/|r_M|³`). The
equilibria of that model differ from the barycentric ones, so 1c computes a **new** set of
L-points consistent with `earth_moon_accel`. The barycentric `lagrange_points` stays as the 1a
tested reference (not used by render).

The 1b L4-balance test already proved the Earth-fixed equilateral point: the point at distance
`D_EM` from Earth, 60° ahead of the Moon **in the Moon's actual (inclined) orbital plane**, has
`earth_moon_accel + centrifugal-about-the-origin ≈ 0` to `< 1e-7 m/s²` (the test rotated the live
Moon position about its real orbit normal, not the z-axis). 1c generalizes this to all five
points and uses the same real-geometry approach.

## Scope (decided)

- **In:** Earth-fixed L1–L5 computation; 5 render markers + labels tracking the rotating
  Earth-Moon line; lightweight target selection (navball TARGET/ANTITARGET + live distance and
  relative-speed readout).
- **Out (deferred):** Jacobi-constant HUD readout; zero-velocity (Hill-region) curves;
  closest-approach time-prediction for L-points; station-keeping / auto-park.

## Components

### 1. Core: Earth-fixed Lagrange points (`core/nbody.py`, pure, TDD)

`earth_fixed_lagrange_points(t_s) -> dict[str, np.ndarray]` — inertial positions of
`"L1".."L5"` (each a float64 `(3,)`) in the Earth-centered inertial frame at `t_s`.

The points are computed relative to the Moon's **actual instantaneous geometry**, not a fixed
z=0 plane: the live Moon (`moon_state_at`, `MOON_ORBIT`) is inclined `i≈0.0898`, so it does not
orbit in z=0. Using the real Moon position and orbit normal keeps the points in the Moon's plane
and makes them exact equilibria of `earth_moon_accel` at every `t_s` (the z-rotation the
barycentric `lagrange_points` uses would not). Concretely:

1. `m = moon_state_at(t_s)`; `rM = m.r`; `d = |rM|` (≈ `D_EM`); `u = rM / d` (Earth→Moon unit).
2. Orbit normal `n_hat = cross(rM, m.v) / |cross(rM, m.v)|`; angular rate is `OMEGA_EM` (circular
   Moon — `|cross(rM, m.v)| / d² == OMEGA_EM`).
- **Collinear L1/L2/L3:** parameterize a point on the Earth-Moon line as `p(s) = s·u` (signed
  distance `s` from Earth along `u`) and solve `net_line(s) = 0` with `scipy.optimize.brentq` in
  three brackets — between the bodies (`eps .. d−eps`), beyond the Moon (`d+eps .. d+0.4·d`), and
  beyond Earth (`−1.6·d .. −eps`). The net along-line acceleration is:

  ```
  net_line(s) = ( earth_moon_accel(s·u, t_s) + OMEGA_EM²·(s·u) ) · u
  ```

  i.e. the live force model plus the centrifugal term `OMEGA_EM²·p` (about the origin), projected
  onto the line. `earth_moon_accel` already carries the indirect term `−μ_M·rM/d³`, so it is not
  re-derived here.
- **Equilateral L4/L5:** rotate `rM` by ±60° about `n_hat` (Rodrigues' formula) — the exact
  equilateral points at distance `d` from both Earth and Moon.
- Return the five inertial `(3,)` positions directly (they are already in the inertial frame —
  no extra rotation step, since `rM`/`u`/`n_hat` are taken from the live Moon state at `t_s`).

`OMEGA_EM`, `D_EM`, `MU_EARTH`, `MU_MOON`, `earth_moon_accel`, `moon_state_at` already exist in
the module (`moon_state_at` is imported).

### 2. Targets (`render/targets.py`, pure)

- `MoonTarget` gains a class attribute `supports_closest_approach = True`.
- New `LagrangePointTarget`:
  - Constructed with a display `name` (e.g. `"L1"`) and `point_id` (the dict key).
  - `supports_closest_approach = False`.
  - `state_at(t_s) -> StateVector`: `r = earth_fixed_lagrange_points(t_s)[point_id]`; the point's
    inertial velocity is its rigid-rotation velocity `v = cross(OMEGA_EM · n_hat, r)`, where
    `n_hat` is the Moon's orbit normal at `t_s` (`m = moon_state_at(t_s)`;
    `n_hat = cross(m.r, m.v)/|cross(m.r, m.v)|`) — the same normal the points rotate about, so the
    velocity is consistent with the inclined geometry, not a z-axis approximation. `mu = MU_EARTH`,
    `epoch_s = t_s`.

### 3. Render: markers + labels (`render/app.py`, sandbox only)

- In `_start_sim` (the non-solar branch): create 5 constant-on-screen-size marker spheres
  (`make_uv_sphere`, fixed render scale, `set_light_off`, a distinct teal color — not the Moon
  gray, CA orange, or node cyan) and 5 billboard `TextNode` labels `"L1".."L5"`, reusing the
  planet-label pattern (`set_billboard_point_eye`, `set_light_off`). Store them in
  `self._lagrange_nps` / `self._lagrange_labels`.
- Each frame in `_update`: `lps = earth_fixed_lagrange_points(self.clock.sim_time_s)`; for each
  name place the marker at `to_render(lps[name])` and its label just above. They rotate with the
  Moon automatically (the positions are recomputed from `sim_time` each frame).

### 4. Render: target selection + readout (`render/app.py`)

- Append the 5 `LagrangePointTarget`s to `self._targets` after `MoonTarget`, so the existing
  click-to-pick (`_try_pick_target`, which projects each target's `state_at(now).r` to pixels)
  selects them with no new picking code.
- The existing per-frame `sas_target_pos = self._target.state_at(now).r` wiring already makes the
  navball TARGET/ANTITARGET markers and SAS hold work for any selected target.
- Branch the target readout block on `self._target.supports_closest_approach`:
  - `True` (Moon) → the existing closest-approach path (markers, sep, rel-vel, countdown),
    unchanged.
  - `False` (L-point) → a **live** readout: current distance `|ship.r − L.r|` and relative speed
    `|ship.v − L.v|` (where `L = self._target.state_at(now)`), no time-forward prediction. Reuse
    `_target_text`. Skip the CA recompute/markers entirely for L-point targets.

## Data flow (per frame, sandbox)

```
... existing _update ...
lps = earth_fixed_lagrange_points(sim_time)
for name in L1..L5: place marker + label at to_render(lps[name])
if target is an L-point:
    L = target.state_at(now); show distance |ship.r-L.r|, rel-speed |ship.v-L.v|
else (Moon): existing closest-approach path
```

## Testing

**Pure (TDD, `tests/core/test_nbody.py`):**
- **Equilibrium:** for each of L1–L5, the net rotating-frame acceleration
  `earth_moon_accel(r, t) − ω×(ω×r)` (with `ω = OMEGA_EM · n_hat`, `n_hat` the Moon's orbit
  normal at `t`) has magnitude `< 1e-6 m/s²`, at `t_s = 0` and at a nonzero `t_s` (proves the
  geometry tracks the real inclined Moon). Do **not** loosen this bound — a failure means the
  collinear solve, the indirect term, or the orbit-normal handling is wrong.
- **L4/L5 are equilateral:** `|L4| ≈ d` and `|L4 − rM| ≈ d` (to `< 1 m`), and the angle between
  `L4` and `rM` is `60°` (`L5` the mirror, on the opposite side of the Earth-Moon line). These
  are frame-independent and hold under the Moon's inclination (so L4/L5 carry a small z-component,
  not the z=0 literals).
- **Collinear ordering / placement:** the L1/L2/L3 signed distances along `u = rM/|rM|` satisfy
  `0 < s(L1) < d < s(L2)` and `s(L3) < 0` (beyond Earth); each collinear point is on the
  Earth-Moon line (`|r × u| < 1 m`).
- **Distance invariance under rotation:** `|L_i(t)|` is constant across several `t_s` (rigid
  rotation about the Moon's normal), `< 1 m` variation.

**Render (headless, `loadPrcFileData offscreen`):**
- 5 markers and 5 labels build in the sandbox; none in solar mode.
- After a `taskMgr.step()`, each L-marker sits at `to_render(earth_fixed_lagrange_points(t)[name])`.
- Selecting an L-point (set `_target`) drives `sas_target_pos` and yields a finite distance +
  rel-speed readout, and does **not** enter the CA recompute branch (`_ca` stays None).

## Risks / notes

- **L1/L2/L3 are unstable** equilibria; the markers are static display points, so instability is
  irrelevant to rendering. (We don't propagate a ship *at* an L-point in any test beyond the
  instantaneous acceleration-null check.)
- **Two `lagrange_points` functions coexist** (barycentric reference + Earth-fixed live). Name the
  new one clearly (`earth_fixed_lagrange_points`) and leave the old one untouched; the 1a tests
  that cover the barycentric version stay green.
- **Marker on-screen size:** use a constant render-space scale (the `CameraRig` sits a fixed
  render distance from the focus), not a tiny world-space radius — same gotcha the Moon/planet
  markers already handle.
