# Continuous-Thrust Flight Model + Controls — Design

**Status:** approved (design phase). Next: implementation plan via writing-plans.
**Date:** 2026-06-24

## Goal

Turn the simulator from an analytic planning tool into a *flyable* game: the player
controls a real rocket in real time — point it, throttle up, watch fuel drain and the
trajectory bend under gravity — while keeping the textbook-accurate analytic engine
(Phases 1–5) intact for coasting, time-warp, and mission planning.

This design covers **sub-project #1 of the "playable game" effort**: the continuous-thrust
flight model **and** its controls/HUD (a fully flyable experience). Missions/objectives,
save/load, and packaging are explicitly out of scope here and will be separate
design→plan→build cycles.

## Decisions (locked during brainstorming)

1. **Flight model:** continuous-thrust free flight (not impulsive). Thrust applies over real
   time; the trajectory integrates numerically under gravity.
2. **Fuel/mass:** real rocket equation (Tsiolkovsky). Vessel has dry mass, fuel mass, max
   thrust, and exhaust velocity. Acceleration = throttle·thrust / current mass; fuel drains
   as throttle·thrust / vₑ. ΔV = vₑ·ln(m₀/m_f) emerges naturally.
3. **Steering:** rotational dynamics. The ship has an orientation and turns toward the
   commanded direction at a max turn rate; thrust fires along the nose.
4. **Navball:** full 3D navball instrument.
5. **Coast/burn architecture:** **Hybrid** — analytic Kepler rails when coasting (time-warp +
   all planning tools intact); real-time RK4 numeric integration only while thrusting, with
   time-warp forced to 1×.

## Architecture (respects the project's strict layering)

```
core/flight.py   NEW. Pure float64/SI functions: rocket equation, powered RK4 integrator,
                 attitude slew, SAS direction vectors. NEVER imports sim/render/panda3d.
sim/world.py     MODIFIED. Vessel gains propulsion + attitude state. World.step branches
                 coast (analytic Kepler) vs burn (numeric). Attitude always slews.
sim/clock.py     MODIFIED (minimal). Warp is forced to 1× while the focused vessel thrusts.
render/app.py    MODIFIED. Flight keybinds, warp-lock enforcement, HUD additions.
render/navball.py NEW. The 3D navball instrument (own 2D-overlay display region).
__main__.py      MODIFIED. Title slider repurposed to fuel load; default propulsion stats.
```

## 1. Vessel data model (sim layer)

`Vessel` (the mutable sim-layer exception) gains, all SI:

| Field | Meaning |
|---|---|
| `dry_mass_kg` | structural + payload mass [kg] |
| `fuel_mass_kg` | remaining propellant [kg] (mutable; drains while thrusting) |
| `max_thrust_n` | engine thrust at full throttle [N] |
| `exhaust_velocity_mps` | vₑ = Isp·g₀ [m/s] |
| `orientation` | unit quaternion, full 3-DOF (so the navball shows pitch/yaw/roll) |
| `max_turn_rate_radps` | reaction-wheel slew rate [rad/s] |
| `throttle` | commanded throttle 0–1 |
| `sas_mode` | OFF / STABILITY / PROGRADE / RETROGRADE / NORMAL / ANTINORMAL / RADIAL_IN / RADIAL_OUT / TARGET |

Existing fields (`name`, `state`, `delta_v_budget_mps`, `nodes`) stay. `delta_v_budget_mps`
becomes a **derived display value** computed from the rocket equation rather than the source
of truth (kept for backward compatibility / HUD; may be deprecated later).

Derived (computed on demand, not stored): `mass_kg = dry + fuel`,
`delta_v_remaining = vₑ·ln(m₀/m_f)`, `twr = max_thrust / (mass·g_local)` where
`g_local = μ / r²`.

The ship's **nose direction** (thrust axis) is `orientation` applied to a body +Z (or +X)
basis vector; only this axis affects dynamics. Roll is controllable but cosmetic.

## 2. Core physics — `core/flight.py` (pure, TDD-first)

All functions float64/SI, raise `ValueError` on invalid input, no graphics imports.

- `tsiolkovsky_dv(ve_mps, m0_kg, mf_kg) -> float` — vₑ·ln(m₀/m_f).
  *Known-answer test:* vₑ=3000, m₀=2000, m_f=1000 ⇒ ΔV = 3000·ln2 ≈ 2079.44 m/s.
- `fuel_burned_kg(throttle, max_thrust_n, ve_mps, dt_s) -> float` — = throttle·thrust·dt / vₑ
  (mass flow ṁ = thrust/vₑ). Clamped so it never exceeds available fuel (caller passes remaining).
- `thrust_accel_mps2(throttle, max_thrust_n, mass_kg) -> float` — throttle·thrust / mass.
- `integrate_powered(state, mass_kg, fuel_kg, thrust_dir_unit, throttle, max_thrust_n, ve_mps, mu, dt_s, substeps) -> (StateVector, fuel_kg)` —
  fixed-substep **RK4** integrating r, v under two-body gravity `−μ r/|r|³` plus thrust
  `(throttle·thrust/mass)·thrust_dir`, with mass decreasing each substep as fuel burns; stops
  thrust when fuel hits zero. Returns the new state and remaining fuel.
  *Tests:* (a) **zero throttle** over a short dt matches `propagate_kepler` to < ~1 mm / energy
  conserved (the integrator reduces to pure gravity); (b) **constant burn in free space**
  (μ=0) matches the analytic rocket equation: final speed = vₑ·ln(m₀/m_f) along thrust_dir,
  straight-line position integral.
- `slew_attitude(orientation_q, target_dir_unit, max_turn_rate_radps, dt_s) -> quaternion` —
  rotate the nose toward `target_dir` by at most `max_rate·dt`; **never overshoot** (clamp to
  the remaining angle). *Tests:* reaches an exactly-opposite target in ≈ π/rate seconds;
  single small step never exceeds the angular gap.
- SAS target-direction helpers (pure, from a `StateVector` + optional target position):
  `sas_target_dir(mode, state, target_pos=None) -> unit vector` for prograde/retrograde/
  normal/antinormal/radial-in/radial-out/target. *Tests:* prograde == v̂, normal == ĥ, etc.

Gravity during a burn uses the **current central body only** (two-body). Mid-burn SOI change
is out of scope (see §7).

## 3. Sim layer — `World.step`

Each tick, for each vessel:
1. **Attitude always slews** toward its commanded direction (from `sas_mode`, or from manual
   pitch/yaw/roll input captured as a commanded quaternion) at `max_turn_rate_radps`.
   Reaction wheels work whether coasting or burning.
2. **Translation:**
   - If `throttle > 0` and `fuel_mass_kg > 0`: call `integrate_powered` for `sim_dt`
     (sim_dt == real_dt here because warp is locked to 1× while thrusting). Update
     `state` and `fuel_mass_kg`.
   - Else: existing analytic `propagate_kepler` (on rails).
3. On the **burn→coast transition** (throttle returns to 0), nothing special is needed —
   `state` already holds the post-burn r, v, and the next coast tick analytically propagates
   from there; the orbit line/prediction recompute from the new elements automatically.

**Warp lock:** while the focused vessel has `throttle > 0`, the clock is held at 1×
(`render/app.py` enforces by not advancing warp and snapping it down; `SimClock` exposes the
current warp and the app refuses warp-up while thrusting). A HUD indicator shows the lock.

## 4. Render — controls, navball, HUD

### Keybinds (added in `render/app.py::_setup_input`)
| Key | Action |
|---|---|
| `Z` / `X` | throttle full / cut |
| `Shift` / `Ctrl` | throttle up / down (increments) |
| `W` / `S` | pitch down / up |
| `A` / `D` | yaw left / right |
| `Q` / `E` | roll left / right |
| `T` | toggle SAS (stability hold) |
| `1`–`7` | SAS hold modes: prograde, retrograde, normal, antinormal, radial-in, radial-out, target |

Manual `W/A/S/D/Q/E` input sets a commanded attitude rate; SAS modes override with a computed
target direction. (Existing camera arrows/zoom/warp/`p`/maneuver keys are preserved; if a key
collides, flight keys win only in sandbox mode.)

### Mouse camera control
**Right-click + drag** orbits the camera around the focus: holding the right mouse button and
moving the mouse changes the rig's azimuth/elevation proportionally to the drag delta; the
scroll wheel still zooms. Implementation: track right-button down/up (`mouse3`), and each
frame read the pointer delta from `base.mouseWatcherNode` and feed it (scaled, with a
sensitivity constant) to `CameraRig.orbit(d_azimuth, d_elevation)`. The default Panda mouse
camera control stays disabled (`disable_mouse()`), so this is the only mouse-look path. Arrow
keys remain as a secondary/no-mouse fallback. Right-drag must not conflict with **left**-click,
which keeps driving the DirectGUI controls (navball is display-only; throttle/SAS are keys).

### Navball — `render/navball.py`
A true 3D sphere instrument rendered into its own bottom-center 2D-overlay display region:
- A textured/colored UV sphere whose rotation = the orbital-frame attitude of the ship
  (so "up" on the ball is radial-out / the orbital reference, KSP-style).
- Overlaid markers fixed in the orbital frame: **prograde/retrograde** (green),
  **normal/antinormal** (purple), **radial-in/out** (cyan), **target/anti-target** (pink),
  plus a fixed **nose reticle** at ball center.
- Implementation: a small sphere parented to a dedicated camera/`DisplayRegion` in the
  lower-center; markers are billboarded nodes positioned on the ball surface by their
  orbital-frame unit vectors, hidden when on the far hemisphere.
- Verified headlessly (offscreen render + screenshot), since it is visual.

### HUD additions (`render/hud`)
Throttle bar (0–100%), fuel remaining (% and kg), current mass, thrust (N), **TWR**,
ΔV remaining (rocket equation), and a "⚠ WARP LOCKED — thrusting" banner when burning.

## 5. Title screen (`render/app.py` + `__main__.py`)

The existing title slider changes from "ΔV budget" to **fuel load (kg)** (range chosen so the
derived ΔV spans a useful band). The ΔV readout becomes the *derived* value shown live as the
slider moves. Defaults for dry mass, max thrust, and vₑ are set so the default ship is
immediately flyable (a sensible TWR > 1 near the start orbit is not required — orbital craft
often have TWR < 1; defaults give ~hundreds of m/s to a few km/s of ΔV).

## 6. Testing strategy

- **TDD-first** for every `core/flight.py` function (known-answer numbers above), per the
  project's mandatory workflow. Plus property/invariant tests: zero-thrust integration
  conserves energy & matches Kepler; powered free-space burn matches the rocket equation;
  attitude slew monotonically approaches the target without overshoot.
- Sim-layer: `World.step` coast path unchanged (existing tests stay green); new tests for the
  burn path (fuel drains, mass drops, speed rises along nose) and attitude slewing.
- Render: navball + HUD wired and exercised headlessly (offscreen `window-type offscreen`,
  `taskMgr.step()`, screenshot) as in Phases 2–5.

## 7. Scope boundaries (YAGNI)

**In:** one player-controllable vessel (vessel 0); two-body gravity during burns; full 3-DOF
attitude with roll controllable but dynamically irrelevant; SAS hold modes; navball + flight
HUD; warp lock while thrusting.

**Out (deferred or per project scope):** staging, multiple engines/part-building, atmosphere
& aerodynamic drag, docking, N-body perturbations during flight, and **mid-burn SOI change**
(the central body is fixed for the duration of a burn and re-evaluated when coasting via the
existing patched-conics path). Missions/objectives, save/load, and PyInstaller packaging are
separate later sub-projects.

## Open implementation notes (not blockers)

- Quaternion math: add a tiny pure helper set in `core/` (or reuse numpy) for slerp/axis-angle;
  keep it float64 and tested. Panda3D's `Quat` is render-only and must not leak into core.
- The navball's orbital reference frame must match the physics frame (J2000/ICRF) re-expressed
  with radial/prograde basis; define it once and share between HUD markers and navball.
```
