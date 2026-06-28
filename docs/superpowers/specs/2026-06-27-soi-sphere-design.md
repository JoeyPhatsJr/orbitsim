# Moon SOI Sphere — design

**Date:** 2026-06-27
**Status:** approved (brainstorm), pending implementation plan
**Part of:** the graphical-improvements effort (a remaining "new visual element" from the original
list, after ship view and HUD readability 2a/2b — all merged).

## Goal

Draw a faint translucent **wireframe sphere** at the Moon's sphere-of-influence radius in the flyable
sandbox, centered on the moving Moon and at true world scale, so the player can see when their
trajectory crosses into the Moon's gravitational regime. It brightens when the vessel is inside the
SOI (a clear "captured by the Moon" cue) and distance-fades so it stays unobtrusive on the zoomed-out
map.

## Constraints / non-goals

- **Render-only.** No physics changes — reuses `core.nbody.MOON_SOI_M` and the existing inside test.
- **Layering preserved:** all new code in `orbitsim/render/`.
- **Sandbox-only:** guarded like the existing Moon / Lagrange visuals (`if not self.solar_system and
  self.world.vessels`). The `--solar` viewer is out of scope (Moon-only this cycle).
- **Moon only.** No Earth SOI (Earth is the fixed central body in the sandbox; its SOI vs the Sun is
  irrelevant here). No solar-viewer planet SOIs.
- **Wireframe only.** Not a solid translucent shell, not a single equatorial ring.
- **No toggle key** — always on, kept unobtrusive by the distance fade (YAGNI).

## Current code (grounding)

- `core/nbody.py::MOON_SOI_M = 3.844e8 * (MU_MOON / MU_EARTH)**0.4` — the Moon SOI radius (~6.61e7 m).
  `dominant_body(state, t)` uses `‖state.r − moon.r‖ < MOON_SOI_M` to pick the dominant body; the same
  test drives the inside/outside cue here.
- `render/app.py`: the Moon is drawn as a constant-size marker positioned each frame via
  `self._moon_np.set_pos(*self.transform.to_render(moon_now.r))` where
  `moon_now = moon_state_at(self.clock.sim_time_s)`. Lagrange/Moon visuals are built in the sandbox
  branch of `_start_sim` and updated in `_update`.
- `render/geometry.py`: holds `make_uv_sphere(...)`; the natural home for a wireframe-sphere builder.
- `render/world_markers.py::distance_fade(distance_m, near_m, far_m, *, minimum)` — pure smoothstep
  fade, reused here for the camera-distance fade.
- `render/floating_origin.py`: `transform.to_render(world_m)` (translate + uniform scale) and
  `transform.scale_m_per_unit` (meters per render unit).
- `render/ship_model.py` shows the "unit mesh, node-scaled to true size, positioned via to_render"
  pattern; the SOI sphere mirrors it but with a large radius that keeps the node scale safe at map
  zoom (the ship model hides at map zoom precisely to avoid a tiny singular node scale).

## Architecture

### 1. Wireframe sphere geometry — `render/geometry.py`

Add `make_wireframe_sphere(n_lat: int = 9, n_lon: int = 12, color=(...)) -> NodePath`:

- Builds a **unit-radius** sphere as `LineSegs` great circles: `n_lat` latitude rings (constant-z
  circles) and `n_lon` longitude meridians (pole-to-pole half-circles). One `LineSegs` is fine.
- Returns a `NodePath`. The caller sets render attributes (depth test on, depth write off,
  transparency M_alpha, light off) so the sphere never punches through the Moon or trajectory and is
  not lit — matching the orbit-line treatment. (Builder may set these itself; either is acceptable as
  long as the final node has them.)

### 2. Build + wire in `app.py` (sandbox-only)

- In `_start_sim` sandbox branch (near the Moon/Lagrange build): create `self._soi_np =
  make_wireframe_sphere(...)`, reparent to `self.render`, set the render attributes, and a base color.
  Default hidden until first placed.
- In `_update` (sandbox path, where the Moon is positioned): each frame
  - `self._soi_np.set_pos(*self.transform.to_render(moon_now.r))`
  - `self._soi_np.set_scale(MOON_SOI_M / self.transform.scale_m_per_unit)` — render units for the SOI
    radius. Large radius keeps this factor well-behaved at every zoom (no singular tiny scale).
  - inside test: `inside = float(np.linalg.norm(v0.state.r - moon_now.r)) < MOON_SOI_M`.
  - camera-distance fade: `fade = distance_fade(self.rig.distance_m, near, far, minimum=...)` with
    `near`/`far` chosen so the sphere is solid up close and recedes when the map is zoomed far out.
  - color/alpha: a faint cool wireframe (e.g. soft blue-white) at `base_alpha * fade`; when `inside`,
    switch to a brighter/warmer accent and a higher alpha. Apply via `set_color`/`set_alpha_scale`.
  - `self._soi_np.show()`.

### 3. Constants

`MOON_SOI_M` imported from `core.nbody`. Colors and fade near/far as `app.py` module/class constants
(e.g. `SOI_COLOR`, `SOI_INSIDE_COLOR`, `SOI_BASE_ALPHA`). Exact values tuned by screenshot.

## Data flow (per frame, additions to `_update`, sandbox path)

1. `moon_now = moon_state_at(...)` (already computed for the Moon marker).
2. Place + scale `self._soi_np` from `to_render(moon_now.r)` and `MOON_SOI_M / scale_m_per_unit`.
3. Compute `inside` (vessel vs Moon distance) and `fade` (camera distance).
4. Set color/alpha and `show()`.

## Error handling / edge cases

- Solar mode: no Moon/vessel → SOI sphere not built (sandbox-only guard).
- Extreme zoom: scale factor `MOON_SOI_M / scale_m_per_unit` stays finite and non-tiny across the
  supported zoom range (SOI radius ~6.6e7 m ≫ the ship-model case), so no singular-transform warning.
- The sphere is camera-distance-faded but **not** decluttered by `world_markers.declutter_indices`
  (that is for point labels, not a volumetric wireframe) — out of scope.

## Testing

- **Structural (headless, panda available):** `make_wireframe_sphere()` returns a `NodePath` whose
  node has non-empty geometry (latitude rings + meridians present), and has a finite tight bounds
  ≈ unit radius.
- **Visual (offscreen screenshots, per CLAUDE.md):**
  - At a map zoom with the Moon in view, the wireframe sphere is centered on the Moon and reads as a
    boundary, not obscuring the Moon/trajectory.
  - Place the vessel inside the SOI (vessel.r within `MOON_SOI_M` of the Moon) → the sphere shows the
    brighter "inside" color.

## Out of scope (YAGNI / later)

- Earth SOI, solar-viewer planet SOIs.
- Solid translucent shell or equatorial-ring styles.
- A show/hide toggle key.
- A HUD "In Moon SOI" text indicator (the osculating-elements HUD already switches to Moon-relative
  Pe/Ap inside the SOI).
