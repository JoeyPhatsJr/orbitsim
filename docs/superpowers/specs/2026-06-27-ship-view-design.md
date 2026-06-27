# Ship View — design

**Date:** 2026-06-27
**Status:** approved (brainstorm), pending implementation plan
**Sub-project of:** the "graphical improvements" effort. Decomposed into two cycles —
**(1) Ship view + camera mode-switching** (this spec) and **(2) HUD/readability pass** (later).

## Goal

Add a 3rd-person **ship view** to the sandbox. Today the renderer is effectively a *map mode*:
the focused vessel is a constant-size fullbright marker and the camera orbits it at a distance.
Ship view lets the player zoom (or snap) all the way in to see an actual **oriented 3D ship model**
against the real Earth/Moon, with attitude and roll visible. It is an additive, render-only feature.

## Constraints / non-goals

- **Render-only.** Touches no `core/` or `sim/` physics. Vessels remain point masses.
- **Layering preserved:** all code lives in `orbitsim/render/`.
- **Stylized placeholder ship**, procedural (built from primitives in code). **No** loaded mesh /
  asset pipeline — explicitly out of scope, swappable later.
- **Free-orbit camera** in ship view (reuse the existing azimuth/elevation rig). No chase cam.
- The thrust **plume is a stretch goal**, built only after the ship itself works.

## Key facts about the current code (grounding)

- The vessel marker is built in `render/app.py::_start_sim` (~L200): `make_uv_sphere(1.0, 8, 12)`,
  fullbright (`set_light_off`), fixed `set_scale(8.0)` so it holds a constant on-screen size at every
  zoom. Positioned each frame via `transform.to_render(vessel.state.r)` (`_update`, ~L988).
- `vessel.state.orientation` is a `[w, x, y, z]` unit quaternion; **nose = body +Z**
  (`core/attitude.py::nose_direction`). Currently consumed only by the navball.
- The **floating origin** (`render/floating_origin.py`, `RenderTransform.to_render`) is a pure
  **translate + uniform scale** — no rotation. Therefore **world orientation = render orientation**;
  the vessel quaternion can be handed straight to the model node via `set_quat`.
- `CameraRig` (`render/camera_rig.py`) orbits the render origin (= focused vessel) at fixed render
  distance; `scale_m_per_unit = distance_m / 1000` encodes zoom. `MIN_DISTANCE_M = 10.0`.
- A directional **sun light** already exists (`_sun_light_np`).
- The orbit-frame code notes Panda flags a **tiny node scale as singular** — relevant below.
- Free keybind confirmed: **`m`** is unused in the sandbox keymap.

## Architecture

New module **`orbitsim/render/ship_model.py`**, following `camera_rig.py`'s pattern: **stdlib-only
top-level imports**; all Panda3D imports happen *inside* functions, so the pure helpers are
importable and unit-testable without graphics installed.

It exposes:

- `view_blend(distance_m) -> (marker_alpha, model_alpha)` — pure, unit-tested (see below).
- `model_node_scale(scale_m_per_unit) -> float` — pure helper returning the node scale that renders
  a meters-built mesh true-size (`1 / scale_m_per_unit`). Unit-tested.
- `build_ship_model() -> NodePath` — procedural, lit, nose along **+Z**.
- thresholds/constants: `SHIP_VIEW_NEAR_M`, `SHIP_VIEW_FAR_M`.

`render/app.py` owns the per-frame wiring (positioning, scaling, orientation, blend, key binding).

## Components

### Dual representation, blended by zoom

Keep the existing constant-size **marker** unchanged (correct for map mode — a true-scale ~10 m ship
is sub-pixel from orbit). Add a true-scale **lit ship model**. A pure function selects visibility:

```
view_blend(distance_m):
    distance >= SHIP_VIEW_FAR_M (~5 km):   (marker=1, model=0)   # map only
    distance <= SHIP_VIEW_NEAR_M (~200 m): (marker=0, model=1)   # ship only
    between:                                linear cross-fade
```

Each frame `app._update` applies the two alphas (via `set_alpha_scale` / transparency) to the marker
and model nodes.

- **Marker:** keeps fixed render-size; only its alpha changes.
- **Model:** parented to a node placed at `to_render(vessel.state.r)`; node scale set to
  `model_node_scale(transform.scale_m_per_unit)` so the meters-built mesh renders true-size; node
  orientation set via `set_quat(LQuaternion(w, x, y, z))` from `vessel.state.orientation`. The model
  is **only active while zoomed in** (model_alpha > 0), where the node scale is large and safe —
  this sidesteps the tiny-node-scale singular warning, since at map zoom the model is hidden.

### The procedural ship (`build_ship_model`)

A stylized but unmistakable vessel from primitives — cone nose + cylinder body + a few fins — built
with **nose along +Z** to match `nose_direction`. Uses scene **lighting** (not fullbright) so its
shape/shading read, and a couple of fins so **roll is visible**. Sized ~10 m. Swappable for a real
mesh later.

**Stretch — thrust plume:** a translucent additive cone off the **−Z** tail, shown when
`vessel.throttle > 0`, length/intensity scaling with throttle. Built only after the ship works.

### Camera & mode switching

The orbit rig already orbits the focused vessel, so ship view is largely "zoom in close." Tune
`MIN_DISTANCE_M` / blend thresholds so the model appears at a sensible on-screen size as you zoom in,
and so the cross-fade window feels natural (continuous-zoom entry).

Add an **`m` toggle**: stores a "map" camera distance and a "ship" camera distance and flips
`rig.distance_m` between them (each remembers its framing). Free-orbit (mouse drag / arrow keys) works
in both states. Bound in the sandbox keymap alongside the other `accept(...)` calls.

## Data flow (per frame, additions to `_update`)

1. Existing: `to_render(vessel.state.r)` → marker position.
2. New: place model node at the same `to_render(r)`; set node scale =
   `model_node_scale(scale_m_per_unit)`; set node quat from `vessel.state.orientation`.
3. New: `(marker_a, model_a) = view_blend(rig.distance_m)`; apply alphas; skip model scale/orient work
   when `model_a == 0`.
4. Stretch: toggle/scale the plume from `vessel.throttle`.

## Error handling / edge cases

- Model active only when visible → no singular tiny-scale transform.
- Solar-system mode has no flyable vessel: ship view is **sandbox-only**; guard like the existing
  `if not self.solar_system and self.world.vessels:` blocks.
- Degenerate/zero quaternion shouldn't occur (orientation is normalized each tick), but `set_quat`
  with a normalized quat is safe regardless.

## Testing

Per project convention: pure helpers get unit tests; visuals get headless screenshots.

- **Unit (no graphics):**
  - `view_blend`: returns map-only beyond `FAR`, ship-only within `NEAR`, monotonic cross-fade
    between, alphas in `[0,1]` and summing sensibly at the endpoints.
  - `model_node_scale`: `1/scale` round-trips a known meters→render-units size.
- **Visual (headless offscreen screenshot, per CLAUDE.md):**
  - Ship reads as a ship; **nose points along the SAS/prograde direction**; roll visible via fins.
  - Cross-fade across the zoom range (marker→ship) looks right; no pop.
  - `m` toggle snaps between map and ship framing.
  - (If built) plume appears under throttle, off the tail.

## Out of scope (YAGNI)

- Loaded 3D mesh / asset pipeline.
- Chase cam (free-orbit only).
- HUD/readability changes (the next cycle).
