# Orbit-Line Caching + Orbit Frame (Phase 6.3, cycle B) — Design

**Date:** 2026-06-25
**Phase:** 6.3 (performance pass).
**Scope:** stop rebuilding orbit polylines every frame; build orbit geometry in world meters once
and place it under a single per-frame "orbit frame" anchor node. Rebuild the vessel orbit only when
its elements change. This also fixes a latent bug: the A3 Moon orbit ring is built once in render
space and never repositioned, so it goes stale when the floating origin moves.

## Problem

`render/app.py::_rebuild_orbit` runs **every frame per vessel**: it samples 256 points, transforms
each through `RenderTransform.to_render`, and rebuilds a `LineSegs` node. On a coasting vessel the
orbit shape is unchanged frame-to-frame; only the floating origin (recentered on the vessel each
frame) changes — which is why the render-space points differ every frame and the node "must" be
rebuilt. Separately, `_moon_orbit_np` is built **once** with `to_render` at `origin=0` and never
moved, so after frame 1 (origin = vessel.r) it is displaced by `−vessel.r / scale`.

## Approach — an "orbit frame" anchor

`to_render(p) = (p − origin) / scale`. A Panda3D node with `pos = to_render(0)` and
`scale = 1 / scale_m_per_unit`, holding geometry whose vertices are **world-meter** positions `p`,
renders each vertex at `pos + scale·p = −origin/scale + p/scale = (p − origin)/scale = to_render(p)`.
So:

- Add `self._orbit_frame: NodePath` (child of `render`), updated **once per frame**:
  `set_pos(*to_render(zeros))` and `set_scale(1.0 / transform.scale_m_per_unit)`.
- Build all Earth-centered **orbit-line** geometry (vessel orbit, Moon ring, maneuver preview) with
  **world-meter vertices** (the raw `sample_orbit_points` output) and parent it under
  `self._orbit_frame`. No per-vertex `to_render` at build time.
- **Vessel orbit:** cache the orbit elements `(a, e, i, raan, argp)` per vessel; rebuild the polyline
  only when they differ beyond a relative tolerance (`1e-9` on `a`, `1e-9` absolute on the angles/`e`)
  — i.e. after a burn, not on coast.
- **Moon ring:** build once in world meters under the frame (no longer stale — the frame repositions
  it each frame).
- **Maneuver preview:** build under the frame in world meters; still rebuilt while a node's Δv is
  being edited (it depends on live Δv), which is fine — it's only active during planning.

Markers (vessel, Moon, node, closest-approach) stay as single `set_pos(to_render(...))` calls —
cheap and already correct; out of scope. `central_np` (Earth) likewise already repositions each
frame; unchanged.

## Precision

`LineSegs` stores vertices in float32. World-meter vertices up to ~3.8e8 m (Moon) → float32 ULP
~32 m on the orbit *line* (~0.5 m at LEO). Visually irrelevant — the line is a guide, and the focus
(vessel) keeps full float64 precision via the existing marker path. (User-approved tradeoff.)

## Components

`render/orbit_lines.py`:
- `build_orbit_node` already builds a `LineSegs` from a list of `(x,y,z)`. It is reused unchanged —
  callers now pass **world-meter** tuples instead of render-space tuples.

`render/app.py`:
- `_start_sim` (sandbox): create `self._orbit_frame`; build the Moon ring under it in world meters;
  init `self._orbit_elem_cache = [None] * len(vessels)`.
- `_rebuild_orbit(idx, vessel)`: compute elements; if within tolerance of the cached tuple, return
  (no rebuild); else rebuild the polyline (world-meter vertices) under `_orbit_frame`, parent it,
  and update the cache.
- Per-frame update: set `_orbit_frame` pos/scale; move the preview build to world-meter + under the
  frame.
- A small pure helper `_orbit_shape_key(elem) -> tuple` = `(a, e, i, raan, argp)` and
  `_orbit_shape_changed(a, b, tol)` to make the change test unit-testable.

## Testing

- **Math (headless, the project's "test the floating origin by math, not eye" rule):** for a set of
  sample world points and a non-zero origin/scale, a node with `pos=to_render(0)`,
  `scale=1/scale_m_per_unit`, holding world-meter vertices, renders each vertex at exactly
  `to_render(p)` within float32 epsilon. Verify by computing the expected render position and the
  node's `get_pos(render) + get_scale()*vertex` equivalently — or, more simply, assert
  `to_render(0) + p/scale == to_render(p)` (the identity the approach relies on) within 1e-3 render
  units.
- **Caching (pure unit test):** `_orbit_shape_changed` returns False for identical elements, True
  when `a`/`e`/`i`/`raan`/`argp` shifts beyond tolerance.
- **Headless smoke:** after `_start_sim` + a few steps on a coasting vessel, `_rebuild_orbit` does
  NOT rebuild the geometry (cache hit — assert the orbit NodePath identity is unchanged across
  frames); after an executed burn, it DOES rebuild (identity changes). The Moon ring stays correctly
  positioned relative to Earth as the vessel moves (assert its world-frame vertex maps to
  `to_render(moon_orbit_point)` within tolerance across two different origins).
- Full suite green.

## Out of scope

- Markers (already cheap/correct); the solar-viewer planets; the Earth body; any change to physics
  or the camera rig. Frame-rate profiling numbers (the win is structural — O(1) vs O(n) per frame on
  coast — not a micro-benchmark).

## Definition of done

- Vessel orbit rebuilt only on element change; Moon ring no longer stale; preview + Moon ring under
  the orbit frame; math + caching tests green; full suite green; a headless run shows the orbit and
  Moon ring correctly placed as the vessel moves.
