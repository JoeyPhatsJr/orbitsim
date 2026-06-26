# Target Selection + Planning Readouts — Design

**Date:** 2026-06-26
**Status:** spec (awaiting build)
**Cycle:** 2 of 3 (Δv controls → **target selection** → intercept node)

## Goal

Generalize the hard-wired Moon "Target" toggle into a **current target** the player
sets by **clicking a body**, and have that target drive: a closest-approach +
relative-velocity readout, the (currently dead) **TARGET/ANTITARGET SAS** hold, and
**navball target markers**. Bodies only this cycle (Moon is the only Earth-centered
body today); ships are a later cycle, so the abstraction must accept them without
rework.

## Background (current state)

- `render/app.py` has `self._target_moon: bool`, a hard-wired Moon marker
  (`_moon_np`), orbit ring, and a CA block that calls `moon_state_at(...)`,
  `closest_approach(...)` and already prints `sep … km   rel … m/s` (relative
  velocity is **already computed** — `ClosestApproach.rel_speed_mps`).
- `core/attitude.py` defines `TARGET`/`ANTITARGET` and `sas_target_dir(mode, state,
  target_pos)` (target_pos is the target's **inertial** position; raises `ValueError`
  if absent). But `sim/world.py` calls `sas_target_dir(mode, state)` with **no
  target_pos**, so those modes are dead.
- `render/navball.py::Navball.update(..., target_pos=None)` already routes
  `target_pos` into `sas_target_dir`; it just isn't passed from `app.py`, and
  `TARGET`/`ANTITARGET` aren't in `_MARKER_COLORS`.

## Design

### Target abstraction (`render/targets.py`, new)

A small value object describing a targetable thing, render-agnostic where possible:

```
class Target:
    name: str
    def state_at(self, t_s: float) -> StateVector   # inertial, Earth-centered
```

The Moon target wraps `core.moon.moon_state_at`. The app holds an ordered
`self._targets: list[Target]` (just `[MoonTarget]` now) and `self._target:
Optional[Target]` (the current selection; `None` = no target). Render assets (the body
marker, the orbit ring) stay in `app.py` keyed off the target; this module stays free
of Panda3D so it's unit-testable.

### Screen-space picking (`render/picking.py`, new — pure, tested)

```
def nearest_marker(click_px, markers_px, tol_px) -> Optional[int]
```
Given a click position in pixels and a list of marker pixel positions, return the
index of the nearest marker within `tol_px`, else `None`. Pure function, unit-tested
(hit, miss-beyond-tol, nearest-of-several, empty list). `app.py` computes
`markers_px` by projecting each target's current render position through the camera
lens.

### Interaction (`render/app.py`)

- Left-click is already camera-orbit-on-drag. Add **tap-to-pick**: record the mouse
  position on button-down; on button-up, if the pointer moved less than a small
  threshold (a tap, not a drag), project all target markers to pixels and call
  `nearest_marker`. A hit sets `self._target`; a tap on empty space (no hit) clears
  it. (This avoids stealing the drag gesture and needs no new mouse button.)
- Drop the old `_toggle_target` button/behavior; keep a key (the freed `T`) as
  "clear target" for convenience.
- HUD always shows the current target name (`Target: Moon` / `Target: none`).

### Generalize the CA block (`render/app.py`)

Replace `if self._target_moon:` and the hard-wired `moon_state_at(...)` with
`if self._target is not None:` using `self._target.state_at(base_epoch)` and
`self._target.state_at(self._ca_abs_epoch)`. Readout text uses `self._target.name`.
Behavior (base-epoch alignment, throttled recompute, cached absolute CA epoch) is
unchanged — only the source of the target state changes. The Moon body marker
(`_moon_np`) keeps rendering regardless of selection.

### Working TARGET/ANTITARGET SAS (`sim/world.py`, `render/app.py`)

- Add `sas_target_pos: Optional[np.ndarray] = None` to `Vessel`.
- Each frame **before** `world.step`, `app.py` sets `vessel.sas_target_pos =
  self._target.state_at(now).r` (or `None` if no target).
- `world.step` passes `vessel.sas_target_pos` into `sas_target_dir(...)`. If a
  `ValueError` is raised (no target / coincident), skip the slew for that tick (hold
  attitude) — do not crash.

### Navball target markers (`render/navball.py`, `render/app.py`)

- Add `TARGET` and `ANTITARGET` to `_MARKER_COLORS` (e.g. magenta / dim-magenta) so
  the existing marker loop draws them.
- `app.py` passes `target_pos=self._target.state_at(now).r` (or `None`) to
  `navball.update(...)`. With `None`, `sas_target_dir` raises and the marker hides —
  already handled.

## Components & boundaries

- `targets.py` (pure): what/where a target is over time. No graphics.
- `picking.py` (pure): nearest-marker hit test. No graphics.
- `Vessel.sas_target_pos` (sim): the per-tick target position the SAS hold consumes;
  set by render, read by `world.step`. Keeps `core` pure and the sim layer ignorant
  of selection UI.
- `app.py` (render): selection state, click plumbing, marker/ring assets, wiring the
  target position into world.step + navball + the CA readout.

## Testing

**Pure:**
- `nearest_marker`: hit within tol; miss beyond tol; nearest of several; empty list.
- `MoonTarget.state_at(t)` matches `moon_state_at(t)` (delegation sanity).
- `world.step` with `sas_mode="TARGET"` and a fixed `sas_target_pos`: after several
  ticks the nose direction converges toward `unit(target_pos - r)` (within slew rate).
  With `sas_target_pos=None`, no exception and attitude is unchanged.

**Render (headless):**
- Calling the pick path with a synthetic click on the Moon marker sets `self._target`;
  a click on empty space clears it.
- With a target set, the navball `TARGET` marker is shown (near hemisphere) and the CA
  markers + readout (`Target: Moon … sep … rel …`) populate.

## Dependencies / sequencing

Independent of Cycle 1. **Cycle 3 (intercept node) depends on this** (`self._target`).

## Out of scope

- Multiple vessels / targeting ships (next cycle).
- Target list panel; phase-angle / transfer-window cue (declined).
