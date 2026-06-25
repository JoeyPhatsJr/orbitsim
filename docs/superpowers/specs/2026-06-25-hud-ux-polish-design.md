# HUD/UX Polish (Phase 6.2, cycle A1) — Design

**Date:** 2026-06-25
**Phase:** 6.2 (HUD/UX polish), first of three cycles.
**Scope:** four self-contained render-layer pieces. NO `core`/`sim` changes.

This is the first slice of Phase 6.2. Two larger, coupled items the phase doc lists under 6.2 are
deliberately split into their own later cycles, because each needs a subsystem that does not exist
yet:
- **Scheduled maneuver nodes** (the "maneuver list" + "auto-warp-down-before-burn"): the sandbox
  currently executes burns immediately (jog sliders → "Execute Burn" applies the impulse at the
  current epoch). A list of pending future nodes and warping down *toward* a future burn both
  require a schedule-a-node-at-a-future-epoch workflow that isn't built. → its own cycle.
- **Rendezvous / target markers**: needs a target-selection subsystem and a second object to
  target (the sandbox has one vessel orbiting Earth). → its own cycle.

## Goal

Polish the running sim's UX with: a keybind help overlay, an inclination readout, a transient
on-screen "toast" message channel (replacing the save/load cycle's `print` fallback), and a small
settings panel with a km↔mi units toggle.

## Pieces

### 1. Keybind help overlay (`render/keybind_overlay.py`, new)

A `KeybindOverlay` component: a semi-transparent `DirectFrame` panel (anchored center/left) listing
control bindings as rows of "KEY — action". Hidden by default; `F1` toggles visibility. Built for
both modes; the content list is passed in at construction so the sandbox shows flight controls and
the solar viewer shows only the controls that apply there.

- Interface: `KeybindOverlay(parent, lines: list[tuple[str, str]])`; `.toggle()`, `.show()`,
  `.hide()`, `.visible` (bool).
- The app owns the binding lists (sandbox vs solar) and registers `self.accept("f1", overlay.toggle)`.
- Sandbox lines include: right-drag = orbit camera, wheel = zoom, arrows = orbit camera,
  W/S = pitch, A/D = yaw, Q/E = roll, Shift/Ctrl = throttle trim, Z = full throttle, X = cut,
  T = SAS toggle, 1–7 = SAS modes, `,`/`.` = warp down/up, F5/F9 = quicksave/quickload,
  Esc = settings, F1 = this help. Solar lines: right-drag/wheel/arrows camera, `,`/`.` warp, Esc
  settings, F1 help.

### 2. Inclination readout (`render/hud/__init__.py`, modify)

Add inclination to the existing orbit-info panel. `Hud.update` gains an `inclination_rad: float`
keyword; the panel renders `f"Inclination: {degrees:.1f}°"`. The app passes `elem.i` (radians; the
`KeplerianElements.i` field) from the value it already computes in the sandbox update loop.

### 3. HUD toast (`render/hud/__init__.py`, modify)

A transient center-screen message. `Hud` gains a `toast` `OnscreenText` (centered, initially empty)
and a method `flash(text: str, seconds: float = 2.0)` that sets the text and schedules clearing via
`base.taskMgr.doMethodLater` (the `Hud` already holds `base`). Calling `flash` again before expiry
replaces the text and reschedules (cancel the prior delayed task so it can't blank a newer message).

The app's existing `_flash_message(text)` (added in the save/load cycle, currently
`print("[orbitsim] …")`) is reimplemented to call `self.hud.flash(text)`. This makes
"Quicksaved"/"Quickloaded" actually appear on screen — closing the loop the save/load plan left
open.

### 4. Settings panel (`render/settings_panel.py`, new) + units in HUD

A `SettingsPanel` component: a small `DirectFrame` (hidden by default) with a single control — a
units toggle (button cycling km ↔ mi). `Esc` toggles the panel.

- Interface: `SettingsPanel(parent, on_units_change: Callable[[str], None])`; `.toggle()`,
  `.visible`. The toggle button flips between `"km"` and `"mi"` and calls `on_units_change(unit)`.
- The `Hud` gains a `units` attribute (default `"km"`) and a `set_units(unit)` method. Distance
  readouts (altitude, periapsis, apoapsis) and speed render in the selected unit: km, or miles
  (× 0.621371) labelled "mi" / "mi/s". The app wires `on_units_change=self.hud.set_units`.
- `Esc` is not otherwise bound (verified — no existing `escape` handler), so binding it to the
  settings toggle is safe.

## Architecture / boundaries

All four pieces live in `render/`. New small focused files for the two stateful widgets
(`keybind_overlay.py`, `settings_panel.py`); the inclination readout, units handling, and toast
extend the existing `Hud` (still small and cohesive). The app constructs the overlay, settings
panel, and (via `Hud`) the toast alongside the existing HUD, and owns the `F1`/`Esc` keybindings
and the sandbox/solar content lists.

Unit conversion stays at the HUD boundary (SI in, km/mi out) — consistent with the project rule
that conversions happen only at the render/HUD edge.

## Testing

Render-layer tests in the style of the existing `tests/render/` suite (DirectGUI node state, no
pixels needed — construct against an offscreen/headless base):

- `Hud.update(..., inclination_rad=…)` → the orbit panel text contains `"Inclination"` and the
  correct degree value (e.g. `0.5 rad → 28.6°`).
- `Hud.set_units("mi")` then `update(...)` → readouts use the `"mi"` label and the converted value;
  `set_units("km")` restores km.
- `Hud.flash("Quicksaved")` → `toast` text becomes `"Quicksaved"`; a scheduled clear is registered.
- `KeybindOverlay` starts hidden; `.toggle()` shows it; `.toggle()` again hides it; the panel text
  contains a known binding (e.g. `"F5"`).
- `SettingsPanel` starts hidden; `.toggle()` shows it; clicking the units control invokes
  `on_units_change` with the new unit string.

App-level wiring (F1/Esc keys, `_flash_message` → toast, passing `elem.i` into `hud.update`) is
verified by the controller with a headless screenshot/smoke run, not unit-tested.

## Out of scope (explicit)

- Scheduled maneuver nodes / maneuver list / auto-warp-down (own cycle).
- Rendezvous / target / closest-approach markers (own cycle).
- Perturbations on/off toggle (N/A — single-body patched conics).
- Any change to physics, the camera rig, or the maneuver-execution model.

## Definition of done

- The four pieces implemented; render tests green; full suite green.
- F1 toggles the keybind overlay; Esc toggles settings; units toggle re-labels HUD distances;
  quicksave/quickload show an on-screen toast; inclination appears in the orbit panel — all
  confirmed by a headless smoke run.
- No `core`/`sim` files modified.
