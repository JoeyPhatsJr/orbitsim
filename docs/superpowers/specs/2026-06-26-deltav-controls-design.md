# Δv Controls (bigger cap + unlimited toggle) — Design

**Date:** 2026-06-26
**Status:** spec (awaiting build)
**Cycle:** 1 of 3 (Δv controls → target selection → intercept node)

## Goal

Let the player carry much more Δv, and toggle **unlimited Δv** on/off — from the
title screen, the Esc settings panel, and a keybind, flippable mid-flight.

## Background

Δv has no stored budget: `Vessel.delta_v_remaining` is derived from fuel via the
rocket equation (`tsiolkovsky_dv`), and fuel mass is the single source of truth
(`sim/world.py`). The title screen's fuel slider caps at **4000 kg**. So "unlimited
Δv" is implemented by special-casing the fuel→Δv derivation and fuel depletion, not
by adding a budget field.

## Design

### Sim layer (`sim/world.py`)

- Add field `unlimited_dv: bool = False` to `Vessel`.
- `delta_v_remaining`: return `float("inf")` when `unlimited_dv` is set (before the
  fuel check).
- `World.step`: a vessel thrusts when `throttle > 0 and (fuel_mass_kg > 0 or
  unlimited_dv)`. When `unlimited_dv`, integrate powered flight as today **but do not
  persist fuel depletion** — keep `fuel_mass_kg` at its pre-step value. This gives
  normal thrust acceleration at the current (real) mass while fuel never drains, so
  Δv is effectively infinite. (Do **not** fake it with a huge fuel mass — that would
  wrongly shrink acceleration via `a = F/m`.)
- `World.any_thrusting`: include `or unlimited_dv` in the per-vessel gate so warp
  still locks to 1× while burning with an empty tank under unlimited.

### Persistence (`sim/persistence.py`)

- Serialize/restore `unlimited_dv` on the vessel record (default `False` for older
  saves). Bump no schema version needed if read defensively with `.get(...)`.

### Render layer (`render/app.py`, `render/settings_panel.py`, `render/hud.py`)

- **Title slider:** raise `range=(0.0, 4000.0)` → `(0.0, 20000.0)` (≈ 9.1 km/s at the
  default dry mass / vₑ; the derived-Δv label already updates live). Add an
  **"Unlimited Δv"** checkbox (`DirectCheckButton`) on the start menu; when checked,
  `_on_play` sets `unlimited_dv = True` on every vessel.
- **Settings panel (Esc):** add an "Unlimited Δv: off/on" toggle button. Expand the
  panel frame to fit a second control. Toggling calls back into the app to flip
  `unlimited_dv` on all vessels live.
- **Keybind:** bind a free key (proposed **`U`** — verify no collision in
  `_setup_input`) to the same toggle, with a toast ("Unlimited Δv ON/OFF").
- **HUD:** when `dv_remaining` is infinite, render it as `∞` (in `hud.update_flight`);
  the maneuver readout's "dV left" likewise shows `∞`. Fuel readout keeps showing the
  actual (held) fuel mass.
- **Execute burn / nodes:** `_execute_burn`'s `0 < dv <= delta_v_remaining` gate
  passes automatically (inf); skip the `fuel_mass_kg -= burned` deduction when
  `unlimited_dv`.

## Components & boundaries

- `Vessel` owns the `unlimited_dv` flag and the derived `delta_v_remaining`/step
  behavior (pure sim, testable headless).
- The render layer only *flips the flag* (title checkbox, Esc toggle, keybind) and
  *formats* the infinite readout. No physics in render.

## Testing

**Pure (sim, no graphics):**
- `delta_v_remaining` returns `inf` when `unlimited_dv`, regardless of fuel.
- `World.step` with `throttle>0, unlimited_dv=True`: `fuel_mass_kg` is unchanged after
  a step, **and** velocity changed (thrust applied) — i.e. acceleration still happens.
- With `unlimited_dv=True` and `fuel_mass_kg == 0`, `any_thrusting()` is True while
  `throttle>0` (warp still locks).
- Persistence round-trip preserves `unlimited_dv`; missing key loads as `False`.

**Render (headless):**
- Toggling via the settings panel / keybind sets `unlimited_dv` on the vessel and the
  HUD shows `∞`.

## Out of scope

- Per-vessel UI (one global toggle hits all vessels — there's one vessel today).
- Changing the fuel/mass model or engine params.
