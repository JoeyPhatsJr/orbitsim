# Save / Load (Phase 6.1) — Design

**Date:** 2026-06-25
**Phase:** 6.1 (Polish, Save/Load & Packaging — save/load slice only)
**Scope decision:** sandbox world only; module + minimal F5/F9 keybinds; no shipped scenario files.

## Goal

Persist and restore the Earth-centered **sandbox** world to/from versioned JSON, and wire
quicksave (F5) / quickload (F9) into the running sandbox so a session can be saved and resumed.

Out of scope (deferred to later 6.x tasks): the `--solar` viewer, interplanetary vessels
(`earth_to_mars`), shipped built-in scenario files, and any load-menu / scenario-picker UI.

## Non-goals / YAGNI

- No serialization of `CelestialBody` internals — bodies are referenced by name and looked up in
  a registry; μ/radius/J2 are code constants, not save data.
- No scenario picker UI, no multiple save slots — a single `saves/quicksave.json`.
- No camera state (rig azimuth/elevation/distance) in the schema for now; camera re-centres on the
  vessel each frame anyway. Can be added under `"camera"` later without a schema bump if optional.

## Schema (versioned JSON, `schema: 1`)

A sandbox save is a flat JSON object:

```json
{
  "schema": 1,
  "kind": "sandbox",
  "central": "Earth",
  "sim_time_s": 0.0,
  "warp": 100.0,
  "vessels": [{
    "name": "Sandbox-1",
    "r_m":   [x, y, z],
    "v_mps": [x, y, z],
    "dry_mass_kg": 1000.0,
    "fuel_mass_kg": 800.0,
    "max_thrust_n": 30000.0,
    "exhaust_velocity_mps": 3000.0,
    "max_turn_rate_radps": 0.8,
    "throttle": 0.0,
    "sas_mode": "OFF",
    "orientation": [w, x, y, z],
    "nodes": [
      {"epoch_s": 0.0, "dv_prograde_mps": 0.0, "dv_normal_mps": 0.0, "dv_radial_mps": 0.0}
    ]
  }]
}
```

Notes:
- **Bodies by name.** `central` is a registry key; the full `CelestialBody` is looked up, not
  serialized.
- **`mu` is not stored.** On load each vessel's `StateVector` is rebuilt with `mu = central.mu`.
- **Exact float round-trip.** Python's `json` emits float64 via `repr()`, which round-trips
  exactly, so tests assert equality (not tolerance) on the numbers.
- **Vessel fields mirror the current `Vessel` dataclass** (post `delta_v_budget_mps` removal):
  `name`, state (`r_m`/`v_mps`), `dry_mass_kg`, `fuel_mass_kg`, `max_thrust_n`,
  `exhaust_velocity_mps`, `max_turn_rate_radps`, `throttle`, `sas_mode`, `orientation`, `nodes`.

## Module API (`orbitsim/sim/persistence.py`)

Pure sim-layer: imports `core` + `sim` only, never `render`/`panda3d`.

```python
BODY_REGISTRY: dict[str, CelestialBody]   # "Sun": SUN, "Earth": EARTH, "Moon": MOON, ...

def save_scenario(world: World, clock: SimClock, path: str | os.PathLike) -> None
def load_scenario(path: str | os.PathLike) -> tuple[World, SimClock]
```

- `save_scenario` builds the dict from `world.central.name`, `world.vessels`, and the clock, then
  writes JSON (creating parent dirs as needed).
- `load_scenario` reads JSON, validates, looks up the central body, rebuilds each `Vessel`
  (reconstructing `StateVector` and `ManeuverNode`s), and returns `(World, SimClock)`.

### Error handling (raise `ValueError`, never silently default)

- `schema` missing or != 1 → `ValueError` (forward/back-compat guard).
- `central` not in `BODY_REGISTRY` → `ValueError`.
- Missing required vessel field / malformed JSON → `ValueError` (wrap `KeyError`/`JSONDecodeError`).

## App wiring (F5 / F9)

Sandbox-only (guarded by the existing non-`solar_system` branch in `render/app.py`).

- **F5 → quicksave:** `save_scenario(self.world, self.clock, "saves/quicksave.json")`. The `saves/`
  directory is gitignored (like `data/`).
- **F9 → quickload:** `load_scenario(...)`, then **restore in place** — copy the loaded vessel's
  fields onto the live `self.world.vessels[0]` (state, masses, thrust, exhaust velocity, turn rate,
  throttle, SAS mode, orientation, nodes) and set `self.clock.sim_time_s` / `self.clock.warp`.

  **Why in-place, not world-swap:** mutating the existing vessel keeps the Panda3D scene graph
  (Earth, vessel marker, orbit lines, navball) intact, avoiding a full rebuild. This is clean
  because the sandbox has exactly one vessel and a fixed central body. The loaded `World` returned
  by `load_scenario` is used only as a data carrier here.

  A tiny on-HUD confirmation ("Quicksaved" / "Quickloaded", or a console print if a transient HUD
  message channel doesn't already exist) acknowledges the action.

## Testing (TDD)

Pure sim-layer tests in `tests/sim/test_persistence.py` — no graphics:

1. **Round-trip (known-answer):** build a `World` (Earth central) + a fully-configured `Vessel`
   (non-default masses, thrust, exhaust velocity, throttle, a non-OFF SAS mode, a non-identity
   orientation, two maneuver nodes) + a `SimClock` with non-default `sim_time_s`/`warp`. Save to a
   tmp path, load, and assert **every** field equal: `r`/`v` arrays, all masses/thrust/exhaust/turn
   rate, throttle, sas_mode, orientation, each node's four components, central body identity,
   `sim_time_s`, `warp`.
2. **`mu` reconstructed:** loaded vessel's `state.mu == EARTH.mu`.
3. **Unknown schema version → `ValueError`.**
4. **Unknown central body name → `ValueError`.**
5. **Malformed / missing-field JSON → `ValueError`.**

App wiring (F5/F9) is verified by the controller via a headless run / manual smoke test, not a
unit test (it touches the render layer).

## Files touched

- `orbitsim/sim/persistence.py` — replace the `# TODO` stub with the module above.
- `tests/sim/test_persistence.py` — new.
- `orbitsim/render/app.py` — add F5/F9 handlers in the sandbox keybind block.
- `.gitignore` — add `saves/`.

## Definition of done

- Round-trip + error-path tests green; full suite still green.
- F5 writes `saves/quicksave.json`; F9 restores it in a live sandbox session (smoke-tested).
- `sim/persistence.py` imports no `render`/`panda3d`.
