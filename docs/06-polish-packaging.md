# 06 — Phase 6: Polish, Save/Load & Packaging

**Gate:** Phase 5 interplanetary flight works.

Goal: turn the working simulator into a shippable desktop app for the author + a friend.

## Task 6.1 — `sim/persistence.py` (scenarios)
- JSON schema for a scenario: central/body set, list of vessels (state vectors + ΔV budgets),
  maneuver nodes, sim time, camera focus. Versioned (`"schema": 1`).
- `save_scenario(world, clock, path)` / `load_scenario(path) -> (World, SimClock)`.
- Ship a few built-in scenarios: `leo_sandbox`, `leo_to_geo`, `earth_to_mars`.
- **Test:** save→load round-trips a world to bit-identical state vectors (within float repr).

## Task 6.2 — HUD/UX polish
- Orbit info panel (Pe/Ap/period/inclination/ΔV remaining), maneuver list, time/warp controls,
  body textures + simple lighting, target/closest-approach markers for rendezvous.
- Keybinding help overlay. Settings (units km/mi, perturbations on/off toggle).
- Pause-on-SOI-change and auto-warp-down-before-burn quality-of-life.

## Task 6.3 — performance pass
- Cache orbit-line geometry; only rebuild on orbit change. Use analytic propagation on-rails,
  reserve `propagate_numeric` for active burns / perturbed vessels.
- Profile the per-frame task; keep 60 FPS with a handful of vessels.

## Task 6.4 — packaging (PyInstaller)
- `pyinstaller` one-folder build bundling Panda3D assets and the DE440 kernel (or download-on-
  first-run with a progress note). Produce a Windows `.exe`.
- Smoke-test the built exe on a clean path: launches, loads a scenario, renders, no missing-DLL
  errors. Document the build command in `README.md`.

## Task 6.5 — docs
- Update `README.md` with controls, screenshots, and the scenario list.
- Short `CONTROLS.md` cheat-sheet.

## Phase 6 exit criteria
- Double-clickable `.exe` launches the sim, loads built-in scenarios, runs smoothly, saves/loads
  state. Ready to hand to a friend.
