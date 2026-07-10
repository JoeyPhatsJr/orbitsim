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
- **Multi-core trajectory/planning work (done).** All heavy CPU work now runs on a single
  persistent `ProcessPoolExecutor` owned by `OrbitApp` (spawn context; falls back to threads
  if workers can't spawn), so the sim uses many cores instead of GIL-serialising on one:
  - **Planning grid search.** `core/optimize.py`'s `porkchop`, `intercept_node`,
    `interplanetary_porkchop`, and the new picklable `interplanetary_departure_node_by_name`
    take an optional `executor=` and fan their departure-time rows across it (result is
    bit-identical to serial — tests pin this). `_plan_intercept` / `_toggle_porkchop` now
    dispatch the search to a background *planning thread* (which fans cells across the pool)
    and apply the node / draw the porkchop on poll — the render thread never freezes.
  - **Flight-time overlap.** `_sample_trajectory`/`_sample_preview` were extracted into pure,
    picklable functions in `render/trajectory_sampling.py`; the live orbit line and the
    maneuver preview submit to the pool and now run in *separate processes*, so they truly
    overlap instead of GIL-serialising.
  - **Still single-core by nature (not a regression):** a single trajectory sample and the
    high-warp `world.step` are sequential ODEs (point *i+1* needs point *i*) and cannot be
    split across cores. Warp stutter is therefore an algorithmic problem (step sizing /
    off-thread on-rails propagation with interpolation), not a multi-core one — out of scope
    for this pass.

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
