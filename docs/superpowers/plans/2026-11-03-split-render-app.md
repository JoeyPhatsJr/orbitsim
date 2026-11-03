# Split `render/app.py` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the 2,542-line `OrbitApp` god object into cohesive render-layer controllers with **zero behavior change**, proven by golden-state tests recorded before the first move.

**Architecture:** First build a deterministic headless harness (fixed-step Panda3D clock, inline executors, fake wall clock) that drives scripted scenarios through real key events and GUI buttons and snapshots physics state + on-screen text + bound events. Then decompose `_start_sim`/`_update` in place into named steps, and move one feature at a time into `orbitsim/render/controllers/<feature>.py` classes that take `app` in their constructor (the existing `Hud(app)` / `CameraRig(app, …)` pattern). Every move must reproduce the goldens exactly.

**Tech Stack:** Python 3.11 (code stays 3.9-compatible), Panda3D, pytest, numpy.

**Spec:** No separate spec doc. Requirements come from the 2026-11-03 repo review: "split `render/app.py` (2,542 lines / 71 methods) into input handling, maneuver editor, planning UI, HUD wiring, scene setup — one piece at a time, each verified by the suite and a headless screenshot."

## Global Constraints

- Layering: `render/` may import `sim` + `core`; `core`/`sim` never import `render` or `panda3d`.
- Process pool: never submit a bound `OrbitApp` **or controller** method to `_process_pool`/executors (controllers hold `app` → pickling drags Panda3D). Pool work stays module-level functions.
- Keep the `if __name__ == "__main__"` guard in `orbitsim/__main__.py` (spawn re-imports it).
- Refactor commits change no behavior: golden files under `tests/render/app_harness/golden/` are **never edited** after Task 1. If a golden fails, the move is wrong.
- `tests/render/app_harness/hooks.py` is the only test code allowed to touch OrbitApp private attributes; it may be edited in a refactor commit only to follow a moved attribute.
- Event bindings stay registered through `app.accept(...)` (so `app.ignoreAll()` in `destroy` still clears them) and in the same order as today.
- No compatibility aliases left on `OrbitApp`; update the few tests that import app internals instead.
- Commit after each task with `refactor(render): …`, push the branch (CLAUDE.md "commit & push after every change"). Stage files by explicit path.
- Commands use the venv: `.venv/bin/python -m pytest …` (from the worktree, run as `<main-checkout>/.venv/bin/python -m pytest …`; `-m` puts the worktree first on `sys.path`).

## Review Focus

1. **Lost or duplicated key bindings** after moving `_setup_input` (e.g. `mouse1-up` is bound by both the maneuver UI and picking): pinned by the `events` field of every golden (Task 1).
2. **Per-frame order changes** inside `_update` (floating-origin recentre must precede marker placement; world step precedes readouts): pinned by goldens of camera/HUD text after multi-frame scenarios, plus the screenshot diff (Task 1 tooling, run in every task).
3. **Solar mode paths** (`--solar`) that sandbox scenarios never reach: pinned by the `solar_boot_warp` scenario (Task 1).
4. **Teardown leaks** (controllers' tasks/futures/pool not shut down → hung process at exit): pinned by `test_app_boots_and_destroys_cleanly` (Task 1), a subprocess that must exit 0 within 60 s.
5. **Escape/modal interplay** (porkchop modal open + Escape; operation overlay cancel): pinned by the `porkchop_escape` scenario (Task 1).

---

## Target layout

| New file (`orbitsim/render/…`) | Class / functions | Moves from `app.py` (lines at base commit `39415a5`) |
|---|---|---|
| `process_pool.py` | `pool_warmup()`, `build_process_pool()` | 99–133 |
| `trajectory_helpers.py` | `coast_chunks()`, `localize_polyline()`, `maneuver_preview_key()` | 89–97, 135–151 |
| `hud/format.py` | `fmt_dist()`, `fmt_countdown()` | 65–87 |
| `controllers/title_screen.py` | `TitleScreen` | 185–280 |
| `controllers/solar_view.py` | `SolarView` | 2503–2549 + solar branches of `_start_sim` |
| `controllers/warp_controls.py` | `WarpControls` | 1432–1511 |
| `controllers/flight_input.py` | `FlightInput` | 1367–1431, 1511–1566, 1606–1635 |
| `controllers/quicksave.py` | `Quicksave` | 1570–1606 |
| `controllers/targeting.py` | `Targeting` | 887–950, 1000–1133, CA block of `_update` |
| `controllers/porkchop_modal.py` | `PorkchopModal` | 1635–1770 |
| `controllers/trajectory_view.py` | `TrajectoryView` | 823–835, 1335–1367, 1828–1921, 2456–2503 |
| `controllers/maneuver_editor.py` | `ManeuverEditor` | 715–886, 951–999, 1192–1236, 1279–1335, 1770–1828, node block of `_update` |
| `controllers/bodies_view.py` | `BodiesView` | 650–715, Moon/Lagrange/planet/SOI blocks of `_update` |
| `controllers/vessel_view.py` | `VesselView` | vessel-marker + ship-view blocks of `_update` |
| `controllers/readouts.py` | `Readouts` | 1133–1192, 1236–1279, osculating-elements block of `_update` |

`app.py` keeps: `OrbitApp.__init__`, `destroy`, `_start_sim` (now a short sequence of controller constructions), `_update` (now a short ordered sequence of controller `frame(...)` calls), `_flash_message`, `run_app`. Target: **< 450 lines**.

## Standard extraction protocol (Tasks 4–10)

Each extraction task lists its methods, state, and wiring; the steps are always:

- [ ] **A. Ownership audit.** For every attribute the moved methods read or write, run `grep -n "self\.<attr>\b" orbitsim/render/app.py`. An attribute moves to the controller only if *all* its readers move; otherwise it stays on `app` and the controller uses `self.app.<attr>`. Record the split in the commit message.
- [ ] **B. Create the controller.** `class <Name>:` with `__init__(self, app: "OrbitApp") -> None` storing `self.app`. Copy method bodies **verbatim**; only rewrite `self.<app-owned>` → `self.app.<app-owned>`. Drop the leading underscore on methods the app calls; keep it on internal helpers.
- [ ] **C. Wire it.** Construct the controller at the exact point in `_start_sim` (or `__init__`) where its build code ran, so scene-graph and binding order are unchanged. Replace call sites with `self.<ctrl>.<method>(…)`; GUI `command=` and `app.accept(...)` targets point at the controller's bound methods.
- [ ] **D. Verify.** Run `.venv/bin/python -m pytest tests/render/app_harness -q` → all PASS, then `.venv/bin/python -m pytest tests -q` → all PASS, then `.venv/bin/python -m tests.render.app_harness.screens --compare` → prints `sandbox: identical` / `solar: identical` (or a diff ratio < 0.001; inspect the PNGs it writes if not).
- [ ] **E. Commit + push.** `git add <exact paths>` then `git commit -m "refactor(render): extract <Name> from OrbitApp (no behavior change)"` and `git push`.

---

### Task 1: Deterministic app harness + goldens (tests only)

**Files:**
- Create: `tests/render/app_harness/__init__.py`, `driver.py`, `hooks.py`, `scenarios.py`, `screens.py`, `test_app_golden.py`, `golden/<scenario>.json`
- Modify: `.gitignore` (add `.refactor-screens/`)

**Interfaces:**
- Produces: `python -m tests.render.app_harness.driver <scenario> <out.json>`; `python -m tests.render.app_harness.screens [--record|--compare]`; `hooks.select_target(app, name: str)`, `hooks.set_jog(app, axis: str, value: float)`; `driver.press(app, label: str)` (finds a live `DirectButton` in `DirectGuiWidget.guiDict` whose `["text"].strip() == label` and calls its `["command"]` with its `["extraArgs"]`; raises `LookupError` if none).

- [ ] **Step 1: Write `driver.py`.** In a fresh process and *before* importing `orbitsim.render.app`: replace `concurrent.futures.ThreadPoolExecutor` and `concurrent.futures.ProcessPoolExecutor` with an `InlineExecutor` (accepts any ctor kwargs; `submit` runs the call immediately and returns a completed `Future`, storing exceptions with `set_exception`; `shutdown(*a, **k)` no-op); replace `time.monotonic` with a counter advanced by exactly `1/60` per `step()`; `loadPrcFileData("", "window-type offscreen\nwin-size 960 600\naudio-library-name null")`; global clock `M_non_real_time` at 60 fps; set `OrbitApp.QUICKSAVE_PATH` to a temp file. Snapshot after the scenario:
  `{"sim_time_s", "warp", "vessel": {"r","v","fuel_mass_kg","throttle","sas_mode","orientation","nodes": [repr(n) …]} | null, "target": name|null, "texts": sorted non-empty texts of every TextNode under aspect2d/a2d*, "rig": {"distance_m","azimuth","elevation"}, "events": sorted(app.messenger.get_events())}` — floats as plain JSON numbers.
- [ ] **Step 2: Write `scenarios.py`** with these scenarios (frames are `app.taskMgr.step()` calls; sandbox = `orbitsim.__main__._default_world()`, `SimClock(sim_time_s=0.0, warp=100.0)`, then `press(app, "LAUNCH SANDBOX")`; solar uses `press(app, "OPEN SOLAR VIEWER")`):
  - `boot_idle`: 120 frames.
  - `burn`: 30 frames; `messenger.send("z")`; 60 frames; `send("x")`; 60 frames.
  - `sas_and_views`: `send("t")`; keys `"1"`…`"9"` each followed by 10 frames; `send("m")` 20 frames; `send("m")`; `send("f1")`; `send("f1")`.
  - `node_edit_execute`: `hooks.set_jog(app, "pro", 0.6)`; 30 frames; `send("mouse1-up")`; `press("Next Ap")`; 10 frames; `press("Execute Burn")`; 900 frames.
  - `warp_keys`: `send("period")` ×3 with 20 frames between; `send("0")`; 60 frames; `send("comma")` ×2; 20 frames.
  - `target_intercept`: `hooks.select_target(app, "Moon")`; 10 frames; `send("i")`; 120 frames.
  - `porkchop_escape`: `hooks.select_target(app, "Moon")`; `send("p")`; 60 frames; `send("escape")`; 10 frames; `send("p")`; 30 frames; `press("CLOSE")`; 10 frames.
  - `quicksave_roundtrip`: 30 frames; `send("f5")`; `send("z")`; 60 frames; `send("x")`; `send("f9")`; 30 frames.
  - `solar_boot_warp`: `World(central=SUN, vessels=[])` + `SimClock(30*365.25*86400, warp=1e6)`, `solar_system=True`; 60 frames; `send("period")`; 60 frames; `send("comma")`; 30 frames.
- [ ] **Step 3: Write `hooks.py`** — the only private-attribute coupling: `select_target(app, name)` sets `app._target` to the entry of `app._targets` whose `.name == name` (calls `app._clear_target()` first); `set_jog(app, axis, value)` sets `app._jog[axis]["value"] = value`.
- [ ] **Step 4: Record goldens** at base commit `39415a5`: run the driver for every scenario into `golden/<name>.json`; run each scenario **twice** and assert byte-identical output before accepting it (determinism gate). If a scenario is not reproducible, fix the harness (not the app) until it is.
- [ ] **Step 5: Write `test_app_golden.py`** — `@pytest.mark.parametrize("scenario", SCENARIOS)`: run the driver in a subprocess (`sys.executable -m tests.render.app_harness.driver`, `cwd=repo root`, `timeout=180`), load JSON, compare to golden: texts/events/sas_mode/target/nodes exact; floats `math.isclose(rel_tol=1e-9, abs_tol=1e-6)` (cross-platform last-bit noise). Plus `test_app_boots_and_destroys_cleanly`: driver scenario `boot_idle` with `app.destroy()` at the end must exit 0 within 60 s. `pytest.importorskip("panda3d")` at module top.
- [ ] **Step 6: Write `screens.py`** — same deterministic setup; renders `boot_idle` (sandbox, rig distance 1.3e9 m) and `solar_boot_warp`; `--record` saves to `.refactor-screens/baseline/{sandbox,solar}.png`, `--compare` saves to `.refactor-screens/current/` and prints per-mode fraction of pixels whose max channel diff > 8 (0 → "identical").
- [ ] **Step 7: Run** `pytest tests/render/app_harness -q` → all PASS; `screens --record` then `screens --compare` → both `identical`. Look at both PNGs.
- [ ] **Step 8: Commit + push** `tests/render/app_harness/**`, `.gitignore`.

### Task 2: Module-level helpers out of `app.py`

**Files:** Create `orbitsim/render/process_pool.py`, `orbitsim/render/hud/format.py`, `orbitsim/render/trajectory_helpers.py`. Modify `orbitsim/render/app.py`, `tests/render/test_maneuver_preview.py` (imports).

**Interfaces:** Produces `process_pool.build_process_pool()` and `process_pool.pool_warmup()` (same bodies and return values as today's `_build_process_pool`/`_pool_warmup`, public names); `hud.format.fmt_dist(meters: float) -> str`, `fmt_countdown(seconds: float) -> str`; `trajectory_helpers.coast_chunks(duration_s: float, max_chunk_s: float = 86400.0) -> list[float]`, `localize_polyline(points)`, `maneuver_preview_key(node, scheduled_epoch_s)`. (`hud/__init__.py` has its own `_dist(meters, units)`; leave both — merging them would change HUD text.)

- [ ] **Step 1:** Add `tests/render/test_process_pool.py::test_pool_warmup_is_picklable_module_function` — `pickle.dumps(process_pool.pool_warmup)` succeeds and `pool_warmup.__module__ == "orbitsim.render.process_pool"`. Run → FAIL (module missing).
- [ ] **Step 2:** Move the functions (bodies verbatim), update call sites in `app.py` and the imports in `test_maneuver_preview.py`.
- [ ] **Step 3:** Protocol step D, then E.

### Task 3: Decompose `_start_sim` and `_update` in place (no new files)

**Files:** Modify `orbitsim/render/app.py` only.

**Interfaces (produced, used by Tasks 4–10):** `_update` becomes, in this order: `_frame_planning()`, `_frame_flight(real_dt)`, `_frame_jogs(real_dt)`, `_frame_origin(scale)`, `_frame_moon_ring(scale)`, `_frame_sun()`, `_frame_vessels()`, `_frame_ship_view()`, `_frame_maneuver_node()`, `_frame_moon_and_lagrange()`, `_frame_planets(scale)`, `_frame_closest_approach()`, `_frame_readouts()` — each the contiguous block under the matching comment header today (lines 1952–2454). `_start_sim` becomes: `_build_central_body()`, `_build_lighting()`, `_build_vessel_markers()`, `_build_ship_view()`, `_build_orbit_frame()`, `_build_mode_specific()` (the `if self.solar_system` split at line 412), `_build_starfield()` — split at its comment headers.

- [ ] **Step 1:** Extract each block into its method verbatim. Locals a block reads from an earlier block become explicit parameters (as named above, plus any others found); locals a later block needs become return values. **Do not** introduce new `self.` attributes. Early `return task.cont` paths stay in `_update`.
- [ ] **Step 2:** Protocol step D, then E (`refactor(render): split _start_sim/_update into named steps`).

### Task 4: `TitleScreen` and `SolarView`

**Methods:** `_build_title_screen`, `_refresh_budget_label`, `_on_play` → `TitleScreen.build()`, `refresh_budget_label()`, `on_play()`; `_helio_body_pos`, `_update_solar_system`, solar branch of `_build_mode_specific` → `SolarView.build()`, `frame()`, `helio_body_pos(name)`.
**State:** `_title_nodes`, `_budget_slider`, `_budget_label`, `_unlimited_check` move to `TitleScreen`; `_fuel_capacity` stays on app.
- [ ] Protocol A–E (one commit per class).

### Task 5: `WarpControls`, `FlightInput`, `Quicksave`

**Methods:** `_build_warp_controls`, `_update_warp_readout`, `_warp_up_guarded`, `_max_warp_guarded` → `WarpControls.build()/update_readout()/warp_up()/max_warp()`. `_setup_input`, `_toggle_ship_view`, `_rmb`, `_handle_escape`, `_apply_mouse_orbit`, `_set_key`, `_throttle_full`, `_throttle_cut`, `_toggle_sas`, `_set_sas`, `_apply_flight_input` → `FlightInput` (`setup()`, `apply(dt)`, `apply_mouse_orbit()`, the rest keep names minus underscore). `_quicksave`, `_quickload` → `Quicksave.save()/load()`; `QUICKSAVE_PATH` stays an `OrbitApp` class attribute (harness sets it).
**Note:** `_setup_input` binds keys handled by other features (`p`, `i`, `f1`): bind them to the *current* owner (`self.app._toggle_porkchop` etc.); later tasks repoint them.
- [ ] Protocol A–E.

### Task 6: `Targeting` and `PorkchopModal`

**Methods:** `_on_mouse1_down`, `_mouse_px`, `_marker_px`, `_try_pick_target`, `_plan_intercept`, `_apply_planned_node`, `_poll_planning`, `_clear_target`, `_ca_marker`, `_frame_closest_approach` → `Targeting` (`frame_closest_approach()`, `poll_planning()`, `clear_target()`, `plan_intercept()`, …). `_toggle_porkchop`, `_poll_porkchop`, `_build_porkchop_modal` → `PorkchopModal.toggle()/poll()`.
**State:** `_target` and `_targets` stay on `app` (read by SAS feed, readouts, maneuver editor); update `hooks.select_target` only if they move.
**Pool check:** `_plan_intercept` / `_toggle_porkchop` submit closures to `_planning_executor` (a thread pool) that call `core.optimize` with `executor=app._process_pool`; keep the grid functions module-level. Add `tests/render/test_controllers_pool_safety.py::test_no_controller_method_reaches_process_pool` — monkeypatch the pool's `submit`/`map` to assert `not inspect.ismethod(fn)` and run the `target_intercept` + `porkchop_escape` driver scenarios in-process.
- [ ] Protocol A–E.

### Task 7: `TrajectoryView`

**Methods:** `_cancel_operation`, `_jump_operation`, `_trajectory_coast_operation`, `_build_encounter_patches`, `_install_trajectory`, `_rebuild_trajectory`, `_update_encounter_cues` → `TrajectoryView`. Update `tests/render/test_maneuver_preview.py` (uses `OrbitApp._build_encounter_patches` + `OrbitApp.ENCOUNTER_COLOR` via a fake self) to call `TrajectoryView.build_encounter_patches` with the same fake.
- [ ] Protocol A–E.

### Task 8: `ManeuverEditor`

**Methods:** `_build_maneuver_ui`, `_toggle_maneuver_drawer`, `_jog_rate_mps_per_s`, `_apply_jogs`, `_release_jogs`, `_current_node`, `_time_to_node`, `_step_node_time`, `_node_to_pe`, `_node_to_ap`, `_clear_node`, `_refresh_readout`, `_sync_maneuver_hud`, `_apply_unlimited`, `_set_unlimited_dv`, `_toggle_unlimited_dv`, `_execute_burn`, `_update_maneuver_preview`, `_frame_jogs`, `_frame_maneuver_node` → `ManeuverEditor`.
**State:** `_dv`, `_jog`, `_dv_value_text`, `_node_marker_np`, `_node_dv_label`, preview futures move; `_node_epoch_s` and `_auto_execute_node` are read by `_frame_flight` → stay on `app` unless Step A shows otherwise. Update `hooks.set_jog` to `app.maneuver_editor._jog`.
- [ ] Protocol A–E.

### Task 9: `BodiesView` and `VesselView`

**Methods:** `_update_starfield`, `_build_planets`, `_frame_moon_ring`, `_frame_sun`, `_frame_moon_and_lagrange`, `_frame_planets`, `_build_central_body`, `_build_lighting`, `_build_starfield`, `_build_orbit_frame` → `BodiesView`. `_frame_vessels`, `_frame_ship_view`, `_build_vessel_markers`, `_build_ship_view` → `VesselView`. Constants (`LAGRANGE_REFRESH_S`, `MOON_RING_REFRESH_S`, `SOI_*`) move with their users.
- [ ] Protocol A–E.

### Task 10: `Readouts`, slim `app.py`, docs

**Methods:** `_place_apsis_markers`, `_update_marker_readability`, `_update_encounter_info`, `_frame_readouts` → `Readouts`.
- [ ] Protocol A–E.
- [ ] **Step F:** `wc -l orbitsim/render/app.py` → < 450. `grep -n "def " orbitsim/render/app.py` lists only `__init__`, `destroy`, `_start_sim`, `_update`, `_flash_message`, `run_app` plus `_frame_planning`/`_frame_flight`/`_frame_origin` if they stayed.
- [ ] **Step G:** Update CLAUDE.md "Data flow each frame" and the architecture section to name the controllers; add a gotcha: "Controllers hold `app` — never submit their bound methods to the pool." Commit + push.

---

## Execution notes

- The harness (Task 1) is the whole safety net: `OrbitApp` currently has almost no direct tests. Do not start Task 2 until the determinism gate in Task 1 Step 4 passes.
- Expect to discover hidden coupling in Step A of Tasks 5, 6 and 8 (`_node_epoch_s`, `_target`, the `mouse1-up` double binding). Leaving state on `app` is always acceptable; narrowing interfaces is a later, separate change.
