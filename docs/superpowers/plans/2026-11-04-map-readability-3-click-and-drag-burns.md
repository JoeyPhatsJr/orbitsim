# Map Readability, Plan 3 of 5: Place and Shape a Burn by Hand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The player places a burn by clicking their orbit ("Add burn here" / "Warp to here"), shapes it by dragging 3D handles or tapping/holding nudge keys, and slides it along the orbit by dragging the node, with the orange line following live.

**Architecture:**
1. **Pure helpers, no Panda3D.** Screen-space orbit picking goes in `render/picking.py`. Drag, nudge and placement math goes in a new `render/node_edit.py`. The lead-time text goes in `render/hud/format.py`. One physics function, `rtn_basis`, goes in `core/maneuvers.py`.
   - Why `render/` and not `core/`: picking a pixel, mapping a drag to a Δv rate, and key-repeat acceleration are input and presentation, not physics.
   - The burn frame (prograde/normal/radial at the burn point) *is* physics, so it lives in `core` with known-answer tests.
2. **Two new stateless controllers**, `node_placement` (hover ghost, click dispatch, orbit menu) and `node_handles` (handles, handle and node drags). `flight_input` gains the input-device queries, `maneuver_editor` gains `set_dv` / `set_node_epoch` / nudges, and `targeting` gains `project_px`, `pick_body_at` and `coast_to`.
3. **All state stays on `OrbitApp`.** Handles, keys and the slider panel all edit the same `app._dv` and `app._node_epoch_s`, so they always agree. Plan 2's estimate-then-exact preview already redraws the orange line whenever those change.

**Tech Stack:** Python 3.11, numpy, Panda3D 1.10.16, pytest (+xdist); the app-golden harness in `tests/render/app_harness/`.

**Spec:** `docs/superpowers/specs/2026-11-04-map-and-planning-readability-design.md`, section "Planning a burn", items 1–3. Plans 1 and 2 are merged (PRs #11, #12). Plan 4 does items 4–6 (aim helpers, auto-plan, slider-panel removal) and the HUD layout. The slider panel **stays working** in this plan.

**Branch:** `map-readability-3-click-and-drag`, from `main` (with this plan committed).

## What planning found (2026-11-05, M1 Pro; the executor need not re-research)

**Mouse and picking today:**
- `flight_input.py:48` binds `"mouse1"` → `targeting.on_mouse1_down` (`targeting.py:73`), which stores `app._mouse1_down_px`.
- `maneuver_editor.py:136` binds `"mouse1-up"` → `_release_jogs` (`:187`), which calls `targeting.try_pick_target()` (`targeting.py:96`) and springs the jog sliders back.
- `try_pick_target`:
  - ignores drags over 6 px;
  - picks a body with `picking.nearest_marker(click, proj, tol_px=22.0)`;
  - otherwise picks a live-line sample with `nearest_future_point(..., tol_px=16.0)` and starts `trajectory_view.trajectory_coast_operation(duration_s)`, with rollback (`targeting.py:117-135`), unless thrusting.
- `nearest_future_point` (`picking.py:22`) has no tests and no other caller.
- `targeting._mouse_px()` (`:77`) and `marker_px(world_r)` (`:85`, one `camLens.project` per point) give pixels **bottom-up** (y = 0 at the bottom).

**Live line cache:**
- `trajectory_view.rebuild_trajectory` (`:111`) fills `app._traj_world_cache[0]` (256×3 geocentric m) and `app._traj_epoch_cache[0]` (absolute epochs, uniform step), refreshed every `TRAJECTORY_REFRESH_S = 2.0` s.
- The default LEO line spans 2 orbits: 13,818 s, step 54 s, about 4.7 px per segment at the opening zoom (rig 88,433 km, 960×600 window).
- `trajectory_coast_operation` sets the cache to `None` when it finishes.

**Node today:**
- `maneuver_editor.current_node()` (`:197`) builds a `ManeuverNode` from `app._dv = {"pro","nrm","rad"}` and `app._node_epoch_s` (`None` = burn-now).
- `frame_maneuver_node` (`:483`) draws the marker only when `_node_epoch_s is not None`:
  - a `make_uv_sphere`, fixed `set_scale(6.0)`, colour (0.3, 1, 1);
  - at `apply_maneuver(v0.state, node).r`, i.e. the two-body (Kepler) position at the epoch;
  - with the label `dV … T-…` via `build_world_label`.
- `_step_node_time` (`:214`) clamps the node to ≥ now.
- `_jog_rate_mps_per_s` (`:145`) is the slider curve: 400 m/s per s at full deflection, curve 4, deadzone 0.02.
- The RTN basis is written inline in `apply_maneuver` and `maneuver_direction` (`core/maneuvers.py:68-73, 99-104`). It divides by |v| with no guard (it gives NaN, not an error, for a landed vessel). **Leave both functions untouched.**

**Frame order** (`app.py:205-249`):
- `_frame_flight` → `apply_jogs` (`:229`) → `_frame_origin` → … → `frame_maneuver_node` (`:236`) → … → `rig.apply()` (`:245`) → `update_marker_readability` (`:246`) → `frame_readouts` (`:248`).
- Camera-dependent hit-testing must run after `rig.apply()`.

**Headless harness:**
- The offscreen window has **`app.mouseWatcherNode is None`**, so today no mouse path runs in goldens.
- `driver.Harness` has `step`, `send`, `press` (DirectButton by label) and nothing for the mouse.
- `snapshot()` records `events` = **every bound messenger event**. Any new `accept()` therefore adds to `events` in every sandbox golden.
- `texts` does **not** include the F1 overlay or DirectButton labels (their text is flattened, so it has no TextNode). `world_labels` does include render-space labels, hidden ones flagged.

**Projection:**
- `points_render_h @ (M(render.get_mat(cam)) @ M(camLens.get_projection_mat()))` gives clip coordinates (row-vector convention). Then px = ((x/w)·0.5+0.5)·W and ((y/w)·0.5+0.5)·H.
- It matches `marker_px` to 3e-5 px on all 256 line points, in **0.11 ms**, against **0.58 ms** for 256 `marker_px` calls.
- Here `M(lm) = [[lm.get_cell(r, c) for c in range(4)] for r in range(4)]`.

**Keys:**
- Panda names punctuation by the character: `","`, `"."`, `"["`, `"]"`, `";"`, `"'"`.
- ShowBase registers shift/control/alt/meta as modifiers (`ShowBase.py:1722-1727`). So `j` pressed while Shift is held arrives as `shift-j`, and a release while Shift is held arrives as `shift-j-up`. Shift and Ctrl are already throttle up/down (held).
- *Found, out of scope:* `flight_input.py:35-36` binds `"period"` / `"comma"`. These are not Panda button names, so the `,` / `.` warp keys most likely never fire from a real keyboard. The `warp_keys` golden sends the literal strings. Reported to the user, not fixed here.

**Frame cost** (real pools, `probe_jog`):
- idle 5.5 ms;
- a jog-edited burn 7.1 ms mean / 8.1 ms p95 (the estimate rebuilds every frame);
- sliding the epoch every frame 6.9 / 8.1 ms.
- Dragging stays well under 16.7 ms. `profile idle`: 7.25 ms mean.

## Decisions (the spec leaves these to the builder)

| What | Decided |
|---|---|
| Nudge keys (left key subtracts, right key adds) | `H`/`J` retro/prograde · `K`/`L` anti-normal/normal · `;`/`'` radial in/out · `[`/`]` node earlier/later. Right hand, clear of WASD/QE/Z/X. No modifier: Shift/Ctrl are throttle. |
| Nudge steps (fine and coarse without a modifier) | **Tap:** 1 m/s, or 10 s. **Hold:** after 0.3 s it repeats at 10 m/s per s (30 s per s), doubling every 0.5 s, capped at 1,000 m/s per s (21,600 s per s). A Moon burn (~3,100 m/s) takes about 6 s of holding. |
| Handle drag | Spring-loaded **rate** control, like the slider jog: pulling up to 150 pt along the handle's on-screen axis gives the slider's curve, up to 400 m/s per s. Release stops it. |
| Snapping | On release, Δv snaps to 0.1 m/s and the node time to 1 s, so values are clean and goldens are bit-stable. |
| Menu with a burn already planned | One node at a time: the first item reads **"Move burn here"** and keeps the Δv (no silent loss). Otherwise it reads "Add burn here". |
| Where a node may go | From now + 10 s to the end of the current orbit: one osculating period, or the first non-Earth SOI entry if that comes sooner, or the line's end if unbound. Beyond that (e.g. past a Moon flyby), the menu offers only "Warp to here". |
| Overlapping loops | The line is 2 orbits, so a pick prefers the **earliest** time among hits within 3 pt of the best one. |
| Sizes and tolerances (pt, × `app.line_px_scale()`) | line hit 14 · node hit 14 · handle hit 12 · node radius 7 · handle radius 5 (− handles 0.75×) · handle offset 36 · ghost radius 6 · tap move 6 · a handle pair is hidden if its on-screen axis is < 8. Marker, handles and ghost are scaled by camera distance, so they stay the same size on screen at any zoom. |
| Colours | Handles: prograde lime (0.75, 1.0, 0.30), normal violet (0.80, 0.50, 1.0), radial coral (1.0, 0.45, 0.45); − handles at alpha 0.55; hovered or dragged handle 1.3×. Ghost: ORBIT cyan at alpha 0.6. |
| Out-of-date pick or thrusting | "Add/Move burn here" when the picked time is no longer ≥ now + 10 s: flash "That point has passed", no node. "Warp to here" while thrusting: flash "Cut throttle before warping". |
| **Fix the broken warp keys (added in plan review, 2026-11-05)** | `flight_input.py:35-36` binds `"period"` / `"comma"`, but Panda3D names those keys `"."` / `","` (`KeyboardButton.ascii_key(',').get_name() == ','`, verified). So `,`/`.` do nothing on a real keyboard; the `warp_keys` golden sends the literal strings, which hides it. **Task 1 also fixes this**: bind `"."`/`","`. Add a unit test that the bound event names equal `KeyboardButton.ascii_key(c).get_name()` for `,` and `.` (RED first). The `warp_keys` scenario then sends `"."`/`","`; the scenario script may change *how* it drives the UI. Re-record it with Task 1's `events` re-record, and keep the physics fields identical. |

## Global Constraints

- **The physics doesn't change.** The flown trajectory, the integrators and the N-body model stay as they are. `apply_maneuver`, `maneuver_direction`, `propagate_*` and `World.step` are untouched.
- **App goldens:** existing scenarios keep `vessel`, `sim_time_s`, `warp`, `target`, `texts`, `world_labels` and `rig` identical. `events` may change in exactly one way: Task 1 adds the 16 nudge-key events (8 keys and their `-up`), proved by script. New scenarios are recorded new.
- **Scenario scripts:** never edit an existing one; add new ones. `hooks.py` is the only test code that touches OrbitApp private attributes.
- **Scope:** one node at a time, placed on your current orbit only. No multi-node and no capture burns, but nothing that makes them harder: keep a single node's state behind `current_node()` / `set_node_epoch()` / `set_dv()`.
- **Spec copy:**
  - menu items "Add burn here" / "Warp to here";
  - the ghost shows "in N min";
  - clicking a body still targets it as today;
  - the nudge keys are shown in the F1 overlay.
- **Speed:** no frame-rate regression. `idle`, `zoom`, `zoom_node_planned`, `high_warp` and `node_planned` (`python -m tests.render.app_harness.profile <w>`) must stay within 10% mean frame ms of `main`, same machine and session. Node dragging stays ≥ 60 fps on the dev Mac: new `node_drag` workload mean ≤ 16.7 ms.
- **Layering and pool:** `core/` never imports Panda3D. Pool jobs stay module-level and picklable; this plan submits no new jobs.
- **Process:** TDD. Commit and push after each task. Stage by explicit path. Trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Final whole-branch review, then PR, then merge.

## Review Focus

1. **Mouse released over a panel or outside the window.** DirectGUI swallows `mouse1-up` there, so a drag would stick and keep changing Δv. Each frame, `apply_drag` ends the drag when `flight_input.mouse1_held()` is `False`. Pinned by `test_drag_ends_when_button_no_longer_held` (Task 3).
2. **A modifier pressed mid-hold.** `j` down, then Shift (throttle), then the release arrives as `shift-j-up` and `j` would ramp forever. `apply_nudges` drops a key when `flight_input.key_held(key)` is `False`. Pinned by `test_nudge_stops_when_key_no_longer_held` (Task 1).
3. **A handle axis pointing at the camera** (an edge-on view of the normal axis): that pair is hidden and can't be grabbed, with no division by a tiny length. Pinned by `test_axis_unit_px_none_when_foreshortened` (Task 3).
4. **The picked time passes while the menu is open** (time keeps running, ×100 by default): no node in the past, no negative coast. Pinned by `test_node_epoch_ok_needs_lead` and `test_add_burn_refused_once_passed` (Task 2).
5. **Overlapping loops, and clicks beyond the current orbit:** the earliest time wins, and past the node window only "Warp to here" is offered. Pinned by `test_pick_prefers_earliest_of_overlapping_loops` and `test_window_stops_at_first_encounter` (Task 2).

Also covered:
- landed or zero-velocity vessel: `rtn_basis` raises, handles hidden (Task 3);
- solar viewer: nothing here is built there; the `solar_*` goldens are unchanged;
- clicks while a coast operation runs are ignored (Task 2).

---

### Task 1: Keyboard nudges for each Δv axis and the node time

**Files:**
- Create: `orbitsim/render/node_edit.py` (pure; this task adds the nudge part and `jog_rate`)
- Modify: `orbitsim/render/controllers/maneuver_editor.py` (`_jog_rate_mps_per_s` delegates; add `set_dv`, `set_node_epoch`, `on_nudge_key`, `apply_nudges`; the node-time text formatting moves into `_fmt_node_t`, which `apply_jogs` reuses)
- Modify: `orbitsim/render/controllers/flight_input.py` (`key_held`; bind the nudge keys in the sandbox branch)
- Modify: `orbitsim/render/app.py:229` (call `apply_nudges(real_dt)` right after `apply_jogs`)
- Modify: `orbitsim/render/keybind_overlay.py` (4 lines; frame `(-0.75, 0.75, -0.95, 0.85)`, text scale 0.045, pos (-0.68, 0.76) so 23+ lines fit)
- Modify: `tests/render/app_harness/hooks.py`, `scenarios.py` (new scenario `nudge_keys`), all 9 sandbox goldens (`events` only)
- Test: `tests/render/test_node_edit.py` (new), `tests/render/test_keybind_overlay.py`, `tests/render/test_node_editing_app.py` (new subprocess probes, modelled on `test_live_preview_app.py`)

**Interfaces:**
- Produces (`node_edit.py`):
  - `@dataclass(frozen=True) class NudgeSpec: tap: float; base_rate: float; doubling_s: float; max_rate: float; hold_delay_s: float`
  - `NUDGE_DV = NudgeSpec(1.0, 10.0, 0.5, 1000.0, 0.3)` and `NUDGE_TIME = NudgeSpec(10.0, 30.0, 0.5, 21600.0, 0.3)`
  - `NUDGE_KEYS: dict[str, tuple[str, int]] = {"j": ("pro", 1), "h": ("pro", -1), "l": ("nrm", 1), "k": ("nrm", -1), "'": ("rad", 1), ";": ("rad", -1), "]": ("time", 1), "[": ("time", -1)}`
  - `def nudge_rate(held_s: float, spec: NudgeSpec) -> float`: 0 before `hold_delay_s`, else `min(max_rate, base_rate * 2 ** ((held_s - hold_delay_s) / doubling_s))`
  - `def jog_rate(value: float, max_rate: float, curve: float, deadzone: float) -> float`: today's `_jog_rate_mps_per_s` body, the same expressions in the same order (bit-identical; `node_edit_execute` pins it)
  - `def snap_dv(x: float) -> float` (`round(x, 1)`) and `def snap_epoch(t: float) -> float` (`float(round(t))`)
- Produces (controllers):
  - `ManeuverEditor.set_dv(axis: str, value_mps: float) -> None`: sets `app._dv[axis]` and `_dv_value_text[axis]` (`f"{v:+.0f}"`), then `refresh_readout()`.
  - `ManeuverEditor.set_node_epoch(epoch_s: float) -> None`: `max(now, epoch_s)` into `app._node_epoch_s`, and updates `_node_time_text`.
  - `ManeuverEditor.on_nudge_key(key: str, down: bool) -> None`:
    - **down:** apply the tap and start `app._nudge_held[key] = 0.0`;
    - **up:** drop the key and snap that axis (`snap_dv`, or `snap_epoch` for time).
    - Time with no node: `set_node_epoch((app._node_epoch_s or now) + sign * tap)`.
    - Δv with no node: a burn-now Δv, exactly as the slider does today.
  - `ManeuverEditor.apply_nudges(real_dt: float) -> None`: for each held key, first drop it if `flight_input.key_held(key) is False`. Then `held += real_dt`, and add `sign * nudge_rate(held, spec) * real_dt` to that axis.
  - `FlightInput.key_held(key: str) -> bool | None`: `mouseWatcherNode.is_button_down(KeyboardButton.ascii_key(key))`, or `None` when there is no mouseWatcher (headless).
  - App state: `app._nudge_held: dict[str, float]`, initialised in `build_maneuver_ui`.
  - Bindings (sandbox branch of `setup_input`): for each `k` in `NUDGE_KEYS`, `accept(k, on_nudge_key, [k, True])` and `accept(f"{k}-up", on_nudge_key, [k, False])`.
  - F1 lines:
    - `("H / J", "Burn retrograde / prograde")`
    - `("K / L", "Burn anti-normal / normal")`
    - `("; / '", "Burn radial in / out")`
    - `("[ / ]", "Burn earlier / later")`
  - `hooks.set_key_held(app, key: str, held: bool | None)`: patches `app.flight_input.key_held` with a per-key override; `None` restores the method.
  - `hooks.node_state(app) -> tuple`: `(app._node_epoch_s, app._dv["pro"], app._dv["nrm"], app._dv["rad"])`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/render/test_node_edit.py
import math, pytest
from orbitsim.render.node_edit import (NUDGE_DV, NUDGE_TIME, NUDGE_KEYS, nudge_rate, jog_rate,
                                       snap_dv, snap_epoch)

def test_nudge_rate_zero_before_hold_delay():
    assert nudge_rate(0.29, NUDGE_DV) == 0.0
def test_nudge_rate_starts_at_base_and_doubles():
    assert nudge_rate(0.3, NUDGE_DV) == 10.0 and nudge_rate(0.8, NUDGE_DV) == pytest.approx(20.0)
def test_nudge_rate_is_capped():
    assert nudge_rate(99.0, NUDGE_DV) == 1000.0 and nudge_rate(99.0, NUDGE_TIME) == 21600.0
def test_nudge_keys_cover_each_axis_both_ways():
    assert sorted(NUDGE_KEYS.values()) == sorted(
        (a, s) for a in ("pro", "nrm", "rad", "time") for s in (1, -1))
def test_jog_rate_is_the_slider_curve():
    assert jog_rate(0.6, 400.0, 4.0, 0.02) == 400.0 * (math.exp(2.4) - 1) / (math.exp(4.0) - 1)
    assert jog_rate(0.01, 400.0, 4.0, 0.02) == 0.0 and jog_rate(-1.0, 400.0, 4.0, 0.02) == -400.0
def test_snaps():
    assert snap_dv(12.345) == 12.3 and snap_epoch(2200.6) == 2201.0
```

```python
# tests/render/test_keybind_overlay.py (append)
import pathlib, re
from orbitsim.render.node_edit import NUDGE_KEYS

def test_nudge_keys_listed():
    keytext = " ".join(k for k, _ in SANDBOX_BINDINGS).lower()
    assert all(k in keytext for k in NUDGE_KEYS)

def test_nudge_keys_do_not_clash():
    src = "".join(p.read_text() for p in pathlib.Path("orbitsim/render").rglob("*.py"))
    literal = set(re.findall(r"accept\(\s*\"([^\"]+)\"", src))
    assert not literal & (set(NUDGE_KEYS) | {f"{k}-up" for k in NUDGE_KEYS})
```

`tests/render/test_node_editing_app.py` (probe `NUDGES`):
- launch, `h.step(10)`;
- send `"]"`, `"]-up"`; send `"j"`, `"j-up"`; send `";"`, `";-up"` → RESULT `node_state` A;
- `h.send("l")`, `h.step(60)`, `h.send("l-up")` → B;
- `h.send("j")`, `hooks.set_key_held(app, "j", False)`, `h.step(30)` → C (key no longer held: no ramp).

```python
def test_nudge_taps_set_exact_steps():
    a, b, c = _run(NUDGES)
    assert a[1:] == (1.0, 0.0, -1.0) and a[0] is not None          # node created, +1 pro, -1 rad
    assert b[2] == pytest.approx(10 * 0.5 / math.log(2) * (2 ** 1.4 - 1) + 1.0, abs=0.5) # tap + 0.7 s ramp
def test_nudge_stops_when_key_no_longer_held():
    a, b, c = _run(NUDGES)
    assert c[1] == b[1] + 1.0                                       # only the tap, no ramp
```

- [ ] **Step 2: Run them, expect FAIL** (`ModuleNotFoundError: orbitsim.render.node_edit`).
- [ ] **Step 3: Implement** `node_edit.py`, the editor and flight-input methods, the bindings, the overlay lines and size, and the hooks. Add scenario `nudge_keys`: launch, `h.step(10)`, then a `]` tap, a `]` hold for 90 frames, a `j` tap, an `l` hold for 60 frames, a `;` tap, then `h.step(30)`.
- [ ] **Step 4: Run** `tests/render/test_node_edit.py tests/render/test_keybind_overlay.py tests/render/test_node_editing_app.py`. Expected: all pass.
- [ ] **Step 5: Goldens.**
  1. `python -m tests.render.app_harness.record nudge_keys`.
  2. Run the golden suite: the 9 existing sandbox scenarios fail on `events` only.
  3. Re-record them: `python -m tests.render.app_harness.record boot_idle burn sas_and_views node_edit_execute warp_keys target_intercept porkchop_escape quicksave_roundtrip zoom_with_target_node`.
  4. **Prove it** with a scratch script (in the scratchpad, `__main__` guard). For each of the 9, compare `json.loads(git show HEAD:<golden>)` with the new file: every key except `events` must be equal, and `sorted(new["events"]) == sorted(old["events"] + NUDGE_EVENTS)`, where `NUDGE_EVENTS = [k for key in NUDGE_KEYS for k in (key, key + "-up")]`.
  5. The `solar_*` goldens must be untouched.
- [ ] **Step 6: Full suite** `-n auto`: green.
- [ ] **Step 7: Commit**:
  - Stage: the files above plus `tests/render/app_harness/golden/{nudge_keys,boot_idle,burn,sas_and_views,node_edit_execute,warp_keys,target_intercept,porkchop_escape,quicksave_roundtrip,zoom_with_target_node}.json`.
  - Message: `feat(render): keyboard nudges for burn delta-V and node time (H/J K/L ;/' [/])`.
  - The body says that the 9 goldens changed in `events` only (+16 key events), with everything else verified identical.

### Task 2: Hover ghost and the click-orbit menu ("Add burn here" / "Warp to here")

**Files:**
- Modify: `orbitsim/render/picking.py`: add `project_points_px`, `PolylinePick`, `pick_polyline_epoch`, `point_at`. Remove `nearest_future_point` (untested; its only caller goes away).
- Modify: `orbitsim/render/node_edit.py`: add `MIN_NODE_LEAD_S`, `node_epoch_window`, `node_epoch_ok`, `screen_constant_scale`, `menu_pos`.
- Modify: `orbitsim/render/hud/format.py`: add `fmt_lead`.
- Create: `orbitsim/render/controllers/node_placement.py` (`NodePlacement`).
- Modify: `orbitsim/render/controllers/targeting.py`:
  - remove `on_mouse1_down`, `_mouse_px`, `try_pick_target`;
  - add `project_px`, `pick_body_at`, `coast_to` (the body and coast code of `try_pick_target`, `:108-135`, moves into these two).
- Modify: `orbitsim/render/controllers/flight_input.py`:
  - add `mouse_px` (moved from targeting), `mouse_over_gui`, `mouse1_held`;
  - `"mouse1"` → `node_placement.on_mouse1_down`;
  - `_handle_escape` closes an open menu first, right after the running-operation branch.
- Modify: `orbitsim/render/controllers/maneuver_editor.py`: add `place_node`; `_release_jogs` calls `node_placement.on_mouse1_up()` instead of `try_pick_target()`.
- Modify: `orbitsim/render/app.py`:
  - construct `self.node_placement = NodePlacement(self)` (the wiring test parses `self.x = Cls(self)`);
  - call `build_node_placement()` after `targeting.build_targeting()` (`:118`);
  - call `node_placement.frame_hover()` after `update_marker_readability` (`:246`).
- Modify: `keybind_overlay.py`: add `("Click orbit", "Add burn / warp to here")`.
- Modify: `tests/render/app_harness/hooks.py`, `scenarios.py`.
- Test: `tests/render/test_picking.py`, `tests/render/test_node_edit.py`, `tests/render/test_hud_format.py`, `tests/render/test_node_editing_app.py`.

**Interfaces:**
- Consumes: Task 1's `set_node_epoch`, `snap_epoch`.
- Produces (pure):
  - `def project_points_px(points_render: np.ndarray, render_to_clip: np.ndarray, width_px: int, height_px: int) -> np.ndarray`: (N, 2) bottom-up px; a row is NaN where clip w ≤ 1e-9 (behind the camera). The recipe is in "What planning found".
  - `@dataclass(frozen=True) class PolylinePick: epoch_s: float; index: int; fraction: float; distance_px: float`
  - `def pick_polyline_epoch(click_px, points_px, epochs_s, min_epoch_s, max_epoch_s, tol_px, tie_px=3.0) -> PolylinePick | None`
    - Each segment with both ends finite has its fraction range limited to where the lerped epoch is inside the window. Take the closest point in that range.
    - Keep hits with distance ≤ `tol_px` (`math.inf` allowed). Among hits within `best + tie_px`, return the earliest epoch.
    - numpy-vectorised.
  - `def point_at(points_m: np.ndarray, pick: PolylinePick) -> np.ndarray`: the lerp of the pick's two samples, shape (3,).
  - `MIN_NODE_LEAD_S = 10.0`
  - `def node_epoch_window(now_s, period_s: float | None, encounter_entry_epochs: list[float], line_end_s: float) -> tuple[float, float]`: `(now + MIN_NODE_LEAD_S, min(now + period, first entry, line_end))`, with a `None` period as +inf.
  - `def node_epoch_ok(epoch_s, now_s) -> bool`: `epoch_s >= now_s + MIN_NODE_LEAD_S`.
  - `def screen_constant_scale(distance_units, size_px, fov_v_deg, viewport_h_px) -> float`: `size_px * 2 * d * tan(fov/2) / h`.
  - `def menu_pos(px, width, height, menu_w=0.46, menu_h=0.17) -> tuple[float, float]`
    - Converts to aspect2d: `x = (px_x/W*2-1)*W/H + 0.03`, `z = py/H*2-1 - 0.03`.
    - Then clamps x to `[-W/H, W/H - menu_w]` and z to `[-1 + menu_h, 1]`.
  - `def fmt_lead(seconds) -> str`: whole seconds, floored; `"now"` under 1 s, otherwise `"in 45 s"`, `"in 36 min"`, `"in 2 h 05 min"`, `"in 3 d 04 h"`.
- Produces (controllers):
  - `FlightInput.mouse_px() -> tuple[float, float] | None`, `mouse_over_gui() -> bool` (`mw.get_over_region() is not None`), `mouse1_held() -> bool | None` (`mw.is_button_down(MouseButton.one())`). All are safe when `mouseWatcherNode is None`.
  - `Targeting.project_px(points_m: np.ndarray) -> np.ndarray`: geocentric m → px via `transform`, the camera matrix and `project_points_px`.
  - `Targeting.pick_body_at(click_px) -> bool`: today's body pick (22 pt): clears the prior target, selects the new one, returns True on a hit.
  - `Targeting.coast_to(epoch_s: float) -> bool`:
    - Thrusting: flash "Cut throttle before warping", return False.
    - `epoch_s <= now`: flash "That point has passed", return False.
    - Otherwise: today's coast operation with rollback (`"Coasting … along trajectory"`), return True.
  - `ManeuverEditor.place_node(epoch_s: float) -> None`: `_auto_execute_node = False`, then `set_node_epoch(snap_epoch(epoch_s))`. Δv is kept.
  - `NodePlacement`:
    - `build_node_placement()`: app state `_hover_pick=None`, `_ghost_np=None`, `_ghost_label=None`, `_orbit_menu=None`, `_orbit_menu_pick=None`.
    - `orbit_pick_at(px, tol_px, window=None) -> tuple[float, np.ndarray] | None`: picks on `_traj_world_cache[0]` / `_traj_epoch_cache[0]`. The default window is `(now + MIN_NODE_LEAD_S, line end)`.
    - `node_window() -> tuple[float, float]`:
      - period from `state_to_elements(v0.state).period_s` (`None` on ValueError or when unbound);
      - entries are the `entry_epoch_s` of `_traj_encounters[0]` for bodies other than Earth and Sun.
    - `on_mouse1_down()`:
      - ignored while `operation.status.running`;
      - with the menu open: close it and swallow the click (`_mouse1_down_px = None`);
      - otherwise record `_mouse1_down_px`. (Task 3 inserts the handle hit-test here.)
    - `on_mouse1_up()`: on a tap (≤ 6 pt move), `pick_body_at` first; otherwise an orbit pick at 14 pt opens the menu on a hit.
    - `frame_hover()`:
      - Hides the ghost (label text `""`) when the menu is open, the mouse is off-window or over the GUI, `_rmb_down`, or there is no line or no hit.
      - Otherwise places the ghost at `point_at(pick)`: a sphere of radius 6 pt via `screen_constant_scale`, with label `fmt_lead(epoch - now)`.
      - An open menu re-anchors to its picked point's px each frame, keeping its last position when the point is off-screen.
    - `menu_open() -> bool` (safe in solar mode), `close_menu() -> None`.
  - **Menu**: a DirectFrame on `aspect2d` (THEME panel), built on open and **destroyed** on close, so no events linger in goldens. Buttons use `button_options()`:
    - `"Add burn here"`, or `"Move burn here"` when a node or Δv exists. Shown only if the epoch is ≤ `node_window()[1]`. It runs `node_epoch_ok`, then `place_node`; otherwise it flashes "That point has passed".
    - `"Warp to here"`: `coast_to(epoch)`.
    - Both close the menu.
  - hooks:
    - `set_mouse_px(app, px)`: patches `app.flight_input.mouse_px`; `None` = off-window.
    - `orbit_point_px(app, i)` and `orbit_point_epoch(app, i)`: live-line sample `i`, via `targeting.marker_px`.
    - `body_px(app, name)`.
    - `orbit_menu_labels(app) -> list[str] | None`.
    - `settings_visible(app) -> bool`.
    - `first_sample_after(app, lead_s) -> int`: the first index whose epoch is ≥ now + `lead_s`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/render/test_picking.py (append)
import math, numpy as np, pytest
from orbitsim.render.picking import pick_polyline_epoch, project_points_px
PTS = np.array([[0, 0], [100, 0], [200, 0], [300, 0]], float); EP = np.array([0., 10., 20., 30.])

def test_pick_interpolates_epoch_along_segment():
    p = pick_polyline_epoch((150.0, 5.0), PTS, EP, 0.0, 30.0, tol_px=10.0)
    assert p.index == 1 and p.fraction == pytest.approx(0.5)
    assert p.epoch_s == pytest.approx(15.0) and p.distance_px == pytest.approx(5.0)
def test_pick_misses_beyond_tolerance():
    assert pick_polyline_epoch((150.0, 20.0), PTS, EP, 0.0, 30.0, tol_px=10.0) is None
def test_pick_clamps_to_time_window():
    assert pick_polyline_epoch((50.0, 0.0), PTS, EP, 12.0, 30.0, tol_px=math.inf).epoch_s == pytest.approx(12.0)
def test_pick_prefers_earliest_of_overlapping_loops():
    loop = np.array([[0, 0], [100, 0], [100, 1], [0, 1]], float)   # segment 2 runs back over segment 0
    assert pick_polyline_epoch((50.0, 0.9), loop, EP, 0.0, 30.0, tol_px=10.0).epoch_s == pytest.approx(5.0)
def test_pick_skips_points_behind_camera():
    pts = PTS.copy(); pts[1] = np.nan
    assert pick_polyline_epoch((50.0, 0.0), pts, EP, 0.0, 30.0, tol_px=10.0) is None
def test_pick_none_for_empty_window():
    assert pick_polyline_epoch((50.0, 0.0), PTS, EP, 25.0, 20.0, tol_px=math.inf) is None
def test_project_identity_and_behind_camera():
    px = project_points_px(np.array([[0, 0, 0], [1, 1, 0]], float), np.eye(4), 960, 600)
    assert px.tolist() == [[480.0, 300.0], [960.0, 600.0]]
    m = np.eye(4); m[:, 3] = [0, 1, 0, 0]                          # clip w = y
    assert np.isnan(project_points_px(np.array([[0, -1, 0]], float), m, 960, 600)).all()
```

```python
# tests/render/test_node_edit.py (append)
def test_window_is_one_period_for_a_bound_orbit():
    assert node_epoch_window(100.0, 5000.0, [], 20000.0) == (110.0, 5100.0)
def test_window_stops_at_first_encounter():
    assert node_epoch_window(100.0, None, [9000.0, 3000.0], 40000.0) == (110.0, 3000.0)
def test_window_runs_to_line_end_when_unbound():
    assert node_epoch_window(100.0, None, [], 40000.0) == (110.0, 40000.0)
def test_node_epoch_ok_needs_lead():
    assert node_epoch_ok(110.0, 100.0) and not node_epoch_ok(109.9, 100.0)
def test_screen_constant_scale():
    assert screen_constant_scale(1000.0, 10.0, 30.0, 600) == pytest.approx(
        10 * 2000 * math.tan(math.radians(15)) / 600)
def test_menu_pos_centre_and_clamp():
    assert menu_pos((480.0, 300.0), 960, 600) == pytest.approx((0.03, -0.03))
    x, z = menu_pos((959.0, 2.0), 960, 600)
    assert x == pytest.approx(1.6 - 0.46) and z == pytest.approx(-1 + 0.17)

# tests/render/test_hud_format.py (append)
def test_fmt_lead():
    assert [fmt_lead(s) for s in (0.4, 45, 36 * 60 + 10, 2 * 3600 + 300, 3 * 86400 + 4 * 3600)] == [
        "now", "in 45 s", "in 36 min", "in 2 h 05 min", "in 3 d 04 h"]
```

Probes go in `test_node_editing_app.py`. Each launches, then `h.step(30)`. `_click(h, px)` means: `set_mouse_px`, step 1, send `mouse1`, send `mouse1-up`, step 1.
- `test_hover_shows_ghost_with_lead_time`: hover sample 40, then step 2. A visible `world_labels` entry starts with `"in "` and ends `" min"`.
- `test_click_orbit_opens_menu_and_adds_burn`:
  1. Click sample 40. The menu labels are `["Add burn here", "Warp to here"]`.
  2. Press "Add burn here". The menu is `None`.
  3. `node_state[0]` is within 60 s of `orbit_point_epoch(40)`, and Δv is all 0.
- `test_menu_says_move_when_a_burn_exists`: after adding a burn, click sample 60. The first label is `"Move burn here"`.
- `test_escape_closes_menu_not_settings`: click, then send `escape`. The menu is `None` and `settings_visible` is False.
- `test_warp_to_here_coasts`: click sample 40, press "Warp to here", step 3. Sim time advances ≥ 2,000 s.
- `test_warp_refused_while_thrusting`: send `"z"`, step 2, click, press "Warp to here", step 2. Sim time advances < 10 s.
- `test_add_burn_refused_once_passed`:
  1. Click sample `i = hooks.first_sample_after(app, 60.0)`, the first live-line sample at least 60 s ahead of now.
  2. Step 80 frames (about 130 s of sim time at ×100).
  3. Press "Add burn here". Expect `node_state[0] is None`.

  Use sample indices only through this hook or with a large lead (sample 40 is about 2,160 s ahead). The line refreshes every 2 s of *real* time, so at ×100 its first sample can be about 200 s of sim time in the past.
- `test_click_body_still_targets`:
  1. 12× `wheel_down`, step 90.
  2. Click `body_px("Moon")`. Target is `"Moon"` and the menu is `None`.
  3. Click (5, 5). The menu is `None` and the target is unchanged.

- [ ] **Step 2: Run them, expect FAIL** (ImportError / missing hooks).
- [ ] **Step 3: Implement** the helpers, the controller, the targeting/flight-input/editor changes, the hooks and the overlay line.
- [ ] **Step 4: Run** `tests/render/test_picking.py tests/render/test_node_edit.py tests/render/test_hud_format.py tests/render/test_node_editing_app.py tests/render/test_controller_wiring.py`. Expected: all pass.
- [ ] **Step 5: New goldens.** Add these scenarios, each starting with launch + step 30:
  - `orbit_menu_add_burn`: click sample 40, press "Add burn here", step 30.
  - `orbit_menu_warp_here`: click sample 40, escape, click sample 40, press "Warp to here", step 10.
  - `click_body_targets`: as in the probe.

  Then run `record orbit_menu_add_burn orbit_menu_warp_here click_body_targets` and the full suite. Task 1's 10 goldens must be unchanged: there are no new events, the menu is destroyed on close, and the ghost text is `""` with no mouse.
- [ ] **Step 6: Eyeball it.**
  - Scratch shooter at 1600×900 (`install_determinism(win_size="1600 900")`): hover and click with the hooks, then `app.win.save_screenshot`. Check that the ghost and "in N min" sit on the line, and that the menu sits at the click, is readable, and is clamped near the edges.
  - Once in a real window (`python -m orbitsim`): clicking a menu button must not reopen or close the menu (DirectGUI suppresses `mouse1` on its regions).
- [ ] **Step 7: Commit** `feat(render): hover your orbit for a ghost node; click for Add burn here / Warp to here`.

### Task 3: 3D Δv handles on the node

**Files:**
- Modify: `orbitsim/core/maneuvers.py`: add `rtn_basis`. Don't touch `apply_maneuver` or `maneuver_direction`: their unguarded NaN on |v| = 0 is what the per-frame marker relies on today.
- Modify: `orbitsim/render/node_edit.py`: add `HANDLES`, `HandleDrag`, `axis_unit_px`, `handle_drag_value`, and the drag constants.
- Create: `orbitsim/render/controllers/node_handles.py` (`NodeHandles`).
- Modify: `orbitsim/render/controllers/node_placement.py`: `on_mouse1_down` asks `node_handles.hit_test` first. A hit starts `begin_drag` and is not a tap; `on_mouse1_up` ends the drag. While the pointer is on a handle or the node, or while dragging, the ghost is hidden.
- Modify: `orbitsim/render/controllers/maneuver_editor.py`: the fixed `set_scale(6.0)` on the marker goes. `node_handles.frame_handles` scales it each frame.
- Modify: `orbitsim/render/app.py`:
  - construct `self.node_handles = NodeHandles(self)`;
  - call `build_node_handles()` beside `build_node_placement()`;
  - call `node_handles.apply_drag(real_dt)` right after `apply_nudges`;
  - call `node_handles.frame_handles()` after `update_marker_readability` and before `frame_hover`.
- Modify: `keybind_overlay.py`: add `("Drag handles", "Shape the burn")`.
- Modify: `hooks.py`, `scenarios.py`.
- Test: `tests/core/test_maneuvers.py`, `tests/render/test_node_edit.py`, `tests/render/test_node_editing_app.py`.

**Interfaces:**
- Consumes: Task 1's `jog_rate`, `snap_dv`, `set_dv`; Task 2's `screen_constant_scale`, `Targeting.marker_px`, `FlightInput.mouse_px` / `mouse1_held`.
- Produces:
  - `def rtn_basis(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (core): `(v_hat, h_hat, r_hat)` with `apply_maneuver`'s formulas (`r_hat = h_hat × v_hat`). Raises `ValueError` if |v| == 0 or |r × v| == 0.
  - `HANDLES: tuple[tuple[str, str, int, tuple[float, float, float]], ...]`: `(kind, axis, sign, rgb)` for `"pro+"`, `"pro-"`, `"nrm+"`, `"nrm-"`, `"rad+"`, `"rad-"`, with the colours from Decisions.
  - `@dataclass(frozen=True) class HandleDrag: kind: str; start_px: tuple[float, float]; axis_px: tuple[float, float] | None` (`kind == "node"` has no axis; Task 4).
  - `def axis_unit_px(node_px, handle_px, min_len_px) -> tuple[float, float] | None`
  - `def handle_drag_value(start_px, mouse_px, axis_unit, max_drag_px) -> float`: `clamp(dot(mouse - start, axis_unit) / max_drag_px, -1, 1)`.
  - Constants: `HANDLE_OFFSET_PX = 36`, `HANDLE_RADIUS_PX = 5`, `NODE_RADIUS_PX = 7`, `HANDLE_HIT_PX = 12`, `NODE_HIT_PX = 14`, `MIN_AXIS_PX = 8`, `MAX_DRAG_PX = 150`, `DRAG_MAX_RATE_MPS = 400.0`.
  - `NodeHandles`:
    - `build_node_handles()`: app state `_drag=None`, `_handle_nps={}` (built on first use), `_handle_px={}`, `_node_px=None`, `_handle_hover=None`.
    - `frame_handles()`. Hide all handles (and `_node_px = None`) when `app._node_marker_np is None` or `rtn_basis` raises. Otherwise:
      1. Find the burn point: `propagate_kepler(v0.state, epoch - v0.state.epoch_s)`.
      2. Set the camera distance `d = (cam.get_pos(render) - marker_pos).length()`, and size from it with `screen_constant_scale(d, …, camLens.get_fov()[1], win.get_y_size())`. Every pt size is multiplied by `app.line_px_scale()`.
      3. Scale the marker to `NODE_RADIUS_PX`.
      4. Put each handle sphere at `marker + dir * offset` (`HANDLE_RADIUS_PX`; the − handles 0.75× and alpha 0.55). The dragged handle sits further out, at `offset * (1 + max(0, value))`.
      5. Cache `_node_px` and `_handle_px` via `marker_px`. A pair whose `axis_unit_px` is `None` is hidden, and its px is `None`.
      6. Scale the hovered or dragged handle to 1.3×.
    - `hit_test(px) -> str | None`: handles first (`HANDLE_HIT_PX`), then the node (`"node"`, `NODE_HIT_PX`), both via `picking.nearest_marker`.
    - `begin_drag(kind, px)`, `end_drag()`, `dragging() -> str | None`.
      - `end_drag` snaps the axis with `snap_dv` and calls `set_dv`.
      - `begin_drag` stores `axis_unit_px(_node_px, _handle_px[kind], ...)`.
    - `apply_drag(real_dt)`:
      - If `mouse1_held() is False`, call `end_drag()` and return.
      - If the mouse is off-window, do nothing.
      - Otherwise apply `set_dv(axis, dv + sign * jog_rate(handle_drag_value(...), DRAG_MAX_RATE_MPS, 4.0, 0.02) * real_dt)`.
  - hooks:
    - `node_px(app)`, `handle_px(app, kind)`;
    - `set_mouse1_held(app, held)` (patches `flight_input.mouse1_held`; `None` restores it);
    - `dragging(app)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_maneuvers.py (append)
from orbitsim.core.maneuvers import rtn_basis

def test_rtn_basis_circular_equatorial():
    pro, nrm, rad = rtn_basis(np.array([7e6, 0, 0]), np.array([0, 7.5e3, 0]))
    np.testing.assert_allclose([pro, nrm, rad], [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-15)

def test_rtn_basis_matches_apply_maneuver():
    s = _periapsis_state()
    for dv, i in (((10.0, 0, 0), 0), ((0, 10.0, 0), 1), ((0, 0, 10.0), 2)):
        after = apply_maneuver(s, ManeuverNode(s.epoch_s, *dv))
        np.testing.assert_allclose(after.v - s.v, 10.0 * rtn_basis(s.r, s.v)[i], atol=1e-9)

def test_rtn_basis_raises_on_zero_velocity():
    with pytest.raises(ValueError):
        rtn_basis(np.array([7e6, 0, 0]), np.zeros(3))
```

```python
# tests/render/test_node_edit.py (append)
def test_axis_unit_px():
    assert axis_unit_px((0, 0), (30, 40), 8) == pytest.approx((0.6, 0.8))
def test_axis_unit_px_none_when_foreshortened():
    assert axis_unit_px((0, 0), (3, 4), 8) is None
def test_handle_drag_value_along_against_and_across():
    assert handle_drag_value((0, 0), (75, 0), (1, 0), 150) == 0.5
    assert handle_drag_value((0, 0), (-300, 0), (1, 0), 150) == -1.0
    assert handle_drag_value((0, 0), (0, 80), (1, 0), 150) == 0.0
def test_handles_cover_each_axis_both_ways():
    assert sorted((a, s) for _k, a, s, _c in HANDLES) == sorted(
        (a, s) for a in ("pro", "nrm", "rad") for s in (1, -1))
```

Probes (`DRAG`). After the Task 2 click flow adds a burn at sample 40, `h.step(5)`. Then `start = handle_px(kind)`, `node = node_px()`, mouse to `start`, step 1, send `mouse1`, mouse to `start + 1.5*(start - node)`, step 60, send `mouse1-up`, step 30.
- `test_prograde_handle_adds_only_prograde`: `"pro+"` gives `pro > 0`, `nrm == rad == 0.0`, the epoch unchanged, and `pro == round(pro, 1)`.
- `test_radial_in_handle_lowers_radial`: `"rad-"` gives `rad < 0` and the others 0.
- `test_drag_ends_when_button_no_longer_held`: after `mouse1` on a handle, `set_mouse1_held(app, False)`, step 2 → `dragging is None` and the Δv frozen over 30 more frames.
- `test_handles_hidden_without_node`: before any burn, `node_px is None`.

- [ ] **Step 2: Run them, expect FAIL.**
- [ ] **Step 3: Implement** `rtn_basis`, the helpers, the controller, the dispatch, the hooks and the overlay line.
- [ ] **Step 4: Run** `tests/core/test_maneuvers.py tests/render/test_node_edit.py tests/render/test_node_editing_app.py tests/render/test_controller_wiring.py`, then the full suite. Every golden is unchanged (handles have no text or events; the marker isn't snapshotted).
- [ ] **Step 5: New golden** `handle_drag_prograde`: the `"pro+"` probe flow as a scenario. Then `record handle_drag_prograde`.
- [ ] **Step 6: Eyeball it.** Screenshot a node with handles: at the opening zoom, zoomed in (`wheel_up` ×15), and with the camera orbited (arrows). Check the colours, that the handles keep their size, and that nothing overlaps the node label. Then drag a handle in a real window: the orange estimate follows every frame and settles to the exact line.
- [ ] **Step 7: Commit** `feat(render): drag 3D prograde/normal/radial handles to shape a burn`.

### Task 4: Drag the node to slide it along the orbit

**Files:**
- Modify: `orbitsim/render/controllers/node_handles.py` (the `"node"` drag kind)
- Modify: `keybind_overlay.py`: add `("Drag node", "Slide the burn along the orbit")`
- Modify: `scenarios.py`
- Test: `tests/render/test_node_editing_app.py`

**Interfaces:**
- Consumes: Task 2's `NodePlacement.orbit_pick_at` and `node_window()`, and `ManeuverEditor.set_node_epoch`; Task 3's `hit_test` returning `"node"`, `HandleDrag(kind="node", axis_px=None)`, `snap_epoch`.
- Produces: `apply_drag` for `"node"`:
  - Each frame, `pick = node_placement.orbit_pick_at(mouse, tol_px=math.inf, window=node_placement.node_window())`. When there's a pick, `set_node_epoch(pick epoch)`. With no line or no mouse, the node stays.
  - `end_drag` calls `set_node_epoch(snap_epoch(epoch))`. Δv is untouched.
  - Because the drag is clamped to the node window, it can't leave the current orbit or go under `MIN_NODE_LEAD_S`.

- [ ] **Step 1: Write the failing probes** (`SLIDE`). Add a burn at sample 40 via the menu, nudge `j` once (+1 m/s), step 5. Then mouse to `node_px`, `mouse1`, mouse to `orbit_point_px(80)`, step 10, `mouse1-up`, step 5.
  - `test_dragging_node_slides_its_time`: `abs(epoch - orbit_point_epoch(80)) <= 60` (one sample), the epoch is a whole second, and Δv is still `(1.0, 0.0, 0.0)`.
  - `test_node_drag_stays_on_current_orbit`: the same, but drop at sample 250 (second loop, past one period). Then `epoch <= now + period` (period via `state_to_elements`, read in the probe) and `epoch >= now + 10`.
- [ ] **Step 2: Run them, expect FAIL** (the epoch doesn't change).
- [ ] **Step 3: Implement** the node kind.
- [ ] **Step 4: Run** `tests/render/test_node_editing_app.py`, then the full suite: green.
- [ ] **Step 5: New golden** `node_drag_slides_time` (the first probe's flow). Then `record node_drag_slides_time`.
- [ ] **Step 6: Commit** `feat(render): drag the burn node to slide it along your orbit`.

### Task 5: Frame-time gates, docs, review, PR

**Files:**
- Modify: `tests/render/app_harness/profile.py` (workload `node_drag`)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the `node_drag` workload.**
  - **setup:** launch, step 30, then add a burn at sample 40 through hooks and the menu, as in Task 2. Then mouse to `handle_px("pro+")`, send `mouse1`.
  - **per_frame:** mouse to `start + (i % 60) * 2 pt` along the handle's axis (a continuous drag, so the estimate redraws every frame).
- [ ] **Step 2: Profiler gate.**
  1. Export `main` with `git archive` into the scratchpad, as in plans 1 and 2.
  2. Run `idle`, `zoom`, `zoom_node_planned`, `high_warp` and `node_planned` on both trees, same session. The mean may rise no more than 10%.
  3. Run `node_drag` on the branch: mean ≤ 16.7 ms (expected about 7–8 ms; see "What planning found").
  4. Put the numbers in the PR body.
- [ ] **Step 3: CLAUDE.md.**
  - Add `node_placement` and `node_handles` to the controller list.
  - New gotchas:
    - Headless windows have no `mouseWatcherNode`. Mouse and keyboard reads go through `flight_input.mouse_px/mouse1_held/mouse_over_gui/key_held`, which the harness patches via `hooks.set_mouse_px` etc.
    - Panda names punctuation keys by the character (`","`, not `"comma"`).
    - Modifiers prefix events (`shift-j`, `shift-j-up`), so held-key and drag state is re-checked each frame against the real button state.
    - Every `accept()` changes the `events` field of every sandbox golden.
    - Node handles, keys and the slider all write `app._dv` / `app._node_epoch_s` through `set_dv` / `set_node_epoch`.
- [ ] **Step 4: Full suite** `-n auto` green. `record --check-only` passes for all scenarios.
- [ ] **Step 5: Commit** `test(render): node-drag frame-time workload; CLAUDE.md notes for burn editing`.
- [ ] **Step 6: Final whole-branch review**, then the fix pass. Then the PR, titled "Map readability 3/5: place and shape burns by hand", ending with the attribution line. Merge.
