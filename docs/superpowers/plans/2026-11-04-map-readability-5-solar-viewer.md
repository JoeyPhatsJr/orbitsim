# Map Readability, Plan 5 of 5: The Solar Viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the solar viewer (`--solar`), every planet's orbit is drawn as a line, planet markers and names stay a constant, readable size on screen at any zoom, and names never pile up on each other, on other planets' dots, or on the HUD.

**Architecture:**
1. **Orbit lines.** The solar viewer gets its own orbit lines, built from one new core helper, `core.planets.heliocentric_state` (DE440, or the circular fallback offline). The geometry is baked in 1e9 m units under one frame node that is rescaled each frame, so zooming never rebuilds it; lines are rebuilt only every 30 sim-days.
2. **Constant-size markers and labels.** Each frame, every marker and label is scaled by `units_per_pixel(depth)`, so its size is set in pixels.
3. **Declutter.** A pure, rect-based helper `render/label_layout.py::layout_labels` places each name in one of four slots around its dot, or hides it. It works by priority, keeps hovered names (`pinned`), and treats other dots and the HUD as obstacles.

**Tech Stack:** Python 3.11, numpy, Panda3D (`LineSegs`, billboards, `PerspectiveLens`), pytest (+xdist), and the app-golden harness in `tests/render/app_harness/`.

**Spec:** `docs/superpowers/specs/2026-11-04-map-and-planning-readability-design.md`, "The look": the "Solar viewer", "Labels" and "Lines" bullets. Plans 1–4 are merged before this one starts. Plan 1 provides `render/line_style.py` (`REFERENCE_STYLE`, `TARGET_REFERENCE_STYLE`, `reference_style`, `px_scale`) and the unlit, pixel-width `build_orbit_node`. Plan 4 lays out the sandbox HUD, fixes the warp readout overflow, and sets the sandbox label rules. Plan 5 touches the solar viewer only.

## What planning found (2026-11-04, M1 Pro, 1600×900 offscreen, DE440 on disk)

Before screenshots were taken at 2030-01 with camera distance 2e12 m (boot), 4e11 m and 8e12 m.

- **Why there are no orbit lines.** The solar viewer never builds any. `OrbitApp._start_sim` calls only `solar_view.build_planets()` in solar mode, and `_update` returns right after `solar_view.update_solar_system()`, which only places markers. The planet orbit lines live in `bodies_view.build_sandbox_planets` / `frame_planets`, and those run in the sandbox only. So the `self.app.solar_system or planet_orbits_visible(...)` test in `frame_planets` is dead code. It isn't scale, frame or visibility.
- **A prototype confirms the fix.** It drew the 8 osculating heliocentric orbits (`sample_relative_orbit_points`, Plan 1's reference style with the existing per-planet tints) under a frame node scaled `1e9 / scale_m_per_unit`. All 8 lines render correctly at 2e12 and 1e13 m. At the boot zoom, the Saturn ellipse fills the view, and the Uranus and Neptune orbits enter as long arcs across the top-left corner.
- **Why the labels pile up.** The labels are world-space billboards at a fixed 15 render units, depth-tested, with a fixed +6 world-Z offset.
  - Because of perspective their size varies wildly: at 8e12 m Neptune's label is about 4× Jupiter's.
  - Their cards occlude each other: Earth's name is drawn behind Venus's card, and "Sun"/"Mercury" overlap.
  - At boot, Sun, Mercury, Venus, Earth and Mars all sit within about 130 px of the screen centre.
- **Why the markers are hard to read.** The markers are fixed at 4 render units, about 6.7 px at the focus depth and different elsewhere. Earth's marker carries the `earth_day` texture, so at about 12 px it reads as a near-black dot. There are also two Sun spheres: `bodies_view.build_central_body` makes `central_np` (10 units) and `solar_view` makes a Sun marker (8 units).
- **The lens.** `camLens.get_fov()` is (50.9°, 30°). One label text unit is 1.475 tall on its card (`get_card_actual()` gives b = −0.355, t = 1.12).
- **Profiler baseline.** `python -m tests.render.app_harness.profile solar` gives a mean of **1.92 ms** per frame. About 1.25 ms of that is 8 `body_state` Skyfield calls per frame. A prototype `layout_labels` over 9 labels costs about 25 µs.

## Global Constraints

- **The physics doesn't change.** App-golden fields `vessel`, `sim_time_s`, `warp`, `target` and `events` stay identical. Only `world_labels` (and `texts`, which isn't expected to change) may be re-recorded for `solar_boot_warp` / `solar_porkchop_key`, with the reason in the commit.
- **No new `accept()` bindings in solar mode.** They would change the frozen `events` field. Hover is polled from `mouseWatcherNode`.
- **No frame-rate regression >10%** on `solar`, `idle`, `zoom`, `zoom_node_planned` and `high_warp` (`.venv/bin/python -m tests.render.app_harness.profile <w>`), measured against `main` in the same session on the same machine.
- **Per-frame ephemeris calls stay as they are today:** one `heliocentric_state` per planet per frame, and none for the Sun. Orbit rebuilds reuse that frame's states.
- **Layering** (CLAUDE.md):
  - `core/` never imports Panda3D;
  - new math lives in `core/` with known-answer tests;
  - `render/label_layout.py` is pure (no Panda3D import);
  - controllers stay stateless and all state lives on `OrbitApp`;
  - test code reaches `OrbitApp` private attributes only through `tests/render/app_harness/hooks.py`.
- **Line look (Plan 1):** reference alpha, constant pixel width × `OrbitApp.line_px_scale()`, unlit (`build_orbit_node` already does all three), and a muted per-planet tint.
- **Pixel sizes** are framebuffer pixels × `OrbitApp.line_px_scale()` (Retina- and window-aware). Projections use `win.get_fb_x_size()` / `get_fb_y_size()`, with y up (as in `Targeting.marker_px`).
- **Process:**
  - branch `map-readability-5-solar-viewer` from up-to-date `main`;
  - TDD, then commit and push after each task;
  - stage by explicit path, never `git add -A` or `.`;
  - commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - at the end, a whole-branch review, then PR, then merge.

## Review Focus

1. **A planet behind the camera or off-screen.** This happens when zoomed in: the outer planets can sit behind the camera. Expected: the label is hidden, the dot is never giant, and nothing crashes. Pinned by `test_units_per_pixel_rejects_bad_input` and `test_offscreen_anchor_is_hidden_and_no_obstacle` (Task 3), and the inner-zoom probe in Task 4.
2. **Offline (no DE440).** Orbits, markers and labels come from the circular fallback, and the orbit lines still pass through the dots. Pinned by `test_heliocentric_state_fallback_*` (Task 1) and `test_solar_viewer_offline_draws_orbits` (Task 2).
3. **High or max warp (1e6× and up).** Orbit lines rebuild once per 30-sim-day bucket, never every frame or on zoom, and names don't hop between slots frame to frame. Pinned by `test_solar_orbits_rebuild_per_bucket_not_on_zoom` (Task 2) and `test_preferred_slot_is_kept_when_still_free` (Task 3).
4. **A name under the warp bar or the HUD text** (a planet near the top of the screen). Expected: the name moves to another slot or hides; it never sits on the buttons. Pinned by `test_obstacles_block_labels` (Task 3) and the obstacle wiring in Task 5.
5. **Retina, or a resized window.** Dots and names keep their pixel size, scaled by `line_px_scale()`. Pinned by `test_units_per_pixel_scales_with_depth_and_viewport` (Task 3). Task 4's probe measures on-screen size at two zooms.

---

### Task 1: Pure pieces — heliocentric states and planet orbit styles

**Files:**
- Modify: `orbitsim/core/planets.py` (add `heliocentric_state`)
- Modify: `orbitsim/render/line_style.py` (add `PLANET_ORBIT_TINTS`, `planet_orbit_style`)
- Modify: `orbitsim/render/controllers/bodies_view.py:376-464`. Use `planet_orbit_style` there, delete the local `_orbit_colors` table and the now-unused `dataclasses.replace` import, and change `if self.app.solar_system or planet_orbits_visible(...)` to `if planet_orbits_visible(...)` (the `solar_system` arm was unreachable).
- Modify: `orbitsim/render/controllers/solar_view.py` (`_helio_body_pos` → `heliocentric_state(...).r`)
- Test: `tests/core/test_planets.py`, `tests/render/test_line_style.py`

**Interfaces:**
- Produces: `heliocentric_state(name: str, t_s: float) -> StateVector` in `orbitsim/core/planets.py`.
  - `name` is a body name as in `core.bodies` (`"Sun"`, `"Mercury"` … `"Neptune"`), case-insensitive.
  - Returns heliocentric J2000/ICRF r [m] and v [m/s], with `mu=MU_SUN` and `epoch_s=t_s`.
  - `"Sun"` returns zero r and v.
  - With DE440: exactly `ephemeris.body_state(NAME, t_s, center="SUN")`, with `r`/`v` bit-identical.
  - Fallback, on `ImportError` (no skyfield) or `EphemerisUnavailableError`: for Earth, `r = -sun_state_at(t).r` and `v = -sun_state_at(t).v`; for the others, `<planet>_state_at(t) − sun_state_at(t)` for both r and v.
  - Unknown name raises `ValueError`.
  - Import `orbitsim.core.ephemeris` lazily inside the function and call `ephemeris.body_state` through the module, so tests can monkeypatch it.
- Produces: in `orbitsim/render/line_style.py`:
  - `PLANET_ORBIT_TINTS: dict[str, tuple[float, float, float]]`, the RGB of today's `bodies_view._orbit_colors`, moved verbatim: Mercury (0.5, 0.5, 0.5), Venus (0.85, 0.75, 0.45), Earth (0.3, 0.5, 1.0), Mars (0.85, 0.4, 0.2), Jupiter (0.8, 0.7, 0.5), Saturn (0.9, 0.8, 0.6), Uranus (0.6, 0.85, 0.9), Neptune (0.3, 0.4, 0.9).
  - `planet_orbit_style(name: str, targeted: bool = False) -> LineStyle`, which returns `reference_style(targeted)` with `color = tint + (that style's alpha,)`. Unknown name raises `ValueError`. The sandbox's lines stay bit-identical.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_planets.py
import pytest
from orbitsim.core import ephemeris
from orbitsim.core.planets import heliocentric_state

T = 958_000_000.0   # 2030-05

def _offline(monkeypatch):
    def boom(*a, **k):
        raise ephemeris.EphemerisUnavailableError("offline")
    monkeypatch.setattr(ephemeris, "body_state", boom)

def test_heliocentric_state_sun_is_origin():
    s = heliocentric_state("Sun", T)
    assert np.all(s.r == 0.0) and np.all(s.v == 0.0) and s.mu == MU_SUN and s.epoch_s == T

def test_heliocentric_state_matches_de440_exactly():
    if not ephemeris.available():
        pytest.skip("DE440 not available")
    s, ref = heliocentric_state("mars", T), ephemeris.body_state("MARS", T, center="SUN")
    assert np.array_equal(s.r, ref.r) and np.array_equal(s.v, ref.v) and s.mu == MU_SUN

def test_heliocentric_state_fallback_earth_is_minus_sun(monkeypatch):
    _offline(monkeypatch)
    s, sun = heliocentric_state("Earth", T), sun_state_at(T)
    np.testing.assert_array_equal(s.r, -sun.r); np.testing.assert_array_equal(s.v, -sun.v)

def test_heliocentric_state_fallback_planet_is_geo_minus_sun(monkeypatch):
    _offline(monkeypatch)
    s, sun, geo = heliocentric_state("Jupiter", T), sun_state_at(T), jupiter_state_at(T)
    np.testing.assert_array_equal(s.r, geo.r - sun.r); np.testing.assert_array_equal(s.v, geo.v - sun.v)
    assert np.linalg.norm(s.r) == pytest.approx(A_JUPITER, rel=1e-10)

def test_heliocentric_state_rejects_unknown_body():
    with pytest.raises(ValueError):
        heliocentric_state("Pluto", T)
```

```python
# tests/render/test_line_style.py
from orbitsim.render.line_style import (PLANET_ORBIT_TINTS, planet_orbit_style,
    REFERENCE_STYLE, TARGET_REFERENCE_STYLE)

def test_planet_orbit_style_is_tinted_reference():
    s = planet_orbit_style("Earth")
    assert s.color == (0.3, 0.5, 1.0, REFERENCE_STYLE.color[3])
    assert (s.width_px, s.halo_px, s.dash) == (REFERENCE_STYLE.width_px, REFERENCE_STYLE.halo_px, None)

def test_planet_orbit_style_targeted_uses_highlight_width_and_alpha():
    s = planet_orbit_style("Mars", targeted=True)
    assert s.color[3] == TARGET_REFERENCE_STYLE.color[3] and s.width_px == TARGET_REFERENCE_STYLE.width_px

def test_every_planet_has_a_tint_and_unknown_raises():
    assert set(PLANET_ORBIT_TINTS) == {"Mercury", "Venus", "Earth", "Mars",
                                       "Jupiter", "Saturn", "Uranus", "Neptune"}
    with pytest.raises(ValueError):
        planet_orbit_style("Sun")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/core/test_planets.py tests/render/test_line_style.py -q`
Expected: FAIL with `ImportError` (`heliocentric_state`, `PLANET_ORBIT_TINTS`).

- [ ] **Step 3: Implement `heliocentric_state` and `planet_orbit_style`, then switch both call sites.** In `bodies_view.frame_planets`, `style = planet_orbit_style(oname, target == oname)`. In `solar_view`, replace `_helio_body_pos(name, t)` with `heliocentric_state(name, t).r` and delete `_helio_body_pos`.

- [ ] **Step 4: Run the tests and the goldens**

Run: `.venv/bin/python -m pytest tests/core/test_planets.py tests/render/test_line_style.py tests/render/app_harness -q -n auto`
Expected: all pass, and every golden is unchanged. Marker positions use the same `body_state` call, and the sandbox styles are bit-identical.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/planets.py orbitsim/render/line_style.py orbitsim/render/controllers/bodies_view.py orbitsim/render/controllers/solar_view.py tests/core/test_planets.py tests/render/test_line_style.py
git commit -m "refactor: heliocentric_state (core) and planet_orbit_style shared by sandbox and solar viewer"
```

### Task 2: Planet orbit lines in the solar viewer

**Files:**
- Modify: `orbitsim/render/controllers/solar_view.py` (`build_planets`, `update_solar_system`, new `frame_orbits`)
- Modify: `tests/render/app_harness/hooks.py` (add `solar_orbit_nodes`, `solar_orbit_frame_scale`)
- Test: `tests/render/test_solar_view_app.py` (new; subprocess probes in the style of `tests/render/test_opening_camera_app.py`)

**Interfaces:**
- Consumes:
  - Task 1's `heliocentric_state` and `planet_orbit_style`;
  - `orbit_lines.sample_relative_orbit_points(body_state, center_state, mu, n)`;
  - `build_orbit_node(points, style, px_scale)`.
- Produces: on `SolarView`:
  - `SOLAR_ORBIT_UNIT_M = 1.0e9` (render baking unit);
  - `SOLAR_ORBIT_REFRESH_S = 30 * 86400.0`;
  - `SOLAR_ORBIT_POINTS = 256`.
- Produces: new `OrbitApp` state, created in `build_planets`:
  - `_solar_orbit_frame`, a NodePath named `solar_orbit_frame` under `render`;
  - `_solar_orbit_nps: dict[str, NodePath]`, where each node is named `solar_orbit_<Planet>`;
  - `_solar_orbit_bucket: int | None = None`.
- Produces: `SolarView.frame_orbits(states: dict[str, StateVector]) -> None`, called from `update_solar_system` with that frame's heliocentric states (the same dict that places the markers). It works in two parts:
  - If `floor(t / SOLAR_ORBIT_REFRESH_S)` differs from `_solar_orbit_bucket`, it removes and rebuilds the 8 orbit nodes: `sample_relative_orbit_points(states[name], <zero StateVector>, MU_SUN, n=SOLAR_ORBIT_POINTS) / SOLAR_ORBIT_UNIT_M`, styled with `planet_orbit_style(name)` and `px_scale=self.app.line_px_scale()`.
  - Every frame: `_solar_orbit_frame.set_pos(*transform.to_render(zeros))` and `_solar_orbit_frame.set_scale(SOLAR_ORBIT_UNIT_M / transform.scale_m_per_unit)`.
- Produces: hooks.
  - `solar_orbit_nodes(app) -> dict[str, NodePath]` returns a copy of `_solar_orbit_nps`.
  - `solar_orbit_frame_scale(app) -> float` returns the frame's x scale.

- [ ] **Step 1: Write the failing probes** in `tests/render/test_solar_view_app.py`.

Each probe runs `python -c PROBE` in `REPO_ROOT` and parses a `RESULT` line, as `test_opening_camera_app.py` does. Each probe calls `install_determinism()`, `build_app("solar")` and `Harness(app).press("OPEN SOLAR VIEWER")`.

```python
PLANETS = ["Earth", "Jupiter", "Mars", "Mercury", "Neptune", "Saturn", "Uranus", "Venus"]

def test_solar_viewer_draws_every_planet_orbit():
    # probe: h.step(5); nodes = hooks.solar_orbit_nodes(app)
    # RESULT: sorted(nodes), all(not n.is_hidden() for n in nodes.values()),
    #         all(n.get_name() == f"solar_orbit_{k}" for k, n in nodes.items())
    assert names == PLANETS and all_shown and all_named

def test_solar_orbits_rebuild_per_bucket_not_on_zoom():
    # probe: h.step(5); keys0 = {k: n.get_key() for k, n in hooks.solar_orbit_nodes(app).items()}
    #        app.rig.set_distance(1.0e13); h.step(3)
    #        keys1 = ...; scale_ok = math.isclose(hooks.solar_orbit_frame_scale(app),
    #                                   1.0e9 / app.transform.scale_m_per_unit, rel_tol=1e-9)
    #        app.clock.sim_time_s += 30 * 86400.0; h.step(1); keys2 = ...
    assert keys1 == keys0          # zooming never rebuilds
    assert scale_ok                # the frame carries the zoom instead
    assert all(keys2[k] != keys0[k] for k in keys0)   # a new 30-day bucket rebuilds

def test_solar_viewer_offline_draws_orbits():
    # probe, BEFORE build_app: import orbitsim.core.ephemeris as e; e._load_failed = True
    #        h.step(5); RESULT sorted(hooks.solar_orbit_nodes(app)),
    #        "circular approx (offline)" in app.hud.text.getText()
    assert names == PLANETS and offline_text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/render/test_solar_view_app.py -q`
Expected: FAIL. The probe raises `AttributeError: ... '_solar_orbit_nps'`.

- [ ] **Step 3: Implement.**
  - `build_planets` creates the frame and the state.
  - `update_solar_system` computes `states = {body.name: heliocentric_state(body.name, t) for body in self.app._planet_bodies}` once (the Sun's entry costs nothing). It places the markers from `states[...].r`, then calls `self.frame_orbits(states)` before `rig.apply()`.

- [ ] **Step 4: Run the probes and the goldens**

Run: `.venv/bin/python -m pytest tests/render/test_solar_view_app.py tests/render/app_harness -q -n auto`
Expected: all pass. Lines aren't snapshotted, so the goldens are unchanged.

- [ ] **Step 5: Eyeball it.**
  1. On `main`, before this task, record a baseline: `.venv/bin/python -m tests.render.app_harness.screens --record`.
  2. Take shots at camera distances 4e11, 2e12 and 1e13 m with a scratch shooter in the session scratchpad. The shooter does `install_determinism(win_size="1600 900")`, `build_app("solar")`, `press("OPEN SOLAR VIEWER")`, `step(60)`, then for each distance `rig.set_distance(d)`, `step(5)` and `win.save_screenshot(Filename.from_os_specific(path))`. It needs an `if __name__ == "__main__":` guard.
  3. Expected: 8 tinted orbit lines, thin and quiet. Each line passes through its planet's dot at every zoom. Nothing is rebuilt while zooming.
  4. Run `.venv/bin/python -m tests.render.app_harness.profile solar`; the mean must stay within 10% of `main`.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/controllers/solar_view.py tests/render/app_harness/hooks.py tests/render/test_solar_view_app.py
git commit -m "feat(render): every planet's orbit drawn in the solar viewer (zoom-scaled frame, 30-day refresh)"
```

### Task 3: Pure label layout and pixel sizing

**Files:**
- Create: `orbitsim/render/label_layout.py` (no Panda3D import)
- Test: `tests/render/test_label_layout.py`

Plan 4 may already have added a generic rect-based label declutter helper (the sandbox's `world_markers.declutter_indices` is point-based). If one exists with the same semantics — priority order, a pinned set that always shows, rect obstacles, hidden when nothing fits — reuse it instead of `layout_labels`, add only what's missing (slots, preferred slot), and point these tests at it. Otherwise create it here.

**Interfaces:**
- Produces: in `orbitsim/render/label_layout.py`.
  - `Rect = tuple[float, float, float, float]`: `(x0, y0, x1, y1)` in pixels, y up, with `x0 < x1` and `y0 < y1`.
  - `SLOTS = ("above", "right", "left", "below")`.
  - `@dataclass(frozen=True) class LabelRequest` with fields:
    - `key: str`
    - `anchor_px: tuple[float, float] | None` (None means off-screen or behind the camera)
    - `size_px: tuple[float, float]` (card width, height)
    - `marker_radius_px: float`
    - `priority: int`
  - `units_per_pixel(depth_units: float, vfov_deg: float, viewport_h_px: float) -> float` returns `2·depth·tan(vfov/2) / viewport_h`. It raises `ValueError` if depth ≤ 0, vfov ∉ (0, 180) or viewport_h ≤ 0.
  - `slot_rect(anchor_px, size_px, clearance_px: float, slot: str) -> Rect` places the card centred above, centred to the right, centred to the left, or centred below the anchor, with its near edge `clearance_px` from the anchor. Unknown slot raises `ValueError`.
  - `rects_overlap(a: Rect, b: Rect) -> bool`, with open intervals (touching edges don't overlap).
  - `layout_labels(requests, viewport_px: tuple[float, float], *, gap_px: float = 3.0, obstacles: Sequence[Rect] = (), pinned: Collection[str] = (), preferred_slots: Mapping[str, str] | None = None) -> dict[str, tuple[str, Rect] | None]` returns, for every request key, the `(slot, rect)` it was placed in, or `None` if hidden.
  - `label_origin_offset_px(rect: Rect, card: tuple[float, float, float, float], font_px: float, anchor_px) -> tuple[float, float]` returns the pixel offset from the anchor to the text origin that puts the card `(l, r, b, t)` (text units × `font_px`) exactly on `rect`.
  - `render2d_rect_px(lo: tuple[float, float], hi: tuple[float, float], viewport_px) -> Rect` maps render2d `(x, z)` in [−1, 1] to y-up pixels.

`layout_labels` algorithm:

```text
order = pinned keys (input order), then the rest by (-priority, input index)
marker_rect(r) = anchor ± marker_radius_px            (only requests with an anchor)
for each request in order:
    anchor None -> result None (and it is no obstacle)
    candidates = [preferred_slots[key]] (if any) + the other SLOTS in SLOTS order
    rect = first slot_rect(anchor, size, marker_radius_px + gap_px, slot) that
           lies fully inside (0, 0, viewport) and overlaps no placed label rect,
           no other request's marker_rect, and no obstacle
    none fits: pinned -> take the first candidate anyway; else -> None
    a placed rect joins the placed list
```

- [ ] **Step 1: Write the failing tests**

```python
import math, random
import pytest
from orbitsim.render.label_layout import (LabelRequest, SLOTS, label_origin_offset_px,
    layout_labels, rects_overlap, render2d_rect_px, slot_rect, units_per_pixel)

VIEW = (400.0, 300.0)

def req(key, x, y, prio=50, w=40.0, h=20.0, r=6.0):
    return LabelRequest(key, None if x is None else (x, y), (w, h), r, prio)

def test_units_per_pixel_matches_lens_geometry():
    assert units_per_pixel(1000.0, 30.0, 900) == pytest.approx(2000.0 * math.tan(math.radians(15)) / 900)

def test_units_per_pixel_scales_with_depth_and_viewport():
    base = units_per_pixel(1000.0, 30.0, 900)
    assert units_per_pixel(2000.0, 30.0, 900) == pytest.approx(2 * base)
    assert units_per_pixel(1000.0, 30.0, 1800) == pytest.approx(base / 2)

def test_units_per_pixel_rejects_bad_input():
    for args in ((0.0, 30.0, 900), (-1.0, 30.0, 900), (1.0, 0.0, 900), (1.0, 180.0, 900), (1.0, 30.0, 0)):
        with pytest.raises(ValueError):
            units_per_pixel(*args)

def test_slot_rects_clear_the_marker():
    assert slot_rect((100, 100), (40, 20), 8, "above") == (80, 108, 120, 128)
    assert slot_rect((100, 100), (40, 20), 8, "right") == (108, 90, 148, 110)
    assert slot_rect((100, 100), (40, 20), 8, "left") == (52, 90, 92, 110)
    assert slot_rect((100, 100), (40, 20), 8, "below") == (80, 72, 120, 92)
    with pytest.raises(ValueError):
        slot_rect((100, 100), (40, 20), 8, "diagonal")

def test_isolated_labels_go_above():
    out = layout_labels([req("a", 100, 100), req("b", 300, 200)], VIEW)
    assert out["a"][0] == "above" and out["b"][0] == "above"

def test_crowded_label_moves_to_another_slot_instead_of_stacking():
    out = layout_labels([req("hi", 200, 150, 90), req("lo", 215, 150, 10)], VIEW)
    assert out["hi"][0] == "above" and out["lo"] is not None and out["lo"][0] != "above"
    assert not rects_overlap(out["hi"][1], out["lo"][1])

def test_label_never_covers_another_marker():
    out = layout_labels([req("a", 200, 150, 90), req("b", 200, 175, 10)], VIEW)
    b_dot = (194, 169, 206, 181)
    assert not rects_overlap(out["a"][1], b_dot)

def test_lowest_priority_label_hidden_when_no_slot_fits():
    reqs = [req(k, 200 + i, 150, p) for i, (k, p) in enumerate(
        [("Earth", 100), ("Venus", 90), ("Mars", 85), ("Mercury", 70), ("Sun", 10)])]
    out = layout_labels(reqs, VIEW)
    assert out["Earth"] is not None and out["Sun"] is None
    shown = [v[1] for v in out.values() if v is not None]
    assert all(not rects_overlap(a, b) for i, a in enumerate(shown) for b in shown[i + 1:])

def test_pinned_label_always_shown_and_others_avoid_it():
    reqs = [req(k, 200 + i, 150, p) for i, (k, p) in enumerate(
        [("Earth", 100), ("Venus", 90), ("Mars", 85), ("Mercury", 70), ("Sun", 10)])]
    out = layout_labels(reqs, VIEW, pinned={"Sun"})
    assert out["Sun"] is not None
    assert all(v is None or k == "Sun" or not rects_overlap(v[1], out["Sun"][1]) for k, v in out.items())

def test_offscreen_anchor_is_hidden_and_no_obstacle():
    out = layout_labels([req("gone", None, None, 100), req("a", 200, 150)], VIEW)
    assert out["gone"] is None and out["a"][0] == "above"

def test_label_must_fit_inside_viewport():
    out = layout_labels([req("top", 200, 295)], VIEW)
    assert out["top"][0] != "above"
    x0, y0, x1, y1 = out["top"][1]
    assert 0 <= x0 and x1 <= VIEW[0] and 0 <= y0 and y1 <= VIEW[1]

def test_obstacles_block_labels():
    out = layout_labels([req("a", 200, 150)], VIEW, obstacles=[(150, 155, 250, 200)])
    assert out["a"][0] != "above"

def test_preferred_slot_is_kept_when_still_free():
    out = layout_labels([req("a", 200, 150)], VIEW, preferred_slots={"a": "left"})
    assert out["a"][0] == "left"

def test_random_layouts_never_overlap():
    rng = random.Random(5)
    for _ in range(200):
        reqs = [req(f"k{i}", rng.uniform(0, 400), rng.uniform(0, 300), rng.randint(0, 100),
                    rng.uniform(20, 70), 22.0, rng.choice([6.0, 7.0, 10.0])) for i in range(9)]
        out = layout_labels(reqs, VIEW)
        shown = [(k, v[1]) for k, v in out.items() if v is not None]
        dots = {r.key: (r.anchor_px[0] - r.marker_radius_px, r.anchor_px[1] - r.marker_radius_px,
                        r.anchor_px[0] + r.marker_radius_px, r.anchor_px[1] + r.marker_radius_px)
                for r in reqs}
        for i, (k, a) in enumerate(shown):
            assert 0 <= a[0] and a[2] <= 400 and 0 <= a[1] and a[3] <= 300
            assert all(not rects_overlap(a, b) for _, b in shown[i + 1:])
            assert all(not rects_overlap(a, d) for kk, d in dots.items() if kk != k)

def test_label_origin_offset_puts_card_on_rect():
    assert label_origin_offset_px((80, 108, 120, 128), (-1.0, 1.0, -0.35, 1.12), 10.0, (100, 100)) \
        == pytest.approx((0.0, 14.15))

def test_render2d_rect_px():
    assert render2d_rect_px((-1, -1), (1, 1), (1600, 900)) == (0, 0, 1600, 900)
    assert render2d_rect_px((-0.5, 0.8), (0.5, 1.0), (1600, 900)) == pytest.approx((400, 810, 1200, 900))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/render/test_label_layout.py -q`
Expected: FAIL with `ModuleNotFoundError: orbitsim.render.label_layout`.

- [ ] **Step 3: Implement `orbitsim/render/label_layout.py`** with the interfaces and algorithm above.

- [ ] **Step 4: Run the tests and verify they pass.** Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/label_layout.py tests/render/test_label_layout.py
git commit -m "feat(render): pure screen-space label layout (priority, pinned, slots, obstacles)"
```

### Task 4: Constant on-screen markers and labels in the solar viewer

**Files:**
- Modify: `orbitsim/render/controllers/solar_view.py` (`build_planets`, `update_solar_system`, new `frame_markers_and_labels`, `_mouse_px`)
- Modify: `tests/render/app_harness/hooks.py` (add `screen_radius_px`, `label_screen_height_px`)
- Test: `tests/render/test_solar_view_app.py`
- Re-record if needed: `solar_boot_warp`, `solar_porkchop_key` (`world_labels` only)

**Interfaces:**
- Consumes: Task 3's `units_per_pixel`, `slot_rect` and `label_origin_offset_px`.
- Produces: constants on `SolarView`.
  - `MARKER_RADIUS_PX = {"Sun": 10.0, "Jupiter": 7.0, "Saturn": 7.0}` and `DEFAULT_MARKER_RADIUS_PX = 6.0`.
  - `LABEL_FONT_PX = 15.0`: pixels per text unit, so a card is about 22 px tall.
  - `LABEL_GAP_PX = 3.0`.
  - All of these are multiplied by `self.app.line_px_scale()`.
- Produces: `build_planets` changes.
  - Markers are named `solar_marker_<Name>`.
  - Earth's marker loses its texture and gets a flat `(0.35, 0.6, 1.0, 1.0)`.
  - `self.app.central_np.hide()`: the Sun marker is the only Sun.
  - Labels get `set_depth_test(False)` and `set_bin("fixed", 60)` (drawn over orbit lines; the layout keeps them off dots).
  - New state `_solar_label_cards: dict[str, tuple[float, float, float, float]]` from each label's `node().get_card_actual()`.
- Produces: `SolarView.frame_markers_and_labels(mouse_px: tuple[float, float] | None) -> None`. It is called at the end of `update_solar_system`, **after** `rig.apply()`, as `self.frame_markers_and_labels(self._mouse_px())`. For each body:
  1. `p = cam.get_relative_point(render, marker_pos)` and `depth = p.y`.
  2. If `depth <= camLens.get_near()`, hide both marker and label, and record `None` in `_solar_marker_px`.
  3. Otherwise compute `upp = units_per_pixel(depth, camLens.get_fov()[1], fb_h)` and set the marker scale to `radius_px · ui · upp`.
  4. The anchor is `camLens.project(p, Point2)` mapped to framebuffer pixels (y up), or `None` if the projection fails.
  5. The label scale is `LABEL_FONT_PX · ui · upp`.
  6. The label position is `marker_pos + (cam_right · dx + cam_up · dy) · upp`, where `(dx, dy) = label_origin_offset_px(slot_rect(anchor, card_size, (radius + gap) · ui, "above"), card, LABEL_FONT_PX · ui, anchor)`, and `cam_right` / `cam_up` come from `self.app.camera.get_quat(self.app.render)`.

  In this task every on-screen label goes "above"; Task 5 replaces that slot choice with `layout_labels`.
- Produces: state `_solar_marker_px: dict[str, tuple[float, float, float] | None]`, holding `(x, y, radius_px)` per body, or `None`.
- Produces: `SolarView._mouse_px() -> tuple[float, float] | None`, the mouse position in framebuffer pixels (y up), or `None` when there is no mouse. Headless has no mouse.
- Produces: hooks.
  - `screen_radius_px(app, node_name: str) -> float | None` projects the node's centre and centre + camera-right × node scale, and returns the pixel distance.
  - `label_screen_height_px(app, text: str) -> float | None` projects the label's card top and bottom (label pos ± camera-up × t/b × label scale), and returns the pixel distance.
  - Both are measured from the scene, independent of `units_per_pixel`. Both return `None` when the node is hidden or its centre doesn't project (off-screen or behind the camera).
  - `solar_marker_px(app) -> dict[str, tuple[float, float, float] | None]` returns a copy of `_solar_marker_px`.

- [ ] **Step 1: Write the failing probe**

```python
def test_markers_and_labels_keep_their_pixel_size_at_any_zoom():
    # probe (default 960x600 => line_px_scale() == 1.0): h.step(5)
    # for d in (4.0e11, 2.0e12, 1.0e13): app.rig.set_distance(d); h.step(3)
    #     row = {name: (hooks.screen_radius_px(app, f"solar_marker_{name}"),
    #                   hooks.label_screen_height_px(app, name)) for name in ("Earth", "Jupiter", "Neptune", "Sun")}
    # RESULT rows
    card_h = (1.12 + 0.355) * 15.0
    for row in rows:
        assert row["Earth"][0] is not None and row["Sun"][0] is not None   # always on screen
        for name, (radius, height) in row.items():
            if radius is None:            # off-screen / behind the camera at this zoom: skip
                continue
            want = {"Sun": 10.0, "Jupiter": 7.0}.get(name, 6.0)
            assert radius == pytest.approx(want, abs=0.3)
            assert height == pytest.approx(card_h, abs=0.5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/render/test_solar_view_app.py -q -k pixel_size`
Expected: FAIL. Neptune's radius at 1e13 m is far above 6 px today, and `solar_marker_*` nodes don't exist.

- [ ] **Step 3: Implement** the build changes and `frame_markers_and_labels`. Remove the old fixed `set_scale(8.0 / 4.0)` and the `label.set_pos(..., rz + 6.0)` placement.

- [ ] **Step 4: Run the probes and the goldens**

Run: `.venv/bin/python -m pytest tests/render/test_solar_view_app.py tests/render/app_harness -q -n auto`
Expected: the probes pass. A solar golden may fail **only** on `world_labels` hidden flags (labels of off-screen bodies, or bodies behind the camera, are now hidden). Any difference in `vessel`, `sim_time_s`, `warp`, `target`, `events`, `texts` or `rig` is a bug: stop and fix it.

- [ ] **Step 5: Re-record any failing solar golden** with `.venv/bin/python -m tests.render.app_harness.record <name>`. Then run Plan 1's physics-field check loop (Plan 1, Task 5, Step 5) over the two solar files. Expected: `ok` for both.

- [ ] **Step 6: Eyeball** the three scratch shots.
  - Every dot is the same size at every zoom, and Earth is clearly blue.
  - There is one Sun.
  - Names are the same height everywhere, and Neptune is no longer giant.
  - Names may still collide here; Task 5 fixes that.

- [ ] **Step 7: Commit** the code, the hooks, the test and any re-recorded golden. The message must say: "solar goldens: world_labels hidden flags only (off-screen bodies now hide their label); physics fields verified identical".

### Task 5: Declutter planet names (priority, hover, HUD obstacles)

**Files:**
- Modify: `orbitsim/render/controllers/solar_view.py` (`frame_markers_and_labels`, new `_ui_obstacles_px`)
- Modify: `tests/render/app_harness/hooks.py` (add `solar_label_rects`, `hover_solar_body`)
- Modify: `tests/render/app_harness/scenarios.py` (add `solar_zoom_out_labels`)
- Create: `tests/render/app_harness/golden/solar_zoom_out_labels.json` (recorded)
- Test: `tests/render/test_solar_view_app.py`
- Re-record if needed: `solar_boot_warp`, `solar_porkchop_key` (`world_labels` only)

**Interfaces:**
- Consumes: Task 3's `LabelRequest`, `layout_labels`, `render2d_rect_px` and `label_origin_offset_px`; `picking.nearest_marker(click_px, markers_px, tol_px)`; Task 4's per-body anchors, sizes and `upp`.
- Produces: constants on `SolarView`.
  - `SOLAR_LABEL_PRIORITY = {"Earth": 100, "Venus": 90, "Mars": 85, "Jupiter": 80, "Saturn": 75, "Mercury": 70, "Uranus": 65, "Neptune": 60, "Sun": 10}`. The order is: home first, then its neighbours, then the giants, then the small or far planets. The Sun's name comes last because the big yellow dot at the centre needs no label.
  - `HOVER_TOL_PX = 14.0` (× ui).
- Produces: `frame_markers_and_labels(mouse_px)` now works in four steps:
  1. Build one `LabelRequest` per body. `size_px` is the card `(r − l, t − b) × LABEL_FONT_PX × ui`, and the marker radius is × ui.
  2. `hovered` is the body name from `nearest_marker(mouse_px, anchors, HOVER_TOL_PX · ui)`, over the bodies with an anchor, or `None`.
  3. Call `layout_labels(requests, (fb_w, fb_h), gap_px=LABEL_GAP_PX · ui, obstacles=self._ui_obstacles_px(), pinned={hovered} - {None}, preferred_slots=self.app._solar_label_slots)`.
  4. Each placed label is positioned on its rect (Task 4's offset math) and shown; each `None` label is hidden. Dots are never hidden by the declutter.
- Produces: new `OrbitApp` state.
  - `_solar_label_slots: dict[str, str]`: last frame's slots, which give hysteresis.
  - `_solar_label_rects: dict[str, Rect | None]`: the last layout, used by hooks.
- Produces: `SolarView._ui_obstacles_px() -> list[Rect]`, the framebuffer-pixel rects of `self.app._warp_frame` and `self.app.hud.text`. It calls `get_tight_bounds(self.app.render2d)` on each and passes the result through `render2d_rect_px`, skipping any node that is hidden or has no bounds. If Plan 4 renamed or moved these nodes, follow the rename; the rects come from live bounds, so a moved HUD is followed automatically.
- Produces: hooks.
  - `solar_label_rects(app) -> dict[str, Rect | None]`.
  - `hover_solar_body(app, name) -> None` calls `app.solar_view.frame_markers_and_labels(mouse_px=app._solar_marker_px[name][:2])`.
- Produces: the scenario below. Never edit it after its golden exists.

```python
def solar_zoom_out_labels(h):
    # Zoomed out to the outer planets, the inner ones crowd the Sun: their names must
    # declutter (hidden or moved, never stacked) while the outer planets stay labelled.
    h.press("OPEN SOLAR VIEWER")
    h.step(10)
    for _ in range(12):
        h.send("wheel_down")
        h.step(5)
    h.step(60)
```

Register it as `"solar_zoom_out_labels": ("solar", solar_zoom_out_labels)`.

- [ ] **Step 1: Write the failing probes**

```python
def test_planet_names_never_overlap_each_other_or_dots():
    # probe: for d in (2.0e12, 1.0e13): app.rig.set_distance(d); h.step(3)
    #        RESULT per zoom: (hooks.solar_label_rects(app), hooks.solar_marker_px(app))
    from orbitsim.render.label_layout import rects_overlap
    for rects, dots in results:
        shown = {k: r for k, r in rects.items() if r is not None}
        assert "Earth" in shown
        items = list(shown.items())
        for i, (k, a) in enumerate(items):
            assert all(not rects_overlap(a, b) for _, b in items[i + 1:])
            assert all(not rects_overlap(a, (x - r, y - r, x + r, y + r))
                       for kk, d in dots.items() if d is not None and kk != k for x, y, r in [d])

def test_hovering_a_hidden_name_reveals_it():
    # probe: app.rig.set_distance(1.0e13); h.step(3)
    #        hidden = [k for k, r in hooks.solar_label_rects(app).items()
    #                  if r is None and hooks.solar_marker_px(app)[k] is not None]
    #        target = hidden[0]; hooks.hover_solar_body(app, target)
    #        RESULT hidden != [], hooks.solar_label_rects(app)[target] is not None
    assert had_hidden and revealed
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/render/test_solar_view_app.py -q -k "overlap or hover"`
Expected: FAIL (`AttributeError: '_solar_label_rects'`).

- [ ] **Step 3: Implement** the layout wiring, the hover and the obstacles.

- [ ] **Step 4: Run the probes, then the goldens**

Run: `.venv/bin/python -m pytest tests/render/test_solar_view_app.py tests/render/app_harness -q -n auto`
Expected: the probes pass. The existing solar goldens may differ **only** in `world_labels` hidden flags.

- [ ] **Step 5: Record and re-record.**
  1. Record `solar_zoom_out_labels`, plus any failing solar golden, with `.venv/bin/python -m tests.render.app_harness.record <names>`.
  2. Open the new JSON and check it before committing. Expected: Earth, Jupiter and Saturn are shown (`false`), and at least one of Sun, Mercury, Venus and Mars is hidden (`true`). If Earth is hidden, the layout is wrong: fix the code, don't keep the golden.
  3. Run the Plan 1 physics-field check over every solar golden. Expected: `ok`.

- [ ] **Step 6: Eyeball the three scratch shots.**
  - At boot, the inner names sit around their dots without overlapping, and the Sun's name gives way first.
  - At 1e13 m, the inner cluster shows "Earth" and hides the rest.
  - No name sits on the warp bar or the HUD text.
  - At 1e13 m, no name hops between slots across 60 frames of warp: take two shots 30 frames apart and compare them.

- [ ] **Step 7: Commit** the code, the hooks, the scenario, the new golden and any re-recorded golden. The message must say: "new golden solar_zoom_out_labels; solar_boot_warp/solar_porkchop_key re-recorded: world_labels hidden flags only (decluttered names); physics fields verified identical".

### Task 6: Finish

- [ ] **Step 1: Run the full suite:** `.venv/bin/python -m pytest tests/ -q -n auto`. Expected: all green.
- [ ] **Step 2: Profiler gate.** Run `solar`, `idle`, `zoom`, `zoom_node_planned` and `high_warp` on this branch and on `main` in the same session. Each mean must be within 10%.
  - The sandbox workloads should be unchanged; their only change is Task 1's bit-identical refactor.
  - If `solar` fails, check the following, which are the only intended per-frame costs:
    1. `heliocentric_state` is called once per planet per frame (8 calls, as before), and the orbit rebuild reuses those states.
    2. Orbits rebuild only on a bucket change.
    3. There are 9 projections and one `layout_labels` per frame.
- [ ] **Step 3: Update CLAUDE.md** with a short "Solar viewer" note under the architecture section. It should say:
  - planets come from `core.planets.heliocentric_state`;
  - orbit lines are baked in `SOLAR_ORBIT_UNIT_M` under `solar_orbit_frame`, which is rescaled each frame, so zoom never rebuilds them (refresh every 30 sim-days);
  - markers and names are sized in pixels per frame via `label_layout.units_per_pixel`;
  - names are placed by `layout_labels`: priority `SOLAR_LABEL_PRIORITY`, the hovered body pinned, other dots and the HUD as obstacles, and hidden when nothing fits.

  Commit with the trailer and push.
- [ ] **Step 4: Before/after screenshots and PR.**
  1. Run `.venv/bin/python -m tests.render.app_harness.screens --compare` against the Task 2 baseline, and take the three scratch shots.
  2. Run a whole-branch review.
  3. Open the PR with the before/after images (boot, 4e11 m, 1e13 m), the golden re-record reasons, and the profiler numbers. The description ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  4. Merge.
