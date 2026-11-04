# Map Readability, Plan 1 of 5: "See Everything" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every orbit line, the background and the player's ship clearly visible at any zoom: thick strongly-coloured lines, a quiet starfield, the real ship model as the map icon, and an opening camera that shows the lit Earth and your orbit.

**Architecture:** One pure style module (`render/line_style.py`) owns colours, pixel widths, fade and dashing. `build_orbit_node` takes a style plus a framebuffer pixel scale, and every call site picks a named style. The starfield is dimmed with a colour scale. The yellow marker sphere becomes a constant-screen-size, unlit, outlined copy of the ship model. A pure `opening_camera` helper points the first view from the Sun's side.

**Tech Stack:** Python 3.11, Panda3D (`LineSegs`, `NodePath`), numpy, pytest (+ xdist). The headless app-golden harness is in `tests/render/app_harness/`.

**Spec:** `docs/superpowers/specs/2026-11-04-map-and-planning-readability-design.md`. This plan covers its "The look" bullets for lines, background, your ship and opening camera. Later plans: 2 live preview line + faster exact preview; 3 click-orbit menu, drag handles, keyboard nudges; 4 aim helpers, auto-plan safe periapsis, slider-panel removal, HUD panel layout and labels; 5 solar viewer orbit lines + label declutter.

## Global Constraints

- Stay on Panda3D; `core/` never imports Panda3D (CLAUDE.md layering).
- The physics doesn't change: app-golden fields `vessel`, `sim_time_s`, `warp`, `target`, `events` must stay identical. Only presentation fields (`rig`, `texts`, `world_labels`) may be re-recorded, each with its reason in the commit.
- Not a KSP clone: genre conventions only, OrbitSim's own colours.
- No frame-rate regression on `idle`, `zoom`, `zoom_node_planned`, `high_warp` (`python -m tests.render.app_harness.profile <w>`). Mean frame ms may not rise more than 10% against a run on `main` from the same machine and session.
- Colours: your orbit bright cyan; planned path orange, dashed; reference orbits muted grey; the targeted body's orbit highlighted; encounter segments yellow.
- Lines keep a constant on-screen width at any zoom (GL line width is in framebuffer pixels, so this already holds) and scale with framebuffer height so Retina screens aren't hairlines.
- TDD; commit + push after each task; stage by explicit path; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Window resized or on a Retina framebuffer:** lines must stay proportionally thick. Pinned by `test_px_scale_*` (Task 1), and the scale is read from `app.win` each time a line is built (Task 2).
2. **Very short dashed polylines** (a 2–3 point preview fragment): must still draw something, not vanish. Pinned by `test_dashed_two_points_still_draws` (Task 2).
3. **Offline (no star texture):** the procedural starfield must be dimmed too. Pinned by `test_procedural_starfield_is_dimmed` (Task 3).
4. **Target switched Moon → none → Moon:** the Moon's orbit highlight must follow, not stick. Pinned by `test_reference_style_follows_target` (Task 1), and the rebuild key in Task 2.
5. **Solar viewer (no vessel):** the ship icon and opening camera must not be built or applied. Already pinned by the `solar_*` goldens, which must stay green unchanged.

---

### Task 1: Line styles (pure)

**Files:**
- Create: `orbitsim/render/line_style.py`
- Test: `tests/render/test_line_style.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class LineStyle: color: tuple[float, float, float, float]; width_px: float; halo_px: float; fade_minimum: float; dash: tuple[int, int] | None` (dash = (points on, points off)).
  - Named styles: `ORBIT_STYLE`, `PLAN_STYLE`, `REFERENCE_STYLE`, `TARGET_REFERENCE_STYLE`, `ENCOUNTER_STYLE`.
  - `def reference_style(targeted: bool) -> LineStyle`
  - `def px_scale(framebuffer_height_px: int) -> float`
  - `REFERENCE_HEIGHT_PX = 900`

Values (the spec leaves exact values to the builder; these are the decided ones):

| style | color | width_px | halo_px | fade_minimum | dash |
|---|---|---|---|---|---|
| ORBIT | (0.22, 0.84, 1.0, 1.0) | 3.5 | 2.5 | 0.65 | None |
| PLAN | (1.0, 0.62, 0.18, 1.0) | 3.5 | 2.5 | 0.75 | (6, 4) |
| REFERENCE | (0.60, 0.64, 0.71, 0.55) | 2.0 | 0.0 | 1.0 | None |
| TARGET_REFERENCE | (0.86, 0.88, 0.92, 0.9) | 2.75 | 1.5 | 1.0 | None |
| ENCOUNTER | (1.0, 0.83, 0.30, 1.0) | 4.5 | 2.5 | 0.8 | None |

- [ ] **Step 1: Write the failing tests**

```python
from orbitsim.render.line_style import (
    ORBIT_STYLE, PLAN_STYLE, REFERENCE_STYLE, TARGET_REFERENCE_STYLE, ENCOUNTER_STYLE,
    px_scale, reference_style,
)

def test_own_orbit_and_plan_are_thick_and_opaque():
    for s in (ORBIT_STYLE, PLAN_STYLE, ENCOUNTER_STYLE):
        assert s.width_px >= 3.5 and s.color[3] == 1.0 and s.fade_minimum >= 0.65

def test_plan_is_dashed_orange_and_orbit_is_solid_cyan():
    assert PLAN_STYLE.dash == (6, 4) and ORBIT_STYLE.dash is None
    r, g, b, _ = PLAN_STYLE.color;  assert r > 0.9 and b < 0.3
    r, g, b, _ = ORBIT_STYLE.color; assert b > 0.9 and r < 0.3

def test_reference_is_quieter_than_own_orbit():
    assert REFERENCE_STYLE.width_px < ORBIT_STYLE.width_px
    assert REFERENCE_STYLE.color[3] < ORBIT_STYLE.color[3]

def test_reference_style_follows_target():
    assert reference_style(True) is TARGET_REFERENCE_STYLE
    assert reference_style(False) is REFERENCE_STYLE
    assert TARGET_REFERENCE_STYLE.width_px > REFERENCE_STYLE.width_px

def test_px_scale_is_one_at_reference_and_grows_with_height():
    assert px_scale(900) == 1.0
    assert px_scale(1800) == 2.0          # Retina 2x framebuffer

def test_px_scale_never_below_one():
    assert px_scale(600) == 1.0 and px_scale(0) == 1.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/render/test_line_style.py -q`
Expected: FAIL with `ModuleNotFoundError: orbitsim.render.line_style`

- [ ] **Step 3: Implement `orbitsim/render/line_style.py`** with the table above. `px_scale` returns `max(1.0, height / REFERENCE_HEIGHT_PX)`. No Panda3D import.

- [ ] **Step 4: Run them to verify they pass.** Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/line_style.py tests/render/test_line_style.py
git commit -m "feat(render): named line styles for orbits, plans, references, encounters"
```

### Task 2: Styled, dashed, DPI-scaled orbit lines at every call site

**Files:**
- Modify: `orbitsim/render/orbit_lines.py` (`build_orbit_node`; drop `TRAJECTORY_COLOR`, `MANEUVER_COLOR`, `REFERENCE_ORBIT_COLOR`)
- Modify: `orbitsim/render/controllers/trajectory_view.py:74,85` (encounter patches, live trajectory)
- Modify: `orbitsim/render/controllers/maneuver_editor.py:11,356` (preview)
- Modify: `orbitsim/render/controllers/bodies_view.py:22-23,264,441` (Moon ring, planet orbits)
- Modify: `orbitsim/render/app.py:40` (`ENCOUNTER_COLOR = ENCOUNTER_STYLE.color`, still referenced by `tests/render/test_maneuver_preview.py:87`)
- Test: `tests/render/test_orbit_lines.py`

**Interfaces:**
- Consumes: Task 1's `LineStyle`, the named styles, `px_scale`, `reference_style`.
- Produces: `build_orbit_node(points_render, style: LineStyle = ORBIT_STYLE, px_scale: float = 1.0) -> NodePath`. The halo stroke is drawn only when `style.halo_px > 0`. Dashing breaks the strip by point index: `dash[0]` points drawn, `dash[1]` skipped, repeating. A polyline shorter than `dash[0] + 1` points is drawn solid.
- Produces: `OrbitApp.line_px_scale() -> float`, `px_scale(self.win.get_fb_y_size())` (1.0 if `self.win` is None). Every call site passes it.

- [ ] **Step 1: Write the failing tests** (append to `tests/render/test_orbit_lines.py`; each builds offscreen like the existing `build_orbit_node` test at line 77)

```python
from panda3d.core import RenderModeAttrib
from orbitsim.render.line_style import ORBIT_STYLE, PLAN_STYLE, REFERENCE_STYLE

def _strokes(node):
    return [c for c in node.get_children()]

def _thickness(np_):
    return np_.node().get_geom_state(0).get_attrib(RenderModeAttrib).get_thickness()

def _strip_count(np_):
    return np_.node().get_geom(0).get_primitive(0).get_num_primitives()

POINTS = [(float(i), 0.0, 0.0) for i in range(40)]

def test_width_scales_with_px_scale():
    node = build_orbit_node(POINTS, ORBIT_STYLE, px_scale=2.0)
    color_stroke = _strokes(node)[-1]
    assert _thickness(color_stroke) == ORBIT_STYLE.width_px * 2.0

def test_halo_only_when_style_has_one():
    assert len(_strokes(build_orbit_node(POINTS, ORBIT_STYLE))) == 2
    assert len(_strokes(build_orbit_node(POINTS, REFERENCE_STYLE))) == 1

def test_dashed_plan_breaks_into_dash_strips():
    solid = _strokes(build_orbit_node(POINTS, ORBIT_STYLE))[-1]
    dashed = _strokes(build_orbit_node(POINTS, PLAN_STYLE))[-1]
    assert _strip_count(solid) == 1
    assert _strip_count(dashed) == 4        # 40 pts, (6 on, 4 off)

def test_dashed_two_points_still_draws():
    node = build_orbit_node([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], PLAN_STYLE)
    assert _strip_count(_strokes(node)[-1]) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/render/test_orbit_lines.py -q`
Expected: FAIL. `build_orbit_node` has no `style` parameter (TypeError).

- [ ] **Step 3: Implement the new `build_orbit_node`**, then switch the call sites:
  - live trajectory → `ORBIT_STYLE`
  - preview → `PLAN_STYLE`
  - encounter patches → `ENCOUNTER_STYLE`
  - Moon ring → `reference_style(target is the Moon)`
  - planet orbits → `reference_style(target is that planet)`, keeping each planet's existing tint by `dataclasses.replace(style, color=_orbit_colors[name][:3] + (style.color[3],))`

  Each call site passes `px_scale=self.app.line_px_scale()`.

  Highlight rebuild: add the current target name to the Moon-ring rebuild condition (`self.app._moon_orbit_target`, initialised to `None` where `_moon_orbit_np` is initialised) and to the planet-orbit rebuild condition (the same pattern, with its own attribute), so a target change rebuilds once.

- [ ] **Step 4: Run the line tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/render/test_orbit_lines.py tests/render/test_maneuver_preview.py -q && .venv/bin/python -m pytest tests/ -q -n auto`
Expected: all pass. Lines aren't in snapshots, so the goldens are unchanged.

- [ ] **Step 5: Eyeball it.** Take screenshots with the scratch shooter used in brainstorming (`start`, `earth_moon`, `intercept`, `solar` at 1600x900). Expected: your orbit clearly cyan and thick, the plan orange and dashed, the Moon ring grey, brighter when targeted. Then the profiler check from Global Constraints on `zoom` and `zoom_node_planned`.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/orbit_lines.py orbitsim/render/controllers/trajectory_view.py orbitsim/render/controllers/maneuver_editor.py orbitsim/render/controllers/bodies_view.py orbitsim/render/app.py tests/render/test_orbit_lines.py
git commit -m "feat(render): thick styled orbit lines (cyan orbit, dashed orange plan, quiet references)"
```

### Task 3: Quiet starfield

**Files:**
- Modify: `orbitsim/render/skybox.py` (`build_starfield`, `_background`)
- Test: `tests/render/test_skybox.py`

**Interfaces:**
- Produces: `STARFIELD_BRIGHTNESS = 0.35`, applied in `_background` as `node.set_color_scale(b, b, b, 1)`, so both the textured sky and the procedural fallback get it.

- [ ] **Step 1: Write the failing tests**

```python
def test_textured_or_procedural_starfield_is_dimmed():   # whichever this machine builds
    ...build base offscreen as in test_build_starfield_returns_a_node...
    from orbitsim.render.skybox import STARFIELD_BRIGHTNESS
    sky = build_starfield(base)
    assert sky.get_color_scale()[0] == pytest.approx(STARFIELD_BRIGHTNESS)
    base.destroy()

def test_procedural_starfield_is_dimmed():
    from orbitsim.render.skybox import _procedural_points, _background, STARFIELD_BRIGHTNESS
    node = _background(_procedural_points())
    assert node.get_color_scale()[0] == pytest.approx(STARFIELD_BRIGHTNESS)
```

- [ ] **Step 2: Run them, expect FAIL** (`ImportError: STARFIELD_BRIGHTNESS`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `tests/render/test_skybox.py`; expect PASS. Eyeball the `start` screenshot**: stars present but clearly behind the lines.
- [ ] **Step 5: Commit** `feat(render): dim the starfield so lines and bodies stand out`.

### Task 4: The real ship as the map icon

**Files:**
- Modify: `orbitsim/render/ship_model.py` (add `build_ship_icon`)
- Modify: `orbitsim/render/controllers/vessel_view.py` (`build_vessel_markers`, `frame_vessels`, `frame_ship_view`)
- Test: `tests/render/test_ship_model.py`

**Interfaces:**
- Produces:
  - `SHIP_ICON_RENDER_UNITS = 22.0`: icon length in render units. The camera always sits 1000 units from the focus, so the size is constant on screen (the old sphere was radius 8).
  - `def build_ship_icon() -> NodePath`. A `ship_icon` root holding:
    - the ship model, unlit (`set_light_off`), colour `(0.92, 0.95, 1.0, 1)`;
    - an `outline` child: the same model, scaled 1.18 about its centre, `CullFaceAttrib.make_reverse()`, colour `ORBIT_STYLE.color`, unlit, drawn behind (`set_bin("fixed", 20)` vs the body at 21).

    The root is scaled so the model's nose-to-tail length equals `SHIP_ICON_RENDER_UNITS`.
- `OrbitApp.vessel_nps[i]` now holds icons instead of spheres. Same attribute, same cross-fade via `view_blend` (marker alpha → icon alpha). Each frame the icon also gets `set_quat` from `vessel.orientation` (`[w, x, y, z]`, as `frame_ship_view` already does for the model).

- [ ] **Step 1: Write the failing tests**

```python
def test_ship_icon_is_fixed_render_length_and_outlined():
    # Pure geometry: no ShowBase/window needed (one ShowBase per process is fragile).
    from orbitsim.render.ship_model import build_ship_icon, SHIP_ICON_RENDER_UNITS
    icon = build_ship_icon()
    lo, hi = icon.get_tight_bounds()
    assert (hi - lo).z == pytest.approx(SHIP_ICON_RENDER_UNITS * 1.18, rel=0.02)  # outline is the outer hull
    assert not icon.find("**/outline").is_empty()
    assert icon.has_light_off() or icon.find("**/+GeomNode").has_light_off()
```

- [ ] **Step 2: Run `tests/render/test_ship_model.py`, expect FAIL** (`ImportError: build_ship_icon`).
- [ ] **Step 3: Implement `build_ship_icon`; switch `build_vessel_markers` to it; add the orientation update in `frame_vessels`.**
- [ ] **Step 4: Run `tests/render/test_ship_model.py` and the full suite with `-n auto`.** Expected: all pass, goldens unchanged (the marker isn't snapshotted).
- [ ] **Step 5: Eyeball `start` and `intercept`.** The ship reads as a ship, pointing its real attitude, outlined against Earth. Then `ship` (`M`): the cross-fade to the true-scale model still works.
- [ ] **Step 6: Commit** `feat(render): the real ship model, outlined, as the constant-size map icon`.

### Task 5: Opening camera shows the lit Earth and your orbit

**Files:**
- Modify: `orbitsim/render/camera_rig.py` (pure helper)
- Modify: `orbitsim/render/app.py` (sandbox start: apply once, after the world exists)
- Test: `tests/render/test_camera_rig.py`
- Re-record: the sandbox goldens in `tests/render/app_harness/golden/` (`rig`, plus any `world_labels` hidden flags that depend on zoom)

**Interfaces:**
- Produces:
  - `OPENING_ELEVATION_RAD = 0.35`
  - `OPENING_DISTANCE_FACTOR = 3.0`
  - `def opening_camera(sun_dir, r_apoapsis_m: float) -> tuple[float, float, float]`, returning `(azimuth_rad, elevation_rad, distance_m)`. It raises `ValueError` for a zero `sun_dir` or a non-positive radius.
  - The app sets `rig.set_distance(d)`, `rig.azimuth = rig.target_azimuth = az`, `rig.elevation = rig.target_elevation = el` once at sandbox start. That's a snap, not an ease. Initial state is allowed to bypass the "only set targets" rule, which governs input handlers.
  - The sun direction comes from `core.nbody._csun(t)` (the source `frame_sun` uses). Apoapsis comes from `state_to_elements(vessel.state)`, or the current radius if that raises.

- [ ] **Step 1: Write the failing tests**

```python
import math, pytest
from orbitsim.render.camera_rig import opening_camera, OPENING_ELEVATION_RAD, OPENING_DISTANCE_FACTOR

def test_camera_sits_on_the_sun_side():
    az, el, d = opening_camera((0.0, 1.0, 0.0), 8.8e6)
    assert az == pytest.approx(math.pi / 2) and el == OPENING_ELEVATION_RAD

def test_distance_frames_the_whole_orbit():
    _, _, d = opening_camera((1.0, 0.0, 0.0), 8.8e6)
    assert d == pytest.approx(OPENING_DISTANCE_FACTOR * 8.8e6)

def test_rejects_degenerate_input():
    with pytest.raises(ValueError):
        opening_camera((0.0, 0.0, 0.0), 8.8e6)
    with pytest.raises(ValueError):
        opening_camera((1.0, 0.0, 0.0), 0.0)
```

- [ ] **Step 2: Run them, expect FAIL** (ImportError).
- [ ] **Step 3: Implement the helper and apply it at sandbox start.**
- [ ] **Step 4: Run the full suite.** Expected: the sandbox goldens fail **only** on `rig` (and possibly `world_labels`). Any difference in `vessel`, `sim_time_s`, `warp`, `target` or `events` means a bug: stop and fix the code.
- [ ] **Step 5: Re-record each failing sandbox golden** with `.venv/bin/python -m tests.render.app_harness.record <name>`. Then prove the physics fields are untouched:

```bash
for f in tests/render/app_harness/golden/*.json; do .venv/bin/python - "$f" <<'EOF'
import json, subprocess, sys
f = sys.argv[1]
old = json.loads(subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout)
new = json.load(open(f))
def phys(d):
    steps = d if isinstance(d, list) else d.get("snapshots", [d])
    return [{k: s.get(k) for k in ("vessel", "sim_time_s", "warp", "target", "events")} for s in steps]
assert phys(old) == phys(new), f"physics changed in {f}"
print("ok", f)
EOF
done
```

  Expected: `ok` for every file. (Adapt `phys()` to the golden file's actual top-level shape if it differs; the five keys are what matters.)
- [ ] **Step 6: Eyeball `start`.** Earth is mostly lit and your whole orbit is in frame.
- [ ] **Step 7: Commit** the code, the test and the re-recorded goldens. The message must state: "re-recorded sandbox goldens: rig (and zoom-dependent world_labels) only, physics fields verified identical".

### Task 6: Finish

- [ ] **Step 1:** Full suite with `-n auto`. Then the profiler gate: `idle`, `zoom`, `zoom_node_planned` and `high_warp` on this branch vs `main`, same session. Each mean must be within 10%.
- [ ] **Step 2:** Update CLAUDE.md. Replace the obsolete "Render markers … constant on-screen size" bullet's marker wording with: lines use `render/line_style.py` named styles with **pixel** widths × `OrbitApp.line_px_scale()`; the ship's map icon is `build_ship_icon` at `SHIP_ICON_RENDER_UNITS`. Commit.
- [ ] **Step 3:** Push, open the PR with before/after screenshots (`start`, `intercept`, `solar`), merge.
