# HUD Overlay Readability (Cycle 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the colliding, hardcoded top-corner readouts with a reusable self-sizing panel component (grouped, color-coded, subtle backgrounds), and add a navball SAS chip + clickable SAS-mode buttons and a velocity readout (orbital/target toggle) above the navball.

**Architecture:** A pure `layout_panel` (computes line positions + background extent, so overlap cannot regress) plus a `HudPanel` DirectGUI class live in `render/hud/panel.py`. `Hud` is rewired to drive a top-left panel (TIME/ORBIT/MANEUVER) and a top-right panel (VESSEL); the maneuver readouts move out of `app.py`'s standalone text nodes into the panel. A pure `heading_pitch` is added to `core/attitude.py`. A new `render/sas_panel.py` holds the navball SAS chip + button grid + velocity readout.

**Tech Stack:** Python 3, Panda3D / DirectGUI, pytest. Venv interpreter only: `.venv/Scripts/python`.

## Global Constraints

- **Layering:** `core/` imports no panda3d; `render/` may import `core`. The pure helpers
  (`layout_panel`, `PanelLayout`, `heading_pitch`) must import NO DirectGUI/panda3d at module top
  level so they unit-test without graphics (mirror `orbit_panel_lines`, `camera_rig.zoom_to_scale`,
  `ship_model.view_blend`). `core/attitude.py` stays pure physics (numpy only) — it already imports numpy.
- **SI / radians in core.** `heading_pitch` returns radians; the HUD converts to degrees at the boundary.
- **Sandbox-only UI:** the SAS panel, velocity readout, and MANEUVER section exist only when
  `not self.solar_system and self.world.vessels` — guard exactly like the existing maneuver/navball UI.
- **Additive controls:** the SAS number keys (1–8) and `T` keep working; new buttons call the same
  `app._set_sas(mode)` / `app._toggle_sas()`. Keep the F1 overlay entries.
- **Backgrounds:** semi-transparent dark `frameColor=(0, 0, 0, 0.45)`.
- **Venv:** run everything with `.venv/Scripts/python -m pytest ...`. Bare `python` lacks deps.
- **Headless visual checks:** prepend `loadPrcFileData("", "window-type offscreen")` before constructing
  `OrbitApp(world, clock, solar_system=False)`, call `app._on_play()` to tear down the title and build
  the scene, drive with `app.taskMgr.step()`, capture `app.win.save_screenshot(Filename.from_os_specific(path))`.
  Set `PYTHONPATH=.`. (Sandbox world/clock construction: copy from `orbitsim/__main__.py::_default_world`.)
- **Commits:** stage explicit paths only (never `git add -A`/`.`). End messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`, then `git push`. Do NOT stage `data/`,
  `saves/`, screenshots, `.hypothesis/`, `.claude/settings.local.json`, `CLAUDE.md`.

---

### Task 1: Pure panel layout — `layout_panel` + `PanelLayout`

**Files:**
- Create: `orbitsim/render/hud/panel.py`
- Test: `tests/render/test_panel.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PanelLayout:` fields `line_ys: list[list[float]]`,
    `frame_top: float`, `frame_bottom: float`.
  - `layout_panel(section_line_counts: list[int], *, top: float, line_height: float, padding: float, section_gap: float) -> PanelLayout`.
    Lays out sections top-to-bottom in corner-relative coords (y decreases downward). Each line占 a
    slot of height `line_height`; an extra `section_gap` precedes each non-empty section after the
    first; a section with count 0 contributes an empty `[]` sublist and consumes no gap. `frame_top =
    top + padding`; `frame_bottom = (last_line_y - line_height) - padding` (or `top - padding` when all
    sections empty).

- [ ] **Step 1: Write the failing tests**

```python
# tests/render/test_panel.py
"""Tests for the pure panel-layout math (no DirectGUI)."""
from orbitsim.render.hud.panel import layout_panel, PanelLayout


def _flat(ys):
    return [y for sec in ys for y in sec]


def test_lines_strictly_decreasing_no_overlap():
    lay = layout_panel([1, 3, 2], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03)
    flat = _flat(lay.line_ys)
    assert len(flat) == 6
    assert all(flat[i] > flat[i + 1] for i in range(len(flat) - 1))  # the overlap-bug guard


def test_section_gap_applied_between_sections():
    lay = layout_panel([1, 1], top=0.0, line_height=0.06, padding=0.0, section_gap=0.03)
    # second section's first line sits one line_height + one section_gap below the first
    assert lay.line_ys[0][0] == 0.0
    assert abs(lay.line_ys[1][0] - (0.0 - 0.06 - 0.03)) < 1e-12


def test_empty_section_collapses():
    lay = layout_panel([2, 0, 1], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03)
    assert lay.line_ys[1] == []
    flat = _flat(lay.line_ys)
    assert len(flat) == 3
    # the empty middle section must not insert a gap before the third section
    expected_third = lay.line_ys[0][1] - 0.06 - 0.03
    assert abs(lay.line_ys[2][0] - expected_third) < 1e-12


def test_frame_encloses_all_lines():
    lay = layout_panel([1, 3, 2], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03)
    flat = _flat(lay.line_ys)
    assert lay.frame_top >= max(flat)
    assert lay.frame_bottom <= min(flat)


def test_all_empty_is_degenerate_but_safe():
    lay = layout_panel([0, 0], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03)
    assert _flat(lay.line_ys) == []
    assert lay.frame_top >= lay.frame_bottom
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/render/test_panel.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (panel module/functions not defined).

- [ ] **Step 3: Implement the pure layout**

```python
# orbitsim/render/hud/panel.py
"""Self-sizing HUD panel: a pure layout helper plus a DirectGUI panel class.

layout_panel imports only stdlib so it unit-tests without graphics (mirrors
hud.orbit_panel_lines / camera_rig.zoom_to_scale). The DirectGUI HudPanel class
(added in a later task) imports panda3d INSIDE its methods.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PanelLayout:
    line_ys: list           # list[list[float]] — per-section line y positions
    frame_top: float
    frame_bottom: float


def layout_panel(section_line_counts, *, top, line_height, padding, section_gap):
    """Lay out stacked sections top-to-bottom (y decreases downward).

    Each line occupies a slot of height `line_height`; `section_gap` precedes each
    non-empty section after the first; a 0-count section is skipped (empty sublist,
    no gap). Returns per-line ys plus the enclosing background frame extent.
    """
    line_ys = []
    cur = top
    first = True
    for count in section_line_counts:
        if count <= 0:
            line_ys.append([])
            continue
        if not first:
            cur -= section_gap
        first = False
        sec = []
        for _ in range(count):
            sec.append(cur)
            cur -= line_height
        line_ys.append(sec)

    flat = [y for sec in line_ys for y in sec]
    frame_top = top + padding
    if flat:
        frame_bottom = (min(flat) - line_height) - padding
    else:
        frame_bottom = top - padding
    return PanelLayout(line_ys=line_ys, frame_top=frame_top, frame_bottom=frame_bottom)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/render/test_panel.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/hud/panel.py tests/render/test_panel.py
git commit -m "HUD 2a: pure panel layout (layout_panel/PanelLayout)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 2: Pure heading/pitch — `core/attitude.py`

**Files:**
- Modify: `orbitsim/core/attitude.py` (add a function near `sas_target_dir`)
- Test: `tests/core/test_attitude.py` (add cases; create the file if absent)

**Interfaces:**
- Consumes: `nose_direction` (already in `core/attitude.py`), `quat_from_axis_angle` (same module).
- Produces: `heading_pitch(orientation_q, state) -> tuple[float, float]` returning `(heading_rad, pitch_rad)`.
  pitch = angle of the nose above the local horizon (`arcsin(nose·radial_out)`), pitch ∈ [-π/2, π/2];
  heading = `atan2(nose·east, nose·prograde)` normalized to `[0, 2π)`, using the local-horizon basis
  (prograde = v̂, radial-out, east = radial_out × v̂) — the same basis as `navball.horizon_frame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/core/test_attitude.py  (add these; keep existing tests if the file exists)
import math
import numpy as np

from orbitsim.core.attitude import heading_pitch, quat_from_axis_angle
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH


def _circular_equatorial():
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU_EARTH / 7.0e6), 0.0])
    return StateVector(r=r, v=v, mu=MU_EARTH)


def test_identity_nose_points_east():
    # identity orientation -> nose = body +Z = [0,0,1]; in this frame east = [0,0,1].
    st = _circular_equatorial()
    hdg, pit = heading_pitch(np.array([1.0, 0.0, 0.0, 0.0]), st)
    assert abs(pit) < 1e-9
    assert abs(hdg - math.pi / 2) < 1e-9


def test_nose_prograde_is_zero_heading_zero_pitch():
    # rotate +Z onto +Y (prograde) via -90 deg about X.
    st = _circular_equatorial()
    q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -math.pi / 2)
    hdg, pit = heading_pitch(q, st)
    assert abs(pit) < 1e-9
    assert abs(hdg) < 1e-9


def test_nose_radial_out_is_pitch_up_90():
    # rotate +Z onto +X (radial-out) via +90 deg about Y.
    st = _circular_equatorial()
    q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi / 2)
    hdg, pit = heading_pitch(q, st)
    assert abs(pit - math.pi / 2) < 1e-9


def test_ranges():
    st = _circular_equatorial()
    for ang in np.linspace(-math.pi, math.pi, 7):
        q = quat_from_axis_angle(np.array([0.3, 0.5, 0.8]), float(ang))
        hdg, pit = heading_pitch(q, st)
        assert 0.0 <= hdg < 2 * math.pi + 1e-9
        assert -math.pi / 2 - 1e-9 <= pit <= math.pi / 2 + 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -q`
Expected: FAIL — `ImportError: cannot import name 'heading_pitch'`.

- [ ] **Step 3: Implement `heading_pitch`**

Add to `orbitsim/core/attitude.py` (it already imports numpy as np and `math`/`nose_direction` exist; add `import math` at the top if not present):

```python
def heading_pitch(orientation_q, state):
    """Heading and pitch [rad] of the ship nose relative to the local horizon.

    pitch = arcsin(nose . radial_out)  -> angle above the local horizontal, in [-pi/2, pi/2].
    heading = atan2(nose . east, nose . prograde), normalized to [0, 2*pi).
    Basis: prograde = v_hat, radial_out, east = radial_out x v_hat — mirrors
    navball.horizon_frame (core cannot import the render layer).
    """
    import math
    r = np.asarray(state.r, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    prograde = v / np.linalg.norm(v)
    radial_out = np.cross(v, np.cross(r, v))
    radial_out = radial_out / np.linalg.norm(radial_out)
    east = np.cross(radial_out, prograde)
    nose = nose_direction(orientation_q)
    pitch = math.asin(max(-1.0, min(1.0, float(np.dot(nose, radial_out)))))
    heading = math.atan2(float(np.dot(nose, east)), float(np.dot(nose, prograde)))
    if heading < 0.0:
        heading += 2.0 * math.pi
    return heading, pitch
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_attitude.py -q`
Expected: PASS (4 added tests; plus any pre-existing).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/attitude.py tests/core/test_attitude.py
git commit -m "HUD 2a: pure heading_pitch (nose vs local horizon)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 3: `HudPanel` DirectGUI class

**Files:**
- Modify: `orbitsim/render/hud/panel.py` (add the class below the pure helper)

**Interfaces:**
- Consumes: `layout_panel`, `PanelLayout`.
- Produces: `HudPanel(parent, *, x, top, scale=0.045, align="left")` and
  `HudPanel.set_sections(sections)` where
  `sections = [{"header": str|None, "header_color": (r,g,b,a), "rows": [(text, (r,g,b,a)), ...]}, ...]`.
  Empty sections (no header and no rows) are omitted. Lays out one `OnscreenText` per visible line
  (pooled/reused), sizes one `DirectFrame` background `(0,0,0,0.45)` to the layout's frame extent.
  `LINE_HEIGHT`, `PADDING`, `SECTION_GAP` are module constants.

- [ ] **Step 1: Add module styling constants + the class**

```python
# append to orbitsim/render/hud/panel.py

LINE_HEIGHT = 0.058   # vertical spacing between HUD text lines (aspect2d units)
PADDING = 0.02        # background padding around content
SECTION_GAP = 0.028   # extra space before each section after the first
_BG_COLOR = (0.0, 0.0, 0.0, 0.45)


class HudPanel:
    """A corner-anchored stack of color-coded sections over one translucent background."""

    def __init__(self, parent, *, x, top, scale=0.045):
        from direct.gui.DirectFrame import DirectFrame
        self._parent = parent
        self._x = x
        self._top = top
        self._scale = scale
        self._texts = []   # pooled OnscreenText rows
        self._bg = DirectFrame(parent=parent, frameColor=_BG_COLOR,
                               frameSize=(0, 0.01, 0, 0.01), pos=(0, 0, 0))
        self._bg.hide()

    def _row(self, i):
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        while i >= len(self._texts):
            t = OnscreenText(text="", scale=self._scale, align=TextNode.ALeft,
                             fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1), mayChange=True,
                             parent=self._parent)
            t.hide()
            self._texts.append(t)
        return self._texts[i]

    def set_sections(self, sections):
        # Build flat (text, color) row list + per-section counts (header counts as a row).
        rows = []
        counts = []
        for sec in sections:
            header = sec.get("header")
            body = sec.get("rows", [])
            n = (1 if header else 0) + len(body)
            counts.append(n)
            if header:
                rows.append((header, sec.get("header_color", (1, 1, 1, 1))))
            rows.extend(body)

        lay = layout_panel(counts, top=self._top, line_height=LINE_HEIGHT,
                           padding=PADDING, section_gap=SECTION_GAP)
        flat_ys = [y for s in lay.line_ys for y in s]

        for i, (text, color) in enumerate(rows):
            t = self._row(i)
            t.setText(text)
            t.setFg(color)
            t.setPos(self._x, flat_ys[i])
            t.show()
        for j in range(len(rows), len(self._texts)):
            self._texts[j].hide()

        if rows:
            # Background spans from a small left margin to a fixed width, framed vertically.
            self._bg["frameSize"] = (self._x - PADDING, self._x + 0.62,
                                     lay.frame_bottom, lay.frame_top)
            self._bg.show()
        else:
            self._bg.hide()
```

- [ ] **Step 2: Headless screenshot of a standalone panel**

Write a scratch script (do not commit): build a `ShowBase` offscreen, create a `HudPanel` on
`base.a2dTopLeft` at `x=0.08, top=-0.10`, call `set_sections` with a TIME row, an ORBIT section (header
+ 6 rows), and a MANEUVER section (header + 3 colored rows), step, screenshot. Then call `set_sections`
again WITHOUT the MANEUVER section and screenshot to confirm it shrinks.

```python
# scratch_panel.py
from panda3d.core import loadPrcFileData, Filename
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "win-size 600 600")
from direct.showbase.ShowBase import ShowBase
from orbitsim.render.hud.panel import HudPanel
OUT = r"<scratchpad>"
base = ShowBase()
p = HudPanel(base.a2dTopLeft, x=0.08, top=-0.10)
CYAN = (0.7, 0.95, 1.0, 1); MAG = (1.0, 0.4, 1.0, 1); ORG = (1.0, 0.7, 0.4, 1); GRN = (0.6, 1.0, 0.6, 1)
full = [
    {"header": None, "rows": [("Sim time: 0 s past J2000", CYAN)]},
    {"header": "ORBIT", "header_color": CYAN, "rows": [(s, (1,1,1,1)) for s in
        ["Altitude: 500.0 km", "Speed: 8.07 km/s", "Periapsis: 500 km",
         "Apoapsis: 2,465 km", "Inclination: 8.1°", "Period: 115.6 min"]]},
    {"header": "MANEUVER", "header_color": MAG, "rows": [
        ("Maneuver dV: 120 m/s  (dV left 1,763)", MAG),
        ("Node in T-02:14", CYAN), ("Target: Moon  CA T-08:00", ORG)]},
]
p.set_sections(full)
for _ in range(3): base.taskMgr.step()
base.win.save_screenshot(Filename.from_os_specific(OUT + r"\panel_full.png"))
p.set_sections(full[:2])  # drop MANEUVER
for _ in range(3): base.taskMgr.step()
base.win.save_screenshot(Filename.from_os_specific(OUT + r"\panel_short.png"))
print("saved")
```

Run: `PYTHONPATH=. .venv/Scripts/python <scratchpad>/scratch_panel.py`
Expected: `panel_full.png` shows three grouped, color-coded sections over a translucent box with no
overlap; `panel_short.png` shows the box shrunk to two sections. Read both PNGs to verify.

- [ ] **Step 3: Suite stays green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add orbitsim/render/hud/panel.py
git commit -m "HUD 2a: HudPanel DirectGUI class (grouped sections, self-sizing bg)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 4: Rewire `Hud` + `app.py` to grouped panels; drop redundant warp line

**Files:**
- Modify: `orbitsim/render/hud/__init__.py`
- Modify: `orbitsim/render/app.py` (remove the 3 standalone maneuver `OnscreenText`s; route their text through `Hud`)
- Modify: `tests/render/test_hud_format.py` (drop the `warp` arg)

**Interfaces:**
- Consumes: `HudPanel` (Task 3).
- Produces (new `Hud` API):
  - `Hud.update(...)` unchanged signature EXCEPT `warp` removed — builds TIME+ORBIT sections of the
    left panel.
  - `Hud.set_maneuver(dv_line: str, node_line: str, target_line: str)` — stores the three maneuver
    lines (empty string = omit that row) and rebuilds the left panel; the MANEUVER section is omitted
    when all three are empty.
  - `Hud.update_flight(...)` unchanged signature — builds the right VESSEL panel.
  - `orbit_panel_lines(...)` loses its `warp` parameter and the `Warp:` line.

- [ ] **Step 1: Drop the warp line from `orbit_panel_lines` + update its test**

In `orbitsim/render/hud/__init__.py`, change `orbit_panel_lines` to remove the `warp` parameter and the
`f"Warp: x{warp:,.0f}"` entry (keep `Sim time` as the first line). In
`tests/render/test_hud_format.py`, remove `warp=1.0` from the `_lines` helper `base` dict. Add:

```python
def test_no_warp_line():
    text = "\n".join(_lines())
    assert "Warp" not in text
```

Run: `.venv/Scripts/python -m pytest tests/render/test_hud_format.py -q` → PASS.

- [ ] **Step 2: Rewire `Hud` to own two `HudPanel`s**

Replace `Hud.__init__`'s `self.text`/`self.flight` `OnscreenText`s with `HudPanel`s and store maneuver
state; keep `self.toast`. Implement the section-building. Colors: TIME/ORBIT cyan-white
`(0.7,0.95,1.0,1)` headers, body white; VESSEL green `(0.6,1.0,0.6,1)`; MANEUVER header magenta, rows
magenta/cyan/orange.

```python
# orbitsim/render/hud/__init__.py  (Hud class, replacing the OnscreenText panels)
from orbitsim.render.hud.panel import HudPanel

_CYAN = (0.7, 0.95, 1.0, 1)
_GREEN = (0.6, 1.0, 0.6, 1)
_MAG = (1.0, 0.4, 1.0, 1)
_ORANGE = (1.0, 0.7, 0.4, 1)

# in __init__:
self._left = HudPanel(base.a2dTopLeft, x=0.08, top=-0.10)
self._right = HudPanel(base.a2dTopRight, x=-0.62, top=-0.10)  # left-aligned text, right region
self._orbit_lines = []      # set by update()
self._mvr = ("", "", "")    # (dv_line, node_line, target_line)
# keep self.toast as before; keep self.units

def _rebuild_left(self):
    time_line = self._orbit_lines[0] if self._orbit_lines else ""
    orbit_rows = [(s, (1, 1, 1, 1)) for s in self._orbit_lines[1:]]
    sections = [{"header": None, "rows": [(time_line, _CYAN)] if time_line else []},
                {"header": "ORBIT", "header_color": _CYAN, "rows": orbit_rows}]
    dv, node, tgt = self._mvr
    mvr_rows = []
    if dv:   mvr_rows.append((dv, _MAG))
    if node: mvr_rows.append((node, _CYAN))
    if tgt:  mvr_rows.append((tgt, _ORANGE))
    if mvr_rows:
        sections.append({"header": "MANEUVER", "header_color": _MAG, "rows": mvr_rows})
    self._left.set_sections(sections)
```

Update `Hud.update(...)`: remove `warp` param; set `self._orbit_lines = orbit_panel_lines(...)` (without
warp) and call `self._rebuild_left()`. Add `Hud.set_maneuver(dv_line, node_line, target_line)` that sets
`self._mvr` and calls `self._rebuild_left()`. Update `Hud.update_flight(...)` to build the VESSEL section
via `self._right.set_sections([{ "header": "VESSEL", "header_color": _GREEN, "rows": [(l, _GREEN) for l in lines] }])`.

Note: `a2dTopRight` x is negative-going-left; with left-aligned text use a negative `x` (e.g. -0.62) so
the block sits inside the right edge. Verify/adjust `x` by screenshot in Step 4.

- [ ] **Step 3: Route the maneuver readouts through `Hud` in `app.py`**

- Delete the three standalone creations: `self._dv_readout` (~L409), `self._node_ttn_text` (~L420),
  `self._target_text` (~L289), and their `OnscreenText` blocks.
- `_refresh_readout` (~L613): replace `self._dv_readout.setText(...)` with building the dv string and
  calling `self.hud.set_maneuver(dv_line, self._node_line, self._target_line)`. Maintain cached
  `self._node_line` / `self._target_line` strings (init to "") so each setter preserves the others.
- Node readout (~L1114/1116): set `self._node_line = f"Node {label}   dV {node.magnitude_mps:,.1f} m/s"`
  (or `""`), then `self.hud.set_maneuver(self._dv_line, self._node_line, self._target_line)`. Cache the
  dv string as `self._dv_line` in `_refresh_readout`.
- Target readouts (~L599 "Target: none" → use ""; ~L1140; ~L1179): set `self._target_line = <string or "">`
  then call `self.hud.set_maneuver(...)`.
- Remove `warp=self.clock.warp` from the `self.hud.update(...)` call (~L1203).

Initialize `self._dv_line = self._node_line = self._target_line = ""` where the other maneuver state is
initialized (near the maneuver UI build).

- [ ] **Step 4: Headless screenshot — the overlap bug, fixed**

Scratch script (not committed): start the sandbox offscreen (`_on_play`), set a planned node and select
the Moon target so all maneuver lines are populated, zoom to a map distance, step, screenshot. Confirm
TIME/ORBIT/MANEUVER stack with no overlap over a translucent background, plus the right VESSEL panel.
Then clear the node + target and screenshot to confirm MANEUVER vanishes and the panel shrinks.

```python
# scratch_hud.py  — sketch; set node/target via the app's own methods
# app._node_epoch_s = app.clock.sim_time_s + 600.0 ; app._refresh_readout()
# app._try_pick_target(...) OR app._target = MoonTarget(); then step + screenshot
```

Run it; read `hud_overlap.png` (no overlap, node+target present) and `hud_clean.png` (no MANEUVER).
Adjust `HudPanel` `x`/`top`/widths if the right panel clips.

- [ ] **Step 5: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: all pass (panel + heading_pitch + hud_format updated).

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/hud/__init__.py orbitsim/render/app.py tests/render/test_hud_format.py
git commit -m "HUD 2a: grouped top-left/right panels; maneuver rows via Hud; drop warp line

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 5: Navball SAS chip + clickable SAS-mode buttons

**Files:**
- Create: `orbitsim/render/sas_panel.py`
- Modify: `orbitsim/render/app.py` (construct + update the SAS panel; sandbox-only)

**Interfaces:**
- Consumes: `core.attitude.heading_pitch`, `core.attitude.SAS_MODES`, `app._set_sas`, `app._toggle_sas`,
  `vessel.sas_mode`.
- Produces: `SasPanel(base, *, on_set_mode, on_toggle)` with
  `SasPanel.update(sas_mode: str, heading_rad: float, pitch_rad: float)` — sets the readout text and
  highlights the active mode button.

- [ ] **Step 1: Implement `SasPanel`**

```python
# orbitsim/render/sas_panel.py
"""Navball-adjacent SAS chip: current mode + heading/pitch readout, and a button
grid (8 hold modes + on/off) mirroring the 1-8 / T keys. Render-only."""
import math

from orbitsim.core.attitude import SAS_MODES

_SHORT = {"PROGRADE": "PRO", "RETROGRADE": "RET", "NORMAL": "NML", "ANTINORMAL": "ANM",
          "RADIAL_IN": "RIN", "RADIAL_OUT": "ROUT", "TARGET": "TGT", "ANTITARGET": "ATG"}
_IDLE = (0.15, 0.15, 0.2, 0.85)
_ACTIVE = (0.2, 0.7, 1.0, 0.95)
_BG = (0.0, 0.0, 0.0, 0.45)


class SasPanel:
    def __init__(self, base, *, on_set_mode, on_toggle):
        from direct.gui.DirectButton import DirectButton
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        self._buttons = {}
        # Readout sits above the navball (bottom-center region).
        self._readout = OnscreenText(
            text="", pos=(0.0, 0.34), scale=0.04, fg=(0.85, 0.95, 1.0, 1),
            shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True,
            parent=base.a2dBottomCenter)
        # SAS on/off button, then a row of 8 mode buttons just above the navball.
        self._buttons["__TOGGLE__"] = DirectButton(
            text="SAS", scale=0.04, pos=(-0.78, 0, 0.30), frameColor=_IDLE,
            text_fg=(1, 1, 1, 1), command=on_toggle, parent=base.a2dBottomCenter)
        for i, mode in enumerate(SAS_MODES):
            self._buttons[mode] = DirectButton(
                text=_SHORT[mode], scale=0.035, pos=(-0.62 + i * 0.155, 0, 0.30),
                frameColor=_IDLE, text_fg=(1, 1, 1, 1),
                command=on_set_mode, extraArgs=[mode], parent=base.a2dBottomCenter)

    def update(self, sas_mode, heading_rad, pitch_rad):
        hdg = math.degrees(heading_rad) % 360.0
        pit = math.degrees(pitch_rad)
        self._readout.setText(f"SAS: {sas_mode}    HDG {hdg:03.0f}°   PIT {pit:+03.0f}°")
        for mode, btn in self._buttons.items():
            if mode == "__TOGGLE__":
                on = sas_mode != "OFF"
                btn["frameColor"] = _ACTIVE if on else _IDLE
            else:
                btn["frameColor"] = _ACTIVE if mode == sas_mode else _IDLE
```

- [ ] **Step 2: Wire into `app.py` (sandbox-only)**

In `_start_sim`, in the sandbox branch where the navball is built (after `self.navball = Navball(self)`),
add:

```python
            from orbitsim.render.sas_panel import SasPanel
            self.sas_panel = SasPanel(self, on_set_mode=self._set_sas, on_toggle=self._toggle_sas)
```

In `_update`, after `self.navball.update(...)` (~L1227), add (sandbox path only — `_update` already
returns early for solar mode):

```python
        from orbitsim.core.attitude import heading_pitch
        hdg, pit = heading_pitch(v0.orientation, v0.state)
        self.sas_panel.update(self.world.vessels[0].sas_mode, hdg, pit)
```

- [ ] **Step 3: Headless screenshot + click behavior**

Scratch script (not committed): start sandbox offscreen, step, screenshot (`sas.png` — readout + button
row above navball, current mode highlighted). Then simulate a mode set: `app._set_sas("PROGRADE")`, step,
screenshot (`sas_pro.png` — PRO highlighted, readout shows PROGRADE). Read both PNGs.

Run: `PYTHONPATH=. .venv/Scripts/python <scratchpad>/scratch_sas.py`
Expected: buttons render above the navball; clicking/`_set_sas` highlights the active mode and updates
the readout. Adjust button positions/scale if they overlap the navball or each other.

- [ ] **Step 4: Full suite green + commit**

Run: `.venv/Scripts/python -m pytest tests/ -q` (all pass).

```bash
git add orbitsim/render/sas_panel.py orbitsim/render/app.py
git commit -m "HUD 2a: navball SAS chip + clickable SAS-mode buttons

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 6: Velocity readout above the navball (orbital/target toggle)

**Files:**
- Modify: `orbitsim/render/sas_panel.py` (add `VelocityReadout`)
- Modify: `orbitsim/render/app.py` (construct + update; compute orbital + target-relative speed)

**Interfaces:**
- Consumes: `hud._speed` (the km/mi formatter) — import `from orbitsim.render.hud import _speed`.
- Produces: `VelocityReadout(base, units_getter)` with
  `VelocityReadout.update(orbital_speed_mps: float, target_rel_speed_mps)` where the second arg is a
  float or `None`. Clicking the chip toggles `_vel_mode` between "ORBITAL" and "TARGET". Shows
  `Orbital  <spd>` or `Target  <spd>`, and `Target  —` when in target mode with `target_rel_speed`
  None.

- [ ] **Step 1: Implement `VelocityReadout`**

```python
# add to orbitsim/render/sas_panel.py

class VelocityReadout:
    """Clickable chip above the navball toggling orbital vs target speed."""

    def __init__(self, base, units_getter):
        from direct.gui.DirectButton import DirectButton
        self._units_getter = units_getter
        self._mode = "ORBITAL"
        self._orbital = 0.0
        self._target = None
        self._btn = DirectButton(
            text="", scale=0.045, pos=(0.0, 0, 0.40), frameColor=_BG,
            text_fg=(1, 1, 1, 1), relief=None, command=self._toggle,
            parent=base.a2dBottomCenter)
        self._refresh()

    def _toggle(self):
        self._mode = "TARGET" if self._mode == "ORBITAL" else "ORBITAL"
        self._refresh()

    def update(self, orbital_speed_mps, target_rel_speed_mps):
        self._orbital = orbital_speed_mps
        self._target = target_rel_speed_mps
        self._refresh()

    def _refresh(self):
        from orbitsim.render.hud import _speed
        units = self._units_getter()
        if self._mode == "ORBITAL":
            self._btn["text"] = f"Orbital  {_speed(self._orbital, units)}"
        elif self._target is None:
            self._btn["text"] = "Target  —"
        else:
            self._btn["text"] = f"Target  {_speed(self._target, units)}"
```

- [ ] **Step 2: Wire into `app.py`**

In `_start_sim` sandbox branch, after the SAS panel:

```python
            from orbitsim.render.sas_panel import VelocityReadout
            self.vel_readout = VelocityReadout(self, lambda: self.hud.units)
```

In `_update`, after the SAS panel update, compute speeds and call update:

```python
        orbital_speed = float(v0.state.v_mag)
        target_rel = None
        if self._target is not None:
            tv = self._target.state_at(self.clock.sim_time_s).v
            target_rel = float(np.linalg.norm(v0.state.v - tv))
        self.vel_readout.update(orbital_speed, target_rel)
```

(`np` is already imported in `app.py`; `self._target` is the current target or `None`.)

- [ ] **Step 3: Headless screenshot — toggle behavior**

Scratch script (not committed): start sandbox offscreen, step, screenshot (`vel_orbital.png` — shows
"Orbital  …"). Call `app.vel_readout._toggle()` with NO target, step, screenshot (`vel_target_none.png`
— "Target  —"). Set `app._target = MoonTarget()`, step, screenshot (`vel_target.png` — "Target  <spd>").
Read the three PNGs.

Run: `PYTHONPATH=. .venv/Scripts/python <scratchpad>/scratch_vel.py`
Expected: the chip shows orbital speed by default; toggling with no target shows the em-dash; with the
Moon targeted it shows a relative speed. Adjust `pos` if it overlaps the SAS readout.

- [ ] **Step 4: Full suite green + commit**

Run: `.venv/Scripts/python -m pytest tests/ -q` (all pass).

```bash
git add orbitsim/render/sas_panel.py orbitsim/render/app.py
git commit -m "HUD 2a: velocity readout above navball (orbital/target toggle)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

## Self-Review notes

- **Spec coverage:** reusable panel + backgrounds (Tasks 1,3); overlap fix structurally via computed
  layout (Task 1 invariant + Task 4 wiring); grouped TIME/ORBIT/MANEUVER + VESSEL with colors, drop
  redundant warp line (Task 4); `heading_pitch` (Task 2); navball SAS chip + clickable buttons mirroring
  1–8/T, active highlight (Task 5); velocity readout orbital/target toggle incl. no-target em-dash
  (Task 6); sandbox-only guards (Tasks 5,6); unit tests for pure helpers + headless screenshots for
  visuals (all). Out-of-scope 2b items (orbit/trajectory lines, in-world markers, navball reskin) absent.
- **Type consistency:** `layout_panel(section_line_counts, *, top, line_height, padding, section_gap) ->
  PanelLayout`; `HudPanel.set_sections(sections)` with the `{header, header_color, rows}` shape;
  `Hud.set_maneuver(dv_line, node_line, target_line)`; `heading_pitch(orientation_q, state) -> (heading,
  pitch)`; `SasPanel.update(sas_mode, heading_rad, pitch_rad)`; `VelocityReadout.update(orbital_speed_mps,
  target_rel_speed_mps)` — used consistently across tasks.
- **Verification points flagged for execution:** exact `a2dTopRight` `x` and panel widths (Task 4 Step 4)
  and SAS/velocity button positions (Tasks 5–6) are tuned by screenshot — the wiring is independent of
  the final pixel values. `core/attitude.py` already imports numpy; confirm `import math` exists at top
  (the function also imports it locally as a safeguard).
```
