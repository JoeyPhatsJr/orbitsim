# HUD Overlay Readability (Cycle 2a) — design

**Date:** 2026-06-27
**Status:** approved (brainstorm), pending implementation plan
**Part of:** the "graphical improvements" effort. Cycle 2 (HUD/readability) was split into
**2a — screen-space overlay readability** (this spec) and **2b — in-world scene readability** (later:
target/CA/Lagrange markers + labels, orbit/trajectory line color + Pe/Ap markers + depth). Cycle 1
(ship view) is already merged.

## Goal

Make the 2D DirectGUI overlay legible and organized. Today the top-left corner stacks four
independently-positioned `a2dTopLeft` readouts whose hardcoded y-positions now collide: the 8-line
orbit panel (starts y=-0.12) grows straight through the magenta dV-left readout (-0.36), the cyan
node-time readout (-0.42), and the orange target readout (-0.48). This cycle replaces hardcoded
stacking with a reusable, self-sizing **panel** component that groups and color-codes readouts over
subtle backgrounds, adds a navball **SAS chip** (mode + heading/pitch readout) with a row of
**clickable SAS-mode buttons** mirroring the existing SAS number keys (1–8) and T, and adds a
**velocity readout above the navball** that click-toggles between **orbital** and **target** speed.

## Constraints / non-goals

- **Render-only**, except one small pure addition to `core/attitude.py` (heading/pitch math — see
  below), which stays SI/radians and is unit-tested.
- **Layering preserved:** overlay code in `orbitsim/render/`; the pure panel-layout helper imports no
  DirectGUI (mirrors `orbit_panel_lines`, `view_blend`).
- **No new orbit/trajectory/in-world marker work** — that is Cycle 2b.
- Keep all existing keybindings working; the new buttons are *additive* to the 1–7 / T keys.

## Current code (grounding)

- `render/hud/__init__.py`: `Hud` owns `self.text` (top-left orbit panel, `a2dTopLeft`, 8 lines via
  `orbit_panel_lines`), `self.flight` (top-right, green), and `self.toast` (center). Pure
  `orbit_panel_lines(...)` builds the orbit text; `_dist`/`_speed` handle km/mi.
- `render/app.py`: separately creates, on `a2dTopLeft`, `self._dv_readout` (magenta, y=-0.36),
  `self._node_ttn_text` (cyan, y=-0.42), `self._target_text` (orange, y=-0.48) — these overlap the
  orbit panel. Warp is shown **twice**: `Warp: x{warp}` in the orbit panel *and* the top-center
  `self._warp_readout` control (`a2dTopCenter`).
- `render/navball.py`: a developed 3D attitude ball (textured sky/ground, colored orbital markers,
  bezel, reticle) in a bottom-center square display region. No SAS-mode or heading text today.
- SAS: `core/attitude.py::SAS_MODES` (8 modes); `vessel.sas_mode` ∈ {"OFF","STABILITY",*SAS_MODES*};
  `app._set_sas(mode)`, `app._toggle_sas()`. Keys: `1`–`8`→modes (bound as `str(i)` for i=1..8),
  `T`→toggle STABILITY/OFF.

## Architecture

### 1. Reusable panel component — `render/hud/panel.py`

Split pure layout from DirectGUI:

- **`layout_panel(section_line_counts, *, line_height, padding, header_gap) -> PanelLayout`** — pure,
  no DirectGUI. `section_line_counts` is a list of ints (lines per section, *including* the header
  line; a section with 0 lines is omitted). Returns a `PanelLayout` (frozen dataclass) with:
  - `line_ys: list[list[float]]` — the y (corner-relative, negative-down) of each line in each
    section, laid out top-to-bottom with `line_height` spacing and `header_gap` extra space before
    each section after the first.
  - `frame_top: float`, `frame_bottom: float` — the background frame's vertical extent, enclosing all
    lines plus `padding`.
  Invariants (the regression guard): line ys strictly decrease within and across sections (no
  overlap); `frame_top >= first line + padding`; `frame_bottom <= last line - padding`; an empty
  section contributes no ys and no gap.

- **`HudPanel`** — DirectGUI class. Constructed with a parent region (e.g. `base.a2dTopLeft`), a left
  margin `x`, a top `y`, and styling constants. `set_sections(sections)` where
  `sections = [(header_text or None, accent_rgba, [body_line, ...]), ...]`: it calls `layout_panel`
  on the non-empty sections, positions one `OnscreenText` per section (header in `accent_rgba`, body
  lines neutral white — multi-line `setText`), and resizes a single semi-transparent `DirectFrame`
  background (`frameColor=(0,0,0,0.45)`) to `[frame_top, frame_bottom]`. Re-callable each frame;
  cheap (text + frame size sets, no geometry rebuild).

### 2. Heading/pitch math — `core/attitude.py`

Add pure **`heading_pitch(orientation_q, state) -> (heading_rad, pitch_rad)`**: pitch =
`arcsin(clamp(nose·radial_out))` (angle of the nose above the local horizon); heading =
`atan2(nose·east, nose·prograde)` normalized to `[0, 2π)`, using the same `horizon_frame` basis the
navball already uses (prograde, east, radial-out). Unit-tested with known-answer cases (nose along
prograde → pitch 0, heading 0; nose along radial-out → pitch +90°). Clamp `arcsin`/`arccos` args to
`[-1,1]` per project convention.

### 3. Navball SAS chip + buttons — `render/navball.py` (or a small `render/sas_panel.py`)

Beside/above the navball:
- a **readout** (one `HudPanel` or `OnscreenText`): current SAS mode (e.g. "SAS: PROGRADE" / "STABILITY"
  / "OFF") + heading/pitch ("HDG 087°  PIT +12°"), updated each frame from `vessel.sas_mode` and
  `heading_pitch`.
- a **button grid**: a SAS on/off `DirectButton` (calls `app._toggle_sas`) plus 8 short-label buttons
  (PRO, RET, NML, ANM, RIN, ROUT, TGT, ATG) that call `app._set_sas(mode)`. The button whose mode ==
  `vessel.sas_mode` is highlighted (distinct frameColor) each frame. Buttons are additive — the 1–8/T
  keys still work and stay in the F1 overlay.

### 4. Velocity readout above the navball — `render/navball.py` (or `render/sas_panel.py`)

A clickable speed readout placed **above** the navball (a `DirectButton` styled as a chip so the
click toggles, with the same `(0,0,0,0.45)` background). It holds a `_vel_mode` ∈ {"ORBITAL",
"TARGET"} (default "ORBITAL"); a click flips it. Each frame it shows:
- **ORBITAL:** `Orbital  8.074 km/s` — magnitude of `vessel.state.v` (speed about the central body),
  formatted with the existing `_speed` (km/mi).
- **TARGET:** `Target  1.231 km/s` — the relative speed `|vessel.state.v − target_v|`, where
  `target_v` is the selected target's velocity (`target.state_at(now).v`, the same source the
  closest-approach / Lagrange rel-vel readout already uses). When **no target is selected**, show
  `Target  —` (em dash) rather than a number.

The toggle is independent of the SAS controls; it only changes which speed is displayed.

## Components & content (the panels)

- **Top-left `HudPanel`** (`a2dTopLeft`), sections in order:
  - **TIME** (accent cyan-white): `Sim time: … s past J2000`. *(Drop the redundant `Warp:` line —
    warp lives in the top-center control.)*
  - **ORBIT** (accent cyan-white): Altitude, Speed, Periapsis, Apoapsis, Inclination, Period (km/mi
    via existing `_dist`/`_speed`).
  - **MANEUVER** (accent magenta header; shown only when a node or target is active): `dV left …`
    (magenta), `Node in T-…` (cyan), `Target: …` (orange). When none active the section is omitted and
    the panel shrinks.
- **Top-right `HudPanel`** (`a2dTopRight`): **VESSEL** section (accent green): Throttle, Fuel,
  Mass, Thrust/TWR, dV-left, and the "WARP LOCKED — thrusting" line when locked.
- **Top-center warp control**: unchanged behavior; add the same `(0,0,0,0.45)` background for
  consistency.

The `Hud` class keeps owning the top-left and top-right panels (now `HudPanel`s instead of bare
`OnscreenText`s) and its `toast`. The maneuver readouts move **out** of `app.py`'s separate
`a2dTopLeft` `OnscreenText`s and **into** the top-left panel's MANEUVER section: `app.py` passes the
maneuver lines (dV-left, node-TTN, target) to the `Hud` each frame instead of setting three separate
text nodes. The old `_dv_readout` / `_node_ttn_text` / `_target_text` nodes are removed.

## Data flow (per frame)

`app._update` already computes orbit + flight + maneuver values. It calls:
- `hud.update(... orbit values ...)` → builds TIME+ORBIT sections.
- `hud.update_flight(...)` → builds VESSEL section.
- `hud.update_maneuver(dv_left, ttn, target_name)` (new) → builds/omits the MANEUVER section.
- `sas_panel.update(sas_mode, heading_rad, pitch_rad)` (new) → readout text + active-button highlight.
- `vel_readout.update(orbital_speed_mps, target_rel_speed_mps_or_None)` (new) → shows the speed for the
  current `_vel_mode`, or `Target  —` when target mode is active with no target.

## Error handling / edge cases

- Empty MANEUVER section → omitted; panel shrinks (covered by `layout_panel` empty-section invariant).
- `heading_pitch` at the velocity/position degeneracies: `horizon_frame` already assumes `v≠0` and
  `r×v≠0` (true for any real orbit); clamp `arcsin` arg. No new degeneracy handling beyond the clamp.
- Solar-system mode has no vessel/navball/maneuver UI → the SAS panel, velocity readout, and MANEUVER
  section are sandbox-only (guard like existing `if not self.solar_system`).
- Velocity readout in TARGET mode with no target selected → `Target  —`; toggling back to ORBITAL
  always shows a number. The `_vel_mode` persists across target select/deselect.

## Testing

- **Unit (no graphics):**
  - `layout_panel`: line ys strictly decreasing within+across sections (no overlap — the bug), header
    gaps applied, empty sections omit ys+gap, frame extent encloses all content + padding.
  - `heading_pitch`: known-answer cases (prograde→(0,0); radial-out→pitch +π/2; retrograde→heading π);
    output ranges (heading ∈ [0,2π), pitch ∈ [-π/2, π/2]).
  - Existing `orbit_panel_lines` / `test_hud` stay green (orbit lines minus the warp line — update that
    test).
- **Visual (headless screenshots, per CLAUDE.md offscreen method):**
  - Top-left panel with TIME+ORBIT+MANEUVER and a node **and** target active → no overlap (the exact
    regression), grouped, color-coded, background legible over Earth.
  - MANEUVER section absent when no node/target → panel shrinks, no gap.
  - Top-right VESSEL panel + backgrounds; warp control background.
  - Navball SAS chip: mode + heading/pitch readout; button grid; clicking a button sets the mode and
    highlights it; active highlight tracks key-driven changes too.
  - Velocity readout above the navball: shows orbital speed by default; clicking toggles to target
    speed (relative) when a target is selected, and to `Target  —` when none is; toggling back returns
    to orbital speed.

## Out of scope (YAGNI / deferred to 2b)

- Orbit/trajectory line color, Pe/Ap markers on the line, depth fading.
- In-world target / closest-approach / Lagrange marker + label legibility, leader lines, decluttering.
- Reskinning the navball ball itself (texture/markers are already good).
