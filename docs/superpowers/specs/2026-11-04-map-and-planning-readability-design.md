# Map & planning readability — design

Date: 2026-11-04. Status: agreed in conversation; reviewed by a fresh builder; awaiting the user's review.

## Why

The user tried to plan a Moon transfer and gave up:

> "couldnt see the lines, they didnt refresh quick enough when plotting the node, and i didnt
> know where to point the node"

They want the map to look and play "like KSP": a readable map view where you plan burns by
dragging nodes and watching the line react. The physics is already right (tested against
Curtis) and fast enough (PRs #9, #10). This pass is about **what the player can see and how
they plan**, so a first-time player can plan and fly a Moon transfer without giving up.

**Success:** starting in LEO, a player can target the Moon, place and shape a burn by hand
(or use the auto-plan button and fine-tune it), see an encounter confirmed on screen, and
execute it. Every line, marker and label involved is clearly visible at any zoom.

## Scope

In: the **map view** (sandbox, Earth–Moon and beyond) and the **solar viewer**, plus
**maneuver planning**.

Out (this pass): the close-up ship view (`M`), built-in scenarios, the Windows .exe,
the docs refresh, the settings/QoL items. These are tabled until this lands.

Also out, **next pass** (user's decision): the **capture burn**, the second burn that puts
you into orbit around the Moon. That means placing nodes on the future transfer path and
holding more than one planned burn. This pass stays **one node at a time, placed on your
current orbit** ("get to the Moon"); "stay at the Moon" comes next. Don't build anything
that makes multiple nodes harder later, but don't build multi-node support now.

## Anti-goals

- **No engine rewrite.** It stays on Panda3D (decided: the problem is presentation, not the
  engine).
- **The physics doesn't change.** The flown trajectory, integrators and the N-body model stay
  as they are; the app goldens' physics fields must not move. (HUD text, labels and visuals
  may change. That's the point of the pass. Re-recording goldens is allowed only for
  presentation fields, with each change explained in its commit.)
- **Not a KSP clone.** It borrows the genre's conventions (thick orbit lines, Ap/Pe markers,
  draggable node handles, navball), not KSP's art, fonts, colours or branding.
- **No frame-rate regression.** The current workloads (`tests.render.app_harness.profile`)
  must not get slower. Node dragging must stay smooth (≥ 60 fps on the dev Mac).

## The look (direction "A · Clean map", chosen by the user)

- **Lines.** Every orbit and trajectory is a thick, solid line with a **constant on-screen
  width** at any zoom (never a hairline). A few strong colours:
  - your current orbit: bright cyan;
  - your planned path after a burn: orange, dashed;
  - the Moon's and planets' orbits: muted grey, there for reference without competing with
    yours (in the solar viewer each planet may get a muted tint);
  - the targeted body's orbit is highlighted relative to the other reference orbits.
  Must render thick on **macOS**: OpenGL ignores line widths above 1 there, which is why
  today's `LineSegs.set_thickness` lines come out 1 px. So the lines need real geometry
  (e.g. screen-space ribbons), not GL line width.
- **Background.** Far fewer and dimmer stars, so lines and bodies pop.
- **Your ship.** The actual 3D ship model (the one from `render/ship_model.py`), pointing the
  way the ship really points, kept at a **fixed, readable on-screen size** at every zoom.
  It replaces the yellow marker sphere. It should stand out against Earth (e.g. a faint
  outline or glow; builder's call).
- **Markers.** Ap and Pe as small ticks on your orbit with short labels ("Pe 500 km"). A
  planned burn is a ring on the orbit with one line of text: size, time until, burn duration
  ("Burn 3,130 m/s · in 23 min · 4m 40s burn"). A predicted encounter is a yellow dot with
  "Moon encounter · Pe 4,317 km".
- **Labels.** Fewer of them, and they never sit on top of buttons or each other.
  Lagrange points fade into the background unless one is targeted.
- **Screen panels.** Small panels in the corners: time and warp top-left, your orbit and
  fuel bottom-left, the navball tucked in at the bottom centre. The maneuver readout
  appears only when a burn exists. Nothing overlaps (today "CA: SHIP" and approach labels
  land on the SAS buttons; the warp text runs into its own buttons at long times).
- **Solar viewer.** Every planet's orbit visible as a line. Planet names must not pile up
  (today Earth's label hides behind Venus's, and Mercury's sits on the Sun's).
- **Opening camera.** The first frame after launch should show Earth lit and your orbit
  clearly, not the night side filling the screen (builder's call on exact framing).

## Planning a burn

1. **Place.** Hovering over your orbit line shows a ghost node that follows the mouse with
   "in N min". **Clicking opens a small menu at that point: "Add burn here" / "Warp to here"**
   (user's decision). "Warp to here" is today's click-to-coast
   (`targeting.try_pick_target` → `trajectory_coast_operation`), kept, just behind the menu.
   Clicking a body still targets it as today. Dragging an existing node slides it along the
   orbit.
2. **Shape.** The node has handles you drag directly: prograde/retrograde, normal/anti-normal,
   radial in/out. The orange line follows **live, every frame** while you drag.
3. **Keyboard nudges** (the user asked to keep these): keys that nudge each Δv axis and the
   node time. Today WASD/QE fly the ship and the arrows orbit the camera, so the nudge keys
   must not clash; the exact keys are the builder's call, shown in the F1 overlay.
4. **Where to aim.** With the Moon (or a planet) targeted, show a faded **"Moon when you
   arrive"** at its position at the time of closest approach, plus a status box: green
   "✓ Moon encounter · Pe 4,317 km" or amber "No encounter · closest 120,000 km".
   *(My call, settling a reviewer question: today two calculations disagree, the Earth-only
   closest approach and the N-body `find_encounters`.)*
   - **Encounter** means the exact (N-body) preview path enters the target's sphere of
     influence. That's what turns the box green.
   - **"Pe"** is the closest altitude above the target's **surface** on that path.
   - **"closest"** (amber case) is the closest distance from the target's centre.
   - If the path hits the surface, the box is red: "✗ Impact".
   - Box and arrival marker come from the exact N-body result when one is available for
     the current node. While the node is being dragged, they show the Earth-only estimate
     greyed out as "checking…", and turn green, amber or red only when the exact result for
     *this* node arrives. That way the colour never flips wrongly.
5. **Auto-plan.** With a target selected, a visible **"Plan Moon transfer"** button (today's
   `Intercept`, `I` key) drops a ready-made node you can then fine-tune by hand.
   *(My call:)* today's planner aims at the Moon's **centre**. The `target_intercept` golden
   shows a 229 km separation, an impact course. Change it to aim for a low, safe pass:
   periapsis altitude of about **200 km** above the target's surface. This is core planner
   code with known-answer tests; it doesn't touch the flown physics.
6. **The slider panel goes away.** The Prograde/Normal/Radial/Node-T jog sliders are replaced
   by the handles + keyboard nudges, so there's one way to edit a burn. Existing actions in
   that panel (Next Pe / Next Ap, Clear, Clear target, Execute/Jump & Execute, unlimited Δv)
   must stay reachable somewhere sensible.

### Live line while editing

Measured today (2026-11-04): the post-burn preview is full N-body, throttled to 5 Hz, one job
in flight at a time. One sample takes **~0.2 s** for a small LEO burn and **~8 s** for a
Moon transfer, so while editing a Moon burn the line is effectively frozen.

Decided behaviour:
- **While the node is changing**, draw an instant two-body (Earth-only Kepler) estimate of the
  post-burn orbit every frame. It's drawn slightly fainter, so it's clear it's provisional.
- **Shortly after editing stops**, replace it with the exact N-body path (Moon, Sun, planets)
  as today. If a stale exact result arrives for an older node, it's discarded (today's
  `_preview_future_key` already does this).
- **Make the exact preview much faster** so the settle is quick. Goal: about a second or
  less for a Moon transfer, down from ~8 s. Profile first: the cause isn't known yet.
  *(My call, settling a reviewer question:)*
  - The preview already uses coarser stepping than flight (`PREDICTION_MIN_SUBSTEP_S`, 24 h
    max step). It may get **its own** cheaper stepping and a shorter drawn horizon. For
    example, it can stop a little after a Moon flyby (showing the pass and the exit leg)
    instead of drawing 2 full orbits / 30 days.
  - It must **not** change `propagate_solar_system` or anything the flown simulation calls,
    unless the change is bit-identical. The physics goldens are the check.
  - There are no preview-accuracy tests today, so add one. On a pinned Moon-transfer state,
    the preview's Moon-periapsis altitude must be within **1 %** of a fine-step reference
    propagation, and its encounter time within **10 minutes**.

## Constraints (from the project, see CLAUDE.md)

- Layering: `core/` never imports Panda3D. Any new math (e.g. sampling the two-body estimate,
  the arrival-time target position) lives in `core/` with known-answer tests; drawing lives in
  `render/`.
- Controllers stay stateless; state lives on `OrbitApp`. New UI goes in the appropriate
  controller (`maneuver_editor`, `trajectory_view`, `vessel_view`, `readouts`,
  `targeting`, …) or a new one.
- Pool jobs stay module-level and picklable (the golden harness's pool stand-in enforces it).
- **Existing app goldens whose inputs disappear** (e.g. `node_edit_execute` drives the jog
  slider via `hooks.set_jog`; `driver.press` finds buttons by their label). *(My call:)* a
  scenario script may change **how** it drives the UI: a hook that sets the same Δv the jog
  produced, or a renamed button. Its recorded **physics** fields must still match exactly,
  and presentation fields may be re-recorded with the reason in the commit. Retire a
  scenario only if its behaviour no longer exists, and record a replacement in that case.
  This is a deliberate, one-time exception to "never edit a scenario", because this pass
  replaces the UI the scenarios click.
- TDD for logic. For visuals, verify with headless screenshots
  (`tests/render/app_harness/screens.py`, offscreen window) and record new app-golden
  scenarios for the new interactions (place node by click, drag handle, nudge key,
  encounter status turning green).
- Small PRs, each merged after review ("PR then merge").

## Left to the builder

- Exact colours, line widths, fonts, panel sizes and positions within the direction above.
- How thick lines are implemented (shader ribbons vs CPU-built quads), as long as they're
  thick on macOS and Windows and don't regress frame time.
- Ship-model highlight style; the ghost-node and handle visuals; hit-testing approach for
  clicking the orbit and grabbing handles.
- The keyboard nudge keys and step sizes.
- The debounce delay before the exact preview replaces the estimate.
- Order of work and how it splits into PRs. (Suggested: lines + background + ship first,
  since they fix "can't see anything"; then the live preview; then node placing/handles;
  then aim helpers and the panel cleanup; solar viewer last.)
