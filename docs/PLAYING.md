# How to Play

A guide to flying in the Orbital Mechanics Simulator: launching, the controls, reading the HUD, and a
first-flight walkthrough. Press **F1** in-app at any time for a quick keybind overlay.

## Launch

```powershell
.venv\Scripts\python -m orbitsim          # the sandbox — a flyable ship in low Earth orbit
.venv\Scripts\python -m orbitsim --solar  # the solar-system viewer — planets from the real ephemeris
```

On the title screen, set your **fuel load** (which sets your ΔV budget via the rocket equation),
optionally tick **Unlimited ΔV**, and press **Play**.

The two modes:

- **Sandbox** (default) — one flyable vessel in low Earth orbit with the full solar system:
  the Sun, Mercury, Venus, Mars, Jupiter, Saturn (with rings), Uranus, Neptune, the Moon, and the
  Earth–Moon Lagrange points, all exerting real N-body gravity. Maneuver-node editor, targeting,
  interplanetary transfers, and gravity-assist encounter display. This is the game.
- **Solar viewer** (`--solar`) — Sun-centered, the planets at their real positions. No vessel; a
  camera/time sandbox for looking at the solar system.

## Controls

### Camera (both modes)

| Input | Action |
|---|---|
| Right-drag mouse | Orbit the camera around the focus |
| Arrow keys | Orbit the camera |
| Mouse wheel | Zoom in / out (smoothly eased) |
| `M` | Snap between **map view** and the 3rd-person **ship view** |

Zooming all the way in cross-fades from the map marker to a lit 3rd-person ship model; zoom out
returns to the map. `M` toggles between your remembered map and ship framings.

### Flight (sandbox)

| Input | Action |
|---|---|
| `W` / `S` | Pitch down / up |
| `A` / `D` | Yaw left / right |
| `Q` / `E` | Roll |
| `Shift` / `Ctrl` | Throttle up / down |
| `Z` | Full throttle |
| `X` | Cut throttle |
| `T` | SAS on/off (stability hold) |
| `1`–`8` | SAS hold mode (see below) |

Manual rotation (`WASDQE`) drops SAS into manual control. While any engine is thrusting, time-warp is
forced to 1× (you can't fast-forward through a burn).

The **SAS hold modes** (keys `1`–`8`, or the clickable buttons by the navball) point the nose at:

| Key | Mode | Points at |
|---|---|---|
| `1` | Prograde | direction of motion |
| `2` | Retrograde | opposite motion (to slow down) |
| `3` | Normal | orbital-plane normal (`+`) |
| `4` | Antinormal | orbital-plane normal (`−`) |
| `5` | Radial-in | toward the body |
| `6` | Radial-out | away from the body |
| `7` | Target | toward the selected target |
| `8` | Antitarget | away from the selected target |

### Targeting & planning (sandbox)

| Input | Action |
|---|---|
| Left-click a body marker | Select / deselect it as your target (Moon, Lagrange point, or planet) |
| `I` | Plan an **intercept** burn to the current target (Lambert solver) |
| `P` | Toggle a **porkchop** ΔV plot for the current target (or a demo) |
| `U` | Toggle the **unlimited-ΔV** cheat |

On-screen, the **maneuver-node editor** (bottom-right sliders + buttons) plots a burn:

- The three sliders set prograde / normal / radial ΔV; the **Node T** slider adjusts the scheduled
  node time (variable rate: gentle near center, fast at full deflection). The magenta preview line
  shows the resulting orbit ("burn now"). Total dV is shown above the sliders.
- **Next Pe / Next Ap** jump the node to the next periapsis/apoapsis; **Clear** removes it;
  **Execute Burn** applies it. The node marker shows a dV + countdown label in world space.

When a target is selected you get a live **closest-approach** readout (separation, relative speed,
countdown) and CA markers; for a Lagrange point or planet you get live distance and relative speed
(in km or AU depending on range).

When inside a planet's SOI, a **flyby encounter** readout appears showing the hyperbolic excess
velocity (v∞), deflection angle, periapsis distance, and equivalent free delta-V from the gravity
assist.

### Time, save, system

| Input | Action |
|---|---|
| `,` / `.` | Time-warp down / up (also the on-screen `<<` / `>>` buttons) |
| `F5` / `F9` | Quicksave / Quickload (sandbox JSON) |
| `Esc` | Settings panel (units km/mi, unlimited-ΔV toggle) |
| `F1` | Toggle the keybind help overlay |

## Reading the HUD

- **Top-left — TIME / ORBIT / MANEUVER** (grouped panel): sim time; altitude, speed, periapsis,
  apoapsis, inclination, period; and, when a node or target is active, the ΔV, node countdown, and
  target / closest-approach lines. All orbital readouts are relative to the **dominant body** — they
  switch to Moon-relative inside the Moon's SOI, Mars-relative near Mars, and Sun-relative in
  heliocentric space. An "Orbiting: ..." label appears when you leave Earth's SOI.
- **Top-right — VESSEL**: throttle, fuel, mass, thrust, thrust-to-weight ratio, and ΔV remaining.
- **Top-center**: the time-warp control and current warp factor.
- **Bottom-center — the navball**: a 3D attitude ball (sky/ground, heading) with your nose fixed at
  center and colored orbital markers (prograde, retrograde, normal, radial, target). Beside it:
  - a **SAS chip** showing the current mode plus heading/pitch, with clickable mode buttons that mirror
    keys `1`–`8` and `T`;
  - a **velocity readout** above the ball — click it to toggle between **orbital** speed and
    **target-relative** speed.

### In the world

- **Pe / Ap markers** sit on your trajectory at periapsis and apoapsis (relative to the dominant body).
- **Target** and **closest-approach** markers + labels appear when you select a target; overlapping
  labels declutter by priority and fade with distance.
- **Lagrange points** L1–L5 are teal markers that rotate with the Moon.
- **Sphere-of-influence shells** — faint translucent spheres around the Moon and each planet mark
  gravitational dominance boundaries. The Moon's shell turns the scene green when you cross inside;
  planet SOI shells tint when approached. These are visual aids only — gravity is continuous N-body,
  not patched conics.
- **True-scale planets** — the Sun, all seven planets (Mercury through Neptune), and the Moon are
  rendered as textured spheres at their real sizes and positions, with heliocentric orbit reference
  lines. Saturn has a textured ring system.
- The bright **trajectory line** is your forward-integrated N-body path; the present is bright and the
  future recedes. In heliocentric space the line extends to show up to 400 days of the transfer arc.
  The faint grey loop is the Moon's reference orbit. A magenta line previews a planned burn.

## First flight: raise your orbit

1. Launch the sandbox and press **Play** (keep the default fuel).
2. Press **`T`** for SAS, then **`1`** to hold **prograde**. The ship aligns with its motion.
3. Drag the **prograde** maneuver slider (bottom-right) to a small positive ΔV. A magenta preview
   orbit appears — note your **apoapsis** rising in the top-left ORBIT panel.
4. Press **Next Pe** so the node sits at your next periapsis (the efficient place to raise the
   opposite side), then **Execute Burn** when you're ready — or just hold prograde and throttle up
   with **`Shift`** to burn manually.
5. Watch the trajectory line and ORBIT readout update. Zoom in (or press **`M`**) to watch your ship
   fire its plume in 3rd-person.

## Reach for the Moon

1. Left-click the **Moon** marker to target it. A closest-approach readout and markers appear.
2. Press **`P`** for a porkchop plot to find a low-ΔV intercept, or plot a prograde burn near
   periapsis to raise apoapsis out toward the Moon.
3. As you approach, the Moon's **SOI shell** turns the scene green when you cross in — your
   periapsis/apoapsis readouts switch to Moon-relative. From here you can capture, or aim for a
   **Lagrange point** (click L1–L5) and null your relative velocity to park there.

## Go interplanetary

1. Click a planet marker (e.g. **Mars**) to target it. The HUD shows a live distance readout in AU.
2. Press **`I`** to plan an **intercept** — the Lambert solver computes a transfer burn with an
   optimal departure time and time-of-flight grid spanning a full synodic period.
3. A maneuver node appears at the optimal departure time. Press **Execute Burn** when the countdown
   reaches zero (or warp to the node — it auto-warps down). The burn escapes Earth and injects you
   onto a transfer orbit.
4. Once in heliocentric space, the HUD switches to "Orbiting: Sun" and shows your heliocentric
   orbital elements. The trajectory line extends to show your full transfer arc. Warp up (`.`)
   through the months-long coast — at 1,000,000× a Mars transfer takes about 22 seconds.
5. As you approach Mars, the planet's SOI sphere appears and tints when you cross in. The HUD
   switches to "Orbiting: Mars" with Mars-relative altitude and periapsis.
6. Burn **retrograde** (`2` for SAS retrograde hold, then throttle up) to capture into Mars orbit.

**Tip:** The default vessel has ~5600 m/s of ΔV, enough for a one-way Mars mission. Toggle
**Unlimited ΔV** (`U`) for round trips or more ambitious maneuvers.

## Tips

- **Burns are most efficient at the apsis opposite the one you want to change** — burn at periapsis to
  raise apoapsis, and vice versa.
- **Time-warp** (`.`) to skip the long coasts, but it auto-drops to 1× during burns and is capped near
  bodies to keep the N-body integration accurate.
- **Interplanetary transfers** are most efficient near the Hohmann window — the `I` intercept planner
  searches a full synodic period to find it automatically.
- Stuck or out of fuel while experimenting? Toggle **Unlimited ΔV** (`U`) or open settings (`Esc`).
- **Quicksave** (`F5`) before a risky maneuver; **Quickload** (`F9`) to retry.
- Press **`P`** with a planet targeted to see a **porkchop plot** — the color map shows ΔV cost across
  departure time vs. flight time, with the optimal window marked.
