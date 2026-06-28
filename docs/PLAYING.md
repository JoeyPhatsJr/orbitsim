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

- **Sandbox** (default) — Earth-centered, one flyable vessel, the Moon, the Earth–Moon Lagrange
  points, and a maneuver-node editor. This is the game.
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
| Left-click a body marker | Select / deselect it as your target (Moon or a Lagrange point) |
| `P` | Toggle a **porkchop** ΔV plot for an intercept transfer |
| `U` | Toggle the **unlimited-ΔV** cheat |

On-screen, the **maneuver-node editor** (bottom-right sliders + buttons) plots a burn:

- The three sliders set prograde / normal / radial ΔV; the magenta preview line shows the resulting
  orbit ("burn now").
- **Node −/+** step the scheduled node time; **Next Pe / Next Ap** jump the node to the next
  periapsis/apoapsis; **Clear** removes it; **Execute Burn** applies it.

When a target is selected you get a live **closest-approach** readout (separation, relative speed,
countdown) and CA markers; for a Lagrange point you get live distance and relative speed.

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
  target / closest-approach lines. Periapsis/apoapsis switch to Moon-relative when you are inside the
  Moon's sphere of influence.
- **Top-right — VESSEL**: throttle, fuel, mass, thrust, thrust-to-weight ratio, and ΔV remaining.
- **Top-center**: the time-warp control and current warp factor.
- **Bottom-center — the navball**: a 3D attitude ball (sky/ground, heading) with your nose fixed at
  center and colored orbital markers (prograde, retrograde, normal, radial, target). Beside it:
  - a **SAS chip** showing the current mode plus heading/pitch, with clickable mode buttons that mirror
    keys `1`–`8` and `T`;
  - a **velocity readout** above the ball — click it to toggle between **orbital** speed and
    **target-relative** speed.

### In the world

- **Pe / Ap markers** sit on your trajectory at periapsis and apoapsis.
- **Target** and **closest-approach** markers + labels appear when you select a target; overlapping
  labels declutter by priority and fade with distance.
- **Lagrange points** L1–L5 are teal markers that rotate with the Moon.
- The **Moon's sphere of influence** is a faint translucent shell (blue from outside); when your ship
  crosses inside it, the scene tints green — you are now in lunar-dominated gravity.
- The bright **trajectory line** is your forward-integrated N-body path; the present is bright and the
  future recedes. The faint grey loop is the Moon's reference orbit. A magenta line previews a planned
  burn.

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

## Tips

- **Burns are most efficient at the apsis opposite the one you want to change** — burn at periapsis to
  raise apoapsis, and vice versa.
- **Time-warp** (`.`) to skip the long coasts, but it auto-drops to 1× during burns and is capped near
  bodies to keep the N-body integration accurate.
- Stuck or out of fuel while experimenting? Toggle **Unlimited ΔV** (`U`) or open settings (`Esc`).
- **Quicksave** (`F5`) before a risky maneuver; **Quickload** (`F9`) to retry.
