# What Is This Game? (Plain English Edition)

This is a space game where you fly a spaceship around Earth, the Moon, and the planets. It's
like Kerbal Space Program, but the math behind the scenes is the real deal — the same equations
NASA uses to plan actual missions.

You start in orbit around Earth (about 500 km up, roughly where the International Space Station
flies) and you can go anywhere: the Moon, Mars, Jupiter, wherever you have enough fuel to reach.
The whole solar system is there, pulling on your ship with real gravity, all at the same time.

There's no building rockets, no launching from the ground, and no air resistance. You start
already in space, and the game is about the flying: planning where to go, deciding when to fire
your engine, and figuring out how to get from one orbit to another without wasting fuel.

---

## The Jargon Dictionary

Here's every technical term you'll run into, explained once and for all.

**Orbit** — The path your ship follows as it falls around a planet. You're not floating still;
you're moving sideways so fast that you keep missing the ground as you fall. That looping path
is your orbit.

**Periapsis (Pe)** — The lowest point of your orbit, where you're closest to whatever you're
going around. The game shortens this to "Pe" on screen.

**Apoapsis (Ap)** — The highest point of your orbit, where you're farthest away. Shown as "Ap"
on screen.

**Altitude** — How high you are above the surface of whatever planet or moon you're near,
measured in kilometers.

**Eccentricity** — How oval-shaped your orbit is. A circle has eccentricity 0. The more
stretched out the oval, the higher the number. Above 1.0 means you're on a path that escapes
and never comes back (a hyperbola — see below).

**Inclination** — How tilted your orbit is compared to the planet's equator, measured in
degrees. 0° means you're going around the equator. 90° means you're going over the poles.

**Period** — How long it takes to go around once. A low Earth orbit takes about 90 minutes.

**Prograde** — The direction you're currently moving. Think of it as "forward along your
orbit." If you fire your engine this way, you speed up and your orbit gets bigger on the
far side.

**Retrograde** — The opposite of prograde — "backward." Burning this way slows you down and
shrinks the far side of your orbit.

**Normal / Anti-normal** — "Up" and "down" relative to the flat plane of your orbit.
Burning this way tilts the plane of your orbit. Expensive and rarely worth doing.

**Radial in / Radial out** — Toward or away from the planet you're orbiting. Burning this way
rotates where the high and low points of your orbit are, without changing the shape much.

**Delta-V (dV or ΔV)** — Literally "change in velocity." It's the currency of space travel.
Every maneuver costs some amount of delta-V (measured in meters per second), and your ship has
a limited supply based on how much fuel you're carrying. Think of it like a gas tank, except
instead of measuring gallons, you measure how much you can change your speed.

**Thrust** — The push force from your engine, measured in Newtons. More thrust means faster
acceleration.

**TWR (Thrust-to-Weight Ratio)** — How strong your engine is compared to the gravity pulling
you down. Above 1.0 means your engine can overpower gravity (important for launches, but you
start in orbit so it's less critical here). The game shows this number in the vessel panel.

**Exhaust velocity** — A measure of how efficient your engine is. Higher exhaust velocity means
you get more delta-V out of the same amount of fuel.

**SAS (Stability Assist System)** — Autopilot for pointing your ship. Turn it on with `T` and
pick a direction (like prograde). The ship automatically rotates to face that direction and
holds steady. Without SAS, you'd have to manually aim while also trying to time your burn.

**Throttle** — How hard your engine is pushing, from 0% (off) to 100% (full). Think of it like
a gas pedal.

**Maneuver node** — A planned future burn. You set one up using the sliders at the bottom-right
of the screen: choose a direction and amount, pick when to do it, and the game shows you a
preview of what your orbit will look like afterward. When the time comes, you execute the burn.

**Navball** — The sphere at the bottom of the screen. It's a 3D compass that shows which way
your ship is pointed relative to your orbit. Blue means you're facing away from the planet,
brown means you're facing toward it. Colored markers show important directions like prograde,
retrograde, and your target.

**Trajectory line** — The bright curved line in 3D space showing where your ship is going.
It's calculated by running the physics forward in time — what you see is what you'll actually
fly through. A magenta (pink-purple) version appears when you're planning a burn, showing
what your orbit would look like after the burn.

**Hohmann transfer** — The most fuel-efficient way to move between two circular orbits. You
burn prograde at the low point to stretch your orbit out to the target altitude, coast up to
the high point, then burn prograde again to circularize. Two burns, one trip.

**Lambert transfer** — A more general way to calculate "how do I get from point A to point B
in a specific amount of time?" The game uses this when you press `I` to plan an intercept —
it figures out exactly what burn to make and when.

**Porkchop plot** — A color-coded chart that shows how much fuel (delta-V) different departure
times and travel times would cost for a trip to another planet. The bright spots are the cheap
options. Named "porkchop" because NASA engineers thought the shape of the low-cost region looked
like a porkchop. Press `P` to see one.

**Synodic period** — How often the alignment between two planets repeats. For Earth and Mars,
it's about 26 months — that's how often you get a fuel-efficient launch window.

**N-body gravity** — The way this game calculates gravity. Instead of pretending only one
planet pulls on you at a time (which is what most space games do), this game calculates the
gravitational pull from Earth, the Moon, the Sun, and all seven other planets simultaneously.
Every object pulls on you, all the time.

**Sphere of influence (SOI)** — An approximate boundary around a planet or moon showing where
its gravity dominates. In this game, you'll see faint translucent bubbles around the Moon and
each planet. These are visual guides only — gravity doesn't actually switch on or off at the
boundary. It's continuous everywhere. When you cross into a planet's SOI, the game starts
showing your orbital information relative to that planet instead of Earth or the Sun.

**Lagrange points (L1–L5)** — Five special spots in the Earth-Moon system where the combined
gravity of Earth and the Moon balance out with the orbital motion, so a spacecraft placed there
(with the right velocity) tends to stay put. The game shows these as teal-colored markers. L1
is between Earth and Moon, L2 is behind the Moon, and L3–L5 are at other balanced positions.
The real James Webb Space Telescope sits at the Sun-Earth L2 point.

**Gravity assist (flyby)** — A trick where you fly close to a planet and let its gravity bend
your path, changing your speed for free — no fuel required. The planet's gravity slings you
in a new direction. This is how NASA's Voyager probes visited four planets on practically no
fuel. When you fly near a planet in the game, the screen shows how much your path is bending
and how much free speed change you're getting.

**Hyperbolic trajectory** — A path that escapes a planet's gravity entirely. Normal orbits are
closed loops (ellipses). When you're going fast enough to escape, your path becomes an open
curve — you fly past and never come back. This happens when you escape Earth heading for
another planet, or during a flyby.

**V-infinity (v∞)** — Your speed relative to a planet "at infinity" — practically, it's how
fast you're approaching or leaving a planet's gravity well. During a flyby, v-infinity tells
you how energetic the encounter is.

**Floating origin** — A behind-the-scenes trick the game uses to stay precise. The solar system
is enormous (billions of kilometers across) but a docking maneuver needs millimeter accuracy.
Computer graphics can only handle about 7 digits of precision. The game solves this by
constantly moving its reference point to wherever you are, so nearby things are always precise.
You'll never notice this — it just works.

**Ephemeris** — A table of where all the planets actually are at any given time. The game uses
NASA's JPL DE440 ephemeris, the same data NASA uses for real missions. This means when you look
at Mars in the game, it's where Mars actually is (or was, or will be) in real life.

**J2000 / ICRF** — The coordinate system the game uses internally. It's the standard reference
frame astronomers use: the origin is at the center of whatever body you're orbiting, and the
axes are fixed relative to distant stars as they were oriented on January 1, 2000. You don't
need to care about this — it just means the physics math is done in the same coordinates as
real space missions.

**Warp (time warp)** — Fast-forwarding time. Press `.` to speed up, `,` to slow down. At
1,000,000× speed, a 7-month trip to Mars takes about 22 seconds. The game automatically slows
warp down near planets (to keep the physics accurate) and forces it to 1× while your engine is
firing.

---

## What You See On Screen

### The 3D View

The main view is a 3D space scene. You can orbit the camera around your ship with the right
mouse button or arrow keys, and zoom in/out with the scroll wheel.

Zoomed out, you see the map: Earth, the Moon, planet markers, orbit lines, your trajectory, and
Lagrange point markers. Zoomed all the way in, you see your actual ship model in third person,
with an engine plume when you're thrusting.

Press `M` to snap between map zoom and ship zoom.

### The HUD (Heads-Up Display)

The HUD is all the text and numbers overlaid on screen:

- **Top-left panels** — Three grouped info boxes:
  - **TIME**: The current simulation time and warp speed.
  - **ORBIT**: Your altitude, speed, periapsis, apoapsis, inclination, and orbital period.
    These numbers update relative to whichever body's gravity dominates — they switch to
    Moon-relative when you're near the Moon, Mars-relative when near Mars, and so on.
  - **MANEUVER**: Shows up when you have a planned burn — the delta-V cost and countdown to
    the burn.

- **Top-right panel (VESSEL)**: Your ship's stats — throttle percentage, fuel remaining, total
  mass, engine thrust, thrust-to-weight ratio, and how much delta-V you have left.

- **Top-center**: Time warp controls and current warp factor.

- **Bottom-center**: The navball (your orientation compass) with:
  - **SAS chip**: Shows which autopilot mode you're in, with clickable buttons.
  - **Velocity readout**: Your speed. Click it to switch between orbital speed and
    speed-relative-to-your-target.

- **Bottom-right**: The maneuver node editor — sliders to plan burns.

### In the 3D World

- **Trajectory line** — Your predicted flight path, bright near the present and fading into
  the future.
- **Pe and Ap markers** — Labels sitting on your orbit at the lowest and highest points.
- **Target and closest-approach markers** — When you've clicked on a target, markers show where
  you'll pass closest to it.
- **Lagrange point markers** — Teal dots at L1 through L5, rotating with the Moon.
- **SOI shells** — Faint transparent spheres around the Moon and each planet showing where
  their gravity dominates.
- **Planets** — The Sun, all eight planets (Mercury through Neptune), and the Moon, drawn as
  textured spheres at their real sizes and positions. Saturn has rings. Earth has a day/night
  line. There's a starfield in the background.

---

## What You Can Do

### Fly Your Ship

Point your ship in a direction using SAS (press `T`, then a number key to pick a direction like
prograde or retrograde), open the throttle (`Shift` to increase, `Z` for full blast, `X` to
cut), and watch your orbit change in real time. Your fuel burns down realistically — the game
uses the actual rocket equation.

### Plan Burns with Maneuver Nodes

Use the sliders at the bottom-right to set up a future burn: pick a direction (prograde, normal,
or radial), set the amount, and adjust the timing. A magenta preview line shows what your orbit
will look like after the burn. When you're happy with it, hit "Execute Burn" and the ship does
it automatically.

Shortcut buttons let you snap the burn to your next periapsis or apoapsis (the most efficient
places to burn).

### Target Things

Click any body (the Moon, a Lagrange point, or a planet) to target it. Once targeted:
- The HUD shows distance, relative speed, and closest-approach information.
- The navball adds target/anti-target markers so you can point toward or away from it.
- You can press `I` to have the game calculate an intercept burn automatically.
- You can press `P` to see a porkchop plot showing the cheapest times to go there.

### Go to the Moon

Burn prograde (forward) at the right time to stretch your orbit out to the Moon's distance.
The Moon's SOI shell turns the scene green when you cross in. Your orbital readouts switch to
Moon-relative. Burn retrograde (backward) to slow down and capture into lunar orbit. Or aim for
a Lagrange point and park there.

### Go Interplanetary

Click a planet, press `I` for an automatic intercept plan, and execute the burn. Warp through
the months-long coast. When you arrive at the planet, burn retrograde to capture. The default
ship has about 5,600 m/s of delta-V — enough for a one-way trip to Mars.

### Use Gravity Assists

Fly close to a planet and its gravity bends your path for free. The game shows you the
deflection angle and how much free velocity change you're getting. This is how real missions
reach the outer planets without carrying impossible amounts of fuel.

### Save and Load

Press `F5` to save, `F9` to load. Everything is preserved: position, speed, fuel, orientation,
and any planned maneuvers.

### Explore the Solar System

Launch with `--solar` for a Sun-centered viewer with all the planets at their real positions.
No ship — just a camera to explore the solar system at any time warp speed.

---

## Two Modes

1. **Sandbox** (the default) — You have a ship in low Earth orbit. The full solar system is
   there. You can fly anywhere, plan maneuvers, target planets, and use all the tools. This is
   the main game.

2. **Solar viewer** (`--solar` flag) — No ship. The camera sits at the center of the solar system
   and you watch the planets move along their real orbits. Good for seeing how the planets are
   arranged and how they move over time.

---

## What the Game Does NOT Do

- **No building rockets.** Your ship is pre-built. The game is about flying, not engineering.
- **No launching from the ground.** You start already in orbit. No atmosphere, no air
  resistance, no runways.
- **No life support, temperature, or communications.** It's purely about the orbits.
- **No multiplayer.** It's a single-player sandbox.

These aren't missing features — they're deliberate choices to keep the focus on orbital
mechanics, the thing the game does with real accuracy.

---

## Quick Reference: Controls

| Key | What it does |
|---|---|
| Right-drag | Rotate the camera |
| Scroll wheel | Zoom in/out |
| Arrow keys | Rotate the camera |
| `M` | Switch between map view and close-up ship view |
| `W/S` | Pitch the ship up/down |
| `A/D` | Turn the ship left/right |
| `Q/E` | Roll the ship |
| `Shift/Ctrl` | Increase/decrease throttle |
| `Z` | Full throttle |
| `X` | Cut throttle (engine off) |
| `T` | Turn SAS autopilot on/off |
| `1`–`8` | SAS direction (1=prograde, 2=retrograde, etc.) |
| `U` | Toggle unlimited fuel (cheat mode) |
| `I` | Calculate an intercept to your target |
| `P` | Show a porkchop plot for your target |
| Click a body | Target/untarget it |
| `,` / `.` | Slow down / speed up time |
| `F5` / `F9` | Quicksave / Quickload |
| `Esc` | Settings menu |
| `F1` | Show/hide this control list |
