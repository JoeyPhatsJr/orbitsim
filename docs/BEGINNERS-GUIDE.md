# The Physics Behind the Simulator — A Beginner's Guide

You don't need a physics degree to enjoy this game, but understanding the basics makes
everything click. This guide explains the real orbital mechanics that power the simulator,
starting from zero.

---

## What even is an orbit?

Imagine throwing a ball. It goes forward, gravity pulls it down, and it lands. Now throw it
harder. It goes further before landing. Throw it *really* hard — say, 28,000 km/h — and
something strange happens: the ball falls toward the Earth, but the Earth curves away beneath
it at the same rate. The ball never hits the ground. It just keeps falling around the planet
forever.

That's an orbit. You're not floating — you're falling. You're just moving sideways fast enough
that you keep missing.

The International Space Station does this at about 400 km altitude, circling the Earth every
90 minutes at 7.7 km/s (that's 27,500 km/h). This simulator puts you in a similar orbit and
lets you fly from there.

## Why can't I just point at the Moon and go?

This is the single most counterintuitive thing about orbital mechanics, and it's the reason
this game is interesting.

In a car, you point where you want to go, hit the gas, and you go there. In orbit, **pointing
at your destination and firing your engine is almost always wrong.** Here's why:

When you fire your engine forward (along your direction of motion — called "prograde"), you
don't go forward. You raise the *opposite* side of your orbit. You make your orbit bigger on
the far side. If you fire backward ("retrograde"), you lower the opposite side.

So to reach the Moon, you don't point at it. You burn prograde at the right time to stretch
your orbit outward until it reaches the Moon's distance. The game's trajectory line shows you
exactly where you'll end up.

## The shape of orbits

Every orbit is an ellipse (an oval) with the planet at one focus — not the center. The closest
point to the planet is called **periapsis** (or "Pe" in the game), and the farthest point is
**apoapsis** (or "Ap").

A perfectly circular orbit has the same altitude everywhere: periapsis equals apoapsis. Most
real orbits are slightly oval, and the game starts you in one of these (a "slightly eccentric"
orbit) so you can see the shape.

The HUD in the top-left constantly shows your current periapsis and apoapsis altitudes. When
you burn prograde, watch the apoapsis number climb. When you burn retrograde, watch it fall.

### Orbital speed isn't constant

Here's another mind-bender: you move **faster** when you're closer to the planet, and
**slower** when you're farther away. This isn't a design choice — it's a law of physics
(conservation of angular momentum, if you want the jargon).

Think of a figure skater pulling their arms in to spin faster. As your orbit brings you closer
to Earth, you speed up; as it carries you away, you slow down. The game's navball and speed
readout show this happening in real time.

## What is delta-V?

Delta-V (written dV or "delta-v") literally means "change in velocity." It's the currency of
spaceflight. Every maneuver costs a certain amount of dV, and your ship has a finite budget
based on how much fuel you're carrying.

The game shows your remaining dV in the top-right VESSEL panel. The default ship has about
5,600 m/s of dV — enough to get to Mars one-way, or to the Moon and back with some to spare.

Here's a rough scale of what things cost:

| Maneuver | Approximate dV |
|---|---|
| Raise a low orbit by 100 km | ~55 m/s |
| Go from low Earth orbit to the Moon | ~3,200 m/s |
| Escape Earth entirely | ~3,200 m/s |
| Transfer to Mars (from Earth orbit) | ~4,300 m/s |
| Transfer to Jupiter | ~6,300 m/s |

When you set up a maneuver node in the game (using the sliders in the bottom-right), the HUD
shows exactly how much dV it costs. The magenta preview line shows what your orbit will look
like after the burn.

### The rocket equation (why fuel is precious)

There's a cruel equation in rocketry called the Tsiolkovsky rocket equation. It says that to
get more dV, you need more fuel. But more fuel means more mass. And more mass means you need
*even more* fuel to push it. It's an exponential — a losing game.

The ship in this game uses realistic rocket physics. The engine has a specific "exhaust
velocity" (3,500 m/s), and the available dV depends logarithmically on the fuel-to-dry-mass
ratio. You can see this in real time: as you burn fuel, the dV remaining drops, but not
linearly — the last drops of fuel give you more dV per kilogram than the first.

If you run out of fuel, that's it. You drift on whatever orbit you're on. (Or toggle Unlimited
dV with `U` and keep experimenting.)

## How burns work

When you open the throttle (`Shift` to increase, `Z` for full), the game integrates the real
rocket equation at every simulation step. Your mass drops as fuel burns, thrust produces
acceleration (F = ma), and your orbit changes continuously.

The game forces time-warp to 1x during burns because the physics integration needs small time
steps to stay accurate. You can't fast-forward through a burn — you watch it happen, which is
actually one of the most satisfying parts.

### Burn directions

The game gives you six directions to burn, organized around your orbit:

- **Prograde** (key `1`): along your velocity. Raises the opposite side of your orbit. The
  most common burn — this is how you go higher.
- **Retrograde** (key `2`): against your velocity. Lowers the opposite side. This is how you
  slow down, shrink your orbit, or capture at a planet.
- **Normal / Anti-normal** (keys `3`/`4`): perpendicular to your orbital plane. Tilts your
  orbit. Expensive and rarely needed (real missions avoid plane changes because they cost a
  lot of dV).
- **Radial in / out** (keys `5`/`6`): toward or away from the planet. Rotates your orbit in
  its plane (moves where periapsis is). Useful for fine-tuning an intercept.

The SAS system (press `T` to enable, then a number key) automatically holds your ship pointed
in any of these directions. This is essential — manually steering a burn would be like trying
to drive a car while holding a compass.

## Transfer orbits — how you get places

Going from one orbit to another is called a "transfer." The simplest transfer is called a
**Hohmann transfer** — two burns that move you from one circular orbit to another.

Here's how it works, step by step:

1. **You're in a low circular orbit.** Periapsis and apoapsis are about equal.
2. **Burn prograde at periapsis.** This raises your apoapsis up to the altitude of your target
   orbit. You're now on an elliptical "transfer orbit."
3. **Coast to apoapsis.** You're at the high point of the ellipse, moving slowly (remember:
   you slow down when you're far from the planet).
4. **Burn prograde again at apoapsis.** This raises your periapsis to match your new altitude.
   Now you're in a higher circular orbit.

This is the most fuel-efficient way to change orbits, and the game's intercept planner (press
`I`) uses a more sophisticated version of this math (Lambert's problem) to compute transfers
to any body in the solar system.

### Porkchop plots — finding the right time

For interplanetary transfers, *when* you depart matters enormously. Earth and Mars (for
example) are on different orbits moving at different speeds. There are windows — roughly every
26 months for Mars — when the geometry is right for a cheap transfer. Miss the window and the
dV cost skyrockets.

Press `P` with a planet targeted to see a **porkchop plot**: a color map of dV cost over all
combinations of departure time and flight time. The bright spot is the optimal window. The
`I` key computes an intercept burn at that optimum automatically.

The name "porkchop" comes from the shape of the low-cost region on the plot, which NASA
engineers thought looked like a porkchop. Seriously.

## N-body gravity — why this isn't "just Kerbal"

Most space games (including Kerbal Space Program) use a simplification called "patched conics":
your ship is only affected by one body's gravity at a time. When you cross an invisible
boundary (the sphere of influence), the game switches which body is pulling you.

This simulator does it differently. **Every body pulls on you all the time.** The Sun, the
Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, and Neptune all exert their real
gravitational force on your ship simultaneously. This is called N-body gravity, and it's what
makes the physics genuinely realistic.

Why does this matter? Because it enables things that patched conics can't:

- **Lagrange points**: there are five special positions in the Earth-Moon system (L1 through
  L5) where the combined gravity of Earth and the Moon, plus the centrifugal effect of the
  rotating frame, balance out. A spacecraft placed there with zero relative velocity will
  (approximately) stay. The game computes these to machine precision and shows them as
  clickable teal markers. You can actually fly to L1 and park there. In a patched-conic game,
  Lagrange points don't exist.

- **Three-body effects**: near the Moon, the Sun's gravity slightly warps your trajectory in
  ways that two-body math can't predict. The game's trajectory line is forward-integrated under
  the full N-body model, so what you see is what you get.

- **Gravity assists**: when you fly past a planet, its gravity bends your trajectory. The game
  shows this with a live flyby readout: your hyperbolic excess velocity, the deflection angle,
  and the free dV you gain from the encounter. Jupiter can bend your path by tens of degrees
  and give you thousands of m/s for free — which is how Voyager got to Neptune.

The translucent spheres you see around the Moon and planets are **sphere-of-influence shells**,
visual aids showing roughly where each body's gravity dominates. But they're just guides —
gravity doesn't switch on and off at those boundaries. It's continuous, everywhere, all the
time.

## The navball — your attitude compass

The sphere at the bottom of the screen is the **navball**, borrowed from real spacecraft
cockpits (and KSP). It shows your ship's orientation relative to the orbit:

- The **blue half** is "sky" (away from the planet), the **brown half** is "ground."
- The **center dot** is where your nose points.
- **Colored markers** show key directions: yellow circle for prograde, yellow X for
  retrograde, magenta triangles for normal/anti-normal, teal squares for radial in/out, and
  pink markers for target/anti-target.

When you enable SAS and pick a hold mode (like prograde), the navball rotates until the
prograde marker sits at the center — meaning your nose is aligned with your velocity. Then
you burn. The SAS chip beside the navball shows your current heading and pitch.

## Time warp — because space is mostly waiting

Real orbital mechanics involves a lot of coasting. A transfer to Mars takes about 7 months.
Even going to the Moon takes a few days.

Press `.` to speed up time (and `,` to slow it down). The game supports extreme warp factors —
at 1,000,000x, a Mars transfer takes about 22 seconds of real time.

But there's a catch: the N-body physics engine needs small time steps to stay accurate. So near
planets, the game automatically caps your warp speed to keep the integration honest. And during
burns, warp is forced to 1x — you can't fast-forward through a rocket firing.

## Saving your progress

Press `F5` to quicksave and `F9` to quickload. The game saves everything: your position,
velocity, fuel, orientation, and any planned maneuver nodes, all as a JSON file. Save before a
risky burn, reload if it goes wrong.

## What the game doesn't simulate

This simulator focuses on **orbital mechanics** — the pure geometry and physics of moving
through space under gravity. It deliberately leaves out:

- **Atmosphere and aerodynamics.** There's no launch from the surface, no drag, no
  aerobraking. You start in orbit.
- **Rocket building.** Your ship is a fixed design with set mass, fuel, and engine stats. The
  focus is on flying, not building.
- **Relativity.** Everything is Newtonian. At the speeds and distances in this game (well
  under the speed of light), relativistic effects are negligible.
- **Life support, communications, heat.** It's about the orbits, not the engineering.

These omissions are deliberate: they keep the game focused on the one thing it does with real
accuracy — the ballet of gravity and velocity that is orbital mechanics.

## Further reading

If this has piqued your interest, here are the rabbit holes:

- **The controls and HUD in detail**: [`PLAYING.md`](PLAYING.md)
- **Showing this to someone non-technical?**: [`PLAIN-ENGLISH.md`](PLAIN-ENGLISH.md) explains the
  whole game with every term defined inline
- **"Orbital Mechanics for Engineering Students"** by Howard Curtis — the textbook this game's
  physics is validated against
- **NASA's Trajectory Browser** (trajbrowser.arc.nasa.gov) — explore real mission trajectories
- **Gravity assists explained**: search "Voyager gravity assist" for the most spectacular
  example in history — one spacecraft visited Jupiter, Saturn, Uranus, and Neptune on a single
  tank of (very little) fuel, using gravity assists at each planet
