# A Virtual Tour of the Orbital Mechanics Simulator

You can't run the game right now, but you want to know what it's like to play. This is a
scene-by-scene walkthrough of what you'd see and do, from launch to Mars and beyond.

---

## Opening the game

You launch the simulator and a title screen appears against a starfield. There's a fuel slider
(which determines how far you can go — the default gives you about 5,600 m/s of delta-V,
enough for a one-way Mars trip), an "Unlimited dV" checkbox for sandbox experimenting, and a
**Play** button.

You press Play.

## First moments: low Earth orbit

The screen opens to a top-down map view. At the center, a textured Earth rotates slowly, with
a visible day/night terminator — the sunlit side bright blues and greens, the night side
showing city lights. A faint blue atmospheric glow rings the planet.

Your ship is a small bright marker circling Earth at about 500 km altitude, moving at
7.6 km/s. A bright curved line trails ahead of it — your **trajectory line**, showing where
you'll be for the next orbit. It loops around Earth in a slightly oval shape (the orbit isn't
perfectly circular; the game starts you with a bit of eccentricity so the orbit has a visible
shape).

Behind everything, a dense starfield fills the sky.

### What the HUD shows

The screen is framed by information panels:

**Top-left** — three grouped sections:

- **TIME**: the simulation clock, ticking in seconds past J2000 (a standard astronomical epoch).
- **ORBIT**: your altitude (502 km), speed (7,642 m/s), periapsis (465 km), apoapsis (732 km),
  inclination (11.2 degrees), and orbital period (95.3 minutes). These numbers update in real
  time.
- **MANEUVER**: empty for now — this fills in when you plan a burn.

**Top-right** — the VESSEL panel: throttle (0%), fuel (100%, 8,000 kg), total mass (10,000 kg),
engine thrust (50.0 kN), thrust-to-weight ratio, and remaining delta-V (5,633 m/s).

**Top-center** — time warp controls. Two buttons (`<<` and `>>`) and a readout showing your
current warp factor. You start at 100x, so the ship visibly moves along its orbit.

**Bottom-center** — the **navball**: a 3D sphere split into blue (sky) and brown (ground)
hemispheres, slowly rotating as your ship orbits. Colored markers dot its surface: a yellow
circle (prograde — your direction of motion), a yellow X (retrograde), magenta triangles
(normal/anti-normal), and teal squares (radial). A small **SAS chip** beside it shows your
current stability mode and heading. Above the ball, a velocity readout shows your orbital speed.

**Bottom-right** — the **maneuver editor**: three horizontal sliders labeled Prograde, Normal,
and Radial, each sitting at zero. Below them, a fourth slider labeled "Node T" for scheduling
when a burn should happen. Below that, buttons: Execute Burn, Next Pe, Next Ap, Clear,
Clear Tgt, Intercept.

### Zooming in: the ship

You scroll the mouse wheel to zoom in. As the camera closes in, the map marker cross-fades
into a 3D ship model — a small cylinder with a nose cone and three tail fins, lit by the Sun.
At this zoom level, the Earth fills the background, enormous and detailed. You can orbit the
camera around the ship by right-dragging.

Press `M` and the camera snaps back out to the map view. Press `M` again and it smoothly
returns to your remembered ship-view framing.

## Your first burn: raising the orbit

You press `T` to enable SAS (stability augmentation), then `1` to hold prograde. The ship
rotates smoothly until its nose points along the velocity vector — the yellow prograde marker
on the navball slides to the center.

Now you drag the Prograde slider in the maneuver editor to the right. A **magenta line**
appears, branching off your white trajectory — this is the predicted orbit *after* the burn.
As you push the slider further, the magenta line stretches outward. The apoapsis number in the
ORBIT panel climbs: 800 km... 1,200 km... 5,000 km. A small magenta label appears above the
sliders: "Total dV: 247 m/s."

You press **Next Pe** and a cyan node marker appears on your orbit at the next periapsis — the
most efficient point to burn. A world-space label floats next to it: "dV 247 m/s T-12:34"
counting down. The MANEUVER section in the HUD now shows the burn details and the countdown.

The countdown reaches zero. You press **Execute Burn**. The ship lights its engine — zoom in
and you see a bright exhaust plume streaming from the tail. The throttle gauge in the VESSEL
panel jumps to 100%. The fuel percentage starts dropping. Time warp is locked to 1x. Your
speed climbs. The trajectory line updates live, stretching outward as the burn reshapes your
orbit. After about 8 seconds the burn completes and the engine cuts off. The plume disappears.

Your orbit is now a long ellipse, apoapsis at 5,000 km, periapsis still down at 465 km. The
trajectory line shows the new oval shape. You warp time forward (press `.` a few times) and
watch the ship climb slowly toward apoapsis, slowing down as it gains altitude, then falling
back toward periapsis and speeding up again.

## The Moon

A grey circle sits farther out — the Moon, about 384,000 km from Earth. A faint grey ellipse
shows its orbit. Five small teal markers (L1 through L5) are arranged around the Earth-Moon
system — the Lagrange points, equilibrium positions computed from the real three-body problem.

A faint translucent blue sphere surrounds the Moon — its **sphere-of-influence shell**, about
66,000 km in radius. This is a visual guide showing where the Moon's gravity dominates; the
actual gravity model is continuous N-body (the Moon pulls on you everywhere, just more
strongly when you're close).

You left-click the Moon marker. It highlights, and the HUD updates: "Target: Moon dist
382,441 km rel 1,022 m/s." Closest-approach markers appear on your trajectory — small markers
showing where your orbit comes nearest to the Moon.

You press `P` and a **porkchop plot** fills part of the screen — a color-coded grid showing
the delta-V cost of every combination of departure time (x-axis) and flight time (y-axis).
The cheapest transfers are bright spots. You close it and press `I` to auto-plan: the game
runs a Lambert solver across the grid and places a maneuver node at the optimal departure.

You warp to the node, execute the burn (about 3,100 m/s — most of your fuel), and coast for
three days of game time (about 4 seconds at 100,000x warp). The trajectory line arcs outward
toward the Moon. As you approach, the translucent SOI sphere looms larger and larger. When you
cross inside, it tints the scene green — a visual signal that you're in the Moon's domain.

The HUD switches: "Orbiting: Moon." Your altitude is now relative to the Moon's surface. Your
periapsis reads 1,847 km. You burn retrograde to slow down and capture into lunar orbit.

Zoomed in, the Moon is a textured grey sphere, craters and maria visible. Your ship circles it
at a few hundred kilometers, the Earth a blue marble in the background.

## Lagrange points

You target L1 (the Lagrange point between Earth and the Moon, about 58,000 km from the Moon).
The HUD shows a live distance and relative velocity. You burn toward it, null your relative
speed, and... you hover. You're balanced at the gravitational equilibrium point between Earth
and Moon.

This is something most space games can't do. L1 doesn't exist in a patched-conic simulation
where only one body's gravity is active at a time. Here, with real N-body gravity, L1 is a
real (if unstable) equilibrium.

## Going interplanetary

Back in Earth orbit (quickload with F9), you click the **Mars** marker — a small reddish dot
far from Earth. The HUD shows: "Target: Mars dist 1.42 AU rel 5,731 m/s." (AU — astronomical
units — appear automatically when distances exceed about 1.5 million km.)

You press `I`. The game searches a grid spanning a full synodic period (the ~26-month cycle
between Earth-Mars alignment windows) and places a node. The computed burn is about 4,100 m/s,
departing in 47 days. You drag the Node T slider to fast-forward to the departure, or warp
time with `.` until the countdown nears zero.

Execute. The burn takes about 12 seconds at full thrust, consuming most of your fuel. The
trajectory line stretches outward from Earth, curving into a long arc through heliocentric
space. You've escaped Earth.

### The heliocentric coast

The HUD switches to "Orbiting: Sun." Your altitude is now in AU. Periapsis and apoapsis are in
AU too. The trajectory line extends to show your full 7-month transfer arc — a gentle curve
from Earth's orbit out to Mars's.

You can see the solar system at this zoom level. The Sun is a bright golden sphere at the
center. Faint circles mark the orbits of all eight planets — Mercury, Venus, Earth, Mars,
Jupiter, Saturn, Uranus, Neptune. Each planet is a tiny textured sphere at its real position:
Mercury a grey dot close to the Sun, Venus a pale yellow, Mars reddish, Jupiter a banded tan
marble, Saturn with its distinctive rings, Uranus a pale cyan, Neptune a deep blue.

Translucent SOI spheres surround each planet, sized by their real gravitational dominance. 
Jupiter's SOI is enormous — about 48 million km in radius. Saturn's is even larger.

You warp time to 1,000,000x. At this rate, the months-long coast takes about 22 seconds. Your
ship's marker crawls along the trajectory line. Earth falls behind. Mars grows ahead.

### Arrival at Mars

Mars's SOI sphere appears as you approach — translucent, reddish. You cross the boundary and
the HUD switches: "Orbiting: Mars." Your altitude is now relative to Mars's surface. You're on
a hyperbolic approach, and the flyby encounter readout appears in green at the bottom of the
MANEUVER panel:

"Flyby Mars: v-inf 2.8 km/s delta 34.2 deg Pe 412 km free dV 1,621 m/s"

This tells you: your hyperbolic excess velocity relative to Mars is 2.8 km/s, Mars's gravity
will bend your path by 34 degrees if you don't intervene, your closest approach will be 412 km,
and the equivalent free delta-V from this gravity assist is 1,621 m/s.

But you don't want a flyby — you want to stay. You press `2` for retrograde SAS hold, then
`Shift` to throttle up. The engine fires, slowing you relative to Mars. Your hyperbolic
trajectory bends into an ellipse, then tightens into a low Mars orbit.

Zoomed in, Mars is a rusty-orange sphere with ice caps and dark markings. Your ship circles it,
300 km up. The Sun is a bright point. Earth is a faint dot. You're 225 million kilometers from
home, and the physics that got you here is textbook-accurate.

## The outer solar system

With Unlimited dV toggled on (`U`), you can go further. Target Jupiter — the intercept planner
searches an 11-year synodic window and finds a transfer. The coast takes about 2.7 years
(a minute or so at maximum warp). Jupiter is enormous when you arrive — a textured gas giant
71,000 km in radius, bigger than 11 Earths. Its SOI is 48 million km across.

Saturn has **rings**: a flat annular disk textured with the characteristic banded pattern,
extending from about 1.1 to 2.3 planetary radii. They're visible as you approach, a thin
bright line against the black of space, resolving into a disk as you orbit.

Uranus and Neptune are smaller, distant, pale spheres — Uranus a muted cyan, Neptune a deep
violet-blue. Getting to Neptune takes over a decade of game time. At maximum warp, it's a few
minutes of watching your trajectory line inch across the outer solar system.

## Gravity assists (flybys)

The encounter display that appeared at Mars works at any planet. Approach Jupiter with high
v-infinity and the readout shows a deflection of 30+ degrees and thousands of m/s of free dV.

In theory (and with the Unlimited dV toggle), you could chain assists — Earth to Jupiter to
Saturn — the way Voyager and Cassini did. Each flyby bends your path and changes your
heliocentric velocity without costing any fuel. The real N-body gravity model makes this
possible; the encounter display tells you what each flyby is worth.

## What it feels like

The overall sensation is one of **patience rewarded by precision.** Long coasts where nothing
visible happens, then a brief window where a well-timed burn reshapes your entire trajectory.
The contrast between the vast scales (AU, months) and the tiny control inputs (m/s, seconds)
is what makes orbital mechanics beautiful.

The physics is unforgiving. A burn 30 seconds too early sends you 10,000 km off target. Too
little delta-V and you sail past the Moon into deep space. But the tools are generous — the
trajectory line always shows you truth, the porkchop plot finds the windows, the intercept
planner does the hard math, and quicksave means you can always try again.

It's a game where you learn by doing. Your first Moon shot will probably miss. Your first Mars
transfer will probably run out of fuel. But every attempt teaches you something about how
gravity and velocity actually work, because the physics under the hood is real.

### A session, condensed

A typical 30-minute session might go:

1. Launch, raise orbit, target the Moon (5 min)
2. Plan and execute a lunar transfer, coast, capture (10 min)
3. Explore the Moon, visit L2, quicksave (5 min)
4. Quickload your Earth orbit, target Mars, find a transfer window (5 min)
5. Execute the departure burn, warp through the coast, arrive, capture (5 min)

Or you might spend the whole time in low Earth orbit, practicing efficient orbit-raising,
learning how prograde and retrograde burns change the shape of an ellipse, watching the
numbers in the ORBIT panel respond to every nudge of thrust. There's no score, no timer, no
enemies. Just you, gravity, and the rocket equation.

## The two modes

The game has two modes, chosen at launch:

**Sandbox** (default) is the full game described above — a flyable ship in Earth orbit with the
entire solar system, maneuver planning, targeting, gravity assists, saves, and the full HUD.

**Solar viewer** (`--solar`) is a camera-only mode. No ship, no flight. Just the Sun and eight
planets at their real positions (computed from JPL ephemerides), with time warp. You can orbit
the camera, zoom from the Sun to Neptune, and watch the planets move along their orbits at
high warp. It's a planetarium.

## The numbers behind the scenes

For anyone curious about what's happening under the hood:

- Physics runs in **64-bit floating point**, in SI units (meters, seconds, kilograms, radians).
  Every distance, every velocity, every gravitational parameter is at full double precision.

- The gravity model is **restricted N-body**: 10 bodies (Sun, Earth, Moon, Mercury, Venus,
  Mars, Jupiter, Saturn, Uranus, Neptune) all pulling on the ship simultaneously. The
  integration uses velocity-Verlet with adaptive sub-stepping — the time step shrinks
  automatically near massive bodies to keep the integration accurate.

- Transfer planning uses **Lambert's problem** — given two positions and a flight time, find
  the orbit that connects them. The game solves this over a grid of departure times and flight
  times, producing the porkchop plot.

- The renderer uses a **floating-origin transform** to handle the scale problem. The solar
  system spans 4.5 billion km, but a docking maneuver needs millimeter precision. Physics
  stays in 64-bit; the renderer subtracts the camera's position in 64-bit *before* casting to
  32-bit for the GPU. This keeps everything sharp at every zoom level.

- There are **275+ unit tests** validating the physics against textbook values (Howard Curtis,
  *Orbital Mechanics for Engineering Students*). Energy conservation, angular momentum
  conservation, period closure, vis-viva identity — all checked to at least 7-digit precision.
  If the physics were wrong, the tests would catch it.

## If you do get to play

See [PLAYING.md](PLAYING.md) for the full controls, keybindings, HUD guide, and a step-by-step
first-flight tutorial. Press F1 in-game for a keybind overlay.

The game requires Python 3.9+ on Windows with some dependencies (Panda3D for rendering,
Skyfield for ephemerides, NumPy/SciPy for math). First launch downloads about 32 MB of
ephemeris data and some texture maps. After that it runs offline.
