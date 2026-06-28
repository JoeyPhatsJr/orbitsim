# Orbital Mechanics Simulator

A realistic 3D orbital mechanics simulator/game for desktop (Windows) — "Kerbal Space Program,
but the physics are real." Built in Python with a unit-tested, textbook-accurate physics core and a
Panda3D renderer. Real Keplerian orbits, Hohmann/bi-elliptic/Lambert transfers, ΔV-budget
optimization with porkchop plots, the real solar system (JPL/Skyfield ephemerides), and a flyable
sandbox that runs a **restricted N-body** model — real gravity from the Sun, Moon, Mercury, Venus,
and Mars — with live maneuver-node prediction, interplanetary transfer planning, and a 3rd-person
ship view. Fly from Earth orbit to Mars under real N-body physics, park at Lagrange points, and
plan your burns with porkchop plots.

Vessels are point masses with a ΔV/fuel budget — **no** rocket-part building, **no**
aerodynamics/atmosphere (deliberately out of scope, to keep the focus on orbital mechanics).

**New here?** See [How to play](docs/PLAYING.md) for controls, the HUD/navball, and a first-flight
walkthrough.

## Features

- **Tested physics core** — Keplerian elements ↔ state vectors, Kepler solvers (elliptic /
  parabolic / hyperbolic), analytic two-body propagation, all in float64 SI units and validated
  against Curtis, *Orbital Mechanics for Engineering Students*.
- **Continuous-thrust flight** — velocity-Verlet integration with the real rocket equation, attitude
  control, navball + HUD. Both coasting and thrusting use the N-body propagator; time-warp is capped
  per-frame to keep the integration accurate.
- **Transfer & ΔV planning** — Hohmann, bi-elliptic, plane-change, and Lambert transfers, plus a
  ΔV optimizer with porkchop diagrams. Press `I` to plan an intercept to any targeted body (Moon or
  planet); press `P` for a porkchop plot with synodic-period search grids for interplanetary targets.
- **Full solar system** — the Sun and all seven planets (Mercury through Neptune) are rendered as
  true-scale textured bodies with heliocentric orbit reference lines and translucent SOI spheres.
  Saturn has a textured ring system. All bodies contribute real gravitational acceleration to the
  N-body model.
- **Interplanetary flight** — escape Earth, coast under N-body gravity, and capture at another
  planet. The HUD adapts to the dominant body (altitude, periapsis, apoapsis, and TWR switch to
  Mars-relative when you enter Mars's SOI). Trajectory lines extend to 400-day horizons in
  heliocentric space. Distances display in AU when appropriate.
- **Sandbox** — maneuver nodes with spring-loaded jog sliders for delta-V and node time, dV labels
  on node markers with countdown, Pe/Ap presets, auto-warp-to-node, and live predicted-orbit
  rendering.
- **Targeting & ΔV** — click a body to target it (Moon, Lagrange points, or planets), with
  target-relative closest-approach + relative velocity, working TARGET/ANTITARGET SAS, a one-click
  porkchop **intercept** planner, and an **unlimited-ΔV** sandbox toggle.
- **Gravity assist planning** — live flyby encounter display when approaching a planet's SOI:
  hyperbolic excess velocity (v-infinity), deflection angle, periapsis distance, and equivalent free
  delta-V. Pure hyperbolic geometry (`core/flyby.py`) validated with 16 tests.
- **Restricted N-body sandbox** (`core/nbody.py`) — the ship flies as a massless test particle under
  the gravity of Earth, Moon, Sun, and all seven planets (velocity-Verlet with adaptive
  sub-stepping). The forward-integrated trajectory line is the source of truth, and time-warp is
  capped near bodies to keep the integration accurate. The **Earth–Moon Lagrange points** (L1–L5)
  balance to machine precision and are shown as selectable navigation targets. Translucent
  **sphere-of-influence shells** around the Moon and each planet mark gravitational boundaries. An
  alternative to KSP's patched conics — no SOI transitions, just continuous N-body gravity.
- **3rd-person ship view** — zoom (or press `M`) all the way in to an oriented, lit ship model with an
  exhaust plume; zoom out to the map. Adaptive camera clip planes keep distant bodies visible at all
  zoom levels. Smoothed orbit camera throughout.
- **Readable HUD** — grouped, self-sizing TIME/ORBIT/MANEUVER and VESSEL panels, a 3D attitude navball
  with prograde/retrograde/normal/radial/target markers, a SAS chip with clickable mode buttons, and a
  toggleable orbital/target velocity readout. In-world Pe/Ap, target, and closest-approach markers with
  decluttering and depth fade. Smart formatting (days/hours for long countdowns, AU for interplanetary
  distances).
- **Realistic Earth** — textured, day/night terminator, atmosphere; starfield skybox. Planet textures
  for all planets, Moon, and Sun. Saturn ring texture.
- **Save/load** — JSON sandbox saves, F5/F9 quicksave/quickload.

## Architecture

Strict one-directional layering keeps the physics unit-testable with zero graphics installed:

```
orbitsim/core/    pure physics. float64, SI units. NEVER imports panda3d, sim, or render.
orbitsim/sim/     stateful world: World, Vessel, SimClock. imports core only.
orbitsim/render/  Panda3D. imports sim + core. ALL graphics live here.
```

Frame is J2000 / ICRF, origin at the central body; sim time is seconds past J2000 TDB. A
floating-origin transform (`render/floating_origin.py`) keeps docking-scale precision while
rendering a solar-system-scale scene in float32.

## Quick start

Requires Python 3.9+ on Windows.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,render]"        # physics + render deps

.venv\Scripts\python -m pytest tests/ -q      # run the test suite (~275 tests)
.venv\Scripts\python -m orbitsim              # launch the sandbox (LEO, flyable)
.venv\Scripts\python -m orbitsim --solar      # launch the solar-system viewer
```

Once installed, the `orbitsim` console script is also available (`orbitsim`, `orbitsim --solar`).

First launch downloads and caches large assets into `data/` (gitignored): the DE440 ephemeris
kernel (~32 MB) and texture maps. All download code degrades gracefully offline.

Press **F1** in-app for the keybind overlay, or read [docs/PLAYING.md](docs/PLAYING.md) for the full
controls, a guide to the HUD/navball, and a first-flight walkthrough.

## Documentation

Start with [`docs/PLAYING.md`](docs/PLAYING.md) to actually fly. The original build is specified
phase-by-phase in [`docs/`](docs/):

| Doc | Purpose |
|---|---|
| [`docs/PLAYING.md`](docs/PLAYING.md) | **How to play** — controls, HUD/navball, first-flight walkthrough |
| [`docs/00-OVERVIEW.md`](docs/00-OVERVIEW.md) | Architecture, conventions, definition of done — **read first** |
| [`docs/01-physics-core.md`](docs/01-physics-core.md) | Two-body physics core (formulas + test values) |
| [`docs/02-rendering-scale.md`](docs/02-rendering-scale.md) | Panda3D render + the floating-origin scale solution |
| [`docs/03-sandbox-maneuvers.md`](docs/03-sandbox-maneuvers.md) | Maneuver nodes, burns, live prediction |
| [`docs/04-transfers-optimizer.md`](docs/04-transfers-optimizer.md) | Hohmann/bi-elliptic/Lambert + ΔV optimizer + porkchop |
| [`docs/05-real-solar-system.md`](docs/05-real-solar-system.md) | Skyfield ephemerides, patched conics, SOI |
| [`docs/06-polish-packaging.md`](docs/06-polish-packaging.md) | Save/load, HUD polish, packaging |

Subsequent "playable game" work is specified under [`docs/superpowers/`](docs/superpowers/)
(specs + plans).

**Golden rule:** the physics core (`orbitsim/core/`) never imports rendering. It is pure,
deterministic, float64, SI units, and fully unit-tested. Get it green before anything draws.
