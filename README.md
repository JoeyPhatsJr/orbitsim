# Orbital Mechanics Simulator

A realistic 3D orbital mechanics simulator/game for desktop (Windows) — "Kerbal Space Program,
but the physics are real." Built in Python with a unit-tested, textbook-accurate two-body physics
core and a Panda3D renderer. Real Keplerian orbits, Hohmann/bi-elliptic/Lambert transfers,
ΔV-budget optimization with porkchop plots, the real solar system (JPL/Skyfield ephemerides +
patched conics with SOI handoffs), and a free-build sandbox with live maneuver-node prediction.

Vessels are point masses with a ΔV/fuel budget — **no** rocket-part building, **no**
aerodynamics/atmosphere (deliberately out of scope, to keep the focus on orbital mechanics).

## Features

- **Tested physics core** — Keplerian elements ↔ state vectors, Kepler solvers (elliptic /
  parabolic / hyperbolic), analytic two-body propagation, all in float64 SI units and validated
  against Curtis, *Orbital Mechanics for Engineering Students*.
- **Continuous-thrust flight** — RK4 integration with the real rocket equation, attitude control,
  navball + HUD. Coasting vessels propagate analytically (time-warp on rails); thrusting vessels
  integrate numerically.
- **Transfer & ΔV planning** — Hohmann, bi-elliptic, plane-change, and Lambert transfers, plus a
  ΔV optimizer with porkchop diagrams.
- **Real solar system** — planet/moon positions from the DE440 ephemeris via Skyfield; patched
  conics with sphere-of-influence transitions.
- **Sandbox** — scheduled maneuver nodes, Pe/Ap presets, auto-warp-to-node, and live
  predicted-orbit rendering.
- **Targeting & ΔV** — click a body to target it, target-relative closest-approach + relative
  velocity, working TARGET/ANTITARGET SAS, a one-click porkchop **intercept** planner, a bigger
  ΔV cap, and an **unlimited-ΔV** sandbox toggle.
- **Restricted N-body engine** (`core/nbody.py`, in progress) — the ship as a massless test
  particle under Earth + Moon gravity (velocity-Verlet, Jacobi constant, **Lagrange points** that
  balance to machine precision). Moving the sandbox toward real lunar gravity and parkable L-points
  — an alternative to KSP's patched conics. *Physics core is built and tested; render integration
  is next.* See `docs/superpowers/specs/2026-06-26-nbody-flyable-design.md`.
- **Realistic Earth** — textured, day/night terminator, atmosphere; starfield skybox.
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

.venv\Scripts\python -m pytest tests/ -q      # run the test suite (~136 tests)
.venv\Scripts\python -m orbitsim              # launch the sandbox (LEO, flyable)
.venv\Scripts\python -m orbitsim --solar      # launch the solar-system viewer
```

First launch downloads and caches large assets into `data/` (gitignored): the DE440 ephemeris
kernel (~32 MB) and texture maps. All download code degrades gracefully offline.

## Documentation

The original build is specified phase-by-phase in [`docs/`](docs/):

| Doc | Purpose |
|---|---|
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
