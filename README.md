# Orbital Mechanics Simulator

A realistic 3D orbital mechanics simulator/game for desktop — "Kerbal Space Program, but
the physics are real." Built in Python. Real Keplerian orbits, Hohmann/bi-elliptic/Lambert
transfers, ΔV-budget optimization, the real solar system (JPL ephemerides), and a free-build
sandbox with maneuver nodes.

## Status
Greenfield. Being built in phases — see [`docs/`](docs/).

## Quick start (once Phase 0 is done)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                          # physics core must be green before rendering work
python -m orbitsim              # launches the app
```

## For implementers (read this first)
The build is specified for incremental implementation in [`docs/`](docs/):

| Doc | Purpose |
|---|---|
| [`docs/00-OVERVIEW.md`](docs/00-OVERVIEW.md) | Architecture, conventions, how to work, definition of done — **read first** |
| [`docs/01-physics-core.md`](docs/01-physics-core.md) | Phase 1: tested two-body physics core (formulas + test values) |
| [`docs/02-rendering-scale.md`](docs/02-rendering-scale.md) | Phase 2: Panda3D render + the floating-origin scale solution |
| [`docs/03-sandbox-maneuvers.md`](docs/03-sandbox-maneuvers.md) | Phase 3: maneuver nodes, burns, live prediction |
| [`docs/04-transfers-optimizer.md`](docs/04-transfers-optimizer.md) | Phase 4: Hohmann/bi-elliptic/Lambert + ΔV optimizer + porkchop |
| [`docs/05-real-solar-system.md`](docs/05-real-solar-system.md) | Phase 5: Skyfield ephemerides, patched conics, SOI |
| [`docs/06-polish-packaging.md`](docs/06-polish-packaging.md) | Phase 6: save/load, HUD polish, PyInstaller exe |

**Golden rule:** the physics core (`orbitsim/core/`) never imports rendering. It is pure,
deterministic, float64, SI units, and fully unit-tested. Get it green before anything draws.
