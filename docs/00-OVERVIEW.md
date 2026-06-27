# 00 — Overview, Architecture & Working Agreement

**Read this entire file before writing any code.** It defines how the project is structured,
the rules every phase must follow, and how to know when a task is actually done.

> **Status note (2026-06-26):** the original six phases are complete; current work is the
> "playable game" effort, now pivoting the sandbox to a **restricted N-body** model (real Moon
> gravity + Lagrange points). For the live feature status, the N-body direction, and the per-cycle
> spec→plan→build workflow, see **`CLAUDE.md`** (the authoritative status) and
> `docs/superpowers/{specs,plans}/`. This file remains the canonical reference for the
> architecture rules and definition of done, which still hold.

---

## 1. What we are building

A desktop (Windows) 3D orbital mechanics simulator/game in Python. Three pillars:

1. **Transfer & ΔV planning** — Hohmann, bi-elliptic, plane-change, and Lambert transfers;
   a ΔV-budget optimizer with porkchop plots.
2. **Real solar system** — real planet/moon positions from JPL ephemerides (Skyfield/DE440),
   patched-conic interplanetary trajectories with sphere-of-influence (SOI) handoffs.
3. **Sandbox** — free experimentation: place a vessel, add maneuver nodes, burn
   prograde/retrograde/normal, watch the predicted orbit update live.

Target users: the author + a friend, on desktop. Priority is **polish + physical accuracy**,
not portability. Point-mass vessels with a ΔV/fuel budget — **no** rocket-part building, **no**
aerodynamics/atmosphere (out of scope; keeps focus on orbital mechanics).

---

## 2. Non-negotiable architecture rule

```
orbitsim/core/   ── pure physics. float64, SI units. NEVER imports panda3d or any UI.
orbitsim/sim/    ── stateful world: clock, vessels, time-warp. imports core, not render.
orbitsim/render/ ── Panda3D. imports sim + core. ALL graphics live here.
```

If you ever feel tempted to `import panda3d` inside `core/`, stop — you are doing it wrong.
The physics core must be runnable and testable with **zero** graphics installed. This
separation is what lets us unit-test the math to textbook accuracy.

### Target package layout
```
orbitsim/
├── __init__.py
├── __main__.py            # `python -m orbitsim` → launches render app
├── core/
│   ├── __init__.py
│   ├── constants.py       # μ (GM), radii, J2 — sourced from astropy, not hand-typed
│   ├── bodies.py          # CelestialBody dataclass, SOI radius, rotation rate
│   ├── state.py           # StateVector (r, v) in an inertial frame
│   ├── elements.py        # KeplerianElements dataclass + conversions
│   ├── kepler.py          # Kepler equation solvers (ellipse/parabola/hyperbola)
│   ├── propagate.py       # two-body propagation (analytic + numeric)
│   ├── frames.py          # frame rotations (perifocal ↔ inertial), helpers
│   ├── ephemeris.py       # Skyfield wrapper (Phase 5)
│   ├── patched_conics.py  # SOI transitions (Phase 5)
│   ├── maneuvers.py       # impulsive ΔV, maneuver nodes (Phase 3)
│   ├── transfers.py       # Hohmann / bi-elliptic / plane change / Lambert (Phase 4)
│   └── optimize.py        # ΔV optimizer, porkchop grids (Phase 4)
├── sim/
│   ├── __init__.py
│   ├── world.py           # body registry + vessels
│   ├── clock.py           # sim time + time-warp
│   └── persistence.py     # JSON save/load
└── render/
    ├── __init__.py
    ├── app.py             # Panda3D ShowBase bootstrap + main loop
    ├── floating_origin.py # the scale solution (see docs/02)
    ├── orbit_lines.py     # conic → polyline
    ├── camera_rig.py      # focus/zoom across huge dynamic range
    └── hud/               # DirectGUI panels
tests/
└── (mirrors orbitsim/, e.g. tests/core/test_kepler.py)
```

---

## 3. Conventions (apply everywhere)

- **Units:** SI everywhere in `core/` — **meters, seconds, radians, kg**. Convert to km/degrees
  only at the UI boundary. Document units in every docstring and append `_m`, `_s`, `_rad`,
  `_mps` to variables when ambiguous (e.g. `radius_m`, `true_anomaly_rad`).
- **Frame:** physics inertial frame is **J2000 / ICRF**, origin at the central body's center
  unless stated. Vectors are `numpy.ndarray` shape `(3,)`, dtype `float64`.
- **Numbers:** always `float64`. Never use Python floats in arrays. Never use `float32` in core.
- **Angles:** radians internally, normalized to `[0, 2π)` for anomalies, `[0, π]` for inclination.
- **Immutability:** `StateVector`, `KeplerianElements`, `CelestialBody` are **frozen
  dataclasses**. Operations return new instances; never mutate in place.
- **No magic numbers:** physical constants come from `core/constants.py` (which pulls from
  `astropy.constants`). Never hard-type `398600` or `6.674e-11` inline.
- **Type hints** on every function. **Docstrings** (NumPy style) on every public function,
  stating units, frame, and the reference/formula used.
- **Style:** format with `black` (line length 100). Names: `snake_case` funcs/vars,
  `PascalCase` classes, `UPPER_SNAKE` constants.
- **Errors:** raise `ValueError` with a clear message on invalid input (e.g. negative `a` for a
  bound orbit). Do not silently clamp or return `None` on bad physics input.

---

## 4. How to work (TDD — mandatory)

This codebase is math. A weaker model gets math wrong silently. The defense is tests with
**known answers**. For every core function:

1. **Write the test first** using the concrete values provided in the phase doc.
2. Run it — watch it fail (`pytest tests/core/test_x.py -q`).
3. Implement until green.
4. Add **round-trip / conservation** tests (these catch errors even when you don't have a
   textbook number): see §5.
5. Only then move on. **Never** implement a second function while the first is red.

Run the whole core suite often: `pytest tests/core -q`. The core suite must be **100% green**
before any rendering/Phase-2 work begins.

### Definition of Done (per task)
- [ ] Function implemented with type hints + NumPy-style docstring stating units & frame.
- [ ] Unit test against the provided textbook value passes (tolerance stated in phase doc).
- [ ] Round-trip and/or conservation test passes.
- [ ] `black` formatted, no unused imports.
- [ ] No `core/` file imports anything from `render/` or `panda3d`.
- [ ] **Committed and pushed to GitHub** (`origin/main`) after the change is verified — stage the
      specific files you changed by explicit path (never `git add -A`/`.`; keep `data/`, scratch
      files, and `.claude/settings.local.json` out), then `git push`.

---

## 5. Universal verification tools (use these constantly)

These invariants must hold regardless of inputs — encode them as `hypothesis` property tests:

- **Round-trip:** `elements → state → elements` returns the original elements (within 1e-6
  relative). `state → elements → state` likewise.
- **Energy conservation:** specific orbital energy `ε = v²/2 − μ/r = −μ/(2a)` is constant along
  an orbit. After propagating any time Δt, `ε` changes by < 1e-9 relative.
- **Angular momentum conservation:** `h = r × v` is constant (vector) under two-body propagation.
- **Period closure:** after exactly one period `T = 2π√(a³/μ)`, the state returns to its start
  (position error < 1 mm for LEO-scale orbits using analytic propagation).
- **Circular orbit:** if `e = 0`, speed is constant `= √(μ/r)` and `r` is constant.
- **Vis-viva:** `v² = μ(2/r − 1/a)` must hold at every propagated point.

If any of these drift, the math is wrong — fix before proceeding.

---

## 6. Phase order & gates

| Phase | Doc | Gate to start |
|---|---|---|
| 0 Scaffold | this file §7 | — |
| 1 Physics core | 01 | Phase 0 done |
| 2 Render + scale | 02 | **core suite 100% green** |
| 3 Sandbox/maneuvers | 03 | Phase 2 renders one orbit |
| 4 Transfers/optimizer | 04 | Phase 3 maneuver nodes work |
| 5 Real solar system | 05 | Phase 4 transfers work |
| 6 Polish/packaging | 06 | Phase 5 interplanetary works |

Do **not** skip ahead. Each later phase doc should be re-read (and expanded if needed) right
before starting it, using the real code that now exists.

---

## 7. Phase 0 — Scaffold (do this first)

1. Create the package layout in §2 (empty `__init__.py` files, stub modules with just
   docstrings + `TODO`).
2. Create `pyproject.toml` configuring `black` (line-length 100) and `pytest` (testpaths=tests).
3. Create the venv, `pip install -r requirements.txt`. Confirm `import numpy, scipy, astropy`
   works. (Panda3D/Skyfield/lamberthub import-check can wait until their phases.)
4. Add `tests/core/test_smoke.py` with `def test_imports(): import orbitsim.core` and make
   `pytest` pass.
5. Add a `.gitignore` (`.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/*.bsp`).

**Done when:** `pytest` runs and is green, repo layout matches §2.
