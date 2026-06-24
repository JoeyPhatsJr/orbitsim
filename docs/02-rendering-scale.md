# 02 — Phase 2: Rendering + the Scale Problem

**Gate:** start only when `pytest tests/core` is 100% green.

Goal of this phase: draw the Earth and one vessel, show its orbit line, fly the camera, and run
a time-warpable clock — without floating-point jitter. The first "wow" milestone.

Engine: **Panda3D** (`ShowBase`). The render layer imports `sim/` and `core/`; the reverse is
forbidden.

---

## The core difficulty: dynamic range

The solar system spans ~1.5e11 m; a docking maneuver needs ~1e-3 m precision. GPUs render with
**float32** (~7 significant digits). If you place objects at their true SI coordinates, anything
past ~1e4 m from the origin jitters visibly ("the Kraken"). We solve this with **two decoupled
representations**:

1. **Physics space** — always true `float64` SI meters, owned by `core/`/`sim/`. Never scaled.
2. **Render space** — `float32`, re-centered every frame on a **floating origin**, and scaled
   by a per-view factor so the visible scene fits comfortably in ~[-1000, 1000] render units.

Physics never sees render space; rendering never feeds back into physics.

---

## Task 2.1 — `sim/clock.py`

```python
class SimClock:
    """Owns simulation time (seconds past J2000, TDB) and the time-warp rate."""
    sim_time_s: float        # float64
    warp: float = 1.0        # sim seconds per real second; allowed steps below
    WARP_STEPS = [1, 5, 10, 50, 100, 1_000, 10_000, 100_000, 1_000_000]

    def advance(self, real_dt_s: float) -> float:
        # returns sim_dt = real_dt_s * warp; caller propagates by it
```
At high warp, vessels advance by **analytic** `propagate_kepler` (huge Δt is fine on-rails). Near
a maneuver or SOI boundary, clamp `warp` down so `propagate_numeric` stays accurate. (Hook the
clamp in Phase 3/5; for now just expose `warp` stepping.)

**Test (no graphics):** `advance` returns `real_dt*warp`; warp stepping stays within `WARP_STEPS`.

---

## Task 2.2 — `sim/world.py`

A registry: central `CelestialBody`, a list of `Vessel` objects. A `Vessel` holds its current
`StateVector` and metadata (name, dry mass, fuel/ΔV budget — budget used in Phase 3+).

```python
@dataclass
class Vessel:
    name: str
    state: StateVector          # mutable here (sim layer), updated each tick
    delta_v_budget_mps: float = 0.0

class World:
    central: CelestialBody
    vessels: list[Vessel]
    def step(self, sim_dt_s: float) -> None:
        # for each vessel: vessel.state = propagate_kepler(vessel.state, sim_dt_s)
```
**Test (no graphics):** stepping a circular-orbit vessel by one period returns it near its start
(reuses the core conservation guarantee at the sim layer).

---

## Task 2.3 — `render/floating_origin.py` (the key abstraction)

```python
class RenderTransform:
    """Maps physics-space SI float64 positions to render-space float32 positions.

    render_pos = (physics_pos_m - origin_m) / scale_m_per_unit
    """
    origin_m: np.ndarray      # float64, the physics point currently mapped to render (0,0,0)
    scale_m_per_unit: float   # meters per render unit (set from camera zoom)

    def to_render(self, physics_pos_m: np.ndarray) -> tuple[float, float, float]:
        # subtract origin in float64 FIRST, then cast to float32 → preserves local precision
```
Each frame: set `origin_m` to the **focused body/vessel's** physics position, pick
`scale_m_per_unit` from camera distance, then convert every object through `to_render`. Because
the subtraction happens in float64 before the float32 cast, local detail near the focus keeps
millimeter precision regardless of absolute solar-system coordinates.

**Test (no graphics, pure math):** a point 1e11 m from origin, with origin set 1e-3 m away from
it, maps to a render position whose implied physics distance is still 1e-3 m within 1e-6 — i.e.
precision is preserved across the float32 cast. This test is the whole reason the renderer works;
write it carefully.

---

## Task 2.4 — `render/orbit_lines.py`

Sample the current orbit into a polyline and build a Panda3D `LineSegs` node.
```
elements = state_to_elements(vessel.state)
for nu in linspace(0, 2π, N=256):           # closed ellipse; for hyperbola sample true-anomaly
    pos = elements_to_state(elements@nu).r   # physics-space point
    add to_render(pos) to the line
```
Recompute when the orbit changes (after a burn); otherwise the ellipse is static in physics space
and only re-mapped through the (cheap) `RenderTransform` each frame. Color by vessel.

---

## Task 2.5 — `render/camera_rig.py`

- Orbit-style camera: focus target (a body or vessel), azimuth/elevation drag, **log-scaled**
  zoom (mouse wheel) spanning vessel-close (~10 m) to whole-system (~1e12 m). Zoom sets
  `RenderTransform.scale_m_per_unit`.
- Key to cycle focus target; smooth re-center when focus changes.
- Near/far clip planes set from `scale` so depth precision stays usable.

---

## Task 2.6 — `render/app.py` + `__main__.py`

`ShowBase` subclass. Build the central body (sphere, textured later), vessels (markers + orbit
lines), a HUD (Task 2.7). Register a `taskMgr` task that each frame:
1. `sim_dt = clock.advance(globalClock.getDt())`
2. `world.step(sim_dt)`
3. update `RenderTransform.origin_m` to focus; re-map all node positions via `to_render`.

`__main__.py` wires `SimClock` + `World` (one Earth + one LEO vessel from a default scenario) and
runs `app.run()`.

## Task 2.7 — `render/hud/` (DirectGUI)

Minimal overlay: current sim time (formatted UTC), warp rate, focused vessel's altitude, speed,
periapsis/apoapsis, period. Pull numbers from `core` (convert SI → km/UTC at this boundary only).
Buttons/keys: warp up/down, pause, cycle focus.

---

## Phase 2 exit criteria

- App launches via `python -m orbitsim`, shows Earth + a vessel on a visible orbit line.
- Camera focuses/zooms across the full range **without jitter** at any zoom (visually verify by
  zooming to the vessel surface — it must be rock-steady).
- Time warp speeds the orbit smoothly; HUD shows correct, live altitude/speed/period.
- The `to_render` precision test and all sim-layer tests pass.

Then proceed to `docs/03-sandbox-maneuvers.md`.
