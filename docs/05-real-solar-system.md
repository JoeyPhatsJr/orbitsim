# 05 — Phase 5: Real Solar System (ephemerides + patched conics)

**Gate:** Phase 4 transfers work.

Goal: replace the toy single-body world with the real solar system — real planet/moon positions
from JPL data — and support interplanetary trajectories via patched conics with SOI handoffs.

---

## Task 5.1 — `core/ephemeris.py` (Skyfield / DE440)

```python
from skyfield.api import load
# Download/caches DE440 once into data/ (de440s.bsp is smaller, 1850–2150 — fine for us).
def body_state(name: str, t_sim_s: float, center: str = "SUN") -> StateVector:
    """Position+velocity of `name` relative to `center` at sim time t (sec past J2000 TDB),
    in J2000 ecliptic/ICRF, returned as SI StateVector. Wraps Skyfield."""
```
- Convert our `sim_time_s` (sec past J2000 TDB) → Skyfield `Time` carefully (Skyfield uses Terrestrial/
  Barycentric scales; document the exact conversion and test it).
- Cache the loaded kernel at module load. Add the `.bsp` to `.gitignore`; download on first run.

**Tests:** Earth's heliocentric distance is ~1 AU (`1.496e11 m`) within 2% across several dates;
Earth's orbital speed ~29.8 km/s within 2%. (Sanity anchors against known values.)

## Task 5.2 — `core/bodies.py` expansion

Add Sun, all 8 planets, Earth's Moon (and optionally major moons) with real μ, radius, J2 (where
relevant), and SOI radii. Source μ from astropy/IAU; cite each in a comment. Parent links form the
tree Sun → planets → moons.

## Task 5.3 — `core/patched_conics.py` (SOI transitions)

Model interplanetary flight as a chain of two-body arcs, switching central body at SOI crossings.
```python
def dominant_body(pos_m, t_sim_s, bodies) -> CelestialBody:
    """Return the body whose SOI currently contains pos (smallest enclosing SOI; default Sun)."""
def propagate_patched(state, dt, bodies) -> StateVector:
    """Two-body propagate in the current body's frame; when the vessel crosses an SOI boundary,
    re-express state relative to the new central body (frame shift using ephemeris) and continue.
    Detect the crossing by bisection on |pos - body_pos| == soi_radius within the step."""
```
Frame shifts must use `ephemeris.body_state` for the body's position/velocity at the crossing time
and add/subtract it (Galilean shift). Keep everything float64.

**Tests:**
- Frame-shift round-trip: shifting a state Earth→Sun→Earth returns the original within 1 m / 1 mm/s.
- A hyperbolic Earth-escape state propagated forward crosses Earth's SOI exactly once; after the
  handoff its heliocentric orbit is bound and sensible (energy < 0 about the Sun).

## Task 5.4 — interplanetary planning

Extend the Phase-4 porkchop/Lambert to heliocentric departure/arrival using `ephemeris` planet
states (e.g. Earth→Mars windows). The optimizer now searches real launch dates. Patched-conic ΔV
includes hyperbolic departure (C3) and arrival capture burns.

**Test:** an Earth→Mars porkchop over 2030–2033 shows the known ~26-month synodic window structure
and a minimum total ΔV in a physically plausible band (rough magnitude check, not exact).

## Task 5.5 — rendering at solar-system scale

The Phase-2 floating origin already handles the scale; now feed planet positions from
`ephemeris.body_state` each frame, draw planet orbit lines, and let the camera focus any body.
Verify no jitter when focused on a vessel near Mars while the Sun is 2.3e11 m away.

## Phase 5 exit criteria
- Real planets sit at correct positions/dates; focusing/zooming across the system has no jitter.
- A vessel can fly Earth→Mars via patched conics, with SOI handoffs occurring at the right places.
- Ephemeris sanity tests and patched-conic frame-shift tests are green.
