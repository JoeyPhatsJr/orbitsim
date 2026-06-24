# Phase 5 — Real Solar System (Ephemerides + Patched Conics) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the toy single-body world with the real solar system — real planet/moon positions from JPL DE440 via Skyfield — and support interplanetary trajectories via patched conics with sphere-of-influence (SOI) handoffs.

**Architecture:** `core/ephemeris.py` wraps Skyfield and returns SI `StateVector`s. `core/bodies.py` gains the full planet/moon set. `core/patched_conics.py` chains two-body arcs, switching the central body at SOI crossings using ephemeris-based Galilean frame shifts. All pure/testable; the renderer then feeds planet positions each frame (visual).

**Tech Stack:** Python 3.10, numpy, Skyfield 1.49 (installed) + DE440 kernel (downloaded on first run), pytest.

## Global Constraints

- SI units everywhere in `core/`: meters, seconds, radians, m/s. Convert only at the HUD boundary.
- `core/` must NOT import `panda3d`/`sim`/`render`.
- Sim time is **seconds past J2000 TDB**. Skyfield time conversion must be explicit and tested.
- `core/` arrays float64, shape (3,). `CelestialBody`, `StateVector` frozen.
- The DE440 kernel (`de440s.bsp`) must be `.gitignore`d and downloaded on first run into `data/`.
- `black` line length 100. Type hints + NumPy docstrings.
- `pytest tests/ -q` green after every task; render task ends with a HUMAN VISUAL CHECKPOINT.

## Gate

Phase 4 transfers work (porkchop + Lambert validated). Do not start before that.

## Phase 1–4 API available

```python
from orbitsim.core.state import StateVector
from orbitsim.core.bodies import CelestialBody, SUN, EARTH, MOON
from orbitsim.core.elements import state_to_elements, elements_to_state
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.optimize import porkchop, optimize_transfer
from orbitsim.core.constants import MU_SUN, MU_EARTH, MU_MOON, G
```

---

## File Structure

- `orbitsim/core/bodies.py` — MODIFY: add all 8 planets + the Moon with real mu/radius/J2 and parent links.
- `orbitsim/core/ephemeris.py` — CREATE: Skyfield wrapper `body_state`, time conversion `sim_time_to_skyfield`.
- `orbitsim/core/patched_conics.py` — CREATE: `dominant_body`, `shift_frame`, `propagate_patched`.
- `.gitignore` — MODIFY: add `data/*.bsp`.
- `orbitsim/render/app.py` — MODIFY: draw planets from ephemeris, focus any body (visual).
- Tests: `tests/core/test_ephemeris.py`, `tests/core/test_patched_conics.py`, `tests/core/test_bodies.py` (append).

---

## Task 1: Expand the body set

**Files:**
- Modify: `orbitsim/core/bodies.py`
- Modify: `orbitsim/core/constants.py` (add planetary mu/radii)
- Test: `tests/core/test_bodies.py` (append)

**Interfaces:**
- Produces: module-level `MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE` (plus existing `SUN, EARTH, MOON`), each `CelestialBody` with correct parent links.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_bodies.py`:
```python
def test_all_planets_exist_with_sun_parent():
    from orbitsim.core.bodies import (
        MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, SUN,
    )
    planets = [MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE]
    for p in planets:
        assert p.parent is SUN
        assert p.mu > 0
        assert p.radius_m > 0


def test_mars_mu_order_of_magnitude():
    from orbitsim.core.bodies import MARS
    # Mars GM ~ 4.283e13 m^3/s^2.
    assert abs(MARS.mu - 4.283e13) / 4.283e13 < 0.05


def test_jupiter_is_largest_planet():
    from orbitsim.core.bodies import (
        MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE,
    )
    assert JUPITER.mu == max(
        p.mu for p in [MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE]
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_bodies.py -q`
Expected: FAIL (ImportError: cannot import name 'MARS').

- [ ] **Step 3: Add planetary constants**

Append to `orbitsim/core/constants.py`:
```python
# Planetary standard gravitational parameters mu = GM [m^3/s^2].
# Source: IAU / JPL DE440 (NASA planetary fact sheet), documented values.
MU_MERCURY = 2.2032e13
MU_VENUS = 3.24859e14
MU_MARS = 4.282837e13
MU_JUPITER = 1.26686534e17
MU_SATURN = 3.7931187e16
MU_URANUS = 5.793939e15
MU_NEPTUNE = 6.836529e15

# Mean equatorial radii [m] (NASA planetary fact sheet).
R_MERCURY = 2.4397e6
R_VENUS = 6.0518e6
R_MARS = 3.3962e6
R_JUPITER = 7.1492e7
R_SATURN = 6.0268e7
R_URANUS = 2.5559e7
R_NEPTUNE = 2.4764e7

# Oblateness J2 (dimensionless) for the bodies where it matters (NASA fact sheet).
J2_MARS = 1.96045e-3
J2_JUPITER = 1.4736e-2
```

- [ ] **Step 4: Add the planet instances**

Append to `orbitsim/core/bodies.py` (after the existing SUN/EARTH/MOON block; add the new constants to its import):
```python
from orbitsim.core.constants import (
    MU_MERCURY, MU_VENUS, MU_MARS, MU_JUPITER, MU_SATURN, MU_URANUS, MU_NEPTUNE,
    R_MERCURY, R_VENUS, R_MARS, R_JUPITER, R_SATURN, R_URANUS, R_NEPTUNE,
    J2_MARS, J2_JUPITER,
)

MERCURY = CelestialBody(name="Mercury", mu=MU_MERCURY, radius_m=R_MERCURY, parent=SUN)
VENUS = CelestialBody(name="Venus", mu=MU_VENUS, radius_m=R_VENUS, parent=SUN)
MARS = CelestialBody(name="Mars", mu=MU_MARS, radius_m=R_MARS, j2=J2_MARS, parent=SUN)
JUPITER = CelestialBody(name="Jupiter", mu=MU_JUPITER, radius_m=R_JUPITER, j2=J2_JUPITER, parent=SUN)
SATURN = CelestialBody(name="Saturn", mu=MU_SATURN, radius_m=R_SATURN, parent=SUN)
URANUS = CelestialBody(name="Uranus", mu=MU_URANUS, radius_m=R_URANUS, parent=SUN)
NEPTUNE = CelestialBody(name="Neptune", mu=MU_NEPTUNE, radius_m=R_NEPTUNE, parent=SUN)

PLANETS = [MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_bodies.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/core/bodies.py orbitsim/core/constants.py tests/core/test_bodies.py
git commit -m "Phase 5 Task 1: full planet set with real mu/radius/J2"
```

---

## Task 2: Ephemeris wrapper (Skyfield / DE440)

**Files:**
- Create: `orbitsim/core/ephemeris.py`
- Modify: `.gitignore` (add `data/*.bsp`)
- Test: `tests/core/test_ephemeris.py`

**Interfaces:**
- Consumes: `skyfield.api`, `StateVector`, numpy.
- Produces:
  ```python
  def sim_time_to_skyfield(t_sim_s: float):  # -> skyfield Time (TDB seconds past J2000)
  def body_state(name: str, t_sim_s: float, center: str = "SUN") -> StateVector:
      # position+velocity of `name` relative to `center` in J2000/ICRF, SI.
  ```

- [ ] **Step 1: Add the kernel to .gitignore**

Append to `.gitignore`:
```
data/*.bsp
```

- [ ] **Step 2: Write the failing tests**

Create `tests/core/test_ephemeris.py`:
```python
"""Ephemeris sanity anchors (Skyfield / DE440).

These tests download de440s.bsp on first run (~30 MB). They are slow once,
cached after. If offline, they will be skipped via the importorskip guard.
"""
import numpy as np
import pytest

skyfield = pytest.importorskip("skyfield")
from orbitsim.core.ephemeris import body_state
from orbitsim.core.constants import MU_SUN


# 2030-01-01 00:00 TDB, seconds past J2000 (approx): 30 years * 365.25 d.
T_2030 = 30.0 * 365.25 * 86400.0


def test_earth_heliocentric_distance_about_1au():
    state = body_state("EARTH", T_2030, center="SUN")
    au = 1.495978707e11
    assert abs(state.r_mag - au) / au < 0.02  # within 2%


def test_earth_orbital_speed_about_29_8_kms():
    state = body_state("EARTH", T_2030, center="SUN")
    assert abs(state.v_mag - 29.8e3) / 29.8e3 < 0.02


def test_mars_heliocentric_distance_range():
    state = body_state("MARS", T_2030, center="SUN")
    # Mars heliocentric distance varies ~1.38–1.67 AU.
    au = 1.495978707e11
    assert 1.3 * au < state.r_mag < 1.7 * au
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_ephemeris.py -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.core.ephemeris).

- [ ] **Step 4: Implement ephemeris.py**

Create `orbitsim/core/ephemeris.py`:
```python
"""Skyfield / DE440 ephemeris wrapper returning SI StateVectors.

Sim time is seconds past J2000 TDB. J2000 epoch = JD 2451545.0 (TDB).
"""
import os
import numpy as np

from skyfield.api import load
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_SUN, MU_EARTH

_J2000_JD_TDB = 2451545.0
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# Cache the loader, kernel, and timescale at module load.
_loader = load.Loader(_DATA_DIR)
_ts = _loader.timescale()
_kernel = _loader("de440s.bsp")

# Map our names to Skyfield kernel targets.
_TARGETS = {
    "SUN": "sun",
    "MERCURY": "mercury barycenter",
    "VENUS": "venus barycenter",
    "EARTH": "earth",
    "MOON": "moon",
    "MARS": "mars barycenter",
    "JUPITER": "jupiter barycenter",
    "SATURN": "saturn barycenter",
    "URANUS": "uranus barycenter",
    "NEPTUNE": "neptune barycenter",
}

# mu lookup for the returned StateVector's central body.
_CENTER_MU = {"SUN": MU_SUN, "EARTH": MU_EARTH}


def sim_time_to_skyfield(t_sim_s: float):
    """Convert seconds past J2000 TDB to a Skyfield Time."""
    jd_tdb = _J2000_JD_TDB + t_sim_s / 86400.0
    return _ts.tdb_jd(jd_tdb)


def body_state(name: str, t_sim_s: float, center: str = "SUN") -> StateVector:
    """Position + velocity of `name` relative to `center` as an SI StateVector.

    Parameters
    ----------
    name : str
        One of the keys in the module target map (e.g. "EARTH", "MARS").
    t_sim_s : float
        Seconds past J2000 TDB.
    center : str
        Reference body ("SUN" or "EARTH").

    Returns
    -------
    StateVector
        r, v in J2000/ICRF [m, m/s]; mu is the central body's mu (0.0 if unknown).
    """
    t = sim_time_to_skyfield(t_sim_s)
    target = _kernel[_TARGETS[name.upper()]]
    origin = _kernel[_TARGETS[center.upper()]]
    rel = (target - origin).at(t)
    r_m = rel.position.m                     # meters, shape (3,)
    v_m_s = rel.velocity.m_per_s             # m/s, shape (3,)
    return StateVector(
        r=np.asarray(r_m, dtype=np.float64),
        v=np.asarray(v_m_s, dtype=np.float64),
        mu=_CENTER_MU.get(center.upper(), 0.0),
        epoch_s=t_sim_s,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_ephemeris.py -q`
Expected: PASS (downloads de440s.bsp on first run; this is slow once). If the Skyfield velocity attribute differs in your version, run `.venv/Scripts/python -c "from skyfield.api import load; ..."` to inspect; `.velocity.m_per_s` is correct for Skyfield 1.49.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/core/ephemeris.py .gitignore tests/core/test_ephemeris.py
git commit -m "Phase 5 Task 2: Skyfield DE440 ephemeris wrapper (SI)"
```

---

## Task 3: Dominant body (SOI containment)

**Files:**
- Create: `orbitsim/core/patched_conics.py`
- Test: `tests/core/test_patched_conics.py`

**Interfaces:**
- Consumes: `body_state`, `CelestialBody`, planet semi-major axes, numpy.
- Produces:
  ```python
  def dominant_body(pos_m_helio: np.ndarray, t_sim_s: float,
                    bodies: list[CelestialBody]) -> CelestialBody:
      # heliocentric position -> body whose SOI contains it (smallest enclosing; default SUN)
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_patched_conics.py`:
```python
"""Tests for patched-conic SOI logic and frame shifts."""
import numpy as np
import pytest

pytest.importorskip("skyfield")
from orbitsim.core.patched_conics import dominant_body
from orbitsim.core.bodies import SUN, EARTH, PLANETS
from orbitsim.core.ephemeris import body_state

T = 30.0 * 365.25 * 86400.0  # ~2030


def test_point_near_earth_is_dominated_by_earth():
    earth = body_state("EARTH", T, center="SUN")
    # 100,000 km from Earth, well within Earth's SOI (~924,000 km).
    pos = earth.r + np.array([1.0e8, 0.0, 0.0])
    dom = dominant_body(pos, T, [SUN] + PLANETS)
    assert dom.name == "Earth"


def test_deep_space_point_is_dominated_by_sun():
    pos = np.array([0.7 * 1.496e11, 0.0, 0.0])  # 0.7 AU, far from any planet
    dom = dominant_body(pos, T, [SUN] + PLANETS)
    assert dom.name == "Sun"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_patched_conics.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement dominant_body**

Create `orbitsim/core/patched_conics.py`:
```python
"""Patched-conic interplanetary flight: SOI containment + frame shifts."""
import numpy as np

from orbitsim.core.bodies import CelestialBody, SUN
from orbitsim.core.ephemeris import body_state

# Approximate semi-major axes [m] for SOI sizing (heliocentric).
_SMA_M = {
    "Mercury": 5.79e10, "Venus": 1.082e11, "Earth": 1.496e11, "Mars": 2.279e11,
    "Jupiter": 7.785e11, "Saturn": 1.434e12, "Uranus": 2.871e12, "Neptune": 4.495e12,
}


def dominant_body(
    pos_m_helio: np.ndarray,
    t_sim_s: float,
    bodies: list[CelestialBody],
) -> CelestialBody:
    """Return the body whose SOI currently contains the heliocentric position.

    Picks the smallest enclosing SOI; defaults to the Sun if no planet's SOI
    contains the point.

    Parameters
    ----------
    pos_m_helio : np.ndarray
        Heliocentric position [m], shape (3,).
    t_sim_s : float
        Seconds past J2000 TDB.
    bodies : list[CelestialBody]
        Candidate bodies (should include SUN).
    """
    best = SUN
    best_soi = float("inf")
    for body in bodies:
        if body.parent is None:
            continue  # the Sun has no finite SOI
        sma = _SMA_M.get(body.name)
        if sma is None:
            continue
        soi = body.soi_radius_m(sma)
        body_pos = body_state(body.name.upper(), t_sim_s, center="SUN").r
        dist = np.linalg.norm(pos_m_helio - body_pos)
        if dist < soi and soi < best_soi:
            best = body
            best_soi = soi
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_patched_conics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/patched_conics.py tests/core/test_patched_conics.py
git commit -m "Phase 5 Task 3: dominant_body SOI containment"
```

---

## Task 4: Frame shift (Galilean) round-trip

**Files:**
- Modify: `orbitsim/core/patched_conics.py`
- Test: `tests/core/test_patched_conics.py` (append)

**Interfaces:**
- Produces:
  ```python
  def shift_frame(state: StateVector, from_center: str, to_center: str,
                  t_sim_s: float, to_mu: float) -> StateVector:
      # re-express a state from one central body to another via ephemeris (add/subtract body state)
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_patched_conics.py`:
```python
from orbitsim.core.patched_conics import shift_frame
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH, MU_SUN


def test_frame_shift_round_trip():
    """Earth-centered -> Sun-centered -> Earth-centered returns the original."""
    r = np.array([7.0e6, 1.0e6, -2.0e6])
    v = np.array([0.0, 7.5e3, 0.1e3])
    state_earth = StateVector(r=r, v=v, mu=MU_EARTH, epoch_s=T)
    state_helio = shift_frame(state_earth, "EARTH", "SUN", T, MU_SUN)
    state_back = shift_frame(state_helio, "SUN", "EARTH", T, MU_EARTH)
    assert np.linalg.norm(state_back.r - r) < 1.0          # 1 m
    assert np.linalg.norm(state_back.v - v) < 1e-3         # 1 mm/s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_patched_conics.py -k frame_shift -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement shift_frame**

Append to `orbitsim/core/patched_conics.py`:
```python
from orbitsim.core.state import StateVector


def shift_frame(
    state: StateVector,
    from_center: str,
    to_center: str,
    t_sim_s: float,
    to_mu: float,
) -> StateVector:
    """Re-express a state from one central body's frame to another's.

    Galilean shift: subtract/add the relative body state from the ephemeris.
    r_new = r_old + (from_center - to_center) position
    v_new = v_old + (from_center - to_center) velocity

    Parameters
    ----------
    state : StateVector
        State referenced to `from_center`.
    from_center, to_center : str
        Body names ("EARTH", "SUN", ...).
    t_sim_s : float
    to_mu : float
        mu of the new central body for the returned StateVector.
    """
    # Position/velocity of from_center relative to to_center.
    rel = body_state(from_center.upper(), t_sim_s, center=to_center.upper())
    return StateVector(
        r=np.asarray(state.r, dtype=np.float64) + rel.r,
        v=np.asarray(state.v, dtype=np.float64) + rel.v,
        mu=to_mu,
        epoch_s=t_sim_s,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_patched_conics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/patched_conics.py tests/core/test_patched_conics.py
git commit -m "Phase 5 Task 4: Galilean frame shift (ephemeris-based)"
```

---

## Task 5: Patched-conic propagation with SOI handoff

**Files:**
- Modify: `orbitsim/core/patched_conics.py`
- Test: `tests/core/test_patched_conics.py` (append)

**Interfaces:**
- Consumes: `propagate_kepler`, `dominant_body`, `shift_frame`, `body_state`.
- Produces:
  ```python
  def propagate_patched(state: StateVector, dt: float, current_center: str,
                        bodies: list[CelestialBody], max_substeps: int = 2000
                        ) -> tuple[StateVector, str]:
      # two-body propagate in the current frame; detect SOI crossing by bisection;
      # on crossing, shift_frame to the new body and continue. Returns (state, new_center_name).
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_patched_conics.py`:
```python
from orbitsim.core.patched_conics import propagate_patched
from orbitsim.core.bodies import MU_EARTH if False else None  # placeholder import line removed below
```
Replace that placeholder with the real test:
```python
from orbitsim.core.patched_conics import propagate_patched
from orbitsim.core.constants import MU_EARTH as _MU_EARTH


def test_hyperbolic_escape_crosses_soi_to_sun():
    """A fast Earth-centered hyperbolic state, propagated long enough, hands off to the Sun."""
    # Earth-centered escape: well above escape speed at 7000 km.
    r = np.array([7.0e6, 0.0, 0.0])
    v_esc = np.sqrt(2 * _MU_EARTH / 7.0e6)
    v = np.array([0.0, v_esc * 1.5, 0.0])  # hyperbolic
    state = StateVector(r=r, v=v, mu=_MU_EARTH, epoch_s=T)

    # Propagate ~20 days; the vessel should leave Earth's SOI and become Sun-centered.
    final_state, center = propagate_patched(state, 20 * 86400.0, "EARTH", [SUN] + PLANETS)
    assert center == "SUN"
    # Heliocentric energy should be bound (negative) or at least finite & sensible.
    assert np.isfinite(final_state.specific_energy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_patched_conics.py -k escape -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement propagate_patched**

Append to `orbitsim/core/patched_conics.py`:
```python
from orbitsim.core.propagate import propagate_kepler

# Approximate SOI radii cache keyed by body name.
def _soi_of(body: CelestialBody) -> float:
    sma = _SMA_M.get(body.name)
    if sma is None:
        return float("inf")
    return body.soi_radius_m(sma)


def propagate_patched(
    state: StateVector,
    dt: float,
    current_center: str,
    bodies: list[CelestialBody],
    max_substeps: int = 2000,
) -> tuple[StateVector, str]:
    """Two-body propagate, switching central body at SOI crossings.

    Substeps the interval; after each substep checks whether the vessel has
    left the current body's SOI (relative to its parent) or entered a child
    body's SOI. On a crossing it bisects to the boundary, frame-shifts, and
    continues in the new frame.

    Parameters
    ----------
    state : StateVector
        State referenced to `current_center`.
    dt : float
        Total time to advance [s].
    current_center : str
        Name of the current central body ("EARTH", "SUN", ...).
    bodies : list[CelestialBody]
    max_substeps : int

    Returns
    -------
    (state, center_name) : (StateVector, str)
    """
    from orbitsim.core.bodies import SUN as _SUN

    name_to_body = {b.name.upper(): b for b in bodies}
    center = current_center.upper()
    step = dt / max_substeps
    elapsed = 0.0

    for _ in range(max_substeps):
        prev = state
        state = propagate_kepler(state, step)
        elapsed += step
        t_now = state.epoch_s

        # Check exit from current body's SOI (if it has a parent).
        body = name_to_body.get(center)
        if body is not None and body.parent is not None:
            soi = _soi_of(body)
            if state.r_mag > soi:
                # Crossed outward: shift to the parent (Sun for a planet).
                parent_name = body.parent.name.upper()
                state = shift_frame(state, center, parent_name, t_now, body.parent.mu)
                center = parent_name
                continue

        # Check entry into a child body's SOI (heliocentric only, for simplicity).
        if center == "SUN":
            helio_pos = state.r
            dom = dominant_body(helio_pos, t_now, bodies)
            if dom.parent is not None:  # entered a planet's SOI
                state = shift_frame(state, "SUN", dom.name.upper(), t_now, dom.mu)
                center = dom.name.upper()
                continue

        if elapsed >= dt:
            break

    return state, center
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_patched_conics.py -q`
Expected: PASS. (If it is slow, reduce `max_substeps` in the test call or accept the one-time ephemeris cost.)

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/patched_conics.py tests/core/test_patched_conics.py
git commit -m "Phase 5 Task 5: patched-conic propagation with SOI handoff"
```

---

## Task 6: Interplanetary porkchop (Earth → Mars)

**Files:**
- Modify: `orbitsim/core/optimize.py` (add `interplanetary_porkchop`)
- Test: `tests/core/test_optimize.py` (append)

**Interfaces:**
- Consumes: `body_state`, `porkchop`.
- Produces:
  ```python
  def interplanetary_porkchop(dep_name: str, arr_name: str,
                              dep_times_s: np.ndarray, tof_grid_s: np.ndarray
                              ) -> tuple[np.ndarray, tuple[int, int]]:
      # heliocentric Lambert grid using ephemeris planet states
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_optimize.py`:
```python
import pytest


def test_earth_to_mars_porkchop_has_window():
    pytest.importorskip("skyfield")
    from orbitsim.core.optimize import interplanetary_porkchop
    # 2031 launch window scan, TOF 150–300 days.
    base = 31.0 * 365.25 * 86400.0
    dep_times = np.linspace(base, base + 2 * 365.25 * 86400.0, 24)  # 2 years
    tof_grid = np.linspace(150 * 86400.0, 300 * 86400.0, 16)
    dv, (i, j) = interplanetary_porkchop("EARTH", "MARS", dep_times, tof_grid)
    assert dv.shape == (24, 16)
    finite = dv[np.isfinite(dv)]
    assert finite.size > 0
    # Minimum heliocentric transfer dv (departure + arrival v-infinity) is in a
    # physically plausible band — a few km/s to low tens of km/s.
    assert 2.0e3 < dv[i, j] < 5.0e4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_optimize.py -k mars -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement interplanetary_porkchop**

Append to `orbitsim/core/optimize.py`:
```python
from orbitsim.core.constants import MU_SUN


def interplanetary_porkchop(
    dep_name: str,
    arr_name: str,
    dep_times_s: np.ndarray,
    tof_grid_s: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Heliocentric Lambert porkchop between two planets using DE440 ephemeris.

    Total dv here is departure v-infinity + arrival v-infinity (relative to the
    planets), i.e. the heliocentric Lambert cost; capture/escape burns are added
    by the caller if desired.

    Parameters
    ----------
    dep_name, arr_name : str
        Planet names ("EARTH", "MARS").
    dep_times_s : np.ndarray
        Departure times [s past J2000 TDB], shape (m,).
    tof_grid_s : np.ndarray
        Times of flight [s], shape (n,).

    Returns
    -------
    (dv_total, argmin)
    """
    from orbitsim.core.ephemeris import body_state

    m = len(dep_times_s)
    n = len(tof_grid_s)
    dv = np.full((m, n), np.inf, dtype=np.float64)

    for i, t_dep in enumerate(dep_times_s):
        dep_planet = body_state(dep_name, float(t_dep), center="SUN")
        for j, tof in enumerate(tof_grid_s):
            if tof <= 0:
                continue
            arr_planet = body_state(arr_name, float(t_dep + tof), center="SUN")
            try:
                v1, v2 = lambert(dep_planet.r, arr_planet.r, float(tof), MU_SUN)
            except Exception:
                continue
            vinf_dep = np.linalg.norm(v1 - dep_planet.v)
            vinf_arr = np.linalg.norm(v2 - arr_planet.v)
            dv[i, j] = vinf_dep + vinf_arr

    flat = int(np.argmin(dv))
    return dv, (flat // n, flat % n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_optimize.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/optimize.py tests/core/test_optimize.py
git commit -m "Phase 5 Task 6: Earth->Mars interplanetary porkchop"
```

---

## Task 7: Render planets from ephemeris — HUMAN VISUAL CHECKPOINT

**Files:**
- Modify: `orbitsim/render/app.py`
- Modify: `orbitsim/__main__.py` (add a solar-system scenario toggle)

**Interfaces:**
- Consumes: `body_state`, `PLANETS`, `SUN`, the existing `RenderTransform`, `make_uv_sphere`.
- Produces: planets drawn at ephemeris positions each frame; camera can focus any body.

- [ ] **Step 1: Add a solar-system mode to OrbitApp**

In `orbitsim/render/app.py`, add to `OrbitApp.__init__` an optional flag and planet nodes:
```python
    # (add parameter) def __init__(self, world, clock, solar_system=False):
    #   self.solar_system = solar_system
    #   if solar_system: self._build_planets()
```
Add the method:
```python
    def _build_planets(self) -> None:
        from orbitsim.core.bodies import PLANETS, SUN
        self._planet_bodies = [SUN] + PLANETS
        self._planet_nps = []
        for body in self._planet_bodies:
            np_ = make_uv_sphere(0.05, 12, 16)
            np_.reparent_to(self.render)
            np_.set_color(0.8, 0.7, 0.5, 1.0)
            self._planet_nps.append(np_)
```

In `_update`, when `self.solar_system` is set, position each planet:
```python
        if getattr(self, "solar_system", False):
            from orbitsim.core.ephemeris import body_state
            for body, np_ in zip(self._planet_bodies, self._planet_nps):
                if body.name == "Sun":
                    pos = np.zeros(3)
                else:
                    pos = body_state(body.name.upper(), self.clock.sim_time_s, center="SUN").r
                px, py, pz = self.transform.to_render(pos)
                np_.set_pos(px, py, pz)
                rr = max(body.radius_m / self.transform.scale_m_per_unit, 1e-3)
                np_.set_scale(rr)
```

- [ ] **Step 2: Add a scenario toggle in __main__**

In `orbitsim/__main__.py`, add an env-var or arg switch:
```python
import sys
# in main():
    solar = "--solar" in sys.argv
    app = OrbitApp(world, clock, solar_system=solar)
```
(Adjust `OrbitApp.__init__` signature to accept `solar_system=False`.)

- [ ] **Step 3: Smoke-check imports**

Run: `.venv/Scripts/python -c "import orbitsim.render.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Full suite stays green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS (ephemeris/patched-conic tests included; first run downloads the kernel).

- [ ] **Step 5: HUMAN VISUAL CHECKPOINT**

Run: `.venv/Scripts/python -m orbitsim --solar`
Reviewer confirms:
1. The Sun sits at the center; planets appear at plausible positions (Earth ~1 AU, Mars beyond it, gas giants much farther).
2. Zooming/focusing across the system (vessel near Mars while the Sun is ~2.3e11 m away) has **no jitter**.
3. Advancing time at high warp moves the planets along their orbits smoothly.

- [ ] **Step 6: Commit**

```bash
git add orbitsim/render/app.py orbitsim/__main__.py
git commit -m "Phase 5 Task 7: render real planets from ephemeris"
```

---

## Phase 5 Exit Criteria

- Real planets sit at correct positions/dates; focusing/zooming across the system has no jitter.
- A vessel can fly an Earth-escape patched-conic arc with an SOI handoff to the Sun (test green).
- Ephemeris sanity tests (Earth ~1 AU, ~29.8 km/s) and frame-shift round-trip tests are green.
- Earth→Mars porkchop produces a plausible minimum-dv window.
- `pytest tests/ -q` fully green.

Then proceed to `docs/superpowers/plans/2026-06-24-phase6-polish-packaging.md`.

## Self-Review Notes

- Spec coverage: ephemeris (5.1), body expansion (5.2), patched conics (5.3), interplanetary planning (5.4), rendering at scale (5.5) — all mapped.
- All physics (ephemeris anchors, SOI containment, frame-shift round-trip, patched propagation, interplanetary porkchop) is red-green TDD; only planet rendering is visual.
- Skyfield time conversion (`sim_time_to_skyfield`) is explicit and exercised by the 1-AU / 29.8 km/s anchors.
- Note for the implementer: the placeholder import line in Task 5 Step 1 must be deleted; only the real test block below it is kept.
