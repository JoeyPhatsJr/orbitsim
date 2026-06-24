# Phase 6 — Polish, Save/Load & Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the working simulator into a shippable Windows desktop app for the author + a friend: JSON save/load, built-in scenarios, HUD/UX polish, a performance pass, and a PyInstaller build.

**Architecture:** `sim/persistence.py` serializes a `World` + `SimClock` to versioned JSON (round-trips to bit-identical state vectors). The render layer gets HUD polish and quality-of-life. PyInstaller bundles it into a one-folder Windows build with the DE440 kernel.

**Tech Stack:** Python 3.10, numpy, Panda3D, PyInstaller (install in Task 5), pytest.

## Global Constraints

- SI units in `core/`/`sim/`; convert only at the HUD boundary.
- `core/` must NOT import `panda3d`/`sim`/`render`. `sim/persistence.py` may import `core/` and `sim/`.
- JSON schema is versioned (`"schema": 1`). Save→load must reproduce state vectors bit-identically (within float repr round-trip).
- `black` line length 100. Type hints + NumPy docstrings.
- `pytest tests/ -q` green after every task; UX/packaging tasks end with a HUMAN VISUAL/MANUAL CHECKPOINT.

## Gate

Phase 5 interplanetary flight works (patched conics + ephemeris rendering verified). Do not start before that.

## Phase 1–5 API available

```python
from orbitsim.sim.world import Vessel, World
from orbitsim.sim.clock import SimClock
from orbitsim.core.state import StateVector
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.bodies import EARTH, SUN, MARS, PLANETS  # plus by-name lookup you will add
from orbitsim.core.constants import MU_EARTH
```

---

## File Structure

- `orbitsim/core/bodies.py` — MODIFY: add a `by_name(name: str) -> CelestialBody` registry helper.
- `orbitsim/sim/persistence.py` — CREATE: `save_scenario`, `load_scenario`, schema constants.
- `orbitsim/scenarios/` — CREATE: built-in scenario JSON files (`leo_sandbox.json`, `leo_to_geo.json`, `earth_to_mars.json`) + a builder script that generates them.
- `orbitsim/render/app.py` — MODIFY: HUD polish, save/load keys, pause-on-SOI, auto-warp-down (visual).
- `README.md`, `CONTROLS.md` — docs.
- `orbitsim.spec` — CREATE: PyInstaller spec.
- Tests: `tests/sim/test_persistence.py`, `tests/core/test_bodies.py` (append).

---

## Task 1: Body registry lookup

**Files:**
- Modify: `orbitsim/core/bodies.py`
- Test: `tests/core/test_bodies.py` (append)

**Interfaces:**
- Produces: `by_name(name: str) -> CelestialBody` (case-insensitive; raises `KeyError` if unknown).

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_bodies.py`:
```python
def test_by_name_lookup():
    from orbitsim.core.bodies import by_name, EARTH, MARS
    assert by_name("Earth") is EARTH
    assert by_name("earth") is EARTH
    assert by_name("MARS") is MARS
    import pytest
    with pytest.raises(KeyError):
        by_name("Pluto")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/core/test_bodies.py -k by_name -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement by_name**

Append to `orbitsim/core/bodies.py` (after all instances and `PLANETS` are defined):
```python
_REGISTRY = {
    b.name.upper(): b
    for b in [SUN, MERCURY, VENUS, EARTH, MOON, MARS, JUPITER, SATURN, URANUS, NEPTUNE]
}


def by_name(name: str) -> "CelestialBody":
    """Look up a pre-built body by name (case-insensitive).

    Raises
    ------
    KeyError
        If no body with that name exists.
    """
    return _REGISTRY[name.upper()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/core/test_bodies.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/core/bodies.py tests/core/test_bodies.py
git commit -m "Phase 6 Task 1: by_name body registry lookup"
```

---

## Task 2: Save/load scenarios (JSON)

**Files:**
- Create: `orbitsim/sim/persistence.py`
- Test: `tests/sim/test_persistence.py`

**Interfaces:**
- Consumes: `World`, `Vessel`, `SimClock`, `StateVector`, `ManeuverNode`, `by_name`, numpy, json.
- Produces:
  ```python
  SCHEMA_VERSION = 1
  def save_scenario(world: World, clock: SimClock, path: str) -> None: ...
  def load_scenario(path: str) -> tuple[World, SimClock]: ...
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/sim/test_persistence.py`:
```python
"""Tests for JSON scenario save/load round-trip."""
import numpy as np
from orbitsim.sim.persistence import save_scenario, load_scenario, SCHEMA_VERSION
from orbitsim.sim.world import Vessel, World
from orbitsim.sim.clock import SimClock
from orbitsim.core.state import StateVector
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.bodies import EARTH
from orbitsim.core.constants import MU_EARTH


def _world() -> World:
    r = np.array([7.0e6, 1.0e5, -2.0e5])
    v = np.array([10.0, 7.5e3, 30.0])
    state = StateVector(r=r, v=v, mu=MU_EARTH, epoch_s=123.0)
    vessel = Vessel(name="Test-1", state=state, delta_v_budget_mps=1500.0)
    vessel.nodes.append(
        ManeuverNode(epoch_s=500.0, dv_prograde_mps=12.0, dv_normal_mps=-3.0, dv_radial_mps=0.5)
    )
    return World(central=EARTH, vessels=[vessel])


def test_round_trip_state_vectors(tmp_path):
    world = _world()
    clock = SimClock(sim_time_s=999.0, warp=100.0)
    path = str(tmp_path / "scenario.json")
    save_scenario(world, clock, path)
    world2, clock2 = load_scenario(path)

    np.testing.assert_array_equal(world2.vessels[0].state.r, world.vessels[0].state.r)
    np.testing.assert_array_equal(world2.vessels[0].state.v, world.vessels[0].state.v)
    assert world2.vessels[0].state.epoch_s == world.vessels[0].state.epoch_s
    assert world2.central.name == "Earth"
    assert clock2.sim_time_s == 999.0
    assert clock2.warp == 100.0


def test_round_trip_nodes_and_budget(tmp_path):
    world = _world()
    clock = SimClock()
    path = str(tmp_path / "s.json")
    save_scenario(world, clock, path)
    world2, _ = load_scenario(path)
    v = world2.vessels[0]
    assert v.delta_v_budget_mps == 1500.0
    assert v.nodes[0].dv_prograde_mps == 12.0
    assert v.nodes[0].dv_normal_mps == -3.0


def test_schema_version_written(tmp_path):
    import json
    path = str(tmp_path / "s.json")
    save_scenario(_world(), SimClock(), path)
    with open(path) as f:
        data = json.load(f)
    assert data["schema"] == SCHEMA_VERSION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/sim/test_persistence.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement persistence.py**

Create `orbitsim/sim/persistence.py`:
```python
"""Versioned JSON save/load for scenarios (World + SimClock)."""
import json
import numpy as np

from orbitsim.sim.world import Vessel, World
from orbitsim.sim.clock import SimClock
from orbitsim.core.state import StateVector
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.bodies import by_name

SCHEMA_VERSION = 1


def _vessel_to_dict(v: Vessel) -> dict:
    return {
        "name": v.name,
        "delta_v_budget_mps": v.delta_v_budget_mps,
        "state": {
            "r": v.state.r.tolist(),
            "v": v.state.v.tolist(),
            "mu": v.state.mu,
            "epoch_s": v.state.epoch_s,
        },
        "nodes": [
            {
                "epoch_s": n.epoch_s,
                "dv_prograde_mps": n.dv_prograde_mps,
                "dv_normal_mps": n.dv_normal_mps,
                "dv_radial_mps": n.dv_radial_mps,
            }
            for n in v.nodes
        ],
    }


def _vessel_from_dict(d: dict) -> Vessel:
    s = d["state"]
    state = StateVector(
        r=np.array(s["r"], dtype=np.float64),
        v=np.array(s["v"], dtype=np.float64),
        mu=s["mu"],
        epoch_s=s["epoch_s"],
    )
    vessel = Vessel(name=d["name"], state=state, delta_v_budget_mps=d["delta_v_budget_mps"])
    for n in d.get("nodes", []):
        vessel.nodes.append(
            ManeuverNode(
                epoch_s=n["epoch_s"],
                dv_prograde_mps=n["dv_prograde_mps"],
                dv_normal_mps=n["dv_normal_mps"],
                dv_radial_mps=n["dv_radial_mps"],
            )
        )
    return vessel


def save_scenario(world: World, clock: SimClock, path: str) -> None:
    """Write a scenario (central body, vessels, nodes, sim time, warp) to JSON."""
    data = {
        "schema": SCHEMA_VERSION,
        "central": world.central.name,
        "sim_time_s": clock.sim_time_s,
        "warp": clock.warp,
        "vessels": [_vessel_to_dict(v) for v in world.vessels],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_scenario(path: str) -> tuple[World, SimClock]:
    """Load a scenario JSON into a (World, SimClock).

    Raises
    ------
    ValueError
        If the schema version is unsupported.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported scenario schema: {data.get('schema')}")
    central = by_name(data["central"])
    vessels = [_vessel_from_dict(d) for d in data["vessels"]]
    world = World(central=central, vessels=vessels)
    clock = SimClock(sim_time_s=data["sim_time_s"], warp=data["warp"])
    return world, clock
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/sim/test_persistence.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orbitsim/sim/persistence.py tests/sim/test_persistence.py
git commit -m "Phase 6 Task 2: versioned JSON scenario save/load"
```

---

## Task 3: Built-in scenarios

**Files:**
- Create: `orbitsim/scenarios/__init__.py` (builders) + generated JSON files
- Test: `tests/sim/test_persistence.py` (append)

**Interfaces:**
- Consumes: `save_scenario`, `World`, `Vessel`, `SimClock`, `StateVector`, `EARTH`, `by_name`.
- Produces: `build_builtin_scenarios(out_dir: str) -> list[str]` and three JSON files loadable by `load_scenario`.

- [ ] **Step 1: Write the failing test**

Append to `tests/sim/test_persistence.py`:
```python
def test_builtin_scenarios_round_trip(tmp_path):
    from orbitsim.scenarios import build_builtin_scenarios
    paths = build_builtin_scenarios(str(tmp_path))
    assert len(paths) == 3
    for p in paths:
        world, clock = load_scenario(p)
        assert len(world.vessels) >= 1
        assert world.vessels[0].state.r.shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/sim/test_persistence.py -k builtin -q`
Expected: FAIL (ModuleNotFoundError: orbitsim.scenarios).

- [ ] **Step 3: Implement the scenario builders**

Create `orbitsim/scenarios/__init__.py`:
```python
"""Built-in scenario builders."""
import os
import numpy as np

from orbitsim.sim.world import Vessel, World
from orbitsim.sim.clock import SimClock
from orbitsim.sim.persistence import save_scenario
from orbitsim.core.state import StateVector
from orbitsim.core.bodies import EARTH
from orbitsim.core.constants import MU_EARTH, R_EARTH


def _leo_sandbox() -> tuple[World, SimClock]:
    r0 = R_EARTH + 400e3
    v = np.sqrt(MU_EARTH / r0)
    state = StateVector(r=np.array([r0, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)
    vessel = Vessel(name="Sandbox-1", state=state, delta_v_budget_mps=3000.0)
    return World(central=EARTH, vessels=[vessel]), SimClock(warp=10.0)


def _leo_to_geo() -> tuple[World, SimClock]:
    r0 = 6678e3
    v = np.sqrt(MU_EARTH / r0)
    state = StateVector(r=np.array([r0, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)
    vessel = Vessel(name="Transfer-1", state=state, delta_v_budget_mps=5000.0)
    return World(central=EARTH, vessels=[vessel]), SimClock(warp=100.0)


def _earth_to_mars() -> tuple[World, SimClock]:
    # Earth-centered parking orbit; the planner (Phase 5) handles the interplanetary leg.
    r0 = R_EARTH + 300e3
    v = np.sqrt(MU_EARTH / r0)
    state = StateVector(r=np.array([r0, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=MU_EARTH)
    vessel = Vessel(name="MarsShip", state=state, delta_v_budget_mps=12000.0)
    # 2031 launch epoch (seconds past J2000 TDB).
    return World(central=EARTH, vessels=[vessel]), SimClock(sim_time_s=31.0 * 365.25 * 86400.0, warp=1000.0)


def build_builtin_scenarios(out_dir: str) -> list[str]:
    """Generate the built-in scenario JSON files in out_dir; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    builders = {
        "leo_sandbox.json": _leo_sandbox,
        "leo_to_geo.json": _leo_to_geo,
        "earth_to_mars.json": _earth_to_mars,
    }
    paths = []
    for fname, builder in builders.items():
        world, clock = builder()
        path = os.path.join(out_dir, fname)
        save_scenario(world, clock, path)
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/sim/test_persistence.py -q`
Expected: PASS.

- [ ] **Step 5: Generate the shipped scenario files + commit**

```bash
.venv/Scripts/python -c "from orbitsim.scenarios import build_builtin_scenarios; print(build_builtin_scenarios('orbitsim/scenarios'))"
git add orbitsim/scenarios/
git commit -m "Phase 6 Task 3: built-in scenarios (leo_sandbox, leo_to_geo, earth_to_mars)"
```

---

## Task 4: HUD/UX polish + save/load keys — HUMAN VISUAL CHECKPOINT

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes: `save_scenario`, `load_scenario`, the existing HUD and clock.
- Produces: orbit info panel (already partly there), maneuver list, F5 quicksave / F9 quickload, a keybinding help overlay (H), pause (space), and auto-warp-down when within 60 s of a node epoch.

- [ ] **Step 1: Add save/load + pause + help keys**

In `orbitsim/render/app.py` `_setup_input`, add:
```python
        self.accept("f5", self._quicksave)
        self.accept("f9", self._quickload)
        self.accept("space", self._toggle_pause)
        self.accept("h", self._toggle_help)
```
Add the methods:
```python
    def _quicksave(self) -> None:
        from orbitsim.sim.persistence import save_scenario
        save_scenario(self.world, self.clock, "quicksave.json")

    def _quickload(self) -> None:
        from orbitsim.sim.persistence import load_scenario
        self.world, self.clock = load_scenario("quicksave.json")

    def _toggle_pause(self) -> None:
        if not hasattr(self, "_paused"):
            self._paused = False
        self._paused = not self._paused

    def _toggle_help(self) -> None:
        if getattr(self, "_help_np", None) is None:
            from direct.gui.OnscreenText import OnscreenText
            self._help_np = OnscreenText(
                text=("Controls:\n  wheel: zoom\n  arrows: orbit cam\n  , / . : warp down/up\n"
                      "  space: pause\n  F5/F9: save/load\n  p: porkchop\n  h: toggle help"),
                pos=(0.0, 0.0), scale=0.05, fg=(1, 1, 1, 1),
            )
        else:
            self._help_np.destroy()
            self._help_np = None
```

- [ ] **Step 2: Honor pause + auto-warp-down in `_update`**

At the top of `OrbitApp._update`, replace the `sim_dt = self.clock.advance(real_dt)` region with:
```python
        real_dt = _global_clock.get_dt()
        # Auto-warp-down when close to a maneuver node so the burn isn't skipped.
        for vessel in self.world.vessels:
            for node in vessel.nodes:
                if 0.0 < (node.epoch_s - self.clock.sim_time_s) < 60.0:
                    while self.clock.warp > 1.0:
                        self.clock.warp_down()
        if getattr(self, "_paused", False):
            sim_dt = 0.0
        else:
            sim_dt = self.clock.advance(real_dt)
        self.world.step(sim_dt)
```

- [ ] **Step 3: Smoke-check imports + full suite green**

Run:
```bash
.venv/Scripts/python -c "import orbitsim.render.app; print('ok')"
.venv/Scripts/python -m pytest tests/ -q
```
Expected: `ok` then all tests PASS.

- [ ] **Step 4: HUMAN VISUAL CHECKPOINT**

Run: `.venv/Scripts/python -m orbitsim`
Reviewer confirms:
1. `H` toggles a readable controls overlay.
2. `space` pauses/unpauses the simulation (orbit motion stops/resumes).
3. `F5` writes `quicksave.json`; change warp, then `F9` restores the saved sim time/warp/orbit.
4. With a maneuver node set ~30 s ahead, warp auto-steps down to x1 as the node approaches.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "Phase 6 Task 4: HUD polish, save/load keys, pause, auto-warp-down"
```

---

## Task 5: Performance pass — HUMAN VISUAL CHECKPOINT

**Files:**
- Modify: `orbitsim/render/app.py`

**Goal:** cache orbit-line geometry; only rebuild when the orbit changes (after a burn), not every frame. Keep 60 FPS.

- [ ] **Step 1: Cache orbit geometry keyed by elements**

In `OrbitApp`, track a cached signature per vessel and only rebuild the orbit node when it changes. Replace `_rebuild_orbit` with:
```python
    def _rebuild_orbit(self, idx, vessel) -> None:
        elem = state_to_elements(vessel.state)
        sig = (round(elem.a, 3), round(elem.e, 9), round(elem.i, 9),
               round(elem.raan, 9), round(elem.argp, 9))
        if not hasattr(self, "_orbit_sigs"):
            self._orbit_sigs = [None] * len(self.world.vessels)
        if self._orbit_sigs[idx] == sig and self.orbit_nps[idx] is not None:
            return  # orbit unchanged; the cheap re-map happens via node reparent only
        pts = sample_orbit_points(elem, n=256)
        pts_render = [self.transform.to_render(p) for p in pts]
        if self.orbit_nps[idx] is not None:
            self.orbit_nps[idx].remove_node()
        node = build_orbit_node(pts_render)
        node.reparent_to(self.render)
        self.orbit_nps[idx] = node
        self._orbit_sigs[idx] = sig
```
Note: because the orbit polyline is built in physics space and re-mapped through `RenderTransform` each frame via `to_render`, the cached geometry must be regenerated when `transform.origin_m`/`scale` change too. Simplest correct approach for the on-rails case: rebuild when EITHER the orbit signature OR the rounded scale changes. Add `round(self.transform.scale_m_per_unit, 3)` into `sig`.

- [ ] **Step 2: Reserve numeric propagation for active burns only**

Confirm `World.step` uses `propagate_kepler` (analytic) — it does (Phase 2). Numeric propagation (`propagate_numeric`) is only invoked explicitly for perturbed/thrusting cases, never in the per-frame loop. No change needed; document this in a comment in `_update`.

- [ ] **Step 3: Full suite green**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 4: HUMAN VISUAL/PERF CHECKPOINT**

Run: `.venv/Scripts/python -m orbitsim` and enable the frame-rate meter (press the default Panda3D `f` if enabled, or add `self.setFrameRateMeter(True)` in `__init__`).
Reviewer confirms: with a handful of vessels the app holds ~60 FPS; the orbit line does not flicker; zooming stays smooth.

- [ ] **Step 5: Commit**

```bash
git add orbitsim/render/app.py
git commit -m "Phase 6 Task 5: cache orbit geometry, hold 60 FPS"
```

---

## Task 6: PyInstaller packaging — HUMAN MANUAL CHECKPOINT

**Files:**
- Create: `orbitsim.spec`
- Modify: `README.md`

- [ ] **Step 1: Install PyInstaller**

Run: `.venv/Scripts/python -m pip install pyinstaller`
Add `pyinstaller>=6.0` to `pyproject.toml` `[project.optional-dependencies] dev`.

- [ ] **Step 2: Create the spec**

Create `orbitsim.spec`:
```python
# PyInstaller spec for orbitsim (one-folder Windows build).
# Build with: .venv\Scripts\pyinstaller orbitsim.spec
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

hiddenimports = collect_submodules("panda3d") + collect_submodules("skyfield")
datas = []
# Bundle the DE440 kernel if present; otherwise it downloads on first run.
import os
if os.path.exists("data/de440s.bsp"):
    datas.append(("data/de440s.bsp", "data"))
# Bundle built-in scenarios.
datas.append(("orbitsim/scenarios", "orbitsim/scenarios"))

a = Analysis(
    ["orbitsim/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="orbitsim",
          debug=False, bootloader_ignore_signals=False, strip=False,
          upx=True, console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name="orbitsim")
```

- [ ] **Step 3: Build**

Run: `.venv/Scripts/pyinstaller orbitsim.spec`
Expected: a `dist/orbitsim/` folder containing `orbitsim.exe`.

- [ ] **Step 4: HUMAN MANUAL CHECKPOINT — run the built exe on a clean path**

Copy `dist/orbitsim/` somewhere outside the repo (e.g. `C:\Temp\orbitsim`) and run `orbitsim.exe`.
Reviewer confirms: it launches, renders Earth + a vessel, loads a built-in scenario, and shows no missing-DLL errors. If the DE440 kernel was not bundled, the solar-system mode downloads it on first run with no crash.

- [ ] **Step 5: Document the build in README + commit**

Add a "Build" section to `README.md`:
```markdown
## Build (Windows)

1. `python -m venv .venv && .venv\Scripts\pip install -e .[dev,render]`
2. `.venv\Scripts\pyinstaller orbitsim.spec`
3. Run `dist\orbitsim\orbitsim.exe`.
The DE440 ephemeris kernel (~30 MB) is bundled if present in `data/`, else
downloaded on first solar-system launch.
```

```bash
git add orbitsim.spec README.md pyproject.toml
git commit -m "Phase 6 Task 6: PyInstaller one-folder Windows build"
```

---

## Task 7: Docs — controls cheat-sheet + README scenario list

**Files:**
- Create: `CONTROLS.md`
- Modify: `README.md`

- [ ] **Step 1: Write CONTROLS.md**

Create `CONTROLS.md`:
```markdown
# Controls

| Key | Action |
|-----|--------|
| Mouse wheel | Zoom in / out |
| Arrow keys | Orbit camera (azimuth / elevation) |
| `,` / `.` | Time warp down / up |
| `space` | Pause / resume |
| `F5` / `F9` | Quicksave / quickload |
| `p` | Show porkchop plot |
| `h` | Toggle controls help |
| `--solar` (launch flag) | Render the full solar system |

Maneuver editor (bottom-right sliders): prograde / normal / radial delta-V,
then **Execute Burn** to commit.
```

- [ ] **Step 2: Update README scenario list**

Add to `README.md`:
```markdown
## Built-in scenarios
- `leo_sandbox` — empty 400 km LEO, full delta-V budget, free maneuvering.
- `leo_to_geo` — 6678 km circular start for a Hohmann transfer to GEO.
- `earth_to_mars` — Earth parking orbit at a 2031 launch epoch for interplanetary planning.

Load a scenario JSON via the API: `load_scenario("orbitsim/scenarios/leo_to_geo.json")`.
```

- [ ] **Step 3: Full suite green + commit**

Run: `.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS.

```bash
git add CONTROLS.md README.md
git commit -m "Phase 6 Task 7: controls cheat-sheet + scenario docs"
```

---

## Phase 6 Exit Criteria

- A double-clickable `orbitsim.exe` launches the sim, loads built-in scenarios, runs smoothly, and saves/loads state.
- Save→load round-trips state vectors bit-identically (test green).
- HUD shows orbit info, controls help, pause, save/load; auto-warp-down protects burns.
- 60 FPS with a handful of vessels.
- `pytest tests/ -q` fully green.

This is the final phase — the simulator is ready to hand to a friend.

## Self-Review Notes

- Spec coverage: persistence (6.1), HUD/UX polish (6.2), performance pass (6.3), packaging (6.4), docs (6.5) — all mapped, plus the `by_name` registry needed by load.
- Save/load + scenario builders are red-green TDD; HUD/perf/packaging are manual checkpoints (inherently non-unit-testable).
- Type consistency: `save_scenario(world, clock, path)` / `load_scenario(path) -> (World, SimClock)` used identically across persistence tests, scenario builders, and the app's quicksave/quickload.
```
