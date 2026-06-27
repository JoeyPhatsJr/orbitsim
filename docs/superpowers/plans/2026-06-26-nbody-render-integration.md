# N-Body Render Integration (Cycle 1b Part 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note:** Tasks 1–3 are pure physics (TDD) — dispatch to **Haiku subagents**. Task 4 is render integration — executed **inline by the controller** with headless verification per project convention. Review each task independently before proceeding to the next.

**Goal:** Wire the tested N-body propagator into the live sandbox so the ship flies under Moon gravity, the trajectory line shows perturbation bending, time-warp caps near bodies, and the HUD reads osculating elements.

**Architecture:** Three pure-physics changes first (new `integrate_powered_nbody` in `core/flight.py`, propagator param in `core/rendezvous.py`, swap in `sim/world.py`), then one render task (`render/app.py`) that replaces the Keplerian orbit line with a forward-integrated trajectory, adds the warp cap, and feeds osculating elements to the HUD.

**Tech Stack:** Python 3, numpy, Panda3D. Tests via `.venv/Scripts/python -m pytest`.

## Global Constraints

- SI, float64 `(3,)` arrays; frame = Earth-centered inertial. (project rule)
- `core/` never imports render/sim/panda3d. (project rule)
- Constants from `core/constants.py` (`MU_EARTH`, `MU_MOON` exist). (project rule)
- TDD for Tasks 1–3; **never loosen a tolerance to pass** — fix the implementation. (project rule)
- Run tests with `.venv/Scripts/python -m pytest`. Commits: explicit paths; end with `Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>` (Tasks 1–3) or `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` (Task 4); then `git push`. (repo discipline)
- All existing tests must remain green after each task. (project rule)

## File Structure

- `orbitsim/core/flight.py` — add `integrate_powered_nbody` (operator-split Verlet + rocket eq under `earth_moon_accel`).
- `orbitsim/core/rendezvous.py` — add `propagator` kwarg to `closest_approach`; backward-compat default `propagate_kepler`.
- `orbitsim/sim/world.py` — swap coast → `propagate_earth_moon`, thrust → `integrate_powered_nbody`.
- `orbitsim/render/app.py` — forward-integrated trajectory line (replacing `_rebuild_orbit`), warp cap, osculating HUD, N-body CA, N-body preview and node marker.
- `tests/core/test_flight.py` — add `integrate_powered_nbody` tests.
- `tests/core/test_rendezvous.py` — add N-body propagator test.
- `tests/sim/test_world.py` — add N-body step tests.

---

## Task 1: `integrate_powered_nbody` (Haiku subagent)

**Files:**
- Modify: `orbitsim/core/flight.py`
- Test: `tests/core/test_flight.py`

**Interfaces:**
- Consumes: `earth_moon_accel(r, t) -> np.ndarray` and `_earth_moon_substeps(state, dt_s, max_step_s) -> int` from `orbitsim.core.nbody`; `mass_flow_rate` from this same file.
- Produces: `integrate_powered_nbody(state, dry_mass_kg, fuel_kg, thrust_dir_unit, throttle, max_thrust_n, ve_mps, dt_s) -> tuple[StateVector, float]` — new state and remaining fuel, same contract as `integrate_powered`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_flight.py`:

```python
from orbitsim.core.nbody import earth_moon_accel, propagate_earth_moon


def test_nbody_free_space_burn_matches_rocket_equation():
    """In free space (mu=0, no gravity), N-body burn dV telescopes to ve*ln(m0/mf)."""
    from orbitsim.core.flight import integrate_powered_nbody
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]),
                    mu=0.0, epoch_s=0.0)
    dry, fuel0, thrust, ve = 1000.0, 1000.0, 30000.0, 3000.0
    mdot = thrust / ve          # 10 kg/s
    burn_time = fuel0 / mdot    # 100 s
    out, fuel = integrate_powered_nbody(
        s, dry_mass_kg=dry, fuel_kg=fuel0,
        thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=thrust, ve_mps=ve, dt_s=burn_time,
    )
    expected_dv = ve * np.log((dry + fuel0) / dry)   # 3000*ln(2) ≈ 2079.44 m/s
    assert abs(out.v[0] - expected_dv) / expected_dv < 1e-3
    assert abs(fuel) < 1e-6     # fuel depleted to exactly 0


def test_nbody_fuel_reaches_zero_exactly():
    """Asking for more burn time than fuel allows: fuel ends at 0, not negative."""
    from orbitsim.core.flight import integrate_powered_nbody
    s = StateVector(r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]),
                    mu=0.0, epoch_s=0.0)
    out, fuel = integrate_powered_nbody(
        s, dry_mass_kg=1000.0, fuel_kg=100.0,
        thrust_dir_unit=np.array([1.0, 0.0, 0.0]),
        throttle=1.0, max_thrust_n=30000.0, ve_mps=3000.0, dt_s=1000.0,
    )
    expected_dv = 3000.0 * np.log(1100.0 / 1000.0)
    assert fuel == 0.0
    assert abs(out.v[0] - expected_dv) / expected_dv < 2e-3


def test_nbody_moon_perturbation_diverges_from_twobody():
    """Near the Moon, N-body trajectory diverges measurably from two-body."""
    from orbitsim.core.flight import integrate_powered_nbody, integrate_powered
    from orbitsim.core.moon import moon_state_at
    # Place ship 5000 km from the Moon (deep in its gravity well).
    t0 = 0.0
    rM = moon_state_at(t0).r
    r_ship = rM + np.array([5.0e6, 0.0, 0.0])
    v_ship = np.array([0.0, 500.0, 0.0])
    s = StateVector(r=r_ship, v=v_ship, mu=MU_EARTH, epoch_s=t0)
    dt = 3600.0   # 1 hour burn
    # N-body burn
    out_nbody, _ = integrate_powered_nbody(
        s, dry_mass_kg=1000.0, fuel_kg=5000.0,
        thrust_dir_unit=np.array([0.0, 1.0, 0.0]),
        throttle=0.1, max_thrust_n=10000.0, ve_mps=3000.0, dt_s=dt,
    )
    # Two-body burn (same call, two-body gravity)
    out_2body, _ = integrate_powered(
        s, dry_mass_kg=1000.0, fuel_kg=5000.0,
        thrust_dir_unit=np.array([0.0, 1.0, 0.0]),
        throttle=0.1, max_thrust_n=10000.0, ve_mps=3000.0, dt_s=dt, substeps=50,
    )
    divergence = np.linalg.norm(out_nbody.r - out_2body.r)
    assert divergence > 1000.0, f"expected N-body divergence near Moon, got {divergence:.1f} m"
```

- [ ] **Step 2: Run to verify they fail**

```
.venv/Scripts/python -m pytest tests/core/test_flight.py -q -k "nbody"
```
Expected: FAIL — `ImportError: cannot import name 'integrate_powered_nbody'`.

- [ ] **Step 3: Implement**

Add to `orbitsim/core/flight.py` (after the existing `integrate_powered` function):

```python
def integrate_powered_nbody(
    state: StateVector,
    dry_mass_kg: float,
    fuel_kg: float,
    thrust_dir_unit: np.ndarray,
    throttle: float,
    max_thrust_n: float,
    ve_mps: float,
    dt_s: float,
) -> tuple:
    """Integrate r, v, fuel over dt_s under earth_moon_accel + thrust.

    Operator splitting per substep: exact rocket-equation velocity impulse,
    then velocity-Verlet drift under earth_moon_accel. Same fuel-reaches-zero
    guarantee as integrate_powered; substep count is proximity-aware.

    Returns
    -------
    (StateVector, float)
        New state (epoch_s + dt_s) and remaining fuel [kg].
    """
    from orbitsim.core.nbody import earth_moon_accel, _earth_moon_substeps

    thrust_dir_unit = np.asarray(thrust_dir_unit, dtype=np.float64)
    r = np.asarray(state.r, dtype=np.float64).copy()
    v = np.asarray(state.v, dtype=np.float64).copy()
    fuel = float(fuel_kg)
    t = state.epoch_s

    n = _earth_moon_substeps(state, dt_s, max_step_s=3600.0)
    h = dt_s / n
    mdot = mass_flow_rate(throttle, max_thrust_n, ve_mps) if ve_mps > 0 else 0.0

    for _ in range(n):
        # Thrust: exact rocket-equation impulse over the burning portion of h.
        if throttle > 0.0 and fuel > 0.0 and mdot > 0.0:
            t_burn = min(h, fuel / mdot)
            m_start = dry_mass_kg + fuel
            m_end = m_start - mdot * t_burn
            v = v + ve_mps * np.log(m_start / m_end) * thrust_dir_unit
            fuel = max(0.0, fuel - mdot * t_burn)
        # Gravity: velocity-Verlet under earth_moon_accel.
        a0 = earth_moon_accel(r, t)
        v_half = v + 0.5 * a0 * h
        r = r + v_half * h
        t = t + h
        a1 = earth_moon_accel(r, t)
        v = v_half + 0.5 * a1 * h

    return StateVector(r=r, v=v, mu=state.mu, epoch_s=state.epoch_s + dt_s), float(fuel)
```

- [ ] **Step 4: Run tests + full suite**

```
.venv/Scripts/python -m pytest tests/core/test_flight.py -q
.venv/Scripts/python -m pytest tests/ -q
```
Expected: all PASS. If `test_nbody_moon_perturbation_diverges_from_twobody` fails with divergence < 1000 m, the N-body Moon force is not being applied — check the import of `earth_moon_accel`. Do NOT loosen the 1000 m bound.

- [ ] **Step 5: Commit**

```
git add orbitsim/core/flight.py tests/core/test_flight.py
git commit -m "$(cat <<'EOF'
N-body: integrate_powered_nbody (operator-split Verlet + rocket eq)

dV telescopes and fuel hits 0 exactly; Moon perturbation diverges from
two-body near the Moon, verifying earth_moon_accel is active.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 2: `closest_approach` with N-body propagator (Haiku subagent)

**Files:**
- Modify: `orbitsim/core/rendezvous.py`
- Test: `tests/core/test_rendezvous.py`

**Interfaces:**
- Consumes: `propagate_kepler` (existing default), `propagate_earth_moon` from `orbitsim.core.nbody` (used by callers).
- Produces: `closest_approach(state_a, state_b, window_s, coarse_samples=720, propagator=propagate_kepler) -> ClosestApproach` — same return type; backward-compat when `propagator` is omitted.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_rendezvous.py`:

```python
def test_nbody_ca_differs_from_keplerian_near_moon():
    """Near the Moon, N-body CA differs measurably from Keplerian CA."""
    from orbitsim.core.nbody import propagate_earth_moon
    from orbitsim.core.moon import moon_state_at
    # Ship and target both near the Moon (strong perturbation).
    rM = moon_state_at(0.0).r
    s_a = StateVector(r=rM + np.array([5.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 900.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    s_b = StateVector(r=rM + np.array([-5.0e6, 0.0, 0.0]),
                      v=np.array([0.0, -900.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    window = 6 * 3600.0    # 6-hour window
    ca_kep = closest_approach(s_a, s_b, window_s=window, coarse_samples=360)
    ca_nbody = closest_approach(s_a, s_b, window_s=window, coarse_samples=360,
                                propagator=propagate_earth_moon)
    # N-body and Keplerian CAs must differ by > 1 km (Moon bends trajectories).
    diff = abs(ca_nbody.separation_m - ca_kep.separation_m)
    assert diff > 1000.0, f"expected N-body CA to differ from Keplerian, got {diff:.1f} m"
```

- [ ] **Step 2: Run to verify it fails**

```
.venv/Scripts/python -m pytest tests/core/test_rendezvous.py -q -k nbody_ca
```
Expected: FAIL — `TypeError: closest_approach() got an unexpected keyword argument 'propagator'`.

- [ ] **Step 3: Implement**

Replace the contents of `orbitsim/core/rendezvous.py` with:

```python
"""Closest approach between two trajectories (coarse scan + refine)."""
from dataclasses import dataclass

import numpy as np

from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.state import StateVector


@dataclass(frozen=True)
class ClosestApproach:
    """Result of a closest-approach search.

    Attributes
    ----------
    t_ca_s : float
        Time of closest approach, seconds from now.
    separation_m : float
        Distance between the two bodies at closest approach [m].
    rel_speed_mps : float
        Relative speed |v_a - v_b| at closest approach [m/s].
    """

    t_ca_s: float
    separation_m: float
    rel_speed_mps: float


def closest_approach(
    state_a: StateVector,
    state_b: StateVector,
    window_s: float,
    coarse_samples: int = 720,
    propagator=propagate_kepler,
) -> ClosestApproach:
    """Minimum separation of two trajectories over ``[0, window_s]``.

    Coarse-scans ``coarse_samples+1`` uniform times, then refines the best one with a
    ternary search over its bracketing interval. Raises ValueError on bad inputs.

    Parameters
    ----------
    propagator : callable
        Function ``(StateVector, float) -> StateVector`` used to advance each state.
        Defaults to ``propagate_kepler``; pass ``propagate_earth_moon`` for N-body CA.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if coarse_samples < 2:
        raise ValueError(f"coarse_samples must be >= 2, got {coarse_samples}")

    def _sep(t):
        ra = propagator(state_a, t).r
        rb = propagator(state_b, t).r
        return float(np.linalg.norm(ra - rb))

    times = np.linspace(0.0, window_s, coarse_samples + 1)
    seps = np.array([_sep(float(t)) for t in times])
    k = int(np.argmin(seps))

    # Ternary-search refine within [t_{k-1}, t_{k+1}].
    lo = times[max(0, k - 1)]
    hi = times[min(len(times) - 1, k + 1)]
    for _ in range(60):
        if hi - lo < 1e-3:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _sep(m1) < _sep(m2):
            hi = m2
        else:
            lo = m1
    t_ca = 0.5 * (lo + hi)

    # Guard: never return a separation worse than the coarse minimum.
    if seps[k] <= _sep(t_ca):
        t_ca = float(times[k])

    sa = propagator(state_a, t_ca)
    sb = propagator(state_b, t_ca)
    sep = float(np.linalg.norm(sa.r - sb.r))
    rel = float(np.linalg.norm(sa.v - sb.v))
    return ClosestApproach(t_ca_s=t_ca, separation_m=sep, rel_speed_mps=rel)
```

- [ ] **Step 4: Run tests + full suite**

```
.venv/Scripts/python -m pytest tests/core/test_rendezvous.py -q
.venv/Scripts/python -m pytest tests/ -q
```
Expected: all PASS. All existing tests still green (default `propagator=propagate_kepler`). Do NOT loosen the 1000 m divergence bound.

- [ ] **Step 5: Commit**

```
git add orbitsim/core/rendezvous.py tests/core/test_rendezvous.py
git commit -m "$(cat <<'EOF'
Rendezvous: propagator param in closest_approach (N-body CA support)

Default propagate_kepler preserves backward compat; pass propagate_earth_moon
from the render layer for N-body closest-approach near the Moon.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 3: `World.step` — swap to N-body propagators (Haiku subagent)

**Files:**
- Modify: `orbitsim/sim/world.py`
- Test: `tests/sim/test_world.py`

**Interfaces:**
- Consumes: `propagate_earth_moon` from `orbitsim.core.nbody`; `integrate_powered_nbody` from `orbitsim.core.flight` (Task 1).
- Produces: `World.step` now propagates under N-body. All existing public API unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/sim/test_world.py`:

```python
def test_world_step_coast_uses_nbody_near_moon():
    """A vessel near the Moon drifts differently from Keplerian after N-body step."""
    from orbitsim.core.moon import moon_state_at
    from orbitsim.core.propagate import propagate_kepler
    rM = moon_state_at(0.0).r
    r_ship = rM + np.array([5.0e6, 0.0, 0.0])
    v_ship = np.array([0.0, 500.0, 0.0])
    st = StateVector(r=r_ship, v=v_ship, mu=MU_EARTH, epoch_s=0.0)
    vessel = Vessel(name="test", state=st)
    world = World(central=EARTH, vessels=[vessel])
    dt = 3600.0
    world.step(dt)
    # N-body result should diverge from Keplerian by > 1 km near the Moon.
    kep = propagate_kepler(st, dt)
    divergence = np.linalg.norm(world.vessels[0].state.r - kep.r)
    assert divergence > 1000.0, f"expected N-body divergence, got {divergence:.1f} m"


def test_world_step_coast_leo_close_to_kepler():
    """In LEO, N-body coast stays within 1 km of Keplerian over one quarter-orbit."""
    from orbitsim.core.propagate import propagate_kepler
    r = 7.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    vessel = Vessel(name="test", state=st)
    world = World(central=EARTH, vessels=[vessel])
    period = 2 * np.pi * np.sqrt(r**3 / MU_EARTH)
    world.step(period / 4)
    kep = propagate_kepler(st, period / 4)
    assert np.linalg.norm(world.vessels[0].state.r - kep.r) < 1000.0


def test_world_step_thrust_nbody_fuel_drains():
    """Thrusting under N-body drains fuel (same contract as two-body)."""
    r = 7.0e6
    st = StateVector(r=np.array([r, 0.0, 0.0]),
                     v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                     mu=MU_EARTH, epoch_s=0.0)
    v = Vessel(name="test", state=st, dry_mass_kg=1000.0, fuel_mass_kg=500.0,
               max_thrust_n=30000.0, exhaust_velocity_mps=3000.0,
               throttle=1.0, sas_mode="PROGRADE")
    world = World(central=EARTH, vessels=[v])
    fuel0 = v.fuel_mass_kg
    speed0 = v.state.v_mag
    for _ in range(60):
        world.step(0.1)
    assert v.fuel_mass_kg < fuel0
    assert v.state.v_mag > speed0
```

- [ ] **Step 2: Run to verify the new tests fail**

```
.venv/Scripts/python -m pytest tests/sim/test_world.py -q -k "nbody"
```
Expected: `test_world_step_coast_uses_nbody_near_moon` FAIL (divergence < 1000 m — still using Keplerian); `test_world_step_coast_leo_close_to_kepler` and `test_world_step_thrust_nbody_fuel_drains` may PASS already — that's fine, they're regression guards.

- [ ] **Step 3: Implement**

Replace the coast and thrust calls in `orbitsim/sim/world.py::World.step`:

```python
def step(self, sim_dt_s: float) -> None:
    """Advance every vessel by sim_dt_s: slew attitude, then translate.

    Coasting vessels propagate under earth_moon_accel (N-body, velocity-Verlet);
    thrusting vessels integrate under earth_moon_accel + rocket equation.
    """
    from orbitsim.core.nbody import propagate_earth_moon
    from orbitsim.core.flight import integrate_powered_nbody
    from orbitsim.core.attitude import (
        slew_toward, sas_target_dir, nose_direction,
    )

    for vessel in self.vessels:
        # 1) Attitude: slew toward the SAS hold direction (if any) each tick.
        if vessel.sas_mode not in ("OFF", "STABILITY"):
            try:
                target = sas_target_dir(vessel.sas_mode, vessel.state, vessel.sas_target_pos)
            except ValueError:
                target = None
            if target is not None:
                vessel.orientation = slew_toward(
                    vessel.orientation, target, vessel.max_turn_rate_radps, sim_dt_s)
        # 2) Translation.
        if vessel.throttle > 0.0 and (vessel.fuel_mass_kg > 0.0 or vessel.unlimited_dv):
            new_state, new_fuel = integrate_powered_nbody(
                vessel.state,
                dry_mass_kg=vessel.dry_mass_kg,
                fuel_kg=vessel.fuel_mass_kg,
                thrust_dir_unit=nose_direction(vessel.orientation),
                throttle=vessel.throttle,
                max_thrust_n=vessel.max_thrust_n,
                ve_mps=vessel.exhaust_velocity_mps,
                dt_s=sim_dt_s,
            )
            vessel.state = new_state
            if not vessel.unlimited_dv:
                vessel.fuel_mass_kg = new_fuel
        else:
            vessel.state = propagate_earth_moon(vessel.state, sim_dt_s)
```

Also remove the now-unused `propagate_kepler` import at the top of `world.py`:

```python
# Remove this line:
from orbitsim.core.propagate import propagate_kepler
```

- [ ] **Step 4: Run tests + full suite**

```
.venv/Scripts/python -m pytest tests/sim/test_world.py -q
.venv/Scripts/python -m pytest tests/ -q
```
Expected: all PASS. `test_world_step_period_closure` was previously exact (analytic Kepler); under N-body it will have a small Moon-perturbation error — if it fails, loosen only THAT test's tolerance from `< 1e-3` to `< 1e3` (1 km over one LEO period is physically correct under N-body, not a bug). Do NOT loosen any other tolerance.

- [ ] **Step 5: Commit**

```
git add orbitsim/sim/world.py tests/sim/test_world.py
git commit -m "$(cat <<'EOF'
World: swap coast/thrust to N-body propagators

Coast -> propagate_earth_moon; thrust -> integrate_powered_nbody.
LEO stays within 1 km of Keplerian; near-Moon diverges > 1 km (correct).

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Task 4: Render integration — trajectory line, warp cap, HUD, CA (controller inline)

**Files:**
- Modify: `orbitsim/render/app.py`

**Interfaces:**
- Consumes (new): `propagate_earth_moon`, `osculating_elements`, `max_safe_warp` from `orbitsim.core.nbody`; `integrate_powered_nbody` is now called by `world.step` (transparent); `propagate_earth_moon` for node marker + CA; `closest_approach(..., propagator=propagate_earth_moon)`.
- Produces: running sandbox with N-body trajectory line, warp cap, osculating HUD, N-body CA.

### Step 4a: Add `_sample_trajectory` helper and `_traj_state_cache`

- [ ] **Step 4a-1: Add `_sample_trajectory` and replace `_orbit_elem_cache` initialization**

In `_start_sim`, replace:
```python
self._orbit_elem_cache = [None for _ in world.vessels]
```
with:
```python
self._traj_state_cache = [None for _ in world.vessels]  # (StateVector, scale) per vessel
```

Add a new private method to `OrbitApp` (place it near `_rebuild_orbit`):

```python
def _sample_trajectory(self, state, n_pts=256, max_horizon_s=7 * 86400):
    """Forward-integrate state under earth_moon_accel and return ~n_pts positions [m].

    Horizon is the osculating orbital period capped at max_horizon_s (7 days).
    Returns an (n_pts, 3) float64 array of world-meter positions.
    """
    from orbitsim.core.nbody import osculating_elements, propagate_earth_moon
    try:
        osc = osculating_elements(state, state.epoch_s)
        horizon_s = min(float(osc.period_s), max_horizon_s)
    except (ValueError, AttributeError):
        horizon_s = float(max_horizon_s)
    dt = horizon_s / n_pts
    pts = np.empty((n_pts, 3), dtype=np.float64)
    pts[0] = state.r
    cur = state
    for i in range(1, n_pts):
        cur = propagate_earth_moon(cur, dt)
        pts[i] = cur.r
    return pts
```

- [ ] **Step 4a-2: Replace `_rebuild_orbit` with `_rebuild_trajectory`**

Remove the existing `_rebuild_orbit` method entirely and add:

```python
def _rebuild_trajectory(self, idx, vessel) -> None:
    """Rebuild the vessel trajectory line if state or zoom changed beyond tolerance."""
    state = vessel.state
    scale = self.transform.scale_m_per_unit
    cached = self._traj_state_cache[idx]
    if cached is not None:
        cached_state, cached_scale = cached
        if cached_scale == scale:
            pos_ok = np.linalg.norm(state.r - cached_state.r) < 100.0
            vel_ok = np.linalg.norm(state.v - cached_state.v) < 0.1
            if pos_ok and vel_ok:
                return
    self._traj_state_cache[idx] = (state, scale)
    pts = [tuple(p / scale) for p in self._sample_trajectory(state)]
    if self.orbit_nps[idx] is not None:
        self.orbit_nps[idx].remove_node()
    node = build_orbit_node(pts)
    node.reparent_to(self._orbit_frame)
    self.orbit_nps[idx] = node
```

- [ ] **Step 4a-3: Update the `_update` call site**

In `_update`, replace:
```python
self._rebuild_orbit(idx, vessel)
```
with:
```python
self._rebuild_trajectory(idx, vessel)
```

- [ ] **Step 4a-4: Run full test suite**

```
.venv/Scripts/python -m pytest tests/ -q
```
Expected: all PASS (trajectory logic is render-only, no physics tests broken).

### Step 4b: Time-warp cap

- [ ] **Step 4b-1: Clamp warp before `clock.advance`**

In `_update`, find the block:
```python
# Flight input, then lock warp to 1x while thrusting (no RK4 through warp).
self._apply_flight_input(real_dt)
if self.world.any_thrusting() and self.clock.warp != 1.0:
    self.clock.warp = 1.0
```

Replace with:
```python
self._apply_flight_input(real_dt)
if self.world.any_thrusting():
    self.clock.warp = 1.0
else:
    from orbitsim.core.nbody import max_safe_warp
    from orbitsim.sim.clock import SimClock
    cap = max_safe_warp(
        self.world.vessels[0].state,
        self.clock.sim_time_s,
        SimClock.WARP_STEPS,
    )
    if self.clock.warp > cap:
        self.clock.warp = cap
```

- [ ] **Step 4b-2: Guard `_warp_up_guarded` against the cap**

Find:
```python
def _warp_up_guarded(self):
    if not self.world.any_thrusting():
        self.clock.warp_up()
```

Replace with:
```python
def _warp_up_guarded(self):
    if self.world.any_thrusting():
        return
    from orbitsim.core.nbody import max_safe_warp
    from orbitsim.sim.clock import SimClock
    cap = max_safe_warp(
        self.world.vessels[0].state,
        self.clock.sim_time_s,
        SimClock.WARP_STEPS,
    )
    if self.clock.warp < cap:
        self.clock.warp_up()
```

- [ ] **Step 4b-3: Run full test suite**

```
.venv/Scripts/python -m pytest tests/ -q
```
Expected: all PASS.

### Step 4c: Osculating elements in HUD

- [ ] **Step 4c-1: Replace `state_to_elements` with `osculating_elements` for HUD**

In `_update`, find the HUD block (~line 1019):
```python
v0 = self.world.vessels[0]
elem = state_to_elements(v0.state)
rp = elem.a * (1 - elem.e)
ra = elem.a * (1 + elem.e)
try:
    period = elem.period_s
except ValueError:
    period = float("nan")
self.hud.update(
    sim_time_s=self.clock.sim_time_s,
    warp=self.clock.warp,
    altitude_m=v0.state.r_mag - self.world.central.radius_m,
    speed_mps=v0.state.v_mag,
    periapsis_m=rp - self.world.central.radius_m,
    apoapsis_m=ra - self.world.central.radius_m,
    period_s=period,
    inclination_rad=elem.i,
)
```

Replace with:
```python
v0 = self.world.vessels[0]
from orbitsim.core.nbody import osculating_elements, MOON_SOI_M
from orbitsim.core.constants import MU_MOON
from orbitsim.core.bodies import MOON as MOON_BODY
from orbitsim.core.moon import moon_state_at
elem = osculating_elements(v0.state, self.clock.sim_time_s)
rp = elem.a * (1 - elem.e)
ra = elem.a * (1 + elem.e)
try:
    period = elem.period_s
except ValueError:
    period = float("nan")
# When inside the Moon's SOI, elements are Moon-relative: use Moon radius for altitude.
moon_now = moon_state_at(self.clock.sim_time_s)
moon_dominant = np.linalg.norm(v0.state.r - moon_now.r) < MOON_SOI_M
ref_radius = MOON_BODY.radius_m if moon_dominant else self.world.central.radius_m
self.hud.update(
    sim_time_s=self.clock.sim_time_s,
    warp=self.clock.warp,
    altitude_m=v0.state.r_mag - self.world.central.radius_m,
    speed_mps=v0.state.v_mag,
    periapsis_m=rp - ref_radius,
    apoapsis_m=ra - ref_radius,
    period_s=period,
    inclination_rad=elem.i,
)
```

Note: `altitude_m` (raw distance from Earth surface) stays Earth-relative always — it's used for the atmosphere indicator and doesn't switch. Only Pe/Ap switch to Moon-relative when Moon-dominant.

- [ ] **Step 4c-2: Run full test suite**

```
.venv/Scripts/python -m pytest tests/ -q
```
Expected: all PASS.

### Step 4d: N-body maneuver preview, node marker, and CA

- [ ] **Step 4d-1: Replace Keplerian preview with N-body forward integration**

In `_update`, find the post-burn orbit preview block:
```python
if node.magnitude_mps > 0.0:
    pred = predict_elements_after(v0.state, node)
    ppts = [tuple(p / self.transform.scale_m_per_unit)
            for p in sample_orbit_points(pred, n=256)]
    if self._preview_np is not None:
        self._preview_np.remove_node()
    self._preview_np = build_orbit_node(ppts, color=(1.0, 0.2, 1.0, 1.0))
    self._preview_np.reparent_to(self._orbit_frame)
elif self._preview_np is not None:
    self._preview_np.remove_node()
    self._preview_np = None
```

Replace with:
```python
if node.magnitude_mps > 0.0:
    from orbitsim.core.maneuvers import apply_maneuver
    post_burn = apply_maneuver(v0.state, node)
    ppts = [tuple(p / self.transform.scale_m_per_unit)
            for p in self._sample_trajectory(post_burn)]
    if self._preview_np is not None:
        self._preview_np.remove_node()
    self._preview_np = build_orbit_node(ppts, color=(1.0, 0.2, 1.0, 1.0))
    self._preview_np.reparent_to(self._orbit_frame)
elif self._preview_np is not None:
    self._preview_np.remove_node()
    self._preview_np = None
```

- [ ] **Step 4d-2: Replace Keplerian node marker with N-body**

Find the node marker position line:
```python
npos = propagate_kepler(v0.state, max(0.0, ttn)).r
```

Replace with:
```python
from orbitsim.core.nbody import propagate_earth_moon
npos = propagate_earth_moon(v0.state, max(0.0, ttn)).r
```

- [ ] **Step 4d-3: Replace Keplerian CA with N-body CA**

Find the closest-approach recompute block and the CA marker position:
```python
self._ca = closest_approach(
    traj, self._target.state_at(base_epoch), window_s=window, coarse_samples=720)
```

Replace with:
```python
from orbitsim.core.nbody import propagate_earth_moon as _pe
self._ca = closest_approach(
    traj, self._target.state_at(base_epoch), window_s=window, coarse_samples=720,
    propagator=_pe)
```

Also find the CA marker position:
```python
ship_at = propagate_kepler(self._ca_traj, ca.t_ca_s).r
```

Replace with:
```python
from orbitsim.core.nbody import propagate_earth_moon
ship_at = propagate_earth_moon(self._ca_traj, ca.t_ca_s).r
```

And the period computation used for the CA window — it currently calls `state_to_elements(traj)`. Since `traj` may now be a post-burn state that is already in an N-body perturbed situation, keep using `state_to_elements` here (it's just a seed for the window duration, not physics-critical):
```python
try:
    period = state_to_elements(traj).period_s
except ValueError:
    period = 14.0 * 86400.0
```
This line is unchanged — leave it as is.

- [ ] **Step 4d-4: Remove now-unused imports**

At the top of `app.py`, check for imports that are no longer used after the replacements:
- `from orbitsim.core.propagate import propagate_kepler` — still used by `apply_maneuver` internally; but in `app.py` it was used directly for the node marker and CA. After the replacements it is no longer used directly in `app.py`. Remove it from the top-level import **only if** no other line in `app.py` still references it. (Search for `propagate_kepler` in the file to confirm.)
- `from orbitsim.core.elements import state_to_elements` — still used for the CA window period and `orbit_shape_changed` context. Keep it.
- `predict_elements_after` import — may no longer be used after Step 4d-1. Remove from imports if so.
- `sample_orbit_points` — still used for the Moon ring. Keep it.

- [ ] **Step 4d-5: Run full test suite**

```
.venv/Scripts/python -m pytest tests/ -q
```
Expected: all PASS.

### Step 4e: Headless verification

- [ ] **Step 4e-1: Verify warp caps in LEO**

Run this verification script (save to the scratchpad, don't commit):

```python
# Headless warp-cap check
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-type offscreen")

import numpy as np
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.bodies import EARTH
from orbitsim.sim.world import Vessel, World
from orbitsim.sim.clock import SimClock
from orbitsim.render.app import OrbitApp

r = 7.0e6
state = StateVector(r=np.array([r, 0.0, 0.0]),
                    v=np.array([0.0, np.sqrt(MU_EARTH / r), 0.0]),
                    mu=MU_EARTH, epoch_s=0.0)
vessel = Vessel(name="ship", state=state, fuel_mass_kg=5000.0,
                max_thrust_n=50000.0, exhaust_velocity_mps=3000.0)
world = World(central=EARTH, vessels=[vessel])
clock = SimClock(warp=1_000_000.0)  # start at max warp
app = OrbitApp(world, clock)
app._on_play()          # skip title screen
app.taskMgr.step()     # runs _update once → warp should be clamped

from orbitsim.core.nbody import max_safe_warp
cap = max_safe_warp(vessel.state, clock.sim_time_s, SimClock.WARP_STEPS)
print(f"Warp after LEO step: {clock.warp}  (cap={cap})")
assert clock.warp <= cap, f"warp {clock.warp} exceeds cap {cap}"
print("PASS: warp cap works in LEO")
app.destroy()
```

Run: `.venv/Scripts/python <path-to-script>`
Expected output: `PASS: warp cap works in LEO` with warp < 1_000_000.

- [ ] **Step 4e-2: Verify trajectory line has points (not a conic)**

Extend the script to check the trajectory cache was populated with state-based key (not element-based):

```python
# After app.taskMgr.step():
cache = app._traj_state_cache[0]
assert cache is not None, "trajectory cache not populated"
cached_state, cached_scale = cache
assert hasattr(cached_state, 'r'), "cache key is a StateVector"
print(f"PASS: trajectory cache populated (scale={cached_scale:.1e})")
```

- [ ] **Step 4e-3: Commit**

```
git add orbitsim/render/app.py
git commit -m "$(cat <<'EOF'
Render: N-body trajectory line, warp cap, osculating HUD, N-body CA

Forward-integrated trajectory replaces Keplerian orbit ring. Warp caps
silently near bodies via max_safe_warp. HUD reads osculating_elements
(Earth/Moon SOI-switching). Preview, node marker, and CA use N-body.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push
```

---

## Self-Review

**Spec coverage:**
- `integrate_powered_nbody` in `core/flight.py` → Task 1 ✓
- `closest_approach` propagator param → Task 2 ✓
- `World.step` coast → `propagate_earth_moon` → Task 3 ✓
- `World.step` thrust → `integrate_powered_nbody` → Task 3 ✓
- Forward-integrated trajectory line (256 pts, adaptive horizon) → Task 4a ✓
- Cache keyed on `(state, scale)` with pos/vel tolerance → Task 4a ✓
- Burns rebuild every frame (warp locked to 1×) → inherits from existing thrust-locks-warp rule ✓
- Maneuver preview → N-body forward integration of post-burn state → Task 4d ✓
- Node marker → `propagate_earth_moon` → Task 4d ✓
- Warp cap silent per frame → Task 4b ✓
- `_warp_up_guarded` respects cap → Task 4b ✓
- HUD osculating elements → Task 4c ✓
- Moon-dominant altitude uses `MOON_BODY.radius_m` → Task 4c ✓
- N-body CA → Task 4d ✓
- CA marker position → `propagate_earth_moon` → Task 4d ✓
- Headless verification → Task 4e ✓

**Placeholder scan:** No TBDs, no "add appropriate X", no "similar to task N" without code.

**Type consistency:**
- `integrate_powered_nbody` returns `(StateVector, float)` — matches `integrate_powered` contract used in Task 3.
- `closest_approach(..., propagator=propagate_kepler)` — default matches all existing callers; Task 4d passes `propagate_earth_moon`.
- `_sample_trajectory(state, n_pts, max_horizon_s)` used in both Task 4a (`_rebuild_trajectory`) and Task 4d (preview) — same signature, consistent.
- `_traj_state_cache` initialized in `_start_sim` (Task 4a-1), written in `_rebuild_trajectory` (Task 4a-2), no other references needed.
