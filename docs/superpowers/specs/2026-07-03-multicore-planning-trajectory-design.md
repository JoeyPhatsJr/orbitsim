# Multi-Core Planning & Trajectory Work — Design

## Problem

On a 16-core machine the sim reads as ~6% CPU under load: Python's GIL confines all heavy
work to one thread. Three user-visible symptoms follow:

1. **Planning freeze** — `_plan_intercept` / `_toggle_porkchop` run the Lambert grid search
   *synchronously on the render thread*, so the whole game hitches for the 1–5 s the sweep
   takes. (An unused `_planning_executor` at `app.py:372` shows this was always the intent.)
2. **Frame drops while flying** — the live orbit line and the maneuver preview run on
   separate `max_workers=1` threads but GIL-serialize whenever both compute at once (common
   with a node planned).
3. **General under-utilization** — 15 idle cores while one is pegged.

## Out of scope (cannot be fixed by cores)

- **High-warp stutter.** `world.step` is a *sequential* ODE: point *i+1* needs point *i*.
  It cannot be split across cores. That is an algorithmic problem (step sizing / off-thread
  on-rails propagation with interpolation), tracked separately.
- A single 256-point trajectory sample is likewise sequential and is **not** internally
  parallelized. We parallelize *across independent samples/grid cells*, never within one.

## Approach (chosen: one persistent process pool)

Rejected alternatives: a background *thread* alone stays GIL-bound (no CPU win); *per-call*
process pools pay Windows `spawn` re-import cost (~seconds importing numpy/scipy/skyfield)
on every invocation.

Chosen: **one persistent `ProcessPoolExecutor`**, owned by `OrbitApp`, created once at
startup and reused. Both planning and trajectory work submit to it; the render thread only
ever *polls* futures — it never blocks. This matches the docs' guidance that the grid fan-out
is "self-contained to `core/optimize.py`", and reuses the existing submit/poll pattern.

## Components

### 1. The pool (`render/app.py`)
- `ProcessPoolExecutor(max_workers = max(1, min(os.cpu_count() - 1, GRID_ROWS)))`, created in
  `__init__` for sandbox mode (`self.solar_system is False`, matching today's executor guard).
- Creation wrapped in `try/except`; on **any** failure (spawn blocked, sandboxed env) fall
  back to today's `max_workers=1` `ThreadPoolExecutor`s. This fallback is also the effective
  "revert switch" if per-frame IPC ever regresses on a given machine. Preserves the
  never-crash-offline rule.
- Shut down in the existing cleanup at `app.py:158` (extend the executor list).

### 2. Planning — fan the grid across cores (Win 1)
`core/optimize.py` gains an optional `executor=None` parameter on:
- `porkchop`, `intercept_node` — already take a plain picklable `StateVector`.
- `interplanetary_porkchop` — already takes ephemeris **name strings**.
- **new** `interplanetary_departure_node` name-based path — refactored to accept
  `target_ephemeris_name` / `sun_ephemeris_name` strings instead of bound-method callables,
  so the work is picklable. Workers reconstruct geocentric states via
  `core.ephemeris.body_state(name, t, center="EARTH")`, falling back to the circular
  `core.planets` approximations offline (the exact logic today's `PlanetTarget.planning_state_at`
  uses — that method becomes a thin wrapper over the shared helper).

When `executor` is given, the **departure-time rows** are mapped across it (row granularity →
one `StateVector` / a few scalars pickled per task, ~24 tasks per sweep). When `executor is
None`, the function runs its current serial loop unchanged — keeping every existing test
graphics-free, deterministic, and offline-safe.

**Dispatch change:** `_plan_intercept` / `_toggle_porkchop` stop calling the search inline.
They submit the search to the existing `_planning_executor` **thread**, which fans grid cells
across the **process pool** and blocks *there* (off the render thread). `_update` polls the
planning future each frame and applies the resulting node / renders the porkchop PNG when the
future resolves. Result: no freeze, and the sweep finishes ~N-cores faster.

### 3. Flight-time overlap (Win 2)
Extract `_sample_trajectory` / `_sample_preview` out of `OrbitApp` into pure, picklable
module-level functions in a new `orbitsim/render/trajectory_sampling.py`. They currently touch
`self` only for `self.world.solar_system`, `self.PREDICTION_MIN_SUBSTEP_S`, and the horizon
helpers — all passable as plain arguments. Imports are pure `core/` (nbody, encounters), so
the layering rule (`render` may import `core`) holds and no Panda3D crosses the process
boundary. The `OrbitApp` methods become thin wrappers that submit to the process pool. Live
line and preview then run in different processes → true overlap instead of GIL serialization.

Worker warm-up: the first solar-propagation sample in a fresh worker loads DE440 once
(persistent pool amortizes it); offline it uses the circular fallback. Return payload is small
(256×3 float64 + a short encounter list).

## Data flow (planning, after)

```
user hits Plan
  -> _plan_intercept builds grids, submits search to _planning_executor (thread)   [render thread returns immediately]
       -> thread calls optimize.intercept_node(..., executor=self._pool)
            -> pool maps 24 departure rows across N worker processes                [cores busy]
       -> thread assembles ManeuverNode
  -> _update polls planning future; when done, applies node + flashes result        [no freeze]
```

## Testing (TDD)

- `core/optimize.py`: for a fixed geometry, the pooled result is **identical** to the serial
  result — same `dv` grid (`np.array_equal`) and same argmin — across worker counts (1, 2, 4).
  Existing serial tests remain untouched (they pass `executor=None` implicitly).
- Name-based `interplanetary_departure_node`: equals the callable-based result for the same
  inputs; offline path returns the circular-approximation answer.
- `render/trajectory_sampling.py`: the extracted function returns arrays identical to the
  pre-refactor method output for a known state (regression pin), in both solar and
  earth-moon modes.
- Fallback: pool-creation failure degrades to thread executors without raising; the full
  existing suite exercises the `executor=None` serial paths.

## Files

- `orbitsim/core/optimize.py` — add `executor` param + row fan-out; name-based interplanetary
  departure helper.
- `orbitsim/render/trajectory_sampling.py` — **new**, pure sampling functions.
- `orbitsim/render/app.py` — persistent pool + lifecycle; async planning dispatch/poll; method
  wrappers delegating to the extracted samplers.
- `orbitsim/render/targets.py` — `PlanetTarget.planning_state_at` delegates to the shared
  name-based ephemeris helper.
- `tests/core/test_optimize.py`, `tests/render/test_trajectory_sampling.py` — parity + fallback.
- `docs/06-polish-packaging.md` — mark the multi-core item done; keep the warp-stutter caveat.
