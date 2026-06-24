# 04 — Phase 4: Transfers & ΔV-Budget Optimization

**Gate:** Phase 3 maneuver nodes work.

Goal: the "mission planner." Closed-form transfers, a Lambert solver for arbitrary intercepts,
and a porkchop-plot-based ΔV optimizer.

All in `core/transfers.py` and `core/optimize.py`. Pure functions, fully testable, SI units.

---

## Task 4.1 — closed-form transfers (`core/transfers.py`)

Between coplanar circular orbits radii `r1`, `r2` about a body with `μ`:

**Hohmann** (two burns):
```
a_t   = (r1 + r2) / 2
dv1   = √(μ/r1) · (√(2 r2/(r1+r2)) − 1)
dv2   = √(μ/r2) · (1 − √(2 r1/(r1+r2)))
dv_total = |dv1| + |dv2|
t_transfer = π · √(a_t³/μ)
```
**Bi-elliptic** (three burns, via apoapsis `rb ≥ r2`): implement the standard 3-burn formula;
return its total ΔV so the optimizer can compare. (Bi-elliptic wins when `r2/r1 ≳ 11.94`.)

**Simple plane change** at speed `v`, angle `Δi`: `dv = 2 v · sin(Δi/2)`.

```python
@dataclass(frozen=True)
class TransferSolution:
    burns: list[ManeuverNode]
    dv_total_mps: float
    time_of_flight_s: float
    kind: str   # "hohmann" | "bielliptic" | "lambert" | ...
```

**Known-answer tests:**
- LEO→GEO Hohmann: `r1=6678 km`, `r2=42164 km`, μ_earth. Expect `dv1≈2.42 km/s`,
  `dv2≈1.47 km/s`, `dv_total≈3.89 km/s`, `t≈5.26 h`. Tolerance 1%.
- Bi-elliptic vs Hohmann crossover near `r2/r1 = 11.94` (assert which is cheaper on each side).

## Task 4.2 — Lambert solver (`core/transfers.py`)

Use **`lamberthub`** (`from lamberthub import izzo2015`). Wrap it:
```python
def lambert(r1, r2, tof_s, mu, prograde=True, revs=0) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's problem: velocities v1, v2 connecting r1→r2 in time tof_s.
    Thin wrapper around lamberthub.izzo2015 with SI in/out. Validate inputs."""
```
Then `intercept(state_chaser, state_target, tof_s)` → the ΔV at departure (`v1 − v_chaser`) and
arrival (`v_target − v2`), returned as a `TransferSolution`.

**Tests:** a Lambert arc whose `tof` equals a Hohmann transfer time, between the matching radii,
must reproduce the Hohmann ΔV within 1% (cross-validates 4.1 and 4.2). Round-trip: propagating
`r1` with the solved `v1` for `tof` lands on `r2` within 1 km.

## Task 4.3 — ΔV optimizer & porkchop (`core/optimize.py`)

```python
def porkchop(departure_states, arrival_states, dep_times, arr_times, mu) -> np.ndarray:
    """Grid of Lambert solves: total ΔV[i,j] over departure_time[i] × arrival_time[j].
    Returns the 2D ΔV array (for plotting) plus the argmin (best window)."""
def optimize_transfer(...) -> TransferSolution:
    """Coarse porkchop grid → scipy.optimize.minimize (Nelder-Mead/L-BFGS-B) refine near argmin."""
```
Render a porkchop contour plot in the HUD (matplotlib offscreen → texture, or Panda3D mesh).
Clicking a cell sets up the corresponding maneuver nodes via Phase 3.

**Tests:** on a synthetic two-circular-orbit case, the porkchop minimum ΔV must match the Hohmann
ΔV (the global optimum for coplanar circular) within a few %.

## Phase 4 exit criteria
- Hohmann/bi-elliptic/plane-change tools produce correct ΔV (textbook-validated).
- Lambert intercepts plan a rendezvous that visibly connects two orbits in the renderer.
- Porkchop plot renders; selecting a window creates executable maneuver nodes.
