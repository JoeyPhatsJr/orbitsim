# 03 — Phase 3: Sandbox & Maneuver Nodes

**Gate:** Phase 2 renders one orbit with a working camera/clock.

Goal: let the user add a maneuver node on the orbit, dial in a ΔV (prograde/normal/radial), and
see the **predicted** resulting orbit live before committing the burn. This is the KSP-style core
of the sandbox.

---

## Task 3.1 — `core/maneuvers.py`

Impulsive ΔV in the **vessel's local orbital frame (RTN/LVLH)**, applied at a point on the orbit.

```python
@dataclass(frozen=True)
class ManeuverNode:
    epoch_s: float            # when the burn happens (sim seconds past J2000)
    dv_prograde_mps: float    # +along velocity
    dv_normal_mps: float      # +along orbital angular momentum (h)
    dv_radial_mps: float      # +along radial (outward)

def apply_maneuver(state: StateVector, node: ManeuverNode) -> StateVector:
    """Propagate state to node.epoch_s, convert (prograde,normal,radial) to inertial using the
    local basis, add to velocity, return new StateVector (same position, new velocity)."""
```

Local basis at the burn point:
```
v_hat = v / |v|                      # prograde
h_hat = (r × v) / |r × v|            # normal (orbit-normal)
r_hat = (h_hat × v_hat)              # radial-out (NOT r/|r|; this gives an orthonormal RTN frame)
dv_inertial = dv_prograde*v_hat + dv_normal*h_hat + dv_radial*r_hat
```

**Tests:**
- A small **prograde** burn at periapsis raises **apoapsis** only (periapsis radius unchanged) —
  assert new `apoapsis > old`, `periapsis ≈ old` within 1 m.
- A **normal** burn changes inclination but not the semi-major axis (energy) — assert `a` constant
  within 1e-6, `i` changed.
- Total ΔV magnitude added equals `√(dvp²+dvn²+dvr²)` (energy bookkeeping sanity).

## Task 3.2 — predicted-orbit preview

`core` function: given current orbit + a `ManeuverNode`, return the post-burn `KeplerianElements`
(via `apply_maneuver` then `state_to_elements`). The renderer draws this as a second, distinct
orbit line (Task 2.4 machinery, different color). Updates live as the user drags ΔV sliders.

## Task 3.3 — node editing UI (`render/hud/`)

- Click the orbit to drop a node (map click → nearest true anomaly on the orbit).
- Drag handles / sliders for prograde, normal, radial ΔV (and node time).
- Readout: ΔV cost vs remaining `delta_v_budget_mps`; new Pe/Ap/period/inclination.
- "Execute" commits: at `node.epoch_s` the clock auto-warps down and the burn is applied
  (impulsive for now; finite-burn modeling is out of scope).

## Task 3.4 — sandbox scenario

A default sandbox: empty Earth orbit, full ΔV budget, free node creation. Wire into
`sim/persistence.py` (Phase 6 formalizes save/load) so a sandbox state can be reloaded.

## Phase 3 exit criteria
- User can add a node, see the predicted orbit update live, execute the burn, and watch the real
  orbit become the predicted one.
- All `core/maneuvers.py` tests green; preview matches post-burn reality within 1 m after execute.
