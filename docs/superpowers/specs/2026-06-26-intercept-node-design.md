# Porkchop Intercept Node — Design

**Date:** 2026-06-26
**Status:** spec (awaiting build)
**Cycle:** 3 of 3 (Δv controls → target selection → **intercept node**)

## Goal

One click generates a maneuver node that sets up a **flyby/intercept** of the current
target, found by a porkchop-style sweep over **burn time × time-of-flight** that
minimizes the **departure** Δv. The player then fine-tunes it (jog sliders, node time)
and reads the resulting close approach off the existing CA markers.

## Background

- `core/optimize.py::porkchop` already does a Lambert grid, but its cost is
  **departure + arrival** Δv (a *rendezvous* — match position **and** velocity). A
  flyby only needs to **match position**, so the intercept solver minimizes the
  **departure burn only**. Reuse the Lambert/propagate machinery, change the cost.
- `lambert` (izzo2015 wrapper, SI) and `propagate_kepler` are the building blocks.
- Maneuver nodes are RTN (`dv_prograde/normal/radial`) at an epoch; `apply_maneuver`
  builds the burn basis `v_hat, h_hat, r_hat = h_hat × v_hat`. Converting an inertial
  Δv to node components is the dot product onto that same basis.

## Design

### Core solver (`core/optimize.py`, new pure function — TDD)

```
def intercept_node(
    ship_state: StateVector,
    target_state_now: StateVector,
    mu: float,
    dep_times_s: np.ndarray,   # burn times relative to ship_state.epoch_s
    tof_grid_s: np.ndarray,    # times of flight to sweep
    refine: bool = True,
) -> ManeuverNode
```

Algorithm:
1. For each `(t_dep, tof)`: `r1, v1 = propagate_kepler(ship_state, t_dep)`;
   `r2 = propagate_kepler(target_state_now, t_dep + tof).r`;
   `vdep, _ = lambert(r1, r2, tof, mu)`; **cost = ‖vdep − v1‖** (departure only).
   Infeasible Lambert cells → `inf`.
2. Take the grid argmin; if `refine`, Nelder-Mead on `(t_dep, tof)` minimizing the
   same departure cost (mirror `optimize_transfer`, but departure-only).
3. At the optimal `(t_dep*, tof*)`: recompute `r1*, v1*` and `vdep*`; the inertial
   burn is `dv_vec = vdep* − v1*`. Project onto the burn basis at `r1*, v1*`:
   `dv_pro = dv_vec·v_hat`, `dv_nrm = dv_vec·h_hat`, `dv_rad = dv_vec·r_hat`.
   Return `ManeuverNode(epoch_s = ship_state.epoch_s + t_dep*, dv_pro, dv_nrm,
   dv_rad)`.
4. If every grid cell is infeasible, raise `ValueError` (no node).

This lives next to `porkchop`/`optimize_transfer` and reuses their imports. No graphics.

### Render wiring (`render/app.py` — controller, headless screenshot)

- Add an **"Intercept"** button to the maneuver UI, enabled only when `self._target`
  is set **and** the ship orbit is bound (elliptical).
- On click: build grids from the live geometry —
  - `dep_times_s`: `linspace(0, P_ship, ~24)` over one ship period (so the burn can be
    placed anywhere on the current orbit);
  - `tof_grid_s`: `linspace(small, ~T_target or 14 d, ~48)` (Moon-scale).
  Call `intercept_node(v0.state, self._target.state_at(now), mu, ...)`. On success,
  set `self._node_epoch_s` and `self._dv["pro"/"nrm"/"rad"]` from the node, refresh the
  preview + readout, toast "Intercept planned (Δv … m/s)". On `ValueError`, toast
  "No intercept found".
- The existing magenta preview orbit and the (Cycle 2) target CA markers then show the
  resulting flyby; the player refines with the jog sliders / node-time controls.

## Components & boundaries

- `intercept_node` (pure core): geometry in, `ManeuverNode` out. Independently
  testable to a closing-the-loop tolerance.
- `app.py` (render): grid construction from live state, button gating, applying the
  node to the editor, user feedback. No physics beyond assembling inputs.

## Testing

**Pure (TDD, the key invariant — not a magic number):**
- **Closing the loop:** for a coplanar ship+target setup, apply the returned node
  (`apply_maneuver`) then `propagate_kepler` by `tof*`; the ship's position is within a
  small separation of the target's position at that time (e.g. < a few km, or a tight
  relative tolerance). This proves the node actually intercepts.
- **Departure-only is cheaper:** the returned departure Δv ≤ the dep+arr Δv at the same
  `(t_dep*, tof*)` (sanity that we optimized the flyby cost, not rendezvous).
- **Infeasible → ValueError:** a target/geometry with no Lambert solution over the grid
  raises rather than returning a bogus node.
- **RTN round-trip:** recomposing `dv_pro·v_hat + dv_nrm·h_hat + dv_rad·r_hat` equals
  the inertial `dv_vec` (within 1e-9 rel) — the projection is lossless.

**Render (headless):**
- Select Moon, click Intercept: a node with `magnitude_mps > 0` is created, and
  `closest_approach` on the predicted post-node orbit is far smaller than the
  un-maneuvered orbit's closest approach to the Moon (the flyby got closer).

## Dependencies / sequencing

Depends on **Cycle 2** (`self._target`, CA markers). Build last.

## Out of scope

- Velocity-matching (rendezvous/capture) — flyby only; matching is the later
  ships/docking cycle.
- Mid-course corrections, multi-burn plans, plane-change optimization beyond what the
  single Lambert burn yields.
