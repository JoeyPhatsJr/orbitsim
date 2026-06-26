# Scheduled Maneuver Nodes (Phase 6.2, cycle A2) — Design

**Date:** 2026-06-25
**Phase:** 6.2 gameplay (second of the 6.2 cycles).
**Scope:** turn the sandbox's immediate-execute maneuver editor into a KSP-style **scheduled**
single-node planner: place one node ahead on the orbit, edit its Δv, watch the predicted
trajectory, warp toward it (with auto-warp-down), and execute the burn at the node.

## Decisions (locked in brainstorming)

- **Execution model:** auto-warp-down as the node approaches, then **manual execute** — the player
  triggers the impulsive burn at/near the node (reusing the existing impulsive apply + fuel spend).
- **One node at a time** (no chaining). The "maneuver list" is therefore a single pending-node
  readout.
- **Timing:** a time-to-node control (+/- buttons) plus **Next Periapsis / Next Apoapsis** presets;
  a node marker on the orbit and the existing magenta preview both reflect the node.

## The core model shift (and the one subtle correctness point)

Today `_current_node()` builds a node at the *current* epoch and "Execute Burn" applies it
immediately. The new model gives the node a **fixed absolute epoch** set when the player places it;
the displayed **time-to-node counts down** as sim time advances:

```
time_to_node = node_epoch_s − clock.sim_time_s        # decreases as you warp forward
```

**Subtlety to get right:** the node epoch must be stored absolutely and held fixed — NOT recomputed
as `now + ttn` each frame (that would make the node perpetually ttn ahead and never arrive). The
+/- buttons and the Pe/Ap presets *re-anchor* `node_epoch_s` relative to the current time; after
that it stays put and the clock walks toward it.

## Components

### Core (pure physics, `orbitsim/core/maneuvers.py`) — TDD

Two functions, bound (elliptical) orbits only (raise `ValueError` for `e ≥ 1` / `a ≤ 0`):

- `time_to_periapsis(state: StateVector) -> float` — seconds until the vessel next passes ν=0.
- `time_to_apoapsis(state: StateVector) -> float` — seconds until it next passes ν=π.

Implementation: `elem = state_to_elements(state)`; current mean anomaly
`M = eccentric_to_mean_anomaly(true_to_eccentric_anomaly(elem.nu, elem.e), elem.e)`; mean motion
`n = 2π / elem.period_s`; then

```
time_to_periapsis = ((2π − M) mod 2π) / n      # M target = 2π (next periapsis)
time_to_apoapsis  = ((π  − M) mod 2π) / n      # M target = π  (apoapsis)
```

Both return a value in `[0, period)`. Known-answer tests: from apoapsis (ν=π), time_to_periapsis =
T/2 and time_to_apoapsis = 0; from periapsis (ν=0), time_to_periapsis = 0 and time_to_apoapsis =
T/2; invariant `0 ≤ t < period` for arbitrary ν.

The node's marker position is `propagate_kepler(state, time_to_node).r` — no new core code.

### Render (`orbitsim/render/app.py` maneuver UI) — controller, headless-verified

Editor state becomes `self._node_epoch_s: float | None` (None = no scheduled node) plus the existing
`self._dv` (jog-slider Δv components). `_current_node()` builds
`ManeuverNode(epoch_s=self._node_epoch_s if set else current epoch, dv…)`.

- **Time-to-node control:** `−`/`+` buttons step `self._node_epoch_s` by a coarse amount (e.g. ±30 s,
  or ±10% of period); a readout shows `T-MM:SS`. A "Clear node" button sets it back to None.
- **Next Periapsis / Next Apoapsis** buttons: set
  `self._node_epoch_s = clock.sim_time_s + time_to_periapsis(state)` (resp. apoapsis).
- **Node marker:** a small marker (distinct color, e.g. cyan) drawn on the orbit at
  `propagate_kepler(state, time_to_node).r`, shown only while a node is scheduled.
- **Preview:** the existing magenta post-burn orbit, now from `predict_elements_after(state, node)`
  with the node's future epoch (already supported).
- **Pending-node readout** (the "list", single entry): `Node in T-MM:SS — Δv X m/s (P+.. N+.. R+..)`,
  or hidden when no node.
- **Auto-warp-down:** in the sandbox update loop, when a node is scheduled and
  `time_to_node ≤ AUTO_WARP_LEAD_S × clock.warp` (i.e. fewer than ~AUTO_WARP_LEAD_S real-seconds
  away at current warp), call `clock.warp_down()` one step per qualifying frame until at 1× by the
  node. `AUTO_WARP_LEAD_S` ≈ 5 s. Never auto-warps *up*.
- **Execute:** the existing "Execute Burn" button. Enabled (and effective) only when no node is
  scheduled (immediate burn, today's behavior) OR `time_to_node ≤ EXECUTE_TOLERANCE_S` (≈ 2 s) — so
  the impulsive burn is applied essentially at the node. On execute: apply impulse + spend fuel
  (existing `_execute_burn`), then clear the node (`self._node_epoch_s = None`). When a node is
  scheduled but not yet due, the button is disabled/greyed.

### Persistence integration (free)

`vessel.nodes` (already saved/loaded) is kept as the single source of truth for the scheduled node:
each frame the editor mirrors its current node into `vessel.nodes = [node]` (or `[]` when none), so
a quicksave during planning restores the scheduled node. The save/load schema is unchanged
(`ManeuverNode` already carries `epoch_s`).

## Architecture / boundaries

- Core: two new pure functions + their tests. No graphics. Imports `state_to_elements`, the kepler
  anomaly helpers, `period_s`.
- Render: changes confined to the maneuver UI region of `app.py` (`_build_maneuver_ui`,
  `_current_node`, `_execute_burn`, `_refresh_readout`, the update loop) plus the new buttons/marker.
- No changes to the physics of `apply_maneuver` / `propagate_kepler` / the flight integrator.

## Testing

- Core (TDD, `tests/core/`): `time_to_periapsis`/`time_to_apoapsis` known answers (T/2 and 0 from
  the apsides), the `0 ≤ t < period` invariant over random ν (hypothesis), and `ValueError` for a
  hyperbolic state.
- Render (controller, headless): set ttn via the buttons → `node_epoch_s` re-anchors and the readout
  shows the right `T-MM:SS`; "Next Periapsis" sets `node_epoch_s ≈ now + time_to_periapsis`;
  stepping the clock toward the node triggers `clock.warp` stepping down to 1×; executing within
  tolerance applies the burn (orbit changes, fuel drops) and clears the node; a node survives a
  quicksave/quickload.

## Out of scope

- Multiple/chained nodes; dragging the node along the orbit in 3D; flown finite burns; node editing
  for vessels other than vessel 0; rendezvous/target (its own cycle, A3).

## Definition of done

- Core functions implemented + tests green; full suite green.
- In a headless run: place a node (manual ttn and via Pe/Ap), see marker + preview + readout, warp
  in and watch auto-warp-down, execute at the node (orbit + fuel change, node clears), and confirm a
  scheduled node round-trips through quicksave/quickload. No physics modules changed.
