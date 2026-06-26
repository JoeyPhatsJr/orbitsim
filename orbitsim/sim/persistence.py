"""Save/load the sandbox world as versioned JSON (Phase 6.1).

Sandbox-only. Bodies are referenced by name via BODY_REGISTRY and never
serialized; each vessel's StateVector mu is rebuilt from the central body
on load. Python's json writes float64 via repr(), so numbers round-trip
exactly.
"""
import json
import os

import numpy as np

from orbitsim.core.bodies import (
    SUN, EARTH, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, URANUS, NEPTUNE,
)
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.state import StateVector
from orbitsim.sim.clock import SimClock
from orbitsim.sim.world import Vessel, World

SCHEMA_VERSION = 1

BODY_REGISTRY = {
    "Sun": SUN, "Earth": EARTH, "Moon": MOON, "Mercury": MERCURY,
    "Venus": VENUS, "Mars": MARS, "Jupiter": JUPITER, "Saturn": SATURN,
    "Uranus": URANUS, "Neptune": NEPTUNE,
}


def _node_to_dict(node: ManeuverNode) -> dict:
    return {
        "epoch_s": node.epoch_s,
        "dv_prograde_mps": node.dv_prograde_mps,
        "dv_normal_mps": node.dv_normal_mps,
        "dv_radial_mps": node.dv_radial_mps,
    }


def _vessel_to_dict(vessel: Vessel) -> dict:
    return {
        "name": vessel.name,
        "r_m": vessel.state.r.tolist(),
        "v_mps": vessel.state.v.tolist(),
        "epoch_s": vessel.state.epoch_s,
        "dry_mass_kg": vessel.dry_mass_kg,
        "fuel_mass_kg": vessel.fuel_mass_kg,
        "max_thrust_n": vessel.max_thrust_n,
        "exhaust_velocity_mps": vessel.exhaust_velocity_mps,
        "max_turn_rate_radps": vessel.max_turn_rate_radps,
        "throttle": vessel.throttle,
        "sas_mode": vessel.sas_mode,
        "orientation": np.asarray(vessel.orientation).tolist(),
        "unlimited_dv": vessel.unlimited_dv,
        "nodes": [_node_to_dict(n) for n in vessel.nodes],
    }


def save_scenario(world: World, clock: SimClock, path) -> None:
    """Serialize the sandbox world + clock to versioned JSON at `path`."""
    data = {
        "schema": SCHEMA_VERSION,
        "kind": "sandbox",
        "central": world.central.name,
        "sim_time_s": clock.sim_time_s,
        "warp": clock.warp,
        "vessels": [_vessel_to_dict(v) for v in world.vessels],
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _vessel_from_dict(d: dict, mu: float) -> Vessel:
    try:
        state = StateVector(
            r=np.array(d["r_m"], dtype=np.float64),
            v=np.array(d["v_mps"], dtype=np.float64),
            mu=mu,
            epoch_s=d["epoch_s"],
        )
        vessel = Vessel(
            name=d["name"],
            state=state,
            dry_mass_kg=d["dry_mass_kg"],
            fuel_mass_kg=d["fuel_mass_kg"],
            max_thrust_n=d["max_thrust_n"],
            exhaust_velocity_mps=d["exhaust_velocity_mps"],
            max_turn_rate_radps=d["max_turn_rate_radps"],
            throttle=d["throttle"],
            sas_mode=d["sas_mode"],
            orientation=np.array(d["orientation"], dtype=np.float64),
            unlimited_dv=d.get("unlimited_dv", False),
        )
        for n in d["nodes"]:
            vessel.nodes.append(ManeuverNode(
                epoch_s=n["epoch_s"],
                dv_prograde_mps=n["dv_prograde_mps"],
                dv_normal_mps=n["dv_normal_mps"],
                dv_radial_mps=n["dv_radial_mps"],
            ))
    except KeyError as exc:
        raise ValueError(f"Save file missing required field: {exc}") from exc
    return vessel


def load_scenario(path) -> tuple[World, SimClock]:
    """Load a sandbox save from `path`; return (World, SimClock).

    Raises ValueError on an unknown schema version, unknown central body,
    or malformed/missing fields.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed save file: {exc}") from exc

    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported save schema {data.get('schema')!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    name = data.get("central")
    if name not in BODY_REGISTRY:
        raise ValueError(f"Unknown central body {name!r} in save file")
    central = BODY_REGISTRY[name]

    try:
        vessels = [_vessel_from_dict(v, central.mu) for v in data["vessels"]]
        clock = SimClock(sim_time_s=data["sim_time_s"], warp=data["warp"])
    except KeyError as exc:
        raise ValueError(f"Save file missing required field: {exc}") from exc

    return World(central=central, vessels=vessels), clock
