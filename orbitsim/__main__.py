"""Entry point for `python -m orbitsim`: launches the render app.

Use ``python -m orbitsim`` for the LEO sandbox, or ``python -m orbitsim --solar``
for the real solar system (JPL DE440 ephemerides).
"""
import sys

import numpy as np

from orbitsim.core.bodies import EARTH, SUN
from orbitsim.core.constants import MU_EARTH, R_EARTH
from orbitsim.core.state import StateVector
from orbitsim.sim.clock import SimClock
from orbitsim.sim.world import Vessel, World


def _default_world() -> World:
    # A slightly eccentric, inclined LEO so the orbit line is visibly non-circular.
    r0 = R_EARTH + 500e3
    v_circ = np.sqrt(MU_EARTH / r0)
    state = StateVector(
        r=np.array([r0, 0.0, 0.0]),
        v=np.array([0.0, v_circ * 1.05, v_circ * 0.15]),
        mu=MU_EARTH,
    )
    vessel = Vessel(name="Sandbox-1", state=state, delta_v_budget_mps=2000.0)
    return World(central=EARTH, vessels=[vessel])


def _solar_world() -> World:
    """A Sun-centered world with no vessels: a viewer for the real solar system."""
    return World(central=SUN, vessels=[])


def main() -> None:
    from orbitsim.render.app import OrbitApp  # imported here so tests can skip graphics

    solar = "--solar" in sys.argv
    if solar:
        world = _solar_world()
        # ~30 years past J2000 (2030) so the planets sit at well-known positions; fast warp.
        clock = SimClock(sim_time_s=30.0 * 365.25 * 86400.0, warp=1_000_000.0)
    else:
        world = _default_world()
        clock = SimClock(sim_time_s=0.0, warp=100.0)
    app = OrbitApp(world, clock, solar_system=solar)
    app.run_app()


if __name__ == "__main__":
    main()
