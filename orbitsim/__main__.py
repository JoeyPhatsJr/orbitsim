"""Entry point for `python -m orbitsim`: launches the render app."""
import numpy as np

from orbitsim.core.bodies import EARTH
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


def main() -> None:
    from orbitsim.render.app import OrbitApp  # imported here so tests can skip graphics

    world = _default_world()
    clock = SimClock(sim_time_s=0.0, warp=100.0)
    app = OrbitApp(world, clock)
    app.run_app()


if __name__ == "__main__":
    main()
