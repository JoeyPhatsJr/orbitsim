"""F5/F9 quicksave and in-place quickload of the sandbox."""
from orbitsim.sim.persistence import load_scenario
from orbitsim.sim.persistence import save_scenario


class Quicksave:
    """F5/F9 quicksave and in-place quickload of the sandbox. Stateless: all state lives on the OrbitApp it is given."""

    def __init__(self, app) -> None:
        self.app = app

    def save(self) -> None:
        """F5: write the current sandbox world + clock to the quicksave slot."""
        try:
            save_scenario(self.app.world, self.app.clock, self.app.QUICKSAVE_PATH)
            self.app._flash_message("Quicksaved")
        except (OSError, ValueError) as exc:
            self.app._flash_message(f"Quicksave failed: {exc}")

    def load(self) -> None:
        """F9: restore the quicksave in place onto the live vessel + clock.

        Mutating the existing vessel (rather than swapping self.app.world) keeps the
        Panda3D scene graph intact. Sandbox-only: one vessel, fixed central body.
        """
        try:
            world, clock = load_scenario(self.app.QUICKSAVE_PATH)
        except (OSError, ValueError) as exc:
            self.app._flash_message(f"Quickload failed: {exc}")
            return
        src = world.vessels[0]
        dst = self.app.world.vessels[0]
        dst.state = src.state
        dst.dry_mass_kg = src.dry_mass_kg
        dst.fuel_mass_kg = src.fuel_mass_kg
        dst.max_thrust_n = src.max_thrust_n
        dst.exhaust_velocity_mps = src.exhaust_velocity_mps
        dst.max_turn_rate_radps = src.max_turn_rate_radps
        dst.throttle = src.throttle
        dst.sas_mode = src.sas_mode
        dst.orientation = src.orientation
        dst.nodes[:] = src.nodes
        self.app.world.solar_system = world.solar_system
        self.app.clock.sim_time_s = clock.sim_time_s
        self.app.clock.warp = clock.warp
        self.app._flash_message("Quickloaded")
