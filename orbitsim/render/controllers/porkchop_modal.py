"""Transfer-window (porkchop) analysis overlay, computed off the render thread."""
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectFrame import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from orbitsim.core.constants import MU_SUN
from orbitsim.core.elements import state_to_elements
from orbitsim.core.optimize import porkchop
from orbitsim.core.planets import A_EARTH
from orbitsim.core.planets import A_JUPITER
from orbitsim.core.planets import A_MARS
from orbitsim.core.planets import A_MERCURY
from orbitsim.core.planets import A_NEPTUNE
from orbitsim.core.planets import A_SATURN
from orbitsim.core.planets import A_URANUS
from orbitsim.core.planets import A_VENUS
from orbitsim.core.state import StateVector
from orbitsim.render.porkchop import render_porkchop_png
from panda3d.core import CardMaker
import numpy as np


class PorkchopModal:
    """Transfer-window (porkchop) analysis overlay, computed off the render thread. Stateless: all state lives on the OrbitApp it is given."""

    def __init__(self, app) -> None:
        self.app = app

    def toggle_porkchop(self) -> None:
        """Build a porkchop plot and overlay it. Uses the current target if one
        is selected (planet or Moon); otherwise falls back to a generic higher-orbit
        demo. Pressing 'p' again removes it.
        """
        if self.app.solar_system:
            return  # transfer planning needs a vessel; the solar viewer has none
        existing = getattr(self.app, "_porkchop_card", None)
        if existing is not None:
            existing.remove_node()
            self.app._porkchop_card = None
            modal = getattr(self.app, "_porkchop_modal", None)
            if modal is not None:
                modal.destroy()
                self.app._porkchop_modal = None
            return

        if self.app._porkchop_future is not None and not self.app._porkchop_future.done():
            self.app._flash_message("Computing porkchop...")
            return

        mu = self.app.world.central.mu
        ship = self.app.world.vessels[0].state
        grid_executor = self.app._grid_executor

        from orbitsim.render.targets import PlanetTarget
        if self.app._target is not None and isinstance(self.app._target, PlanetTarget):
            from orbitsim.core.optimize import interplanetary_porkchop
            from orbitsim.core.constants import MU_SUN
            from orbitsim.core.planets import A_EARTH
            target_a = {
                "Mercury": A_MERCURY, "Venus": A_VENUS, "Mars": A_MARS,
                "Jupiter": A_JUPITER, "Saturn": A_SATURN,
                "Uranus": A_URANUS, "Neptune": A_NEPTUNE,
            }.get(self.app._target.name)
            if target_a is not None:
                from orbitsim.core.planets import _N_EARTH
                w_target = np.sqrt(MU_SUN / target_a**3)
                synodic = 2.0 * np.pi / abs(_N_EARTH - w_target)
                hohmann_tof = np.pi * np.sqrt(((A_EARTH + target_a) / 2.0)**3 / MU_SUN)
            else:
                synodic = 365.25 * 86400.0
                hohmann_tof = 180.0 * 86400.0
            dep_times = np.linspace(0.0, min(synodic, 2 * 365.25 * 86400.0), 36)
            tof_grid = np.linspace(0.3 * hohmann_tof, 2.0 * hohmann_tof, 36)
            label = self.app._target.name
            target_name = self.app._target.name.upper()
            abs_dep = ship.epoch_s + dep_times

            def job():
                dv, _ = interplanetary_porkchop(
                    "EARTH", target_name, abs_dep, tof_grid, executor=grid_executor)
                return dv
        elif self.app._target is not None:
            now = self.app.clock.sim_time_s
            from orbitsim.core.elements import state_to_elements
            try:
                period = state_to_elements(ship).period_s
            except ValueError:
                self.app._flash_message("Unbound orbit — can't compute porkchop")
                return
            dep_times = np.linspace(0.0, period, 24)
            tof_grid = np.linspace(3.0e3, 14.0 * 86400.0, 36)
            target_state = self.app._target.state_at(now)
            label = self.app._target.name

            def job():
                dv, _ = porkchop(
                    ship, target_state, dep_times, tof_grid, mu, executor=grid_executor)
                return dv
        else:
            r1 = ship.r_mag
            r2 = r1 * 2.0
            v2 = np.sqrt(mu / r2)
            arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, v2, 0.0]), mu=mu)
            w1 = np.sqrt(mu / r1**3)
            w2 = np.sqrt(mu / r2**3)
            t_syn = 2.0 * np.pi / abs(w1 - w2)
            t_hohmann = np.pi * np.sqrt(((r1 + r2) / 2.0) ** 3 / mu)
            dep_times = np.linspace(0.0, t_syn, 24)
            tof_grid = np.linspace(0.4 * t_hohmann, 1.6 * t_hohmann, 36)
            label = "transfer"

            def job():
                dv, _ = porkchop(ship, arr, dep_times, tof_grid, mu, executor=grid_executor)
                return dv

        self.app._flash_message(f"Computing {label} porkchop...")
        self.app._porkchop_render = (dep_times, tof_grid)
        self.app._porkchop_future = self.app._planning_executor.submit(job)

    def poll_porkchop(self) -> None:
        """Draw the porkchop overlay once its background grid finishes."""
        future = getattr(self.app, "_porkchop_future", None)
        if future is None or not future.done():
            return
        self.app._porkchop_future = None
        try:
            dv = future.result()
        except Exception:
            import traceback
            traceback.print_exc()
            self.app._flash_message("Porkchop failed")
            self.app._porkchop_render = None
            return
        dep_times, tof_grid = self.app._porkchop_render
        self.app._porkchop_render = None
        self._build_porkchop_modal(dv, dep_times, tof_grid)

    def _build_porkchop_modal(self, dv, dep_times, tof_grid) -> None:
        """Render the porkchop PNG and build its overlay modal (render thread only)."""
        png = render_porkchop_png(dv, dep_times, tof_grid, "porkchop.png")
        from orbitsim.render.ui.theme import THEME, button_options
        self.app._porkchop_modal = DirectFrame(
            frameColor=THEME.panel_alt,
            frameSize=(-0.92, 0.92, -0.72, 0.72),
            parent=self.app.aspect2d,
        )
        OnscreenText(
            text="TRANSFER WINDOW ANALYSIS", pos=(0, 0.64), scale=0.052,
            fg=THEME.cyan, parent=self.app._porkchop_modal,
        )
        tex = self.app.loader.load_texture(png)
        cm = CardMaker("porkchop")
        cm.set_frame(-0.78, 0.78, -0.48, 0.50)
        card = self.app._porkchop_modal.attach_new_node(cm.generate())
        card.set_texture(tex)
        self.app._porkchop_card = card
        DirectButton(
            text="CREATE MANEUVER", scale=0.048, pos=(0.38, 0, -0.61),
            command=self.app.targeting.plan_intercept, parent=self.app._porkchop_modal, **button_options(),
        )
        DirectButton(
            text="CLOSE", scale=0.048, pos=(-0.38, 0, -0.61),
            command=self.toggle_porkchop, parent=self.app._porkchop_modal, **button_options(),
        )
