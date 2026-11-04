"""Time-warp buttons, readout, and guarded warp-up/max (blocked while thrusting)."""

from direct.gui.DirectButton import DirectButton
from direct.gui.DirectFrame import DirectFrame
from direct.gui.OnscreenText import OnscreenText


class WarpControls:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    def __init__(self, app) -> None:
        self.app = app

    def build_warp_controls(self) -> None:
        """Top-center on-screen warp control: slower / faster buttons + readout.
        (The ',' and '.' keys do the same.) Works in both sandbox and solar modes."""
        from orbitsim.render.ui.theme import THEME, button_options
        self.app._warp_frame = DirectFrame(
            frameColor=THEME.panel,
            frameSize=(-0.64, 0.64, -0.16, -0.02),
            parent=self.app.a2dTopCenter,
        )
        self.app._warp_readout = OnscreenText(
            text="", pos=(0.0, -0.09), scale=0.045, fg=THEME.text,
            shadow=(0, 0, 0, 1), mayChange=True, parent=self.app._warp_frame,
        )
        self.app._warp_btns = [
            DirectButton(text="<<", scale=0.05, pos=(-0.28, 0.0, -0.085),
                         command=self.app.clock.warp_down, parent=self.app._warp_frame,
                         **button_options()),
            DirectButton(text=">>", scale=0.05, pos=(0.28, 0.0, -0.085),
                         command=self.warp_up_guarded, parent=self.app._warp_frame,
                         **button_options()),
            DirectButton(text="MAX", scale=0.042, pos=(0.48, 0.0, -0.085),
                         command=self.max_warp_guarded, parent=self.app._warp_frame,
                         **button_options()),
        ]
        self.update_warp_readout()

    def update_warp_readout(self) -> None:
        if getattr(self.app, "_warp_readout", None) is not None:
            locked = not self.app.solar_system and self.app.world.any_thrusting()
            suffix = "  (LOCKED)" if locked else ""
            days = self.app.clock.sim_time_s / 86400.0
            self.app._warp_readout.setText(
                f"T+ {days:,.1f} DAYS     WARP  x{self.app.clock.warp:,.0f}{suffix}"
            )

    def warp_up_guarded(self):
        if self.app.world.any_thrusting():
            return
        if self.app.world.vessels:
            from orbitsim.sim.clock import SimClock
            if self.app.world.solar_system:
                from orbitsim.core.nbody import max_safe_warp_solar
                cap = max_safe_warp_solar(
                    self.app.world.vessels[0].state, self.app.clock.sim_time_s, SimClock.WARP_STEPS)
            else:
                from orbitsim.core.nbody import max_safe_warp
                cap = max_safe_warp(
                    self.app.world.vessels[0].state, self.app.clock.sim_time_s, SimClock.WARP_STEPS)
            if self.app.clock.warp >= cap:
                return
        self.app.clock.warp_up()

    def max_warp_guarded(self):
        """Jump directly to the highest safe warp (up to 100,000,000x in deep space)."""
        if self.app.world.any_thrusting():
            return
        from orbitsim.sim.clock import SimClock

        cap = float(SimClock.WARP_STEPS[-1])
        if self.app.world.vessels:
            if self.app.world.solar_system:
                from orbitsim.core.nbody import max_safe_warp_solar

                cap = max_safe_warp_solar(
                    self.app.world.vessels[0].state,
                    self.app.clock.sim_time_s,
                    SimClock.WARP_STEPS,
                )
            else:
                from orbitsim.core.nbody import max_safe_warp

                cap = max_safe_warp(
                    self.app.world.vessels[0].state,
                    self.app.clock.sim_time_s,
                    SimClock.WARP_STEPS,
                )
        self.app.clock.warp = float(cap)
        self.update_warp_readout()
