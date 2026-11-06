"""Time + warp panel (top-left): mission clock, warp buttons and readout; guarded
warp-up/max (blocked while thrusting)."""

from direct.gui.DirectButton import DirectButton
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from orbitsim.render.hud.format import fmt_mission_time, fmt_warp

# Inside PANELS["time_warp"] (anchor-relative, unscaled). Row 1: the time text;
# row 2: << . warp cell . >> . MAX. The warp text never shares space with a button.
MIN_WINDOW_PX = (1280, 720)      # the smallest supported window
WARP_TEXT_SCALE = 0.048
TIME_X = 0.07
TIME_ROW_W = 0.74                # the panel's inner width
TIME_Y = -0.105                  # baseline of the time row
BUTTON_Y = -0.180                # baseline of the warp row
WARP_DOWN_X = 0.09
WARP_CELL_X = 0.355
WARP_CELL_W = 0.42
WARP_UP_X = 0.61
MAX_X = 0.75
BUTTON_HALF_W = 0.035            # generous half-width of << / >> / MAX at their scale


class WarpControls:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    def __init__(self, app) -> None:
        self.app = app

    def build_warp_controls(self) -> None:
        """The time + warp panel. (The ',' and '.' keys do the same as its buttons.)
        Works in both sandbox and solar modes."""
        from orbitsim.render.ui.hud_layout import PANELS
        from orbitsim.render.ui.theme import THEME, button_options
        from orbitsim.render.ui.widgets import layout_panel_frame, panel_pos
        rect = PANELS["time_warp"]
        frame = layout_panel_frame(self.app, rect)
        self.app._time_warp_panel = frame
        self.app._hud_panels["time_warp"] = frame
        self.app._time_text = OnscreenText(
            text="", pos=panel_pos(rect, TIME_X, TIME_Y), scale=WARP_TEXT_SCALE,
            fg=THEME.text, shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True,
            parent=frame)
        self.app._warp_text = OnscreenText(
            text="", pos=panel_pos(rect, WARP_CELL_X, BUTTON_Y), scale=WARP_TEXT_SCALE,
            fg=THEME.cyan, shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True,
            parent=frame)
        buttons = (("<<", WARP_DOWN_X, self.app.clock.warp_down, 0.06),
                   (">>", WARP_UP_X, self.warp_up_guarded, 0.06),
                   ("MAX", MAX_X, self.max_warp_guarded, 0.05))
        self.app._warp_btns = []
        for label, x, command, scale in buttons:
            bx, bz = panel_pos(rect, x, BUTTON_Y + 0.005)
            self.app._warp_btns.append(DirectButton(
                text=label, scale=scale, pos=(bx, 0.0, bz), command=command, parent=frame,
                **button_options()))
        self.update_warp_readout()

    def update_warp_readout(self) -> None:
        if getattr(self.app, "_warp_text", None) is not None:
            locked = not self.app.solar_system and self.app.world.any_thrusting()
            self.app._time_text.setText(fmt_mission_time(self.app.clock.sim_time_s))
            self.app._warp_text.setText(fmt_warp(self.app.clock.warp, locked))

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
