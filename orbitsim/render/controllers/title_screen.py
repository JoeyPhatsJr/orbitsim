"""Title screen: fuel-load slider, unlimited-dV toggle, launch button."""
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectCheckButton import DirectCheckButton
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectSlider import DirectSlider
from direct.gui.OnscreenText import OnscreenText


class TitleScreen:
    """Title screen: fuel-load slider, unlimited-dV toggle, launch button. Stateless: all state lives on the OrbitApp it is given."""

    def __init__(self, app) -> None:
        self.app = app

    def _build_title_screen(self) -> None:
        """Show the start menu: title, a delta-V budget slider, and a Play button.

        The sim scene and update loop are not built until Play is clicked, so the
        chosen budget is applied to every vessel before flight begins.
        """
        from orbitsim.render.ui.theme import THEME, button_options
        backdrop = DirectFrame(
            frameColor=(0.012, 0.025, 0.040, 1.0),
            frameSize=(-2.0, 2.0, -1.2, 1.2),
            parent=self.app.aspect2d,
        )
        title = OnscreenText(
            text="ORBITSIM",
            pos=(0.0, 0.55),
            scale=0.13,
            fg=THEME.text,
            parent=self.app.aspect2d,
        )
        subtitle = OnscreenText(
            text="REAL GRAVITY  /  REAL ORBITS  /  YOUR MISSION",
            pos=(0.0, 0.42),
            scale=0.05,
            fg=THEME.cyan,
            parent=self.app.aspect2d,
        )
        # Fuel-load control (delta-V is the derived readout via the rocket equation).
        default_fuel = self.app.world.vessels[0].fuel_mass_kg if self.app.world.vessels else 800.0
        self.app._budget_label = OnscreenText(
            text="",
            pos=(0.0, 0.06),
            scale=0.06,
            fg=(1.0, 0.9, 0.4, 1.0),
            mayChange=True,
            parent=self.app.aspect2d,
        )
        self.app._budget_slider = DirectSlider(
            pos=(0.0, 0.0, -0.08),
            scale=0.6,
            range=(0.0, 20000.0),
            value=default_fuel,
            pageSize=100.0,
            command=self._refresh_budget_label,
            parent=self.app.aspect2d,
        )
        hint = OnscreenText(
            text="fuel load  (drag to set)",
            pos=(0.0, -0.22),
            scale=0.045,
            fg=(0.6, 0.7, 0.85, 1.0),
            parent=self.app.aspect2d,
        )
        self.app._unlimited_check = DirectCheckButton(
            text="Unlimited dV", scale=0.05, pos=(0.0, 0.0, -0.32),
            text_fg=(1, 1, 1, 1), boxPlacement="left", parent=self.app.aspect2d,
        )
        play = DirectButton(
            text=("  OPEN SOLAR VIEWER  " if self.app.solar_system else "  LAUNCH SANDBOX  "),
            scale=0.1,
            pos=(0.0, 0.0, -0.45),
            command=self._on_play,
            parent=self.app.aspect2d,
            **button_options(),
        )
        self.app._title_nodes = [backdrop, title, subtitle, self.app._budget_label,
                             self.app._budget_slider, hint, self.app._unlimited_check, play]
        self._refresh_budget_label()

    def _refresh_budget_label(self) -> None:
        from orbitsim.core.flight import tsiolkovsky_dv

        fuel = float(self.app._budget_slider["value"])
        if self.app.world.vessels:
            v = self.app.world.vessels[0]
            dry, ve = v.dry_mass_kg, v.exhaust_velocity_mps
        else:
            dry, ve = 1000.0, 3000.0
        dv = tsiolkovsky_dv(ve, dry + fuel, dry) if fuel > 0 else 0.0
        self.app._budget_label.setText(f"Fuel: {fuel:,.0f} kg   (dV {dv:,.0f} m/s)")

    def _on_play(self) -> None:
        """Apply the chosen fuel load to all vessels, tear down the menu, start the sim."""
        fuel = float(self.app._budget_slider["value"])
        for vessel in self.app.world.vessels:
            vessel.fuel_mass_kg = fuel
        self.app._fuel_capacity = fuel if self.app.world.vessels else 0.0
        want_unlimited = bool(self.app._unlimited_check["indicatorValue"])
        for node in self.app._title_nodes:
            node.destroy() if hasattr(node, "destroy") else node.remove_node()
        self.app._title_nodes = []
        self.app._start_sim()                # builds the HUD + settings panel
        if want_unlimited:
            self.app._set_unlimited_dv(True)  # now applies flag + syncs panel + readout
