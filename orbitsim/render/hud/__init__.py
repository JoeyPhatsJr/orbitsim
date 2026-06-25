"""Minimal DirectGUI overlay. Converts SI -> km/UTC at this boundary only."""
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class Hud:
    """On-screen text panel showing time, warp, and focused-vessel orbit info."""

    def __init__(self, base) -> None:
        # Anchor to the top-left corner; pos is corner-relative (x right, y down),
        # so a small +x / -y nudge places the text just inside the corner.
        self.text = OnscreenText(
            text="",
            pos=(0.08, -0.12),
            scale=0.05,
            fg=(1, 1, 1, 1),
            shadow=(0, 0, 0, 1),
            align=TextNode.ALeft,
            mayChange=True,
            parent=base.a2dTopLeft,
        )
        # Flight readout in the top-right corner.
        self.flight = OnscreenText(
            text="",
            pos=(-0.05, -0.12),
            scale=0.05,
            fg=(0.8, 1.0, 0.8, 1),
            shadow=(0, 0, 0, 1),
            align=TextNode.ARight,
            mayChange=True,
            parent=base.a2dTopRight,
        )

    def update(
        self,
        *,
        sim_time_s: float,
        warp: float,
        altitude_m: float,
        speed_mps: float,
        periapsis_m: float,
        apoapsis_m: float,
        period_s: float,
    ) -> None:
        lines = [
            f"Sim time: {sim_time_s:,.0f} s past J2000",
            f"Warp: x{warp:,.0f}",
            f"Altitude: {altitude_m / 1000.0:,.1f} km",
            f"Speed: {speed_mps / 1000.0:,.3f} km/s",
            f"Periapsis: {periapsis_m / 1000.0:,.1f} km",
            f"Apoapsis: {apoapsis_m / 1000.0:,.1f} km",
            f"Period: {period_s / 60.0:,.1f} min",
        ]
        self.text.setText("\n".join(lines))

    def update_flight(
        self,
        *,
        throttle: float,
        fuel_kg: float,
        fuel_frac: float,
        mass_kg: float,
        thrust_n: float,
        twr: float,
        dv_remaining: float,
        warp_locked: bool,
    ) -> None:
        lines = [
            f"Throttle: {throttle * 100:,.0f}%",
            f"Fuel: {fuel_frac * 100:,.0f}%  ({fuel_kg:,.0f} kg)",
            f"Mass: {mass_kg:,.0f} kg",
            f"Thrust: {thrust_n / 1000:,.1f} kN   TWR: {twr:,.2f}",
            f"dV left: {dv_remaining:,.0f} m/s",
        ]
        if warp_locked:
            lines.append("WARP LOCKED - thrusting")
        self.flight.setText("\n".join(lines))
