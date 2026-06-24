"""Minimal DirectGUI overlay. Converts SI -> km/UTC at this boundary only."""
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class Hud:
    """On-screen text panel showing time, warp, and focused-vessel orbit info."""

    def __init__(self, base) -> None:
        self.text = OnscreenText(
            text="",
            pos=(-1.3, 0.9),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
            parent=base.a2dTopLeft if hasattr(base, "a2dTopLeft") else None,
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
