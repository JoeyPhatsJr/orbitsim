"""Minimal DirectGUI overlay. Converts SI -> km/UTC at this boundary only."""
import math

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

_MI_PER_KM = 0.621371


def _dist(meters: float, units: str) -> str:
    """Format a distance [m] as km or mi (1 km = 0.621371 mi), one decimal."""
    km = meters / 1000.0
    if units == "mi":
        return f"{km * _MI_PER_KM:,.1f} mi"
    return f"{km:,.1f} km"


def _speed(mps: float, units: str) -> str:
    """Format a speed [m/s] as km/s or mi/s, three decimals."""
    kms = mps / 1000.0
    if units == "mi":
        return f"{kms * _MI_PER_KM:,.3f} mi/s"
    return f"{kms:,.3f} km/s"


def orbit_panel_lines(
    *, sim_time_s: float, warp: float, altitude_m: float, speed_mps: float,
    periapsis_m: float, apoapsis_m: float, period_s: float,
    inclination_rad: float, units: str,
) -> list[str]:
    """Build the orbit-info panel text lines. Pure (no DirectGUI) so it is unit-testable."""
    return [
        f"Sim time: {sim_time_s:,.0f} s past J2000",
        f"Warp: x{warp:,.0f}",
        f"Altitude: {_dist(altitude_m, units)}",
        f"Speed: {_speed(speed_mps, units)}",
        f"Periapsis: {_dist(periapsis_m, units)}",
        f"Apoapsis: {_dist(apoapsis_m, units)}",
        f"Inclination: {np.degrees(inclination_rad):,.1f}°",
        f"Period: {period_s / 60.0:,.1f} min",
    ]


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
        self.units = "km"  # distance/speed units for readouts ("km" or "mi")
        # Transient center-screen "toast" message (e.g. "Quicksaved").
        self._base = base
        self._toast_task = None
        self.toast = OnscreenText(
            text="", pos=(0.0, 0.6), scale=0.07, fg=(1.0, 1.0, 0.6, 1),
            shadow=(0, 0, 0, 1), mayChange=True, parent=base.aspect2d,
        )

    def set_units(self, units: str) -> None:
        """Set distance units for HUD readouts ('km' or 'mi')."""
        self.units = units

    def flash(self, text: str, seconds: float = 2.0) -> None:
        """Show a transient center-screen message that clears after `seconds`."""
        if self._toast_task is not None:
            self._base.taskMgr.remove(self._toast_task)
            self._toast_task = None
        self.toast.setText(text)
        self._toast_task = self._base.taskMgr.doMethodLater(
            seconds, self._clear_toast, "hud-toast-clear"
        )

    def _clear_toast(self, task):
        self.toast.setText("")
        self._toast_task = None
        return task.done

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
        inclination_rad: float,
    ) -> None:
        lines = orbit_panel_lines(
            sim_time_s=sim_time_s, warp=warp, altitude_m=altitude_m, speed_mps=speed_mps,
            periapsis_m=periapsis_m, apoapsis_m=apoapsis_m, period_s=period_s,
            inclination_rad=inclination_rad, units=self.units,
        )
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
            ("dV left: ∞" if not math.isfinite(dv_remaining)
             else f"dV left: {dv_remaining:,.0f} m/s"),
        ]
        if warp_locked:
            lines.append("WARP LOCKED - thrusting")
        self.flight.setText("\n".join(lines))
