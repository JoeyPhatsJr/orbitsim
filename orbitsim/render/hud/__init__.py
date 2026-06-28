"""Minimal DirectGUI overlay. Converts SI -> km/UTC at this boundary only."""
import math

import numpy as np
from direct.gui.OnscreenText import OnscreenText

from orbitsim.render.hud.panel import HudPanel

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
    *, sim_time_s: float, altitude_m: float, speed_mps: float,
    periapsis_m: float, apoapsis_m: float, period_s: float,
    inclination_rad: float, units: str, dominant_body: str = "Earth",
) -> list[str]:
    """Build the orbit-info panel text lines. Pure (no DirectGUI) so it is unit-testable."""
    lines = [
        f"Sim time: {sim_time_s:,.0f} s past J2000",
        f"Altitude: {_dist(altitude_m, units)}",
        f"Speed: {_speed(speed_mps, units)}",
    ]
    if dominant_body != "Earth":
        lines.append(f"Orbiting: {dominant_body}")
    lines += [
        f"Periapsis: {_dist(periapsis_m, units)}",
        f"Apoapsis: {_dist(apoapsis_m, units)}",
        f"Inclination: {np.degrees(inclination_rad):,.1f}°",
        f"Period: {period_s / 60.0:,.1f} min",
    ]
    return lines


class Hud:
    """Grouped orbit, maneuver, and vessel HUD panels."""

    def __init__(self, base) -> None:
        self._left = HudPanel(base.a2dTopLeft, x=0.08, top=-0.10, width=0.92)
        self._right = HudPanel(base.a2dTopRight, x=-0.62, top=-0.10)
        self._orbit_lines = []
        self._maneuver_lines = ("", "", "")
        self._encounter_line = ""
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

    def _rebuild_left(self) -> None:
        cyan = (0.7, 0.95, 1.0, 1.0)
        magenta = (1.0, 0.4, 1.0, 1.0)
        orange = (1.0, 0.7, 0.4, 1.0)
        time_line = self._orbit_lines[0] if self._orbit_lines else ""
        sections = [
            {"header": None, "rows": [(time_line, cyan)] if time_line else []},
            {
                "header": "ORBIT",
                "header_color": cyan,
                "rows": [(line, (1.0, 1.0, 1.0, 1.0)) for line in self._orbit_lines[1:]],
            },
        ]
        dv_line, node_line, target_line = self._maneuver_lines
        maneuver_rows = []
        if dv_line:
            maneuver_rows.append((dv_line, magenta))
        if node_line:
            maneuver_rows.append((node_line, cyan))
        if target_line:
            maneuver_rows.append((target_line, orange))
        enc_line = getattr(self, "_encounter_line", "")
        if enc_line:
            maneuver_rows.append((enc_line, (0.4, 1.0, 0.6, 1.0)))
        if maneuver_rows:
            sections.append(
                {"header": "MANEUVER", "header_color": magenta, "rows": maneuver_rows}
            )
        self._left.set_sections(sections)

    def set_maneuver(self, dv_line: str, node_line: str, target_line: str,
                     encounter_line: str = "") -> None:
        self._maneuver_lines = (dv_line, node_line, target_line)
        self._encounter_line = encounter_line
        self._rebuild_left()

    def update(
        self,
        *,
        sim_time_s: float,
        altitude_m: float,
        speed_mps: float,
        periapsis_m: float,
        apoapsis_m: float,
        period_s: float,
        inclination_rad: float,
        dominant_body: str = "Earth",
    ) -> None:
        lines = orbit_panel_lines(
            sim_time_s=sim_time_s, altitude_m=altitude_m, speed_mps=speed_mps,
            periapsis_m=periapsis_m, apoapsis_m=apoapsis_m, period_s=period_s,
            inclination_rad=inclination_rad, units=self.units,
            dominant_body=dominant_body,
        )
        self._orbit_lines = lines
        self._rebuild_left()

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
        green = (0.6, 1.0, 0.6, 1.0)
        self._right.set_sections(
            [{"header": "VESSEL", "header_color": green, "rows": [(line, green) for line in lines]}]
        )
