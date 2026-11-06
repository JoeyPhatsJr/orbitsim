"""Minimal DirectGUI overlay. Converts SI -> km/UTC at this boundary only."""
import math

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

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


def orbit_fuel_lines(
    *, altitude_m: float, speed_mps: float, periapsis_m: float, apoapsis_m: float,
    period_s: float, inclination_rad: float, units: str, dominant_body: str,
    fuel_frac: float, dv_remaining: float, throttle: float, twr: float,
    warp_locked: bool = False,
) -> list[str]:
    """The bottom-left panel's rows, top to bottom. Pure (no DirectGUI) so it is
    unit-testable. Mass and thrust are constants of the one ship, so they're not here."""
    period = (f"{period_s / 60.0:,.1f} min" if math.isfinite(period_s) else "\u2014")
    lines = [
        f"Altitude {_dist(altitude_m, units)}",
        f"Speed {_speed(speed_mps, units)}",
        f"Pe {_dist(periapsis_m, units)} \u00b7 Ap {_dist(apoapsis_m, units)}",
        f"Inc {np.degrees(inclination_rad):,.1f}\u00b0 \u00b7 Period {period}",
    ]
    if dominant_body != "Earth":
        lines.append(f"Orbiting {dominant_body}")
    dv_left = "INF" if not math.isfinite(dv_remaining) else f"{dv_remaining:,.0f} m/s"
    lines += [
        f"Fuel {fuel_frac * 100:,.0f}% \u00b7 dV left {dv_left}",
        f"Throttle {throttle * 100:,.0f}% \u00b7 TWR {twr:,.2f}",
    ]
    if warp_locked:
        lines.append("WARP LOCKED \u2014 thrusting")
    return lines


class Hud:
    """The orbit + fuel panel (bottom-left), the toast, and the solar viewer's text."""

    def __init__(self, base) -> None:
        from orbitsim.render.ui.hud_layout import PANELS
        rect = PANELS["orbit_fuel"]
        # A plain node at the anchor carries the UI scale (OrbitApp._hud_panels).
        self._root = base.a2dBottomLeft.attach_new_node("orbit_fuel_panel")
        base._hud_panels["orbit_fuel"] = self._root
        self._panel = HudPanel(self._root, x=rect.x0 + 0.03, bottom=rect.y0 + 0.02,
                               grow="up", width=rect.x1 - rect.x0 - 0.05)
        self._orbit = None
        self.units = "km"  # distance/speed units for readouts ("km" or "mi")
        # Transient center-screen "toast" message (e.g. "Quicksaved").
        self._base = base
        self._toast_task = None
        self.toast = OnscreenText(
            text="", pos=(0.0, 0.6), scale=0.07, fg=(1.0, 1.0, 0.6, 1),
            shadow=(0, 0, 0, 1), mayChange=True, parent=base.aspect2d,
        )
        # Free-form persistent readout (used by the vesselless solar viewer, which has
        # no orbit/vessel panels to populate): below the time + warp panel.
        self.text = OnscreenText(
            text="", pos=(0.08, -0.28), scale=0.045, align=TextNode.ALeft,
            fg=(1.0, 1.0, 1.0, 1.0), shadow=(0, 0, 0, 1), mayChange=True,
            parent=base.a2dTopLeft,
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
        altitude_m: float,
        speed_mps: float,
        periapsis_m: float,
        apoapsis_m: float,
        period_s: float,
        inclination_rad: float,
        dominant_body: str = "Earth",
    ) -> None:
        """Orbit values for the next ``update_flight`` (which redraws the panel)."""
        self._orbit = dict(
            altitude_m=altitude_m, speed_mps=speed_mps, periapsis_m=periapsis_m,
            apoapsis_m=apoapsis_m, period_s=period_s, inclination_rad=inclination_rad,
            dominant_body=dominant_body,
        )

    def update_flight(
        self,
        *,
        throttle: float,
        fuel_frac: float,
        twr: float,
        dv_remaining: float,
        warp_locked: bool,
    ) -> None:
        """Fuel rows; redraws the orbit + fuel panel."""
        if self._orbit is None:
            return
        lines = orbit_fuel_lines(
            **self._orbit, units=self.units, fuel_frac=fuel_frac,
            dv_remaining=dv_remaining, throttle=throttle, twr=twr, warp_locked=warp_locked,
        )
        n_orbit = len(lines) - (3 if warp_locked else 2)
        white = (1.0, 1.0, 1.0, 1.0)
        green = (0.6, 1.0, 0.6, 1.0)
        amber = (1.0, 0.72, 0.24, 1.0)
        rows = [(line, white) for line in lines[:n_orbit]]
        fuel = [(line, green) for line in lines[n_orbit:n_orbit + 2]]
        if warp_locked:
            fuel.append((lines[-1], amber))
        self._panel.set_sections([{"rows": rows}, {"rows": fuel}])
