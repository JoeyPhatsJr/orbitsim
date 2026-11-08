"""The one place HUD panel rectangles live (pure: no Panda3D).

Every corner panel is a ``PanelRect`` in aspect2d units relative to its anchor (the
matching ``a2d*`` node). ``to_absolute`` / ``to_pixels`` turn them into screen
rectangles for a window aspect and UI scale; ``tests/render/test_hud_layout.py`` pins
that no two overlap and none leaves the screen from 3:2 to 21:9.
"""
from dataclasses import dataclass

MIN_ASPECT = 1.5
MAX_ASPECT = 21 / 9
# Where the camera focus sits in normalised screen y: the middle of the free band
# between the bottom cluster (navball + speed chip) and the top panels.
FOCUS_SCREEN_Y = 0.20

_ANCHORS = ("top_left", "top_right", "bottom_left", "bottom_center", "bottom_right")


@dataclass(frozen=True)
class PanelRect:
    """A panel rectangle in aspect2d units relative to ``anchor``.

    It scales by the UI scale about ``pivot`` (anchor-relative; default the anchor
    itself). ``scales=False`` pins it (the navball's display region is a fixed
    fraction of the window height, whatever the UI scale)."""
    name: str
    anchor: str
    x0: float
    y0: float
    x1: float
    y1: float
    scales: bool = True
    pivot: tuple = (0.0, 0.0)

    def __post_init__(self):
        if self.anchor not in _ANCHORS:
            raise ValueError(f"unknown anchor {self.anchor!r}")
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(f"empty rect {self.name!r}")


PANELS: dict[str, PanelRect] = {r.name: r for r in (
    # time row; << . warp cell . >> . MAX row
    PanelRect("time_warp", "top_left", 0.04, -0.21, 0.84, -0.03),
    # target name, status (icon + text), Plan / Clear target buttons, in-SOI flyby row
    PanelRect("target", "top_right", -0.98, -0.38, -0.04, -0.03),
    # orbit rows + fuel / dV left / throttle . TWR
    PanelRect("orbit_fuel", "bottom_left", 0.04, 0.04, 0.84, 0.66),
    # the ball's own display region (30% of the window height), never scaled
    PanelRect("navball", "bottom_center", -0.31, 0.03, 0.31, 0.63, scales=False),
    # "Orbital 8.053 km/s", sits on top of the ball
    PanelRect("speed_chip", "bottom_center", -0.30, 0.63, 0.30, 0.71, scales=False),
    # SAS readout + 3x3 buttons, grows outward from its edge next to the ball
    PanelRect("sas", "bottom_center", 0.36, 0.04, 0.92, 0.52, pivot=(0.36, 0.0)),
    # burn readout, Next Pe / Next Ap / Delete burn, Execute
    PanelRect("burn", "bottom_right", -1.10, 0.58, -0.04, 0.92),
)}


def _anchor_point(anchor: str, aspect: float) -> tuple[float, float]:
    x = {"top_left": -aspect, "bottom_left": -aspect, "bottom_center": 0.0,
         "top_right": aspect, "bottom_right": aspect}[anchor]
    y = 1.0 if anchor.startswith("top") else -1.0
    return x, y


def to_absolute(rect: PanelRect, aspect: float, ui_scale: float = 1.0):
    """Absolute aspect2d ``(x0, y0, x1, y1)`` of ``rect`` for this window aspect."""
    ax, ay = _anchor_point(rect.anchor, aspect)
    s = ui_scale if rect.scales else 1.0
    px, py = rect.pivot
    return (ax + px + (rect.x0 - px) * s, ay + py + (rect.y0 - py) * s,
            ax + px + (rect.x1 - px) * s, ay + py + (rect.y1 - py) * s)


def to_pixels(rect: PanelRect, width_px, height_px, ui_scale: float = 1.0):
    """Pixel ``(x0, y0, x1, y1)``, origin bottom-left (as ``Targeting.marker_px``)."""
    aspect = width_px / height_px
    x0, y0, x1, y1 = to_absolute(rect, aspect, ui_scale)
    return ((x0 / aspect + 1.0) * 0.5 * width_px, (y0 + 1.0) * 0.5 * height_px,
            (x1 / aspect + 1.0) * 0.5 * width_px, (y1 + 1.0) * 0.5 * height_px)

