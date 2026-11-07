"""Navball-adjacent SAS controls and flight readouts."""
import math

from orbitsim.core.attitude import SAS_MODES

_SHORT = {
    "PROGRADE": "PRO",
    "RETROGRADE": "RET",
    "NORMAL": "NML",
    "ANTINORMAL": "ANM",
    "RADIAL_IN": "RIN",
    "RADIAL_OUT": "ROUT",
    "TARGET": "TGT",
    "ANTITARGET": "ATG",
}
_IDLE = (0.15, 0.15, 0.2, 0.85)
_ACTIVE = (0.2, 0.7, 1.0, 0.95)
_BG = (0.0, 0.0, 0.0, 0.45)


class SasPanel:
    """Current SAS attitude plus buttons mirroring the keyboard controls, in
    PANELS["sas"] beside the navball."""

    def __init__(self, base, *, on_set_mode, on_toggle):
        from direct.gui.DirectButton import DirectButton
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        from orbitsim.render.ui.hud_layout import PANELS
        from orbitsim.render.ui.widgets import layout_panel_frame, panel_pos

        rect = PANELS["sas"]
        self.frame = layout_panel_frame(base, rect)
        base._hud_panels["sas"] = self.frame
        self._buttons = {}
        self._readout = OnscreenText(
            text="",
            pos=panel_pos(rect, 0.64, 0.445),
            scale=0.038,
            fg=(0.85, 0.95, 1.0, 1.0),
            shadow=(0.0, 0.0, 0.0, 1.0),
            align=TextNode.ACenter,
            mayChange=True,
            parent=self.frame,
        )
        controls = [("__TOGGLE__", "SAS", on_toggle, [])]
        controls.extend((mode, _SHORT[mode], on_set_mode, [mode]) for mode in SAS_MODES)
        for index, (key, label, command, args) in enumerate(controls):
            row, column = divmod(index, 3)
            x, z = panel_pos(rect, 0.47 + column * 0.17, 0.335 - row * 0.115)
            self._buttons[key] = DirectButton(
                text=label,
                scale=0.042,
                pos=(x, 0.0, z),
                frameColor=_IDLE,
                text_fg=(1.0, 1.0, 1.0, 1.0),
                command=command,
                extraArgs=args,
                parent=self.frame,
            )

    def update(self, sas_mode: str, heading_rad: float, pitch_rad: float) -> None:
        heading = math.degrees(heading_rad) % 360.0
        pitch = math.degrees(pitch_rad)
        self._readout.setText(
            f"SAS {_SHORT.get(sas_mode, sas_mode)} \N{MIDDLE DOT} HDG {heading:03.0f}\N{DEGREE SIGN}"
            f" \N{MIDDLE DOT} PIT {pitch:+03.0f}\N{DEGREE SIGN}"
        )
        for mode, button in self._buttons.items():
            active = sas_mode != "OFF" if mode == "__TOGGLE__" else mode == sas_mode
            button["frameColor"] = _ACTIVE if active else _IDLE


class VelocityReadout:
    """Clickable chip toggling between orbital and target-relative speed, sitting on
    top of the navball (PANELS["speed_chip"])."""

    def __init__(self, base, units_getter):
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectButton import DirectButton
        from orbitsim.render.ui.hud_layout import PANELS

        rect = PANELS["speed_chip"]
        self._units_getter = units_getter
        self._mode = "ORBITAL"
        self._orbital = 0.0
        self._target = None
        # A plain holder carries the (pinned) UI scale; the button keeps its own.
        self.root = base.a2dBottomCenter.attach_new_node("speed_chip")
        base._hud_panels["speed_chip"] = self.root
        self._button = DirectButton(
            text="",
            scale=0.045,
            pos=(0.5 * (rect.x0 + rect.x1), 0.0, rect.y0 + 0.022),
            frameColor=_BG,
            text_fg=(1.0, 1.0, 1.0, 1.0),
            relief=DGG.FLAT,
            command=self._toggle,
            parent=self.root,
        )
        self._refresh()

    def _toggle(self) -> None:
        self._mode = "TARGET" if self._mode == "ORBITAL" else "ORBITAL"
        self._refresh()

    def update(self, orbital_speed_mps: float, target_rel_speed_mps) -> None:
        self._orbital = orbital_speed_mps
        self._target = target_rel_speed_mps
        self._refresh()

    def _refresh(self) -> None:
        from orbitsim.render.hud import _speed

        if self._mode == "ORBITAL":
            text = f"Orbital  {_speed(self._orbital, self._units_getter())}"
        elif self._target is None:
            text = "Target  \N{EM DASH}"
        else:
            text = f"Target  {_speed(self._target, self._units_getter())}"
        self._button["text"] = text
