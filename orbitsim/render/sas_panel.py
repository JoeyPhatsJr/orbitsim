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
    """Current SAS attitude plus buttons mirroring the keyboard controls."""

    def __init__(self, base, *, on_set_mode, on_toggle):
        from direct.gui.DirectButton import DirectButton
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self._buttons = {}
        self._readout = OnscreenText(
            text="",
            pos=(0.72, 0.80),
            scale=0.04,
            fg=(0.85, 0.95, 1.0, 1.0),
            shadow=(0.0, 0.0, 0.0, 1.0),
            align=TextNode.ACenter,
            mayChange=True,
            parent=base.a2dBottomCenter,
        )
        controls = [("__TOGGLE__", "SAS", on_toggle, [])]
        controls.extend((mode, _SHORT[mode], on_set_mode, [mode]) for mode in SAS_MODES)
        for index, (key, label, command, args) in enumerate(controls):
            row, column = divmod(index, 3)
            self._buttons[key] = DirectButton(
                text=label,
                scale=0.037,
                pos=(0.48 + column * 0.18, 0.0, 0.68 - row * 0.10),
                frameColor=_IDLE,
                text_fg=(1.0, 1.0, 1.0, 1.0),
                command=command,
                extraArgs=args,
                parent=base.a2dBottomCenter,
            )

    def update(self, sas_mode: str, heading_rad: float, pitch_rad: float) -> None:
        heading = math.degrees(heading_rad) % 360.0
        pitch = math.degrees(pitch_rad)
        self._readout.setText(
            f"SAS: {sas_mode}    HDG {heading:03.0f}\N{DEGREE SIGN}   "
            f"PIT {pitch:+03.0f}\N{DEGREE SIGN}"
        )
        for mode, button in self._buttons.items():
            active = sas_mode != "OFF" if mode == "__TOGGLE__" else mode == sas_mode
            button["frameColor"] = _ACTIVE if active else _IDLE


class VelocityReadout:
    """Clickable chip toggling between orbital and target-relative speed."""

    def __init__(self, base, units_getter):
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectButton import DirectButton

        self._units_getter = units_getter
        self._mode = "ORBITAL"
        self._orbital = 0.0
        self._target = None
        self._button = DirectButton(
            text="",
            scale=0.045,
            pos=(0.0, 0.0, 0.70),
            frameColor=_BG,
            text_fg=(1.0, 1.0, 1.0, 1.0),
            relief=DGG.FLAT,
            command=self._toggle,
            parent=base.a2dBottomCenter,
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
