"""Toggleable on-screen keybind help panel (F1)."""
from direct.gui.DirectFrame import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

SANDBOX_BINDINGS = [
    ("Right-drag", "Orbit camera"),
    ("Wheel", "Zoom"),
    ("Arrows", "Orbit camera"),
    ("W / S", "Pitch"),
    ("A / D", "Yaw"),
    ("Q / E", "Roll"),
    ("Shift / Ctrl", "Throttle up / down"),
    ("Z", "Full throttle"),
    ("X", "Cut throttle"),
    ("T", "SAS on/off"),
    ("M", "Toggle ship view / map"),
    ("1-7", "SAS mode (pro/retro/normal/...)"),
    (", / .", "Warp down / up"),
    ("F5 / F9", "Quicksave / Quickload"),
    ("Esc", "Settings"),
    ("F1", "Toggle this help"),
]

SOLAR_BINDINGS = [
    ("Right-drag", "Orbit camera"),
    ("Wheel", "Zoom"),
    ("Arrows", "Orbit camera"),
    (", / .", "Warp down / up"),
    ("Esc", "Settings"),
    ("F1", "Toggle this help"),
]


class KeybindOverlay:
    """A hidden-by-default panel listing key bindings; toggled with F1."""

    def __init__(self, parent, lines):
        self.visible = False
        self._frame = DirectFrame(
            frameColor=(0, 0, 0, 0.6), frameSize=(-0.7, 0.7, -0.75, 0.75),
            pos=(0, 0, 0), parent=parent,
        )
        body = "\n".join(f"{k:>14}   {desc}" for k, desc in lines)
        self._text = OnscreenText(
            text="Controls\n\n" + body, scale=0.05, fg=(1, 1, 1, 1),
            align=TextNode.ALeft, pos=(-0.62, 0.66), parent=self._frame,
            mayChange=False,
        )
        self._frame.hide()

    def show(self):
        self._frame.show()
        self.visible = True

    def hide(self):
        self._frame.hide()
        self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()
