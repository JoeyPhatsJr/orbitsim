"""Toggleable settings panel (Esc): currently a km/mi units toggle."""
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectButton import DirectButton
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class SettingsPanel:
    """A hidden-by-default settings panel; toggled with Esc."""

    def __init__(self, parent, on_units_change, on_unlimited_toggle=None):
        self.visible = False
        self.units = "km"
        self._on_units_change = on_units_change
        self._on_unlimited_toggle = on_unlimited_toggle or (lambda on: None)
        self._unlimited_on = False
        self._frame = DirectFrame(
            frameColor=(0, 0, 0, 0.7), frameSize=(-0.45, 0.45, -0.32, 0.25),
            pos=(0, 0, 0), parent=parent,
        )
        OnscreenText(
            text="Settings", scale=0.06, fg=(1, 1, 1, 1), align=TextNode.ACenter,
            pos=(0, 0.15), parent=self._frame, mayChange=False,
        )
        self._units_btn = DirectButton(
            text="Units: km", scale=0.05, pos=(0, 0, 0.0),
            command=self._cycle_units, parent=self._frame,
        )
        self._unlimited_btn = DirectButton(
            text="Unlimited dV: off", scale=0.05, pos=(0, 0, -0.12),
            command=self._toggle_unlimited, parent=self._frame,
        )
        self._frame.hide()

    def _cycle_units(self):
        self.units = "mi" if self.units == "km" else "km"
        self._units_btn["text"] = f"Units: {self.units}"
        self._on_units_change(self.units)

    def _toggle_unlimited(self):
        self._unlimited_on = not self._unlimited_on
        self._unlimited_btn["text"] = f"Unlimited dV: {'on' if self._unlimited_on else 'off'}"
        self._on_unlimited_toggle(self._unlimited_on)

    def show(self):
        self._frame.show()
        self.visible = True

    def hide(self):
        self._frame.hide()
        self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()
