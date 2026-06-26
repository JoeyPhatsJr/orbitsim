"""Toggleable settings panel (Esc): currently a km/mi units toggle."""
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectButton import DirectButton
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class SettingsPanel:
    """A hidden-by-default settings panel; toggled with Esc."""

    def __init__(self, parent, on_units_change, on_unlimited_toggle=None,
                 enable_unlimited=True):
        self.visible = False
        self.units = "km"
        self._on_units_change = on_units_change
        self._on_unlimited_toggle = on_unlimited_toggle or (lambda on: None)
        self._unlimited_on = False
        self._unlimited_btn = None
        # Taller frame only when the (sandbox-only) unlimited button is present.
        bottom = -0.32 if enable_unlimited else -0.25
        self._frame = DirectFrame(
            frameColor=(0, 0, 0, 0.7), frameSize=(-0.45, 0.45, bottom, 0.25),
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
        if enable_unlimited:
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
        self.sync(not self._unlimited_on)
        self._on_unlimited_toggle(self._unlimited_on)

    def sync(self, on: bool):
        """Reflect externally-changed unlimited-dV state on the button (no callback).

        Lets the title-screen checkbox and the U keybind keep this label in step."""
        self._unlimited_on = on
        if self._unlimited_btn is not None:
            self._unlimited_btn["text"] = f"Unlimited dV: {'on' if on else 'off'}"

    def show(self):
        self._frame.show()
        self.visible = True

    def hide(self):
        self._frame.hide()
        self.visible = False

    def toggle(self):
        self.hide() if self.visible else self.show()
