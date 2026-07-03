"""Themed composite widgets used across gameplay and menus."""
from orbitsim.render.ui.theme import THEME, button_options


class OperationOverlay:
    """Modal progress card for long planning/coast operations."""

    def __init__(self, base, on_cancel):
        from direct.gui.DirectButton import DirectButton
        from direct.gui.DirectFrame import DirectFrame
        from direct.gui.OnscreenText import OnscreenText

        self.frame = DirectFrame(
            frameColor=THEME.panel_alt,
            frameSize=(-0.52, 0.52, -0.18, 0.18),
            pos=(0, 0, 0.12),
            parent=base.aspect2d,
        )
        self.label = OnscreenText(
            text="", pos=(0, 0.08), scale=0.05, fg=THEME.text,
            mayChange=True, parent=self.frame,
        )
        self.progress = DirectFrame(
            frameColor=THEME.control,
            frameSize=(-0.40, 0.40, -0.025, 0.025),
            pos=(0, 0, -0.01), parent=self.frame,
        )
        self.fill = DirectFrame(
            frameColor=THEME.cyan,
            frameSize=(-0.40, -0.40, -0.018, 0.018),
            parent=self.progress,
        )
        self.cancel = DirectButton(
            text="CANCEL", scale=0.045, pos=(0, 0, -0.115),
            command=on_cancel, parent=self.frame, **button_options(),
        )
        self.frame.hide()

    def update(self, status):
        if not status.running:
            self.frame.hide()
            return
        self.label.setText(f"{status.label.upper()}   {status.progress * 100:3.0f}%")
        right = -0.40 + 0.80 * status.progress
        self.fill["frameSize"] = (-0.40, right, -0.018, 0.018)
        self.frame.show()


class PanelDock:
    """Compact left-edge buttons for session-local panel visibility."""

    def __init__(self, base, actions):
        from direct.gui.DirectButton import DirectButton

        self.buttons = []
        for index, (label, command) in enumerate(actions):
            self.buttons.append(DirectButton(
                text=label, scale=0.042, pos=(0.17, 0, 0.16 - index * 0.12),
                command=command, parent=base.a2dLeftCenter, **button_options(),
            ))
