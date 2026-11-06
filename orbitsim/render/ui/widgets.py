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


_ANCHOR_NODES = {
    "top_left": "a2dTopLeft", "top_right": "a2dTopRight", "bottom_left": "a2dBottomLeft",
    "bottom_center": "a2dBottomCenter", "bottom_right": "a2dBottomRight",
}


def layout_panel_frame(base, rect, *, color=None):
    """A DirectFrame filling ``rect`` (a ``hud_layout.PanelRect``) on its anchor node.

    The frame sits at the rect's pivot, so ``set_scale(ui_scale)`` scales it the way
    ``hud_layout.to_absolute`` does; children go at ``panel_pos`` (rect coordinates,
    i.e. anchor-relative and unscaled)."""
    from direct.gui.DirectFrame import DirectFrame
    px, py = rect.pivot
    return DirectFrame(
        frameColor=THEME.panel if color is None else color,
        frameSize=(rect.x0 - px, rect.x1 - px, rect.y0 - py, rect.y1 - py),
        pos=(px, 0.0, py),
        parent=getattr(base, _ANCHOR_NODES[rect.anchor]),
    )


def panel_pos(rect, x, y):
    """(x, z) inside a ``layout_panel_frame`` for anchor-relative rect coordinates."""
    return x - rect.pivot[0], y - rect.pivot[1]


def scale_panel(frame, rect, ui_scale: float) -> None:
    """Apply the UI scale to a ``layout_panel_frame`` (pinned rects stay at 1)."""
    frame.set_scale(ui_scale if rect.scales else 1.0)
