"""Panel visibility, collapse, and modal-stack management."""


class GamePanel:
    def __init__(self, node, *, visible=True, collapsed=False):
        self.node = node
        self.visible = visible
        self.collapsed = collapsed
        self.dirty = True
        self._sync()

    def _sync(self):
        self.node.show() if self.visible else self.node.hide()

    def show(self):
        self.visible = True; self.dirty = True; self._sync()

    def hide(self):
        self.visible = False; self._sync()

    def toggle(self):
        self.hide() if self.visible else self.show()

    def set_collapsed(self, collapsed: bool):
        self.collapsed = bool(collapsed); self.dirty = True


class PanelManager:
    """Own session-local panel visibility and topmost-modal Escape behavior."""

    def __init__(self):
        self.panels = {}
        self.modals = []

    def register(self, name, panel):
        self.panels[name] = panel
        return panel

    def toggle(self, name):
        self.panels[name].toggle()

    def push_modal(self, panel):
        if panel not in self.modals:
            self.modals.append(panel)
        panel.show()

    def close_topmost(self) -> bool:
        while self.modals:
            panel = self.modals.pop()
            if panel.visible:
                panel.hide()
                return True
        return False
