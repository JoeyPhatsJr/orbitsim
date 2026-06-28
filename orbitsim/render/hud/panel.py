"""Self-sizing HUD panel layout and DirectGUI rendering."""
from dataclasses import dataclass


LINE_HEIGHT = 0.06
PADDING = 0.02
SECTION_GAP = 0.03


@dataclass(frozen=True)
class PanelLayout:
    line_ys: list[list[float]]
    frame_top: float
    frame_bottom: float


def layout_panel(
    section_line_counts: list[int],
    *,
    top: float,
    line_height: float,
    padding: float,
    section_gap: float,
) -> PanelLayout:
    """Lay out stacked sections from top to bottom in corner-relative coordinates."""
    line_ys: list[list[float]] = []
    current_y = top
    have_section = False

    for count in section_line_counts:
        if count <= 0:
            line_ys.append([])
            continue
        if have_section:
            current_y -= section_gap
        have_section = True
        section = []
        for _ in range(count):
            section.append(current_y)
            current_y -= line_height
        line_ys.append(section)

    flat = [y for section in line_ys for y in section]
    frame_top = top + padding
    frame_bottom = min(flat) - line_height - padding if flat else top - padding
    return PanelLayout(line_ys, frame_top, frame_bottom)


class HudPanel:
    """Reusable grouped HUD text with a self-sizing translucent background."""

    def __init__(
        self,
        parent,
        *,
        x: float,
        top: float,
        scale: float = 0.045,
        align="left",
        width: float = 0.62,
    ):
        from direct.gui.DirectFrame import DirectFrame

        if align != "left":
            raise ValueError("HudPanel currently supports left alignment only")
        self._parent = parent
        self._x = x
        self._top = top
        self._scale = scale
        self._width = width
        self._texts = []
        self._bg = DirectFrame(
            frameColor=(0.0, 0.0, 0.0, 0.45),
            frameSize=(x - PADDING, x + width, top - PADDING, top + PADDING),
            parent=parent,
        )
        self._bg.set_bin("fixed", 0)
        self._bg.hide()

    def _row(self, index):
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        while index >= len(self._texts):
            text = OnscreenText(
                text="",
                scale=self._scale,
                align=TextNode.ALeft,
                fg=(1.0, 1.0, 1.0, 1.0),
                shadow=(0.0, 0.0, 0.0, 1.0),
                mayChange=True,
                parent=self._parent,
            )
            text.hide()
            text.set_bin("fixed", 1)
            self._texts.append(text)
        return self._texts[index]

    def set_sections(self, sections) -> None:
        """Render `{header, header_color, rows}` section dictionaries."""
        rows = []
        counts = []
        for section in sections:
            header = section.get("header")
            body = section.get("rows", [])
            counts.append((1 if header else 0) + len(body))
            if header:
                rows.append((header, section.get("header_color", (1.0, 1.0, 1.0, 1.0))))
            rows.extend(body)

        layout = layout_panel(
            counts,
            top=self._top,
            line_height=LINE_HEIGHT,
            padding=PADDING,
            section_gap=SECTION_GAP,
        )
        ys = [y for section in layout.line_ys for y in section]
        for index, ((label, color), y) in enumerate(zip(rows, ys)):
            text = self._row(index)
            text.setText(label)
            text.setFg(color)
            text.setPos(self._x, y)
            text.show()
        for text in self._texts[len(rows):]:
            text.hide()

        if rows:
            self._bg["frameSize"] = (
                self._x - PADDING,
                self._x + self._width,
                layout.frame_bottom,
                layout.frame_top,
            )
            self._bg.show()
        else:
            self._bg.hide()
