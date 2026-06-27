"""Self-sizing HUD panel layout and DirectGUI rendering."""
from dataclasses import dataclass


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
