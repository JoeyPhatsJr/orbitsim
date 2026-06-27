"""Tests for the pure panel-layout math (no DirectGUI)."""
from orbitsim.render.hud.panel import PanelLayout, layout_panel


def _flat(ys):
    return [y for section in ys for y in section]


def test_lines_strictly_decreasing_no_overlap():
    layout = layout_panel(
        [1, 3, 2], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03
    )
    flat = _flat(layout.line_ys)
    assert isinstance(layout, PanelLayout)
    assert len(flat) == 6
    assert all(flat[i] > flat[i + 1] for i in range(len(flat) - 1))


def test_section_gap_applied_between_sections():
    layout = layout_panel(
        [1, 1], top=0.0, line_height=0.06, padding=0.0, section_gap=0.03
    )
    assert layout.line_ys[0][0] == 0.0
    assert abs(layout.line_ys[1][0] - (0.0 - 0.06 - 0.03)) < 1e-12


def test_empty_section_collapses():
    layout = layout_panel(
        [2, 0, 1], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03
    )
    assert layout.line_ys[1] == []
    flat = _flat(layout.line_ys)
    assert len(flat) == 3
    expected_third = layout.line_ys[0][1] - 0.06 - 0.03
    assert abs(layout.line_ys[2][0] - expected_third) < 1e-12


def test_frame_encloses_all_lines():
    layout = layout_panel(
        [1, 3, 2], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03
    )
    flat = _flat(layout.line_ys)
    assert layout.frame_top >= max(flat)
    assert layout.frame_bottom <= min(flat)


def test_all_empty_is_degenerate_but_safe():
    layout = layout_panel(
        [0, 0], top=-0.10, line_height=0.06, padding=0.02, section_gap=0.03
    )
    assert _flat(layout.line_ys) == []
    assert layout.frame_top >= layout.frame_bottom
