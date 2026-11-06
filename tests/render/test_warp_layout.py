"""The time + warp panel's text always fits its cell (Rajdhani metrics, no ShowBase)."""
import pytest

from orbitsim.render.hud.format import fmt_mission_time, fmt_warp


def _width(text: str, scale: float) -> float:
    """Rendered width in aspect2d units of ``text`` at ``scale`` in the bundled font."""
    from panda3d.core import Filename, FontPool, TextNode
    from orbitsim.render.ui.theme import font_path
    node = TextNode("measure")
    font = FontPool.load_font(Filename.from_os_specific(font_path()).get_fullpath())
    assert font is not None and font.is_valid()
    node.set_font(font)
    node.set_text(text)
    return node.calc_width(text) * scale


def test_warp_cell_fits_worst_case():
    from orbitsim.render.controllers.warp_controls import (
        TIME_ROW_W, WARP_CELL_W, WARP_TEXT_SCALE)
    assert _width(fmt_warp(1e8, True), WARP_TEXT_SCALE) <= WARP_CELL_W
    assert _width(fmt_mission_time(9_999_999 * 86400.0), WARP_TEXT_SCALE) <= TIME_ROW_W


def test_warp_cell_clear_of_its_buttons():
    # The cell (centred at WARP_CELL_X) sits between the << and >> buttons' inner edges.
    from orbitsim.render.controllers import warp_controls as wc
    left_edge = wc.WARP_DOWN_X + wc.BUTTON_HALF_W
    right_edge = wc.WARP_UP_X - wc.BUTTON_HALF_W
    assert left_edge <= wc.WARP_CELL_X - wc.WARP_CELL_W / 2
    assert wc.WARP_CELL_X + wc.WARP_CELL_W / 2 <= right_edge
    assert wc.MAX_X - wc.BUTTON_HALF_W >= wc.WARP_UP_X + wc.BUTTON_HALF_W


def test_min_window():
    from orbitsim.render.controllers.warp_controls import MIN_WINDOW_PX
    assert MIN_WINDOW_PX == (1280, 720)
