"""The in-SOI flyby row fits the target panel and uses only glyphs the font has.

Review (4b): at Jupiter the old copy ("... δ 123.4° Pe 1,234,567 km free dV ...")
ran past the right edge of the screen, and Rajdhani has no "δ".
"""
import math

from orbitsim.render.hud.format import flyby_line


def _font_node():
    from panda3d.core import Filename, FontPool, TextNode
    from orbitsim.render.ui.theme import font_path
    node = TextNode("measure")
    font = FontPool.load_font(Filename.from_os_specific(font_path()).get_fullpath())
    assert font is not None and font.is_valid()
    node.set_font(font)
    return node


WORST = flyby_line("Jupiter", 99_999.0, math.radians(179.9), 9.99e9, 99_999.0)


def test_flyby_line_copy():
    line = flyby_line("Moon", 812.0, math.radians(37.26), 1_234_000.0, 515.4)
    assert line == "Moon · v-inf 0.8 km/s · turns 37° · Pe 1,234 km · +515 m/s"


def test_flyby_line_fits_the_target_panel_row():
    from orbitsim.render.controllers.targeting import Targeting
    node = _font_node()
    assert node.calc_width(WORST) * Targeting.FLYBY_SCALE <= Targeting.FLYBY_ROW_W, WORST


def test_flyby_line_has_every_glyph():
    node = _font_node()
    missing = [ch for ch in WORST if not node.has_character(ch)]
    assert missing == []
