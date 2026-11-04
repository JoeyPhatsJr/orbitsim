"""Named line styles: own orbit/plan/encounters are bold, references are quiet (pure)."""
from orbitsim.render.line_style import (
    ENCOUNTER_STYLE,
    ORBIT_STYLE,
    PLAN_STYLE,
    REFERENCE_STYLE,
    TARGET_REFERENCE_STYLE,
    px_scale,
    reference_style,
)


def test_own_orbit_and_plan_are_thick_and_opaque():
    for s in (ORBIT_STYLE, PLAN_STYLE, ENCOUNTER_STYLE):
        assert s.width_px >= 3.5 and s.color[3] == 1.0 and s.fade_minimum >= 0.65


def test_plan_is_dashed_orange_and_orbit_is_solid_cyan():
    assert PLAN_STYLE.dash == (6, 4) and ORBIT_STYLE.dash is None
    r, g, b, _ = PLAN_STYLE.color
    assert r > 0.9 and b < 0.3
    r, g, b, _ = ORBIT_STYLE.color
    assert b > 0.9 and r < 0.3


def test_reference_is_quieter_than_own_orbit():
    assert REFERENCE_STYLE.width_px < ORBIT_STYLE.width_px
    assert REFERENCE_STYLE.color[3] < ORBIT_STYLE.color[3]


def test_reference_style_follows_target():
    assert reference_style(True) is TARGET_REFERENCE_STYLE
    assert reference_style(False) is REFERENCE_STYLE
    assert TARGET_REFERENCE_STYLE.width_px > REFERENCE_STYLE.width_px


def test_px_scale_is_one_at_reference_and_grows_with_window_height():
    assert px_scale(900, 900) == 1.0
    assert px_scale(1440, 1440) == 1.6     # tall 1x monitor: proportionally thicker


def test_px_scale_doubles_on_retina_at_any_window_size():
    # The framebuffer/window ratio is the display density; a small Retina window
    # (600 pt tall, 1200 px framebuffer) still needs 2x, not 1200/900 = 1.33x.
    assert px_scale(1200, 600) == 2.0
    assert px_scale(1800, 900) == 2.0


def test_px_scale_never_below_one():
    assert px_scale(600, 600) == 1.0
    assert px_scale(0, 0) == 1.0


def test_planet_orbits_hidden_at_earth_moon_zoom_shown_at_planetary_zoom():
    from orbitsim.render.line_style import PLANET_ORBITS_MIN_DISTANCE_M, planet_orbits_visible
    assert not planet_orbits_visible(1.3e9)          # whole Earth-Moon system in view
    assert planet_orbits_visible(PLANET_ORBITS_MIN_DISTANCE_M)
    assert planet_orbits_visible(3.0e11)             # ~2 AU
