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


def test_px_scale_is_one_at_reference_and_grows_with_height():
    assert px_scale(900) == 1.0
    assert px_scale(1800) == 2.0          # Retina 2x framebuffer


def test_px_scale_never_below_one():
    assert px_scale(600) == 1.0 and px_scale(0) == 1.0
