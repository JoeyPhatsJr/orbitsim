"""Pure-logic tests for the HUD orbit panel line builder (no DirectGUI needed)."""
from orbitsim.render.hud import orbit_panel_lines


def _lines(**over):
    base = dict(
        sim_time_s=0.0, altitude_m=500_000.0, speed_mps=7600.0,
        periapsis_m=400_000.0, apoapsis_m=600_000.0, period_s=5400.0,
        inclination_rad=0.5, units="km",
    )
    base.update(over)
    return orbit_panel_lines(**base)


def test_inclination_line_in_degrees():
    text = "\n".join(_lines(inclination_rad=0.5))
    assert "Inclination: 28.6°" in text  # 0.5 rad -> 28.6 deg


def test_km_units_default():
    text = "\n".join(_lines(units="km", altitude_m=500_000.0))
    assert "Altitude: 500.0 km" in text
    assert "Speed: 7.600 km/s" in text


def test_mi_units_conversion():
    text = "\n".join(_lines(units="mi", altitude_m=500_000.0, speed_mps=7600.0))
    # 500 km * 0.621371 = 310.7 mi ; 7.6 km/s * 0.621371 = 4.722 mi/s
    assert "Altitude: 310.7 mi" in text
    assert "Speed: 4.722 mi/s" in text


def test_period_always_minutes_unit_agnostic():
    text = "\n".join(_lines(period_s=5400.0))
    assert "Period: 90.0 min" in text


def test_no_warp_line():
    assert "Warp" not in "\n".join(_lines())
