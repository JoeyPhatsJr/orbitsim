"""Pure-logic tests for the HUD orbit panel line builder (no DirectGUI needed)."""
from orbitsim.render.hud import orbit_panel_lines


def _lines(**over):
    base = dict(
        altitude_m=500_000.0, speed_mps=7600.0,
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


def test_fmt_lead():
    from orbitsim.render.hud.format import fmt_lead
    assert [fmt_lead(s) for s in (0.4, 45, 36 * 60 + 10, 2 * 3600 + 300, 3 * 86400 + 4 * 3600)] == [
        "now", "in 45 s", "in 36 min", "in 2 h 05 min", "in 3 d 04 h"]


def test_no_sim_time_row():
    # Mission time lives in the time + warp panel now.
    assert not any("Sim time" in line for line in _lines())


def test_mission_time_formats():
    from orbitsim.render.hud.format import fmt_mission_time
    assert fmt_mission_time(12015.0) == "T+ 03:20:15"
    assert fmt_mission_time(12 * 86400 + 3 * 3600 + 20 * 60) == "T+ 12d 03h 20m"
    assert fmt_mission_time(969_100 * 86400 + 3 * 3600) == "T+ 969,100d 03h"
    assert fmt_mission_time(0.0) == "T+ 00:00:00"
    assert fmt_mission_time(-12015.0) == "T\u2212 03:20:15"


def test_warp_formats():
    from orbitsim.render.hud.format import fmt_warp
    assert fmt_warp(100.0, False) == "\u00d7100" and fmt_warp(1e6, False) == "\u00d71,000,000"
    assert fmt_warp(1.0, True) == "\u00d71 LOCKED"
