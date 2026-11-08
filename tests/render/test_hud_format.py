"""Pure-logic tests for the HUD line builders and formatters (no DirectGUI needed)."""
import numpy as np

from orbitsim.render.hud import orbit_fuel_lines


def _lines(**over):
    base = dict(
        altitude_m=500_000.0, speed_mps=7600.0,
        periapsis_m=400_000.0, apoapsis_m=600_000.0, period_s=5400.0,
        inclination_rad=0.5, units="km", dominant_body="Earth", fuel_frac=1.0,
        dv_remaining=5633.0, throttle=0.0, twr=0.6,
    )
    base.update(over)
    return orbit_fuel_lines(**base)


def test_orbit_fuel_lines_compact():
    lines = orbit_fuel_lines(altitude_m=520.9e3, speed_mps=8053.0, periapsis_m=500e3,
                             apoapsis_m=2465.2e3, period_s=115.6 * 60,
                             inclination_rad=np.radians(8.1),
                             units="km", dominant_body="Earth", fuel_frac=1.0,
                             dv_remaining=5633.0, throttle=0.0, twr=0.6)
    assert lines == ["Altitude 520.9 km", "Speed 8.053 km/s", "Pe 500.0 km · Ap 2,465.2 km",
                     "Inc 8.1° · Period 115.6 min", "Fuel 100% · dV left 5,633 m/s",
                     "Throttle 0% · TWR 0.60"]


def test_orbiting_row_only_away_from_earth():
    assert "Orbiting Moon" in _lines(dominant_body="Moon")
    assert _lines(dominant_body="Moon").index("Orbiting Moon") == 4
    assert not any(line.startswith("Orbiting") for line in _lines())


def test_unlimited_dv_and_warp_lock_rows():
    lines = _lines(dv_remaining=float("inf"), warp_locked=True)
    assert "Fuel 100% · dV left INF" in lines
    assert lines[-1] == "WARP LOCKED — thrusting"
    assert not any("WARP" in line for line in _lines())


def test_unbound_orbit_has_no_period_number():
    assert "Inc 28.6° · Period —" in _lines(period_s=float("nan"))


def test_no_mass_or_thrust_rows():
    text = "\n".join(_lines())
    assert "Mass" not in text and "Thrust" not in text


def test_inclination_line_in_degrees():
    assert "Inc 28.6° · Period 90.0 min" in _lines(inclination_rad=0.5)


def test_km_units_default():
    lines = _lines(units="km", altitude_m=500_000.0)
    assert "Altitude 500.0 km" in lines
    assert "Speed 7.600 km/s" in lines


def test_mi_units_conversion():
    lines = _lines(units="mi", altitude_m=500_000.0, speed_mps=7600.0)
    # 500 km * 0.621371 = 310.7 mi ; 7.6 km/s * 0.621371 = 4.722 mi/s
    assert "Altitude 310.7 mi" in lines
    assert "Speed 4.722 mi/s" in lines


def test_no_sim_time_row():
    # Mission time lives in the time + warp panel now.
    assert not any("Sim time" in line for line in _lines())


def test_fmt_lead():
    from orbitsim.render.hud.format import fmt_lead
    assert [fmt_lead(s) for s in (0.4, 45, 36 * 60 + 10, 2 * 3600 + 300, 3 * 86400 + 4 * 3600)] == [
        "now", "in 45 s", "in 36 min", "in 2 h 05 min", "in 3 d 04 h"]


def test_mission_time_formats():
    from orbitsim.render.hud.format import fmt_mission_time
    assert fmt_mission_time(12015.0) == "T+ 03:20:15"
    assert fmt_mission_time(12 * 86400 + 3 * 3600 + 20 * 60) == "T+ 12d 03h 20m"
    assert fmt_mission_time(969_100 * 86400 + 3 * 3600) == "T+ 969,100d 03h"
    assert fmt_mission_time(0.0) == "T+ 00:00:00"
    assert fmt_mission_time(-12015.0) == "T− 03:20:15"


def test_warp_formats():
    from orbitsim.render.hud.format import fmt_warp
    assert fmt_warp(100.0, False) == "×100" and fmt_warp(1e6, False) == "×1,000,000"
    assert fmt_warp(1.0, True) == "×1 LOCKED"


def test_burn_summary():
    from orbitsim.render.hud.format import burn_summary, fmt_until
    assert burn_summary(3130.0, 23 * 60, 413.8) == "Burn 3,130 m/s · in 23 min · 6m 54s burn"
    # (Ruling: the plan's 7.5 s case reads 7.4 here -- 7.5 s rounds to 8 s at
    # 'nearest second', which the interface states.)
    assert burn_summary(37.4, None, 7.4) == "Burn 37 m/s · 7s burn"
    assert fmt_until(1.0) == "now" and fmt_until(2 * 3600 + 5 * 60) == "in 2h 05m"


def test_fmt_until():
    from orbitsim.render.hud.format import fmt_until
    assert [fmt_until(s) for s in (2.0, 45.0, 23 * 60 + 5, 3 * 86400 + 4 * 3600 + 59)] == [
        "now", "in 45 s", "in 23 min", "in 3d 04h"]


def test_fmt_burn_duration():
    from orbitsim.render.hud.format import fmt_burn_duration
    assert [fmt_burn_duration(s) for s in (35.2, 413.8, 3720.0)] == [
        "35s burn", "6m 54s burn", "1h 02m burn"]


def test_burn_summary_without_duration():
    from orbitsim.render.hud.format import burn_summary
    assert burn_summary(12.0, 90.0, None) == "Burn 12 m/s · in 1 min"


def test_apsis_label():
    from orbitsim.render.hud.format import apsis_label
    assert apsis_label("PE", 500.0e3) == "Pe 500 km" and apsis_label("AP", 2_465.2e3) == "Ap 2,465 km"


def test_encounter_label():
    from orbitsim.render.hud.format import encounter_label
    assert encounter_label("Moon", 4_317.4e3, False) == "Moon encounter \u00b7 Pe 4,317 km"
    assert encounter_label("Moon", -10.0, True) == "Moon impact"


def test_burn_summary_without_delta_v_says_so():
    # After "Add burn here" / Next Ap the node has a time but no delta-V yet.
    from orbitsim.render.hud.format import burn_summary
    assert burn_summary(0.0, 23 * 60, None) == "No dV yet · in 23 min"
    assert burn_summary(0.0, None, None) == "No dV yet"


def test_burn_summary_for_a_passed_node_says_it_burns_now():
    from orbitsim.render.hud.format import burn_summary
    assert burn_summary(120.0, -30.0, 7.4) == "Burn 120 m/s · node passed, burns now · 7s burn"
    assert burn_summary(120.0, -1.0, 7.4) == "Burn 120 m/s · now · 7s burn"   # within tolerance


def test_map_labels_follow_the_hud_units():
    from orbitsim.render.hud.format import apsis_label, encounter_label
    assert apsis_label("PE", 500.0e3, units="mi") == "Pe 311 mi"
    assert encounter_label("Moon", 4_317.4e3, False, units="mi") == "Moon encounter · Pe 2,683 mi"
    assert apsis_label("AP", 2_465.2e3, units="km") == "Ap 2,465 km"
