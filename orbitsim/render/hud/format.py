"""Compact distance / countdown formatting shared by HUD readouts and markers."""


def fmt_dist(meters: float) -> str:
    AU = 1.496e11
    if meters >= 0.01 * AU:
        return f"{meters / AU:.2f} AU"
    km = meters / 1000.0
    if km >= 1e6:
        return f"{km / 1e6:.2f} M km"
    return f"{km:,.0f} km"


def fmt_countdown(seconds: float) -> str:
    t = int(seconds)
    if t >= 86400:
        d, rem = divmod(t, 86400)
        h = rem // 3600
        return f"{d}d {h:02d}h"
    if t >= 3600:
        h, rem = divmod(t, 3600)
        m = rem // 60
        return f"{h}h {m:02d}m"
    m, s = divmod(t, 60)
    return f"{m:02d}:{s:02d}"


def fmt_lead(seconds) -> str:
    """Time until something, for map labels: now / in 45 s / in 36 min / in 2 h 05 min /
    in 3 d 04 h (whole seconds, floored)."""
    s = int(seconds)
    if s < 1:
        return "now"
    if s < 60:
        return f"in {s} s"
    if s < 3600:
        return f"in {s // 60} min"
    if s < 86400:
        return f"in {s // 3600} h {s % 3600 // 60:02d} min"
    return f"in {s // 86400} d {s % 86400 // 3600:02d} h"


def fmt_mission_time(sim_time_s: float) -> str:
    """Mission clock (seconds past J2000): T+ 03:20:15 under a day, T+ 12d 03h 20m under
    1,000 days, T+ 969,100d 03h beyond; negative times read T− with the same fields."""
    sign = "T+ " if sim_time_s >= 0.0 else "T− "
    t = int(abs(sim_time_s))
    d, rem = divmod(t, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d == 0:
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"
    if d < 1000:
        return f"{sign}{d}d {h:02d}h {m:02d}m"
    return f"{sign}{d:,}d {h:02d}h"


def fmt_warp(warp: float, locked: bool) -> str:
    """Warp factor: ×100, ×1,000,000; ×1 LOCKED while thrusting."""
    text = f"×{warp:,.0f}"
    return f"{text} LOCKED" if locked else text


def fmt_until(seconds) -> str:
    """Time until a burn: now (<= 2 s) / in 45 s / in 23 min / in 2h 05m / in 3d 04h."""
    s = int(seconds)
    if seconds <= 2.0:
        return "now"
    if s < 60:
        return f"in {s} s"
    if s < 3600:
        return f"in {s // 60} min"
    if s < 86400:
        return f"in {s // 3600}h {s % 3600 // 60:02d}m"
    return f"in {s // 86400}d {s % 86400 // 3600:02d}h"


def fmt_burn_duration(seconds) -> str:
    """Burn length, nearest second: 35s burn / 6m 54s burn / 1h 02m burn."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s burn"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s burn"
    return f"{s // 3600}h {s % 3600 // 60:02d}m burn"


PASSED_NODE_TOL_S = 2.0   # the app's EXECUTE_TOLERANCE_S: later than this is "passed"


def burn_summary(dv_mps, ttn_s, duration_s) -> str:
    """One line for a planned burn: Burn 3,130 m/s · in 23 min · 6m 54s burn.
    No time part for a burn-now node (ttn_s None), no duration part when unknown.
    A node with a time but no delta-V says "No dV yet"; one more than
    PASSED_NODE_TOL_S in the past says it burns now (Execute burns where you are)."""
    if dv_mps <= 0.0:
        parts = ["No dV yet"]
        if ttn_s is not None:
            parts.append(fmt_until(ttn_s))
        return " · ".join(parts)
    parts = [f"Burn {dv_mps:,.0f} m/s"]
    if ttn_s is not None:
        parts.append("node passed, burns now" if ttn_s < -PASSED_NODE_TOL_S
                     else fmt_until(ttn_s))
    if duration_s is not None:
        parts.append(fmt_burn_duration(duration_s))
    return " · ".join(parts)


def apsis_label(kind: str, altitude_m: float) -> str:
    """Map label of an apsis tick: Pe 500 km / Ap 2,465 km (whole km above the surface)."""
    return f"{'Pe' if kind == 'PE' else 'Ap'} {altitude_m / 1000.0:,.0f} km"


def encounter_label(body_name: str, periapsis_alt_m: float, impact: bool) -> str:
    """Map label of a predicted encounter: Moon encounter \u00b7 Pe 4,317 km / Moon impact."""
    if impact:
        return f"{body_name} impact"
    return f"{body_name} encounter \u00b7 Pe {periapsis_alt_m / 1000.0:,.0f} km"


def flyby_line(body_name: str, v_inf_mps: float, deflection_rad: float,
               periapsis_m: float, dv_equivalent_mps: float) -> str:
    """In-SOI flyby row: speed at infinity, how far the pass turns you, periapsis and
    the free dV. Short enough for the target panel at any planet; only glyphs the
    bundled font has (no "δ")."""
    import math
    return (f"{body_name} · v-inf {v_inf_mps / 1000.0:,.1f} km/s · "
            f"turns {math.degrees(deflection_rad):,.0f}° · Pe {fmt_dist(periapsis_m)} · "
            f"+{dv_equivalent_mps:,.0f} m/s")
