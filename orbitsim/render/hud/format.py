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
