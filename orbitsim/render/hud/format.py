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
