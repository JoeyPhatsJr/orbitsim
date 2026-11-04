"""Pure helpers for trajectory display and teleport coasts (no Panda3D)."""
import numpy as np


def maneuver_preview_key(node, scheduled_epoch_s):
    """Stable preview identity; burn-now epochs move with the vessel and are ignored."""
    return (
        node.dv_prograde_mps,
        node.dv_normal_mps,
        node.dv_radial_mps,
        scheduled_epoch_s,
    )


def coast_chunks(duration_s: float, max_chunk_s: float = 86400.0) -> list[float]:
    """Split a teleport coast into bounded positive steps that land exactly on its end."""
    if duration_s <= 0.0:
        return []
    count = max(1, int(np.ceil(duration_s / max_chunk_s)))
    chunk = duration_s / count
    return [chunk] * count


def localize_polyline(points):
    """Return (local points, origin) with float64 subtraction before rendering."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("polyline points must have shape (n, 3) with n > 0")
    origin = pts[0].copy()
    return pts - origin, origin


def kepler_estimate_points(state, n: int = 256):
    """Geocentric (n, 3) points of the two-body orbit through ``state``, or None.

    The instant stand-in for the maneuver preview while the burn is being edited
    (ignores the Moon/Sun; the exact N-body path replaces it once editing settles).
    None for a degenerate state (zero velocity, etc.) -- then nothing is drawn.
    """
    from orbitsim.core.elements import state_to_elements
    from orbitsim.render.orbit_lines import sample_orbit_points
    try:
        return sample_orbit_points(state_to_elements(state), n)
    except (ValueError, ZeroDivisionError):
        return None


def should_submit_exact(current_key, exact_key, inflight_key, last_change_t, now, settle_s):
    """Whether to start the exact N-body preview for ``current_key`` now.

    Only once the burn has been still for ``settle_s``, when no job is in flight, and
    when the drawn exact path isn't already for this burn (no endless resubmits).
    """
    if current_key == exact_key or inflight_key is not None:
        return False
    return now - last_change_t >= settle_s
