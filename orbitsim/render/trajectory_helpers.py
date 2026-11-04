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
