"""Pure screen-space marker hit-testing (no Panda3D)."""
from math import hypot


def nearest_marker(click_px, markers_px, tol_px):
    """Index of the nearest marker within tol_px of the click, else None.

    Parameters
    ----------
    click_px : (float, float)        Click position in pixels.
    markers_px : list[(float, float)] Marker positions in pixels.
    tol_px : float                    Max hit distance in pixels.
    """
    best_i, best_d = None, tol_px
    for i, (mx, my) in enumerate(markers_px):
        d = hypot(mx - click_px[0], my - click_px[1])
        if d <= best_d:
            best_i, best_d = i, d
    return best_i


# --- Screen-space polyline picking (plan 3: click/hover your orbit) ---
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402


def project_points_px(points_render, render_to_clip, width_px, height_px):
    """Project (N, 3) render-space points to (N, 2) bottom-up pixel coordinates.

    ``render_to_clip`` is the row-vector 4x4 (render->camera times camera->clip). Rows
    whose clip w <= 1e-9 (at or behind the camera) are NaN.
    """
    pts = np.asarray(points_render, dtype=np.float64)
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    clip = homo @ np.asarray(render_to_clip, dtype=np.float64)
    w = clip[:, 3]
    out = np.full((len(pts), 2), np.nan)
    ok = w > 1e-9
    out[ok, 0] = (clip[ok, 0] / w[ok] * 0.5 + 0.5) * width_px
    out[ok, 1] = (clip[ok, 1] / w[ok] * 0.5 + 0.5) * height_px
    return out


@dataclass(frozen=True)
class PolylinePick:
    """Where a click lands on a time-sampled polyline."""

    epoch_s: float        # interpolated time at the picked point
    index: int            # segment start sample
    fraction: float       # position along the segment [0, 1]
    distance_px: float    # click-to-line distance


def pick_polyline_epoch(click_px, points_px, epochs_s, min_epoch_s, max_epoch_s, tol_px,
                        tie_px=3.0):
    """Nearest point on a time-sampled screen polyline within a time window, or None.

    Each segment is limited to the part whose interpolated epoch lies in
    [min_epoch_s, max_epoch_s]; the closest point of that part is taken. Hits farther
    than ``tol_px`` are ignored. Overlapping loops: among hits within ``tie_px`` of the
    best one, the earliest epoch wins. Segments with a NaN end (behind the camera) skip.
    """
    p = np.asarray(points_px, dtype=np.float64)
    e = np.asarray(epochs_s, dtype=np.float64)
    if len(p) < 2 or not min_epoch_s <= max_epoch_s:
        return None
    a, b = p[:-1], p[1:]
    e0, e1 = e[:-1], e[1:]
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    span = e1 - e0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_lo = np.where(span != 0, (min_epoch_s - e0) / span, 0.0)
        f_hi = np.where(span != 0, (max_epoch_s - e0) / span, 1.0)
    inside = (span != 0) | ((e0 >= min_epoch_s) & (e0 <= max_epoch_s))
    valid = finite & inside & (f_lo <= 1.0) & (f_hi >= 0.0) & (f_lo <= f_hi)
    if not valid.any():
        return None
    lo, hi = np.clip(f_lo, 0.0, 1.0), np.clip(f_hi, 0.0, 1.0)
    ab = b - a
    seg_len2 = (ab * ab).sum(axis=1)
    c = np.asarray(click_px, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(seg_len2 > 0, ((c - a) * ab).sum(axis=1) / seg_len2, 0.0)
    f = np.clip(np.nan_to_num(f), lo, hi)
    dist = np.hypot(*(a + f[:, None] * ab - c).T)
    dist = np.where(valid, dist, np.inf)
    hits = dist <= tol_px
    if not hits.any():
        return None
    best = dist[hits].min()
    near = hits & (dist <= best + tie_px)
    epoch = e0 + f * span
    i = int(np.flatnonzero(near)[np.argmin(epoch[near])])
    return PolylinePick(float(epoch[i]), i, float(f[i]), float(dist[i]))


def point_at(points_m, pick):
    """World position [m] of a pick: the lerp of its segment's two samples."""
    pts = np.asarray(points_m, dtype=np.float64)
    i = pick.index
    return pts[i] + pick.fraction * (pts[i + 1] - pts[i])
