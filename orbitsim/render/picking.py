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


def nearest_future_point(click_px, points_px, epochs_s, now_s, tol_px=14.0, min_lead_s=1.0):
    """Index of the nearest visible trajectory sample that is safely in the future."""
    candidates = [
        i for i, (point, epoch) in enumerate(zip(points_px, epochs_s))
        if point is not None and epoch >= now_s + min_lead_s
    ]
    hit = nearest_marker(click_px, [points_px[i] for i in candidates], tol_px)
    return None if hit is None else candidates[hit]
