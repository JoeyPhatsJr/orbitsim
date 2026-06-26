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
