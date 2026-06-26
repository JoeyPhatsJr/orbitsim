"""Sample an orbit into a polyline and (Task 8) build a Panda3D LineSegs node."""
import numpy as np

from orbitsim.core.elements import KeplerianElements, elements_to_state


def _angle_diff(x: float, y: float) -> float:
    """Smallest absolute difference between two angles [rad], accounting for 2π wrap."""
    d = abs(x - y) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def orbit_shape_changed(a, b, tol: float = 1e-6) -> bool:
    """True if the orbit *shape* (a, e, i, raan, argp) differs beyond tolerance.

    `a` is compared relative, `e`/`i` absolute, and `raan`/`argp` with 2π-wrap
    awareness (they flip between ~0 and ~2π under recovery noise near the
    boundary). True anomaly (position along the orbit) is ignored. A None on
    either side counts as changed (forces an initial build). The default `tol`
    absorbs element-recovery noise on a coasting orbit while still flagging any
    real burn (which shifts elements far more).
    """
    if a is None or b is None:
        return True
    if abs(a.a - b.a) > tol * max(abs(a.a), 1.0):
        return True
    return (abs(a.e - b.e) > tol or abs(a.i - b.i) > tol
            or _angle_diff(a.raan, b.raan) > tol or _angle_diff(a.argp, b.argp) > tol)


def sample_orbit_points(elements: KeplerianElements, n: int = 256) -> np.ndarray:
    """Sample physics-space positions along an orbit.

    Parameters
    ----------
    elements : KeplerianElements
    n : int
        Number of samples.

    Returns
    -------
    np.ndarray
        (n, 3) float64 positions [m] in the inertial frame.

    Notes
    -----
    Ellipse (e < 1): true anomaly spans [0, 2pi).
    Hyperbola (e >= 1): true anomaly spans the open interval bounded by the
    asymptote angle nu_max = arccos(-1/e); we sample (1 - margin)*nu_max so the
    radius stays finite.
    """
    e = elements.e
    if e < 1.0:
        nus = np.linspace(0.0, 2.0 * np.pi, n)
    else:
        nu_max = np.arccos(-1.0 / e)
        limit = 0.99 * nu_max
        nus = np.linspace(-limit, limit, n)

    pts = np.empty((n, 3), dtype=np.float64)
    for idx, nu in enumerate(nus):
        sampled = KeplerianElements(
            a=elements.a, e=elements.e, i=elements.i, raan=elements.raan,
            argp=elements.argp, nu=float(nu), mu=elements.mu, epoch_s=elements.epoch_s,
        )
        pts[idx] = elements_to_state(sampled).r
    return pts


from panda3d.core import LineSegs, NodePath


def build_orbit_node(
    points_render: list[tuple[float, float, float]],
    color: tuple[float, float, float, float] = (0.3, 0.7, 1.0, 1.0),
) -> NodePath:
    """Build a Panda3D LineSegs polyline from render-space points.

    Parameters
    ----------
    points_render : list of (x, y, z)
        Render-space points (already passed through RenderTransform.to_render).
    color : (r, g, b, a)

    Returns
    -------
    NodePath
        A NodePath holding the line strip.
    """
    segs = LineSegs()
    segs.set_color(*color)
    segs.set_thickness(1.5)
    for idx, (x, y, z) in enumerate(points_render):
        if idx == 0:
            segs.move_to(x, y, z)
        else:
            segs.draw_to(x, y, z)
    return NodePath(segs.create())
