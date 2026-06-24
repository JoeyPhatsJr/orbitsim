"""Sample an orbit into a polyline and (Task 8) build a Panda3D LineSegs node."""
import numpy as np

from orbitsim.core.elements import KeplerianElements, elements_to_state


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
