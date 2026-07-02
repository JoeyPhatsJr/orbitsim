"""Sample an orbit into a polyline and (Task 8) build a Panda3D LineSegs node."""
import numpy as np

from orbitsim.core.elements import KeplerianElements, elements_to_state


TRAJECTORY_COLOR = (0.20, 0.78, 1.0, 1.0)
MANEUVER_COLOR = (1.0, 0.25, 0.90, 1.0)
REFERENCE_ORBIT_COLOR = (0.48, 0.52, 0.62, 0.82)


def path_fade_alphas(points, minimum: float = 0.28) -> np.ndarray:
    """Alpha ramp by cumulative path distance: present is bright, future recedes."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    minimum = max(0.0, min(1.0, minimum))
    if len(pts) == 1:
        return np.ones(1, dtype=np.float64)
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    if distance[-1] <= 0.0:
        return np.ones(len(pts), dtype=np.float64)
    t = distance / distance[-1]
    return 1.0 + (minimum - 1.0) * t * t


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


def sample_relative_orbit_points(body_state, center_state, mu: float, n: int = 256) -> np.ndarray:
    """Sample the osculating orbit of `body_state` about `center_state`.

    Both states must share one frame (the sandbox passes geocentric states);
    the orbit is computed from the relative state with the given mu and the
    returned (n, 3) float64 positions [m] are relative to the center — ready
    to parent to a frame node placed at the center's rendered position.

    Falls back to a circle at the current relative radius if the relative
    state is degenerate (zero relative velocity or radius), so the reference
    line never vanishes.
    """
    from orbitsim.core.elements import state_to_elements
    from orbitsim.core.state import StateVector

    rel = StateVector(
        r=np.asarray(body_state.r, dtype=np.float64) - np.asarray(center_state.r, dtype=np.float64),
        v=np.asarray(body_state.v, dtype=np.float64) - np.asarray(center_state.v, dtype=np.float64),
        mu=mu,
        epoch_s=body_state.epoch_s,
    )
    try:
        return sample_orbit_points(state_to_elements(rel), n=n)
    except (ValueError, ZeroDivisionError):
        radius = float(np.linalg.norm(rel.r))
        angles = np.linspace(0.0, 2.0 * np.pi, n)
        return np.stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)], axis=1
        )


from panda3d.core import LineSegs, NodePath


def build_orbit_node(
    points_render: list[tuple[float, float, float]],
    color: tuple[float, float, float, float] = TRAJECTORY_COLOR,
    thickness: float = 2.25,
    fade_minimum: float = 0.28,
) -> NodePath:
    """Build an outlined, depth-tested trajectory polyline.

    A dark, wider under-stroke keeps the line legible over Earth, stars, and the
    bright limb without making it look like a foreground HUD overlay.

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
    from panda3d.core import AntialiasAttrib, TransparencyAttrib

    root = NodePath("trajectory_line")
    fade = path_fade_alphas(points_render, fade_minimum)

    def stroke(name, stroke_color, stroke_thickness, sort):
        segs = LineSegs(name)
        segs.set_thickness(stroke_thickness)
        for idx, (x, y, z) in enumerate(points_render):
            segs.set_color(
                stroke_color[0], stroke_color[1], stroke_color[2],
                stroke_color[3] * float(fade[idx]),
            )
            if idx == 0:
                segs.move_to(x, y, z)
            else:
                segs.draw_to(x, y, z)
        node = root.attach_new_node(segs.create())
        node.set_bin("fixed", sort)
        node.set_depth_test(True)
        node.set_depth_write(False)
        node.set_transparency(TransparencyAttrib.M_alpha)
        node.set_antialias(AntialiasAttrib.M_line)

    stroke("trajectory_halo", (0.0, 0.01, 0.03, 0.72), thickness + 3.0, 10)
    stroke("trajectory_color", color, thickness, 11)
    return root
