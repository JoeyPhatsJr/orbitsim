"""Encounter detection along a predicted trajectory (pure, float64, SI).

Given a forward-propagated path sampled as ``(points, epochs)`` in the frame's
central-body frame, find the runs of samples that fall inside another body's
sphere of influence — a Moon or planet flyby in the geocentric sandbox — and
summarize each: closest approach to that body, when it happens, the hyperbolic
excess speed, and whether the pass is an impact.

The heavy lifting (which body dominates at a given point/time) is injected as a
``dominant_fn`` callable, so this module stays pure and unit-testable with a
synthetic classifier and never imports the render layer. Convenience wrappers
(`earth_moon_dominant`, `solar_dominant`) bind the sandbox's real classifiers.

Periapsis is measured against the body's center **at each sample's own epoch**
(the body moves). With per-sample velocities and the body's state function it comes
from the body-relative conic at the nearest sample (``core.flyby.conic_periapsis``):
drawn samples are hours apart, and a parabola through them read a 190 km Moon pass as
3,258 km. Without them it is refined with a parabola through the three samples
bracketing the minimum (fine for wide passes and synthetic tests). ``v_inf`` is the hyperbolic excess speed
from vis-viva at the SOI-entry sample, using a finite-difference of the
body-relative position for the relative velocity — accurate enough for a
planning readout, and the live in-SOI HUD (`core/flyby.py`) shows the exact
value once the vessel actually arrives.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Encounter:
    """A single SOI pass detected along a predicted trajectory.

    Attributes
    ----------
    body_name : str
        The body whose SOI the trajectory enters.
    start_index, end_index : int
        Inclusive sample-index bounds of the run inside the SOI.
    entry_epoch_s : float
        Epoch of the first in-SOI sample. Equals ``epochs[0]`` (and is not a
        true crossing) when the trajectory begins already inside the SOI.
    exit_epoch_s : float or None
        Epoch of the last in-SOI sample, or ``None`` when the trajectory ends
        still inside the SOI (an open / unresolved encounter).
    periapsis_index : int
        Sample index nearest the closest approach (marker anchor).
    periapsis_epoch_s : float
        Sub-sample-refined epoch of closest approach.
    periapsis_radius_m : float
        Sub-sample-refined closest-approach distance to the body center [m].
    periapsis_point_m : np.ndarray, shape (3,)
        Where the "Pe" marker is drawn (a fixed point in the central-body frame):
        the conic periapsis when refined, else the vessel at ``periapsis_index``.
    v_inf_mps : float
        Hyperbolic excess speed relative to the body [m/s] (nan if unresolved).
    body_mu : float
        Gravitational parameter of the body [m^3/s^2].
    body_radius_m : float
        Physical radius of the body [m].
    closed : bool
        True when the trajectory both enters and exits the SOI within the samples.
    impact : bool
        True when the closest approach is below the body's surface radius.
    """

    body_name: str
    start_index: int
    end_index: int
    entry_epoch_s: float
    exit_epoch_s: float | None
    periapsis_index: int
    periapsis_epoch_s: float
    periapsis_radius_m: float
    periapsis_point_m: np.ndarray
    v_inf_mps: float
    body_mu: float
    body_radius_m: float
    closed: bool
    impact: bool


def _parabola_vertex(x0, x1, x2, y0, y1, y2):
    """Vertex (x*, y*) of the parabola through three points; falls back to the
    middle sample if the points are degenerate (collinear / coincident)."""
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0.0:
        return x1, y1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    if a <= 0.0:
        return x1, y1                      # not a minimum (or a line) — keep sample
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1
         + x0 * x1 * (x0 - x1) * y2) / denom
    xv = -b / (2.0 * a)
    lo, hi = min(x0, x2), max(x0, x2)
    if not (lo <= xv <= hi):               # vertex outside the bracket — keep sample
        return x1, y1
    return xv, a * xv * xv + b * xv + c


def _build_encounter(points, epochs, centers, start, end, name, mu, radius,
                     velocities=None, body_state_fn=None):
    """Assemble one Encounter from a resolved [start, end] run inside an SOI."""
    idx = np.arange(start, end + 1)
    rel = points[idx] - centers[idx]
    dist = np.linalg.norm(rel, axis=1)
    k = int(np.argmin(dist))               # local index within the run
    pe_index = start + k

    # Sub-sample refinement of the closest-approach distance and epoch.
    if 0 < k < len(dist) - 1:
        te, de = _parabola_vertex(
            epochs[pe_index - 1], epochs[pe_index], epochs[pe_index + 1],
            float(dist[k - 1]), float(dist[k]), float(dist[k + 1]))
        pe_epoch, pe_radius = float(te), float(de)
    else:
        pe_epoch, pe_radius = float(epochs[pe_index]), float(dist[k])
    pe_point = np.asarray(points[pe_index], dtype=np.float64).copy()

    # Conic refinement: the body-relative two-body conic through the nearest sample.
    if velocities is not None and body_state_fn is not None and mu > 0.0:
        from orbitsim.core.flyby import conic_periapsis
        t_k = float(epochs[pe_index])
        body = body_state_fn(name, t_k)
        try:
            pe = conic_periapsis(points[pe_index] - body.r, velocities[pe_index] - body.v, mu)
        except ValueError:
            pe = None                                # radial line: keep the sampled value
        if pe is not None:
            pe_epoch = t_k + pe.time_to_periapsis_s
            pe_radius = pe.radius_m
            pe_point = np.asarray(body_state_fn(name, pe_epoch).r, dtype=np.float64) \
                + pe.point_rel_m

    # v_inf from vis-viva at the entry sample: v_inf^2 = v_rel^2 - 2 mu / r_rel.
    # The relative velocity is a finite difference of the body-relative position
    # (which already removes the body's own motion). Use the entry sample where
    # the geometry is nearly straight, so the difference is well conditioned.
    v_inf = float("nan")
    if mu > 0.0 and end > start:
        e0 = start if start + 1 <= end else start - 1
        e1 = e0 + 1
        dt = epochs[e1] - epochs[e0]
        if dt != 0.0:
            v_rel = (rel[e1 - start] - rel[e0 - start]) / dt
            r_rel = np.linalg.norm(rel[e0 - start])
            if r_rel > 0.0:
                v_inf = float(np.sqrt(max(0.0, float(v_rel @ v_rel) - 2.0 * mu / r_rel)))

    closed = end < len(points) - 1 and start > 0
    return Encounter(
        body_name=name,
        start_index=start,
        end_index=end,
        entry_epoch_s=float(epochs[start]),
        exit_epoch_s=float(epochs[end]) if end < len(points) - 1 else None,
        periapsis_index=pe_index,
        periapsis_epoch_s=pe_epoch,
        periapsis_radius_m=pe_radius,
        periapsis_point_m=pe_point,
        v_inf_mps=v_inf,
        body_mu=float(mu),
        body_radius_m=float(radius),
        closed=bool(closed),
        impact=bool(pe_radius < radius),
    )


def find_encounters(points, epochs, dominant_fn, primary_name="Earth", velocities=None,
                    body_state_fn=None):
    """Detect SOI passes along a sampled trajectory.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Predicted positions [m] in the central-body frame.
    epochs : array-like, shape (N,)
        Sample epochs [s] (monotonic).
    dominant_fn : callable
        ``dominant_fn(r_m, t_s) -> (name, center_m, mu, radius_m)`` — the body
        that dominates at position ``r_m`` and time ``t_s``, its center in the
        same frame, and its μ and physical radius.
    primary_name : str
        The frame's central body; runs dominated by it are not encounters.
    velocities : array-like, shape (N, 3), optional
        Predicted velocities [m/s] at the samples.
    body_state_fn : callable, optional
        ``body_state_fn(name, t_s) -> StateVector`` of a body in the same frame. With
        ``velocities``, each periapsis comes from the body-relative conic at the
        nearest sample (radius, epoch, point and the impact test).

    Returns
    -------
    list[Encounter]
        One per run whose dominant body differs from ``primary_name``, in the
        order encountered along the path.
    """
    points = np.asarray(points, dtype=np.float64)
    epochs = np.asarray(epochs, dtype=np.float64)
    if velocities is not None:
        velocities = np.asarray(velocities, dtype=np.float64)
    n = len(points)
    if n == 0 or points.shape[1:] != (3,) or len(epochs) != n:
        return []

    names = []
    centers = np.empty((n, 3), dtype=np.float64)
    props = {}                              # name -> (mu, radius)
    for i in range(n):
        name, center, mu, radius = dominant_fn(points[i], float(epochs[i]))
        names.append(name)
        centers[i] = np.asarray(center, dtype=np.float64)
        props[name] = (mu, radius)

    encounters = []
    i = 0
    while i < n:
        name = names[i]
        j = i
        while j + 1 < n and names[j + 1] == name:
            j += 1
        if name != primary_name:
            mu, radius = props[name]
            encounters.append(
                _build_encounter(points, epochs, centers, i, j, name, mu, radius,
                                 velocities, body_state_fn))
        i = j + 1
    return encounters


@dataclass(frozen=True)
class TargetApproach:
    """Closest approach of a predicted path to a chosen body.

    distance_m is from the body's **centre**; ship_point_m and body_point_m are the
    ship's and the body's positions [m] at epoch_s, in the path's frame.
    """

    body_name: str
    distance_m: float
    epoch_s: float
    ship_point_m: np.ndarray
    body_point_m: np.ndarray


def closest_approach_on_path(points, epochs, body_name, body_state_fn,
                             encounters=None) -> TargetApproach:
    """Closest sample of ``(points, epochs)`` to ``body_name`` (minimum over samples).

    When ``encounters`` holds a pass of that body, the approach is the first such pass's
    periapsis instead (sub-sample: its conic periapsis when the encounters were refined).
    """
    for enc in encounters or ():
        if enc.body_name == body_name:
            body_r = np.asarray(body_state_fn(body_name, enc.periapsis_epoch_s).r,
                                dtype=np.float64)
            return TargetApproach(body_name, float(enc.periapsis_radius_m),
                                  float(enc.periapsis_epoch_s),
                                  np.asarray(enc.periapsis_point_m, dtype=np.float64).copy(),
                                  body_r.copy())
    points = np.asarray(points, dtype=np.float64)
    epochs = np.asarray(epochs, dtype=np.float64)
    if len(points) == 0 or len(points) != len(epochs):
        raise ValueError("closest approach needs matching, non-empty points and epochs")
    bodies = np.array([np.asarray(body_state_fn(body_name, float(t)).r, dtype=np.float64)
                       for t in epochs])
    dist = np.linalg.norm(points - bodies, axis=1)
    k = int(np.argmin(dist))
    return TargetApproach(body_name, float(dist[k]), float(epochs[k]),
                          points[k].copy(), bodies[k].copy())


# ---------------------------------------------------------------------------
# Sandbox classifiers (bound to the geocentric N-body model).
# ---------------------------------------------------------------------------
def solar_dominant(r_m, t_s):
    """dominant_fn for the full solar-system sandbox (geocentric frame)."""
    from orbitsim.core.nbody import dominant_body_solar

    body, center = dominant_body_solar(r_m, t_s)
    return body.name, np.asarray(center, dtype=np.float64), body.mu, body.radius_m


def earth_moon_dominant(r_m, t_s):
    """dominant_fn for the Earth+Moon sandbox: Moon inside its SOI, else Earth."""
    import numpy as _np
    from orbitsim.core.moon import moon_state_at
    from orbitsim.core.nbody import MOON_SOI_M
    from orbitsim.core.bodies import EARTH as _EARTH, MOON as _MOON

    rM = moon_state_at(t_s).r
    if _np.linalg.norm(_np.asarray(r_m, dtype=_np.float64) - rM) < MOON_SOI_M:
        return _MOON.name, _np.asarray(rM, dtype=_np.float64), _MOON.mu, _MOON.radius_m
    return _EARTH.name, _np.zeros(3), _EARTH.mu, _EARTH.radius_m
