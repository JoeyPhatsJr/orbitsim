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

A fast, shallow flyby can have its whole SOI crossing fall *between* two
display samples, so a sample-only scan misses it. Each gap whose endpoints are
both primary-dominated is therefore probed at a few interpolated sub-points
(`refine_steps`); a run of probes inside a body's SOI becomes a "graze"
encounter anchored to its bracketing sample pair (so render indices stay valid).

Periapsis is measured against the body's center **at each sample's own epoch**
(the body moves), refined to sub-sample accuracy with a parabola through the
three samples bracketing the minimum. ``v_inf`` is the hyperbolic excess speed
from vis-viva at the SOI-entry sample, using a finite-difference of the
body-relative position for the relative velocity — accurate enough for a
planning readout, and the live in-SOI HUD (`core/flyby.py`) shows the exact
value once the vessel actually arrives.
"""

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
        The vessel's predicted position at ``periapsis_index`` (a fixed point in
        the central-body frame — where a "Pe" marker is drawn).
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
        return x1, y1  # not a minimum (or a line) — keep sample
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
    xv = -b / (2.0 * a)
    lo, hi = min(x0, x2), max(x0, x2)
    if not (lo <= xv <= hi):  # vertex outside the bracket — keep sample
        return x1, y1
    return xv, a * xv * xv + b * xv + c


def _periapsis_and_vinf(rel, epochs_run, mu):
    """Closest-approach and hyperbolic-excess-speed of a run of body-relative
    samples ``rel`` (shape (M, 3)) at epochs ``epochs_run``.

    Returns ``(k, pe_epoch_s, pe_radius_m, v_inf_mps)`` where ``k`` is the local
    index of the sample nearest closest approach; the epoch and radius are
    parabola-refined to sub-sample accuracy when the minimum is interior.

    ``v_inf`` is vis-viva at the entry sample: ``v_inf^2 = v_rel^2 - 2 mu/r_rel``,
    with the relative velocity a finite difference of the (already body-relative)
    position — well conditioned near entry where the geometry is nearly straight.
    """
    dist = np.linalg.norm(rel, axis=1)
    m = len(dist)
    k = int(np.argmin(dist))
    if 0 < k < m - 1:
        te, de = _parabola_vertex(
            epochs_run[k - 1],
            epochs_run[k],
            epochs_run[k + 1],
            float(dist[k - 1]),
            float(dist[k]),
            float(dist[k + 1]),
        )
        pe_epoch, pe_radius = float(te), float(de)
    else:
        pe_epoch, pe_radius = float(epochs_run[k]), float(dist[k])

    v_inf = float("nan")
    if mu > 0.0 and m >= 2:
        dt = epochs_run[1] - epochs_run[0]
        if dt != 0.0:
            v_rel = (rel[1] - rel[0]) / dt
            r_rel = np.linalg.norm(rel[0])
            if r_rel > 0.0:
                v_inf = float(np.sqrt(max(0.0, float(v_rel @ v_rel) - 2.0 * mu / r_rel)))
    return k, pe_epoch, pe_radius, v_inf


def _build_encounter(points, epochs, centers, start, end, name, mu, radius):
    """Assemble one Encounter from a resolved [start, end] run inside an SOI."""
    idx = np.arange(start, end + 1)
    rel = points[idx] - centers[idx]
    k, pe_epoch, pe_radius, v_inf = _periapsis_and_vinf(rel, epochs[idx], mu)
    pe_index = start + k
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
        periapsis_point_m=np.asarray(points[pe_index], dtype=np.float64).copy(),
        v_inf_mps=v_inf,
        body_mu=float(mu),
        body_radius_m=float(radius),
        closed=bool(closed),
        impact=bool(pe_radius < radius),
    )


def _catmull_rom(p0, p1, p2, p3, u):
    """Position at parameter ``u`` in [0, 1] between ``p1`` and ``p2`` on a
    uniform Catmull-Rom spline; falls back to linear at path ends (``p0`` or
    ``p3`` is None). Bows the interpolant toward a nearby body so a graze
    hiding between two coarse samples is more likely to be probed inside it."""
    if p0 is None or p3 is None:
        return p1 + u * (p2 - p1)
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def _build_graze(rs, ts, centers, a, b, gap_i, epochs, name, mu, radius):
    """Assemble an Encounter for an SOI pass found *between* samples gap_i and
    gap_i+1 by gap probing. Indices reported (start/end/periapsis) stay valid in
    the original sample arrays — the render layer draws the segment and anchors
    the marker off them — while the closest-approach point/epoch/radius come from
    the finer probe samples ``rs``/``ts``/``centers`` over local range [a, b]."""
    run_r = np.asarray(rs[a : b + 1], dtype=np.float64)
    run_t = np.asarray(ts[a : b + 1], dtype=np.float64)
    rel = run_r - np.asarray(centers[a : b + 1], dtype=np.float64)
    k, pe_epoch, pe_radius, v_inf = _periapsis_and_vinf(rel, run_t, mu)
    # Anchor the marker to whichever bracketing original sample is nearer in time.
    pe_index = gap_i if (pe_epoch - epochs[gap_i]) <= (epochs[gap_i + 1] - pe_epoch) else gap_i + 1
    return Encounter(
        body_name=name,
        start_index=gap_i,
        end_index=gap_i + 1,
        entry_epoch_s=float(run_t[0]),
        exit_epoch_s=float(run_t[-1]),
        periapsis_index=int(pe_index),
        periapsis_epoch_s=pe_epoch,
        periapsis_radius_m=pe_radius,
        periapsis_point_m=run_r[k].copy(),
        v_inf_mps=v_inf,
        body_mu=float(mu),
        body_radius_m=float(radius),
        closed=True,  # entered and exited within the gap
        impact=bool(pe_radius < radius),
    )


def _refine_gaps(points, epochs, names, dominant_fn, primary_name, refine_steps):
    """Recover SOI passes whose whole crossing falls between two consecutive
    samples that are *both* dominated by the primary body — the case the
    sample-only scan misses on a fast, shallow flyby. Each such gap is probed at
    ``refine_steps`` interpolated sub-points; a contiguous run of probes inside
    some body's SOI becomes one graze Encounter."""
    n = len(points)
    grazes = []
    for i in range(n - 1):
        if names[i] != primary_name or names[i + 1] != primary_name:
            continue  # a real crossing already lands on a sample
        p1, p2 = points[i], points[i + 1]
        p0 = points[i - 1] if i - 1 >= 0 else None
        p3 = points[i + 2] if i + 2 < n else None
        t1, t2 = float(epochs[i]), float(epochs[i + 1])
        rs, ts, pnames, pcenters, props = [], [], [], [], {}
        for step in range(1, refine_steps + 1):
            u = step / (refine_steps + 1.0)
            rs.append(_catmull_rom(p0, p1, p2, p3, u))
            ts.append(t1 + u * (t2 - t1))
            nm, center, mu, radius = dominant_fn(rs[-1], ts[-1])
            pnames.append(nm)
            pcenters.append(np.asarray(center, dtype=np.float64))
            props[nm] = (mu, radius)
        j = 0
        while j < len(pnames):
            if pnames[j] == primary_name:
                j += 1
                continue
            nm = pnames[j]
            end = j
            while end + 1 < len(pnames) and pnames[end + 1] == nm:
                end += 1
            mu, radius = props[nm]
            grazes.append(_build_graze(rs, ts, pcenters, j, end, i, epochs, nm, mu, radius))
            j = end + 1
    return grazes


_GAP_REFINE_STEPS = 6


def find_encounters(
    points, epochs, dominant_fn, primary_name="Earth", refine_steps=_GAP_REFINE_STEPS
):
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
    refine_steps : int
        Sub-samples probed inside each primary-to-primary gap to recover a fast,
        shallow SOI pass whose whole crossing falls between two samples (the
        adaptive integrator keeps this rare, but coarse *display* sampling can
        still hide one). 0 disables refinement (sample-only scan). The residual:
        a pass narrower than ~1/(refine_steps+1) of a gap can still slip through;
        the live ``core/flyby.py`` readout is exact once the vessel arrives.

    Returns
    -------
    list[Encounter]
        One per SOI pass whose dominant body differs from ``primary_name``, in
        path order. ``start_index``/``end_index``/``periapsis_index`` are always
        valid indices into ``points`` (grazes anchor to their bracketing pair).
    """
    points = np.asarray(points, dtype=np.float64)
    epochs = np.asarray(epochs, dtype=np.float64)
    n = len(points)
    if n == 0 or points.shape[1:] != (3,) or len(epochs) != n:
        return []

    names = []
    centers = np.empty((n, 3), dtype=np.float64)
    props = {}  # name -> (mu, radius)
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
            encounters.append(_build_encounter(points, epochs, centers, i, j, name, mu, radius))
        i = j + 1

    if refine_steps > 0 and n >= 2:
        encounters.extend(
            _refine_gaps(points, epochs, names, dominant_fn, primary_name, refine_steps)
        )
        encounters.sort(key=lambda e: (e.start_index, e.periapsis_epoch_s))
    return encounters


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
