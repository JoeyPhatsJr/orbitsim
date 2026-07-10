"""Pure trajectory-sampling functions for the render layer's orbit lines.

These forward-integrate a vessel state under the N-body model to produce the
points drawn as the live orbit line and the maneuver preview. They are kept as
plain module-level functions (no ``self``, no Panda3D) for two reasons:

1. **Picklability.** ``OrbitApp`` submits them to a ``ProcessPoolExecutor`` so the
   live line and the maneuver preview run in *separate processes* and stop
   GIL-serializing against each other (and against the render thread). A method
   bound to the un-picklable ``OrbitApp`` (which owns Panda3D nodes) could not
   cross a process boundary; a module-level function taking a picklable
   ``StateVector`` + plain scalars can.
2. **Isolation.** All imports here are pure ``core`` physics — the render/sim
   layering rule holds and nothing graphical is dragged into a worker.

A single sample is a *sequential* ODE (point i+1 needs point i) and is NOT
internally parallel; parallelism comes only from running whole samples
concurrently.
"""
from contextlib import nullcontext

import numpy as np


def trajectory_horizon_s(state, solar_system: bool) -> float:
    """Prediction horizon that keeps escape trajectories continuous across Earth SOI."""
    if not solar_system:
        return 7.0 * 86400.0
    from orbitsim.core.constants import MU_EARTH
    from orbitsim.core.planets import EARTH_SOI_M

    radius = float(np.linalg.norm(state.r))
    energy = 0.5 * float(np.dot(state.v, state.v)) - MU_EARTH / radius
    if energy >= 0.0:
        return 400.0 * 86400.0
    semi_major = -MU_EARTH / (2.0 * energy)
    eccentricity_vector = (
        ((float(np.dot(state.v, state.v)) - MU_EARTH / radius) * state.r
         - float(np.dot(state.r, state.v)) * state.v)
        / MU_EARTH
    )
    apoapsis = semi_major * (1.0 + float(np.linalg.norm(eccentricity_vector)))
    if apoapsis >= 0.8 * EARTH_SOI_M:
        return 400.0 * 86400.0
    period = 2.0 * np.pi * np.sqrt(semi_major**3 / MU_EARTH)
    return min(2.0 * period, 30.0 * 86400.0)


def sample_trajectory(
    state, solar_system: bool, min_substep_s: float,
    n_pts=256, max_horizon_s=7 * 86400, n_orbits=1,
    with_times=False, with_encounters=False,
):
    """Forward-integrate ``state`` under N-body and return ~n_pts positions [m].

    Horizon is ``n_orbits`` osculating orbital periods capped at max_horizon_s; for an
    Earth-bound orbit this draws that many closed loops (successive loops drift slightly
    under perturbation), for a translunar/hyperbolic arc (no period) it shows the
    next ``max_horizon_s`` of the perturbed path. Returns an (n_pts, 3) float64 array of
    world-meter positions in the Earth-centered inertial frame.

    ``solar_system`` selects the full solar propagator (with a stable-prediction
    ephemeris context) vs. the Earth-Moon two-body-plus-Moon propagator;
    ``min_substep_s`` is the coarse substep floor for the solar visual prediction.
    """
    if solar_system:
        from orbitsim.core.nbody import (
            osculating_elements_solar,
            propagate_solar_system,
            stable_prediction_ephemeris,
        )
        osc_fn = osculating_elements_solar
        prop_fn = propagate_solar_system
    else:
        from orbitsim.core.nbody import osculating_elements, propagate_earth_moon
        osc_fn = osculating_elements
        prop_fn = propagate_earth_moon
    context = stable_prediction_ephemeris() if solar_system else nullcontext()
    with context:
        try:
            osc = osc_fn(state, state.epoch_s)
            horizon_s = min(n_orbits * float(osc.period_s), max_horizon_s)
        except (ValueError, AttributeError):
            horizon_s = float(max_horizon_s)
        dt = horizon_s / n_pts
        pts = np.empty((n_pts, 3), dtype=np.float64)
        pts[0] = state.r
        cur = state
        for i in range(1, n_pts):
            if solar_system:
                # The visual prediction is far cheaper than the on-rails sim: a larger
                # deep-space ceiling AND a coarse substep floor so a near-Earth escape
                # climb-out (which the periapsis cap would otherwise integrate at ~25 s
                # steps, ~4.6 s per refresh) stays responsive. The line loses only
                # sub-periapsis detail it never draws; adaptive stepping still tightens
                # for encounters/flybys above the floor.
                cur = prop_fn(cur, dt, max_step_s=24.0 * 3600.0,
                              min_substep_s=min_substep_s)
            else:
                cur = prop_fn(cur, dt)
            pts[i] = cur.r
        epochs = state.epoch_s + np.arange(n_pts, dtype=np.float64) * dt
        encounters = []
        if with_encounters:
            # Classify inside the same ephemeris context that produced the
            # path, so far-future planet positions match what was integrated.
            from orbitsim.core.encounters import (
                find_encounters, solar_dominant, earth_moon_dominant,
            )
            dominant = solar_dominant if solar_system else earth_moon_dominant
            encounters = find_encounters(pts, epochs, dominant, primary_name="Earth")
    if with_encounters:
        return pts, epochs, encounters
    if with_times:
        return pts, epochs
    return pts


def sample_preview(state, solar_system: bool, min_substep_s: float):
    """Compute preview points + encounters without touching Panda3D scene objects."""
    horizon = trajectory_horizon_s(state, solar_system)
    pts, _epochs, encounters = sample_trajectory(
        state, solar_system, min_substep_s,
        n_pts=256, max_horizon_s=horizon, n_orbits=2, with_encounters=True,
    )
    return pts, encounters
