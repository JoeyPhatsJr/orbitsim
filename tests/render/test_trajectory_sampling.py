"""Trajectory sampling: pure, deterministic, and picklable across processes.

These functions were extracted from ``OrbitApp`` so the render layer can run the
live orbit line and the maneuver preview in worker processes. The properties that
matter for that: they need no graphics, they are deterministic, and their inputs
and outputs survive a process boundary.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH
from orbitsim.render.trajectory_sampling import (
    sample_trajectory, sample_preview, trajectory_horizon_s,
)

MIN_SUBSTEP_S = 120.0


def _leo_state():
    r0 = 7.0e6
    return StateVector(
        r=np.array([r0, 0.0, 0.0]),
        v=np.array([0.0, np.sqrt(MU_EARTH / r0), 0.0]),
        mu=MU_EARTH, epoch_s=0.0,
    )


def test_sample_trajectory_shape_and_start_earth_moon():
    state = _leo_state()
    pts = sample_trajectory(state, solar_system=False, min_substep_s=MIN_SUBSTEP_S, n_pts=128)
    assert pts.shape == (128, 3)
    assert pts.dtype == np.float64
    assert np.array_equal(pts[0], state.r)   # first point is exactly the seed
    assert np.all(np.isfinite(pts))


def test_sample_trajectory_is_deterministic():
    state = _leo_state()
    a = sample_trajectory(state, False, MIN_SUBSTEP_S, n_pts=96)
    b = sample_trajectory(state, False, MIN_SUBSTEP_S, n_pts=96)
    assert np.array_equal(a, b)


def test_sample_trajectory_matches_across_process_boundary():
    # The whole point of the extraction: identical result whether run in-process or
    # dispatched to a worker process (validates picklability of args + return).
    state = _leo_state()
    local = sample_trajectory(state, False, MIN_SUBSTEP_S, n_pts=96)
    with ProcessPoolExecutor(max_workers=1) as ex:
        remote = ex.submit(sample_trajectory, state, False, MIN_SUBSTEP_S, 96).result()
    assert np.array_equal(local, remote)


def test_sample_preview_returns_points_and_encounters():
    state = _leo_state()
    pts, encounters = sample_preview(state, solar_system=False, min_substep_s=MIN_SUBSTEP_S)
    assert pts.shape == (256, 3)
    assert isinstance(encounters, list)   # empty list for a bound LEO with no flyby


def test_trajectory_horizon_earth_moon_is_one_week():
    assert trajectory_horizon_s(_leo_state(), solar_system=False) == 7.0 * 86400.0
