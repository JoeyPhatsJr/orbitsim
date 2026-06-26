"""Tests for closest-approach between two Keplerian trajectories."""
import numpy as np
import pytest
from orbitsim.core.rendezvous import ClosestApproach, closest_approach
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH
from orbitsim.core.propagate import propagate_kepler


def _circular(r, mu=MU_EARTH, plane="xy"):
    v = np.sqrt(mu / r)
    if plane == "xy":
        return StateVector(r=np.array([r, 0.0, 0.0]), v=np.array([0.0, v, 0.0]), mu=mu)
    return StateVector(r=np.array([r, 0.0, 0.0]), v=np.array([0.0, 0.0, v]), mu=mu)


def test_identical_states_zero_separation_now():
    s = _circular(7.0e6)
    ca = closest_approach(s, s, window_s=6000.0)
    assert ca.separation_m < 1.0
    assert ca.t_ca_s < 60.0
    assert ca.rel_speed_mps < 1e-6


def test_concentric_circles_min_is_radius_difference():
    r1, r2 = 7.0e6, 2.0e7
    a, b = _circular(r1), _circular(r2)
    n1 = np.sqrt(MU_EARTH / r1**3)
    n2 = np.sqrt(MU_EARTH / r2**3)
    synodic = 2.0 * np.pi / (n1 - n2)
    ca = closest_approach(a, b, window_s=1.3 * synodic, coarse_samples=2000)
    # Coplanar concentric circles: closest possible separation is |r2 - r1|.
    assert abs(ca.separation_m - (r2 - r1)) < 0.02 * (r2 - r1)


def test_refine_not_worse_than_coarse():
    r1, r2 = 7.0e6, 1.1e7
    a, b = _circular(r1), _circular(r2)
    n1 = np.sqrt(MU_EARTH / r1**3)
    n2 = np.sqrt(MU_EARTH / r2**3)
    synodic = 2.0 * np.pi / (n1 - n2)
    window = 1.3 * synodic
    ca = closest_approach(a, b, window_s=window, coarse_samples=500)
    times = np.linspace(0.0, window, 501)
    seps = [np.linalg.norm(propagate_kepler(a, t).r - propagate_kepler(b, t).r) for t in times]
    assert ca.separation_m <= min(seps) + 1.0


def test_rejects_bad_window():
    s = _circular(7.0e6)
    with pytest.raises(ValueError):
        closest_approach(s, s, window_s=0.0)
    with pytest.raises(ValueError):
        closest_approach(s, s, window_s=100.0, coarse_samples=1)
