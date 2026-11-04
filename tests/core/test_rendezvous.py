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


def test_nbody_ca_differs_from_keplerian_near_moon():
    """Near the Moon, N-body CA differs measurably from Keplerian CA."""
    from orbitsim.core.nbody import propagate_earth_moon
    from orbitsim.core.moon import moon_state_at
    # Ship and target both near the Moon (strong perturbation).
    rM = moon_state_at(0.0).r
    s_a = StateVector(r=rM + np.array([5.0e6, 0.0, 0.0]),
                      v=np.array([0.0, 900.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    s_b = StateVector(r=rM + np.array([-5.0e6, 0.0, 0.0]),
                      v=np.array([0.0, -900.0, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    window = 6 * 3600.0    # 6-hour window
    ca_kep = closest_approach(s_a, s_b, window_s=window, coarse_samples=360)
    ca_nbody = closest_approach(s_a, s_b, window_s=window, coarse_samples=360,
                                propagator=propagate_earth_moon)
    # N-body and Keplerian CAs must differ by > 1 km (Moon bends trajectories).
    diff = abs(ca_nbody.separation_m - ca_kep.separation_m)
    assert diff > 1000.0, f"expected N-body CA to differ from Keplerian, got {diff:.1f} m"


# ---------------------------------------------------------------------------
# The default (Kepler) path must give the same answer as the generic per-sample
# path, without re-deriving orbital elements for every sample (it took ~270 ms
# on the render thread every 0.5 s while a target was selected).
# ---------------------------------------------------------------------------
def _ca_cases():
    from orbitsim.core.moon import moon_state_at
    leo = StateVector(r=np.array([7.0e6, 0.0, 0.0]), v=np.array([0.0, 7.546e3, 300.0]), mu=MU_EARTH, epoch_s=0.0)
    tli = StateVector(r=np.array([6.7e6, 0.0, 0.0]), v=np.array([0.0, 10.85e3, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    other = StateVector(r=np.array([7.1e6, 1.0e5, 0.0]), v=np.array([-100.0, 7.4e3, 0.0]), mu=MU_EARTH, epoch_s=0.0)
    moon = moon_state_at(0.0)
    return {"leo-vs-moon": (leo, moon, 5.0 * 86400.0), "tli-vs-moon": (tli, moon, 6.0 * 86400.0),
            "leo-vs-leo": (leo, other, 6000.0)}


@pytest.mark.parametrize("case", ["leo-vs-moon", "tli-vs-moon", "leo-vs-leo"])
def test_closest_approach_fast_path_matches_generic_path(case):
    a, b, window = _ca_cases()[case]
    fast = closest_approach(a, b, window_s=window)
    slow = closest_approach(a, b, window_s=window, propagator=lambda s, t: propagate_kepler(s, t))
    assert fast.t_ca_s == pytest.approx(slow.t_ca_s, abs=1e-3)
    assert fast.separation_m == pytest.approx(slow.separation_m, rel=1e-9, abs=1e-3)
    assert fast.rel_speed_mps == pytest.approx(slow.rel_speed_mps, rel=1e-9, abs=1e-6)


def test_closest_approach_does_not_rederive_elements_per_sample(monkeypatch):
    from orbitsim.core import elements as _elements
    from orbitsim.core import propagate as _prop
    calls = []
    real = _elements.state_to_elements

    def counting(state):
        calls.append(1)
        return real(state)

    monkeypatch.setattr(_prop, "state_to_elements", counting)
    a, b, window = _ca_cases()["leo-vs-moon"]
    closest_approach(a, b, window_s=window)
    assert len(calls) <= 4, f"state_to_elements called {len(calls)} times"
