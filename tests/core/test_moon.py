"""Tests for the idealized Keplerian Moon."""
import numpy as np
from orbitsim.core.moon import MOON_ORBIT, moon_state_at


def test_moon_distance_in_apsis_range():
    for t in (0.0, 1.0e5, 5.0e5, 1.0e6):
        r = np.linalg.norm(moon_state_at(t).r)
        assert 3.6e8 < r < 4.05e8, (t, r)


def test_moon_is_periodic():
    T = MOON_ORBIT.period_s
    a = moon_state_at(12345.0).r
    b = moon_state_at(12345.0 + T).r
    assert np.linalg.norm(a - b) < 1.0e3  # < 1 km after one period


def test_moon_state_geocentric_mu():
    assert moon_state_at(0.0).mu == MOON_ORBIT.mu


def test_moon_orbit_is_circular_at_earth_moon_rate():
    from orbitsim.core.constants import MU_EARTH, MU_MOON
    import numpy as np
    from orbitsim.core.moon import MOON_ORBIT, moon_state_at
    assert MOON_ORBIT.e == 0.0
    assert MOON_ORBIT.mu == MU_EARTH + MU_MOON
    # Circular => distance is constant across the orbit.
    dists = [np.linalg.norm(moon_state_at(t).r) for t in (0.0, 5.0e5, 1.0e6, 1.5e6)]
    assert max(dists) - min(dists) < 1.0e3   # < 1 km variation


def test_closed_form_matches_general_kepler_propagation():
    from orbitsim.core.elements import elements_to_state
    from orbitsim.core.propagate import propagate_kepler

    epoch_state = elements_to_state(MOON_ORBIT)
    for t_s in (0.0, 1.0e5, 1.0e6, 3.0e6):
        actual = moon_state_at(t_s)
        expected = propagate_kepler(epoch_state, t_s - epoch_state.epoch_s)
        np.testing.assert_allclose(actual.r, expected.r, rtol=0.0, atol=1e-3)
        np.testing.assert_allclose(actual.v, expected.v, rtol=0.0, atol=1e-9)
