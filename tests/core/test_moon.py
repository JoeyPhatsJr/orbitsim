"""Tests for the geocentric Moon: real DE440 ephemeris + circular offline fallback."""
import numpy as np
import pytest

from orbitsim.core import moon
from orbitsim.core.constants import MU_EARTH, MU_MOON
from orbitsim.core.moon import MOON_ORBIT, moon_circular_state_at, moon_state_at


def _kernel_or_skip():
    from orbitsim.core import ephemeris
    if not ephemeris.available():
        pytest.skip("DE440 kernel not available")
    return ephemeris


# ---------------------------------------------------------------------------
# Circular fallback (used offline). Pure analytic model, always testable.
# ---------------------------------------------------------------------------

def test_circular_moon_distance_is_mean_distance():
    for t in (0.0, 1.0e5, 5.0e5, 1.0e6):
        r = np.linalg.norm(moon_circular_state_at(t).r)
        assert abs(r - MOON_ORBIT.a) < 1.0e3, (t, r)


def test_circular_moon_is_periodic():
    T = MOON_ORBIT.period_s
    a = moon_circular_state_at(12345.0).r
    b = moon_circular_state_at(12345.0 + T).r
    assert np.linalg.norm(a - b) < 1.0e3  # < 1 km after one period


def test_moon_orbit_elements_are_circular_at_earth_moon_rate():
    assert MOON_ORBIT.e == 0.0
    assert MOON_ORBIT.mu == MU_EARTH + MU_MOON


def test_circular_closed_form_matches_general_kepler_propagation():
    from orbitsim.core.elements import elements_to_state
    from orbitsim.core.propagate import propagate_kepler

    epoch_state = elements_to_state(MOON_ORBIT)
    for t_s in (0.0, 1.0e5, 1.0e6, 3.0e6):
        actual = moon_circular_state_at(t_s)
        expected = propagate_kepler(epoch_state, t_s - epoch_state.epoch_s)
        np.testing.assert_allclose(actual.r, expected.r, rtol=0.0, atol=1e-3)
        np.testing.assert_allclose(actual.v, expected.v, rtol=0.0, atol=1e-9)


def test_circular_fallback_is_fitted_to_the_real_moon_at_j2000():
    # The fallback must sit in the real Moon's plane and phase at J2000 in the
    # ICRF (equatorial) frame -- not on the far side of Earth (the old model was
    # ~730,000 km off). Residual is the real orbit's eccentricity: ~18,000 km.
    ephemeris = _kernel_or_skip()
    real = ephemeris.body_state("MOON", 0.0, center="EARTH")
    circ = moon_circular_state_at(0.0)
    assert np.linalg.norm(real.r - circ.r) < 25.0e6
    h_real = np.cross(real.r, real.v)
    h_circ = np.cross(circ.r, circ.v)
    cos_angle = np.dot(h_real, h_circ) / (np.linalg.norm(h_real) * np.linalg.norm(h_circ))
    assert np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))) < 0.5


# ---------------------------------------------------------------------------
# Real Moon (DE440, hourly cubic-Hermite interpolation).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_s", [0.0, 1234.5, 10 * 86400.0 + 1800.3, 3.0e8 + 2711.9])
def test_moon_matches_de440(t_s):
    # Interpolation error bound for a 1 h step is ~1 cm in position; 1 m / 1e-4
    # m/s leave margin without hiding a wrong formula (which is off by km).
    ephemeris = _kernel_or_skip()
    real = ephemeris.body_state("MOON", t_s, center="EARTH")
    got = moon_state_at(t_s)
    assert np.linalg.norm(got.r - real.r) < 1.0
    assert np.linalg.norm(got.v - real.v) < 1.0e-4
    assert got.epoch_s == t_s


def test_moon_is_continuous_across_sample_knots():
    _kernel_or_skip()
    knot = 37 * moon._SAMPLE_STEP_S
    before = moon_state_at(knot - 1e-6)
    after = moon_state_at(knot + 1e-6)
    assert np.linalg.norm(after.r - before.r) < 1e-2
    assert np.linalg.norm(after.v - before.v) < 1e-6


def test_moon_state_keeps_earth_moon_mu():
    assert moon_state_at(0.0).mu == MOON_ORBIT.mu


def test_moon_falls_back_to_circular_offline(monkeypatch):
    monkeypatch.setattr(moon, "_ephemeris_ready", lambda: False)
    for t_s in (0.0, 5.0e5):
        got = moon_state_at(t_s)
        expected = moon_circular_state_at(t_s)
        np.testing.assert_array_equal(got.r, expected.r)
        np.testing.assert_array_equal(got.v, expected.v)


def test_moon_beyond_kernel_range_falls_back_instead_of_raising():
    # de440s covers 1849-2150; long high-warp runs can leave it. ~2200 AD:
    _kernel_or_skip()
    t_s = 6.3e9
    got = moon_state_at(t_s)
    np.testing.assert_array_equal(got.r, moon_circular_state_at(t_s).r)


def test_moon_sample_cache_is_bounded(monkeypatch):
    _kernel_or_skip()
    monkeypatch.setattr(moon, "_SAMPLE_CACHE_MAX", 16)
    monkeypatch.setattr(moon, "_sample_cache", {})
    for k in range(40):
        moon_state_at(k * moon._SAMPLE_STEP_S + 1.0)
    assert len(moon._sample_cache) <= 16


def test_moon_knots_are_fetched_in_batches(monkeypatch):
    # A 1e8x frame spans ~463 sim-hours: one Skyfield call per hourly knot made a
    # deep-space frame cost ~100 ms. Knots now come in vectorized blocks.
    ephemeris = _kernel_or_skip()
    monkeypatch.setattr(moon, "_sample_cache", {})
    calls = []
    real_states = ephemeris.body_states
    monkeypatch.setattr(ephemeris, "body_state",
                        lambda *a, **k: calls.append(a) or pytest.fail("scalar call"))
    monkeypatch.setattr(ephemeris, "body_states",
                        lambda *a, **k: calls.append(a) or real_states(*a, **k))
    for hour in range(463):
        moon_state_at(1.0e8 + hour * moon._SAMPLE_STEP_S + 0.5)
    assert len(calls) <= 3


def test_batched_moon_knots_match_per_knot_samples():
    ephemeris = _kernel_or_skip()
    moon._sample_cache.clear()
    for t_s in (1.0e8 + 1234.5, 1.0e8 + 77 * 3600.0 + 1.0):
        got = moon_state_at(t_s)
        bucket = int(np.floor(t_s / moon._SAMPLE_STEP_S))
        from orbitsim.core.hermite import hermite_state
        start = ephemeris.body_state("MOON", bucket * 3600.0, center="EARTH")
        end = ephemeris.body_state("MOON", (bucket + 1) * 3600.0, center="EARTH")
        want = hermite_state(start, end, bucket * 3600.0, 3600.0, t_s, mu=MOON_ORBIT.mu)
        np.testing.assert_array_equal(got.r, want.r)
        np.testing.assert_array_equal(got.v, want.v)
