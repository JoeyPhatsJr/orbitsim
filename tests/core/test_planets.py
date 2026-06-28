"""Tests for core.planets — circular orbit approximations for inner solar system."""
import numpy as np
import pytest
from orbitsim.core.planets import (
    sun_state_at, mercury_state_at, venus_state_at, mars_state_at,
    earth_position_helio, A_MERCURY, A_VENUS, A_EARTH, A_MARS,
    MERCURY_SOI_M, VENUS_SOI_M, EARTH_SOI_M, MARS_SOI_M,
    _N_MERCURY, _N_VENUS, _N_EARTH, _N_MARS,
)
from orbitsim.core.constants import MU_SUN


class TestPlanetPositions:
    """Planet states are circular orbits at the correct radius and speed."""

    def test_sun_at_t0_is_minus_earth(self):
        s = sun_state_at(0.0)
        np.testing.assert_allclose(np.linalg.norm(s.r), A_EARTH, rtol=1e-10)

    def test_mercury_orbital_radius(self):
        st = mercury_state_at(0.0)
        r_helio = st.r + earth_position_helio(0.0)
        np.testing.assert_allclose(np.linalg.norm(r_helio), A_MERCURY, rtol=1e-10)

    def test_venus_orbital_radius(self):
        st = venus_state_at(0.0)
        r_helio = st.r + earth_position_helio(0.0)
        np.testing.assert_allclose(np.linalg.norm(r_helio), A_VENUS, rtol=1e-10)

    def test_mars_orbital_radius(self):
        st = mars_state_at(0.0)
        r_helio = st.r + earth_position_helio(0.0)
        np.testing.assert_allclose(np.linalg.norm(r_helio), A_MARS, rtol=1e-10)

    def test_circular_speed_mercury(self):
        st = mercury_state_at(0.0)
        r_helio = st.r + earth_position_helio(0.0)
        v_helio = st.v + A_EARTH * _N_EARTH * np.array([0.0, 1.0, 0.0])
        expected = np.sqrt(MU_SUN / A_MERCURY)
        np.testing.assert_allclose(np.linalg.norm(v_helio), expected, rtol=1e-10)

    def test_period_earth(self):
        """Earth completes one orbit in ~365.25 days."""
        period = 2 * np.pi / _N_EARTH
        np.testing.assert_allclose(period / 86400.0, 365.25, rtol=0.01)

    def test_sun_geocentric_moves(self):
        """The Sun's geocentric position changes with time (Earth orbits it)."""
        s0 = sun_state_at(0.0)
        s1 = sun_state_at(86400.0 * 90)  # 90 days later
        angle = np.arccos(np.clip(np.dot(s0.r, s1.r) / (np.linalg.norm(s0.r) * np.linalg.norm(s1.r)), -1, 1))
        assert angle > 0.1  # should be ~pi/2 (90 deg)


class TestSOIRadii:
    """SOI radii are physically reasonable."""

    def test_earth_soi(self):
        np.testing.assert_allclose(EARTH_SOI_M, 9.25e8, rtol=0.02)

    def test_mercury_soi_smaller_than_earth(self):
        assert MERCURY_SOI_M < EARTH_SOI_M

    def test_mars_soi_smaller_than_earth(self):
        assert MARS_SOI_M < EARTH_SOI_M

    def test_venus_soi_order(self):
        assert MERCURY_SOI_M < VENUS_SOI_M


class TestSolarSystemAccel:
    """The solar system accelerator conserves energy to acceptable tolerance."""

    def test_solar_system_accel_matches_earth_moon_near_earth(self):
        """Near Earth, the solar system accel should be close to earth_moon_accel
        (the Sun/planet perturbations are small compared to Earth's gravity in LEO)."""
        from orbitsim.core.nbody import earth_moon_accel, solar_system_accel
        from orbitsim.core.constants import R_EARTH
        r = np.array([R_EARTH + 500e3, 0.0, 0.0])
        a_em = earth_moon_accel(r, 0.0)
        a_ss = solar_system_accel(r, 0.0)
        # Dominant Earth term is ~8 m/s^2; solar perturbations are ~6e-3 m/s^2.
        np.testing.assert_allclose(a_ss, a_em, atol=0.01)

    def test_energy_conservation_leo_coast(self):
        """Specific energy is approximately conserved over a short coast under
        the solar system model (to the same accuracy as the Earth-Moon model)."""
        from orbitsim.core.nbody import propagate_solar_system
        from orbitsim.core.constants import R_EARTH, MU_EARTH
        r0 = R_EARTH + 400e3
        v0 = np.sqrt(MU_EARTH / r0)
        state = __import__("orbitsim.core.state", fromlist=["StateVector"]).StateVector(
            r=np.array([r0, 0.0, 0.0]),
            v=np.array([0.0, v0, 0.0]),
            mu=MU_EARTH,
        )
        eps0 = 0.5 * v0**2 - MU_EARTH / r0
        # Propagate 1/10 orbit (~550 s).
        period = 2 * np.pi * np.sqrt(r0**3 / MU_EARTH)
        dt = period / 10
        n_steps = 100
        for _ in range(n_steps):
            state = propagate_solar_system(state, dt / n_steps)
        r_mag = np.linalg.norm(state.r)
        v_mag = np.linalg.norm(state.v)
        eps1 = 0.5 * v_mag**2 - MU_EARTH / r_mag
        # Solar perturbation introduces a slow drift, but over 1/10 orbit it's tiny.
        assert abs(eps1 - eps0) / abs(eps0) < 1e-4


class TestDominantBody:
    """dominant_body_solar returns the correct body for various positions."""

    def test_leo_is_earth(self):
        from orbitsim.core.nbody import dominant_body_solar
        from orbitsim.core.constants import R_EARTH
        r = np.array([R_EARTH + 400e3, 0.0, 0.0])
        body, _ = dominant_body_solar(r, 0.0)
        assert body.name == "Earth"

    def test_near_moon_is_moon(self):
        from orbitsim.core.nbody import dominant_body_solar
        from orbitsim.core.moon import moon_state_at
        rM = moon_state_at(0.0).r
        r = rM + np.array([1e6, 0.0, 0.0])  # 1000 km from Moon center
        body, _ = dominant_body_solar(r, 0.0)
        assert body.name == "Moon"

    def test_interplanetary_is_sun(self):
        from orbitsim.core.nbody import dominant_body_solar
        r = np.array([2e11, 0.0, 0.0])  # way beyond Earth SOI
        body, _ = dominant_body_solar(r, 0.0)
        assert body.name == "Sun"
