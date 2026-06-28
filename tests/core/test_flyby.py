"""Tests for core.flyby — gravity-assist deflection geometry."""
import numpy as np
import pytest
from orbitsim.core.flyby import (
    v_infinity, flyby_periapsis, flyby_deflection, rotate_v_infinity,
    flyby_dv_equivalent, max_deflection, flyby_exit_velocity,
    encounter_parameters,
)
from orbitsim.core.constants import MU_JUPITER, R_JUPITER, MU_MARS, R_MARS


class TestVInfinity:
    def test_zero_when_comoving(self):
        v = np.array([1e4, 0, 0])
        np.testing.assert_allclose(v_infinity(v, v), 0.0, atol=1e-10)

    def test_subtraction(self):
        v_ship = np.array([3e4, 0, 0])
        v_planet = np.array([1.3e4, 0, 0])
        vi = v_infinity(v_ship, v_planet)
        np.testing.assert_allclose(vi, [1.7e4, 0, 0], atol=1e-6)


class TestDeflection:
    def test_round_trip_deflection_periapsis(self):
        """flyby_periapsis(v, mu, delta) and flyby_deflection(v, mu, r_p) are inverses."""
        v_inf = 10e3
        delta = 0.5
        r_p = flyby_periapsis(v_inf, MU_JUPITER, delta)
        delta_back = flyby_deflection(v_inf, MU_JUPITER, r_p)
        np.testing.assert_allclose(delta_back, delta, rtol=1e-10)

    def test_larger_periapsis_less_deflection(self):
        v_inf = 8e3
        d1 = flyby_deflection(v_inf, MU_JUPITER, 2 * R_JUPITER)
        d2 = flyby_deflection(v_inf, MU_JUPITER, 5 * R_JUPITER)
        assert d1 > d2

    def test_higher_vinf_less_deflection(self):
        d1 = flyby_deflection(5e3, MU_JUPITER, 2 * R_JUPITER)
        d2 = flyby_deflection(20e3, MU_JUPITER, 2 * R_JUPITER)
        assert d1 > d2

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            flyby_periapsis(10e3, MU_JUPITER, 0.0)
        with pytest.raises(ValueError):
            flyby_periapsis(10e3, MU_JUPITER, np.pi)

    def test_invalid_periapsis_raises(self):
        with pytest.raises(ValueError):
            flyby_deflection(10e3, MU_JUPITER, 0.0)

    def test_jupiter_flyby_realistic(self):
        """A Jupiter flyby at 2 R_J with v_inf ~10 km/s gives a substantial deflection."""
        v_inf = 10e3
        r_p = 2 * R_JUPITER
        delta = flyby_deflection(v_inf, MU_JUPITER, r_p)
        assert 0.1 < delta < np.pi
        dv = flyby_dv_equivalent(v_inf, delta)
        assert dv > 1000.0


class TestRotation:
    def test_magnitude_preserved(self):
        v_inf = np.array([10e3, 5e3, 0])
        v_out = rotate_v_infinity(v_inf, 0.5)
        np.testing.assert_allclose(np.linalg.norm(v_out), np.linalg.norm(v_inf), rtol=1e-10)

    def test_zero_deflection_no_change(self):
        v_inf = np.array([10e3, 0, 0])
        v_out = rotate_v_infinity(v_inf, 0.0)
        np.testing.assert_allclose(v_out, v_inf, atol=1e-6)


class TestFlybyExitVelocity:
    def test_vinf_magnitude_conserved(self):
        """|v_inf| is the same before and after the flyby (energy conservation)."""
        v_planet = np.array([1.3e4, 0, 0])
        v_ship_in = np.array([2.3e4, 2e3, 0])
        v_inf_in = v_ship_in - v_planet
        v_out = flyby_exit_velocity(v_ship_in, v_planet, MU_JUPITER, 2 * R_JUPITER)
        v_inf_out = v_out - v_planet
        np.testing.assert_allclose(
            np.linalg.norm(v_inf_out), np.linalg.norm(v_inf_in), rtol=1e-10)

    def test_heliocentric_speed_changes(self):
        """Heliocentric speed changes after flyby (that's the point of a gravity assist)."""
        v_planet = np.array([1.3e4, 0, 0])
        v_ship_in = np.array([2.3e4, 2e3, 0])
        v_out = flyby_exit_velocity(v_ship_in, v_planet, MU_JUPITER, 2 * R_JUPITER)
        assert abs(np.linalg.norm(v_out) - np.linalg.norm(v_ship_in)) > 100.0


class TestEncounterParameters:
    def test_returns_all_keys(self):
        r_ship = np.array([1e9, 0, 0])
        v_ship = np.array([0, 1e4, 0])
        r_planet = np.zeros(3)
        v_planet = np.array([0, 1.3e4, 0])
        params = encounter_parameters(r_ship, v_ship, r_planet, v_planet, MU_JUPITER)
        assert "v_inf_mag" in params
        assert "periapsis_m" in params
        assert "deflection_rad" in params
        assert "dv_equivalent" in params
        assert "e" in params

    def test_eccentricity_greater_than_one(self):
        r_ship = np.array([5e8, 0, 0])
        v_ship = np.array([0, 2e4, 0])
        r_planet = np.zeros(3)
        v_planet = np.array([0, 1.3e4, 0])
        params = encounter_parameters(r_ship, v_ship, r_planet, v_planet, MU_JUPITER)
        assert params["e"] > 1.0


class TestMaxDeflection:
    def test_surface_gives_maximum(self):
        v_inf = 10e3
        d_max = max_deflection(v_inf, MU_JUPITER, R_JUPITER)
        d_far = max_deflection(v_inf, MU_JUPITER, 5 * R_JUPITER)
        assert d_max > d_far

    def test_mars_flyby_modest(self):
        """Mars has much less mass — deflection at 2 R_Mars with 5 km/s v_inf is modest."""
        delta = max_deflection(5e3, MU_MARS, 2 * R_MARS)
        delta_jup = max_deflection(5e3, MU_JUPITER, 2 * R_JUPITER)
        assert delta < delta_jup
