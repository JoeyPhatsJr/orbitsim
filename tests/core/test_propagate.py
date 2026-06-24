"""Tests for two-body propagation (analytic + numeric)."""
import numpy as np
import pytest
from orbitsim.core.propagate import propagate_kepler, propagate_numeric
from orbitsim.core.state import StateVector
from orbitsim.core.elements import state_to_elements, elements_to_state, KeplerianElements
from orbitsim.core.constants import MU_EARTH, R_EARTH, J2_EARTH


def _circular_leo_state(r_m: float = 7.0e6) -> StateVector:
    """Circular LEO state for testing."""
    v = np.sqrt(MU_EARTH / r_m)
    return StateVector(
        r=np.array([r_m, 0.0, 0.0]),
        v=np.array([0.0, v, 0.0]),
        mu=MU_EARTH,
    )


def _eccentric_state() -> StateVector:
    """Eccentric inclined orbit for testing."""
    elements = KeplerianElements(
        a=1.0e7, e=0.3, i=0.5, raan=1.0, argp=2.0, nu=0.5, mu=MU_EARTH
    )
    return elements_to_state(elements)


class TestPropagateKepler:
    """Tests for analytic two-body propagation."""

    def test_period_closure(self):
        """Propagate by one period -> return to start within 1 mm."""
        state0 = _circular_leo_state()
        period = state_to_elements(state0).period_s
        state1 = propagate_kepler(state0, period)

        pos_error = np.linalg.norm(state0.r - state1.r)
        assert pos_error < 1e-3, f"position error {pos_error} m exceeds 1 mm"

    def test_half_period_apoapsis(self):
        """Half period -> opposite side (apoapsis for circular)."""
        state0 = _circular_leo_state()
        period = state_to_elements(state0).period_s
        state_half = propagate_kepler(state0, period / 2.0)

        np.testing.assert_allclose(state_half.r[0], -state0.r[0], rtol=1e-6)

    def test_energy_conservation(self):
        """Specific energy constant to 1e-9 relative over 10 random dt."""
        state0 = _eccentric_state()
        e0 = state0.specific_energy
        rng = np.random.default_rng(42)
        period = state_to_elements(state0).period_s
        for dt in rng.uniform(0, period, size=10):
            state_t = propagate_kepler(state0, dt)
            e_t = state_t.specific_energy
            rel_error = abs(e_t - e0) / abs(e0)
            assert rel_error < 1e-9, f"energy drift {rel_error} at dt={dt}"

    def test_angular_momentum_conservation(self):
        """Angular momentum vector constant to 1e-9 relative over 10 random dt."""
        state0 = _eccentric_state()
        h0 = state0.angular_momentum
        h0_mag = np.linalg.norm(h0)
        rng = np.random.default_rng(42)
        period = state_to_elements(state0).period_s
        for dt in rng.uniform(0, period, size=10):
            state_t = propagate_kepler(state0, dt)
            h_t = state_t.angular_momentum
            rel_error = np.linalg.norm(h_t - h0) / h0_mag
            assert rel_error < 1e-9, f"h drift {rel_error} at dt={dt}"

    def test_vis_viva(self):
        """v^2 = mu*(2/r - 1/a) at every propagated point."""
        state0 = _eccentric_state()
        a = state_to_elements(state0).a
        rng = np.random.default_rng(42)
        period = state_to_elements(state0).period_s
        for dt in rng.uniform(0, period, size=10):
            state_t = propagate_kepler(state0, dt)
            v_sq = np.dot(state_t.v, state_t.v)
            v_sq_expected = MU_EARTH * (2.0 / state_t.r_mag - 1.0 / a)
            np.testing.assert_allclose(v_sq, v_sq_expected, rtol=1e-9)

    def test_epoch_advances(self):
        """Epoch should advance by dt."""
        state0 = _circular_leo_state()
        dt = 1000.0
        state1 = propagate_kepler(state0, dt)
        np.testing.assert_allclose(state1.epoch_s, state0.epoch_s + dt)


class TestPropagateNumeric:
    """Tests for numeric (ODE) two-body propagation."""

    def test_period_closure_numeric(self):
        """Propagate by one period -> return to start within 1 m."""
        state0 = _circular_leo_state()
        period = state_to_elements(state0).period_s
        state1 = propagate_numeric(state0, period)

        pos_error = np.linalg.norm(state0.r - state1.r)
        assert pos_error < 1.0, f"position error {pos_error} m exceeds 1 m"

    def test_energy_conservation_numeric(self):
        """Specific energy constant to 1e-7 relative over 10 random dt."""
        state0 = _eccentric_state()
        e0 = state0.specific_energy
        rng = np.random.default_rng(42)
        period = state_to_elements(state0).period_s
        for dt in rng.uniform(100, period / 2, size=5):
            state_t = propagate_numeric(state0, dt)
            e_t = state_t.specific_energy
            rel_error = abs(e_t - e0) / abs(e0)
            assert rel_error < 1e-7, f"energy drift {rel_error} at dt={dt}"

    def test_angular_momentum_conservation_numeric(self):
        """Angular momentum constant to 1e-7 relative (numeric)."""
        state0 = _eccentric_state()
        h0 = state0.angular_momentum
        h0_mag = np.linalg.norm(h0)
        rng = np.random.default_rng(42)
        period = state_to_elements(state0).period_s
        for dt in rng.uniform(100, period / 2, size=5):
            state_t = propagate_numeric(state0, dt)
            h_t = state_t.angular_momentum
            rel_error = np.linalg.norm(h_t - h0) / h0_mag
            assert rel_error < 1e-7, f"h drift {rel_error} at dt={dt}"


class TestAnalyticVsNumeric:
    """Agreement between analytic and numeric propagation."""

    def test_agreement_one_period(self):
        """With j2=False, analytic and numeric agree to < 1 m over one period."""
        state0 = _eccentric_state()
        period = state_to_elements(state0).period_s
        rng = np.random.default_rng(42)
        for dt in rng.uniform(100, period, size=5):
            s_analytic = propagate_kepler(state0, dt)
            s_numeric = propagate_numeric(state0, dt)
            pos_diff = np.linalg.norm(s_analytic.r - s_numeric.r)
            assert pos_diff < 1.0, f"analytic/numeric position differ by {pos_diff} m at dt={dt}"


class TestJ2Perturbation:
    """J2 perturbation sanity checks."""

    def test_j2_raan_regression(self):
        """With J2 on an inclined LEO, RAAN should regress over many orbits.

        Expected rate: dOmega/dt ~ -1.5 * n * J2 * (R/p)^2 * cos(i)
        For i < 90 deg, dOmega/dt < 0 (regression).
        """
        inc = np.deg2rad(45.0)
        a = R_EARTH + 400e3
        elements0 = KeplerianElements(
            a=a, e=0.01, i=inc, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        state0 = elements_to_state(elements0)

        period = elements0.period_s
        dt = 10 * period

        state_j2 = propagate_numeric(state0, dt, j2=True)
        elem_j2 = state_to_elements(state_j2)

        n = np.sqrt(MU_EARTH / a**3)
        p = a * (1 - 0.01**2)
        expected_rate = -1.5 * n * J2_EARTH * (R_EARTH / p) ** 2 * np.cos(inc)
        expected_raan_change = expected_rate * dt

        actual_raan = elem_j2.raan
        if actual_raan > np.pi:
            actual_raan -= 2 * np.pi

        assert actual_raan < 0, "RAAN should regress (decrease) for i < 90 deg"
        np.testing.assert_allclose(
            actual_raan, expected_raan_change, rtol=0.15,
        )
