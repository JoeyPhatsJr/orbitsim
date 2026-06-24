"""Tests for Kepler equation solvers and anomaly conversions."""
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from orbitsim.core.kepler import (
    true_to_eccentric_anomaly,
    eccentric_to_mean_anomaly,
    solve_kepler_elliptic,
    mean_to_true_anomaly,
    true_to_hyperbolic_anomaly,
    hyperbolic_to_mean_anomaly,
    solve_kepler_hyperbolic,
    mean_to_true_anomaly_hyperbolic,
)


class TestEllipticAnomalies:
    """Anomaly conversions for elliptical orbits (e < 1)."""

    def test_circular_identity(self):
        """e=0: E == M == nu exactly."""
        for nu in [0.0, 0.5, 1.0, np.pi, 5.0]:
            E = true_to_eccentric_anomaly(nu, 0.0)
            np.testing.assert_allclose(E, nu, atol=1e-14)
            M = eccentric_to_mean_anomaly(E, 0.0)
            np.testing.assert_allclose(M, nu, atol=1e-14)

    def test_solve_kepler_known_answer(self):
        """solve_kepler_elliptic(M=1.0, e=0.5) -> E ~ 1.498701 (tol 1e-6).

        Verify residual E - e*sin(E) - M < 1e-12.
        """
        M = 1.0
        e = 0.5
        E = solve_kepler_elliptic(M, e)
        np.testing.assert_allclose(E, 1.498701, atol=1e-6)
        residual = abs(E - e * np.sin(E) - M)
        assert residual < 1e-12, f"residual {residual} exceeds 1e-12"

    @given(
        nu=st.floats(min_value=0.01, max_value=2 * np.pi - 0.01),
        e=st.sampled_from([0.0, 0.3, 0.7, 0.95]),
    )
    @settings(max_examples=50)
    def test_round_trip_anomalies(self, nu, e):
        """mean_to_true(eccentric_to_mean(true_to_eccentric(nu, e), e), e) == nu."""
        E = true_to_eccentric_anomaly(nu, e)
        M = eccentric_to_mean_anomaly(E, e)
        nu_recovered = mean_to_true_anomaly(M, e)
        diff = abs(nu - nu_recovered)
        diff = min(diff, 2 * np.pi - diff)
        assert diff < 1e-10, f"round-trip error {diff} for nu={nu}, e={e}"

    def test_solve_kepler_convergence(self):
        """Newton-Raphson converges for a range of M and e values."""
        for e in [0.0, 0.1, 0.5, 0.8, 0.95, 0.99]:
            for M in np.linspace(0, 2 * np.pi, 20):
                E = solve_kepler_elliptic(M, e)
                M_norm = M % (2 * np.pi)
                residual = abs(E - e * np.sin(E) - M_norm)
                assert residual < 1e-12, f"e={e}, M={M}: residual={residual}"


class TestHyperbolicAnomalies:
    """Anomaly conversions for hyperbolic orbits (e > 1)."""

    def test_round_trip_hyperbolic(self):
        """Hyperbolic anomaly round-trip for several e values."""
        for e in [1.2, 1.5, 2.0, 5.0]:
            for nu in [0.1, 0.5, 1.0]:
                F = true_to_hyperbolic_anomaly(nu, e)
                M = hyperbolic_to_mean_anomaly(F, e)
                nu_recovered = mean_to_true_anomaly_hyperbolic(M, e)
                diff = abs(nu - nu_recovered)
                assert diff < 1e-10, f"e={e}, nu={nu}: error={diff}"

    def test_solve_kepler_hyperbolic_residual(self):
        """Newton on hyperbolic Kepler equation: M = e*sinh(F) - F."""
        for e in [1.5, 2.0, 5.0]:
            for M in [0.5, 1.0, 5.0]:
                F = solve_kepler_hyperbolic(M, e)
                residual = abs(e * np.sinh(F) - F - M)
                assert residual < 1e-12, f"e={e}, M={M}: residual={residual}"
