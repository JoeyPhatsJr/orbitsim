"""Tests for KeplerianElements and state ↔ elements conversions."""
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from orbitsim.core.elements import (
    KeplerianElements,
    state_to_elements,
    elements_to_state,
)
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH, R_EARTH


class TestKeplerianElements:
    """Tests for KeplerianElements dataclass."""

    def test_period_elliptical(self):
        """Period T = 2π√(a³/μ) for elliptical orbit (a > 0)."""
        a = 7.0e6  # meters
        elements = KeplerianElements(
            a=a, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        expected_period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        np.testing.assert_allclose(elements.period_s, expected_period, rtol=1e-12)

    def test_period_parabolic_raises(self):
        """Period should raise ValueError for parabolic orbit (a = 0)."""
        elements = KeplerianElements(
            a=0.0, e=1.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        with pytest.raises(ValueError):
            _ = elements.period_s

    def test_period_hyperbolic_raises(self):
        """Period should raise ValueError for hyperbolic orbit (a < 0)."""
        elements = KeplerianElements(
            a=-1.0e6, e=1.5, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        with pytest.raises(ValueError):
            _ = elements.period_s

    def test_semi_latus_rectum(self):
        """Semi-latus rectum p = a(1 − e²)."""
        a = 8.0e6
        e = 0.1
        elements = KeplerianElements(
            a=a, e=e, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        expected = a * (1.0 - e**2)
        np.testing.assert_allclose(
            elements.semi_latus_rectum, expected, rtol=1e-12
        )

    def test_periapsis_radius(self):
        """Periapsis radius r_p = a(1 − e)."""
        a = 8.0e6
        e = 0.1
        elements = KeplerianElements(
            a=a, e=e, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        np.testing.assert_allclose(elements.periapsis_radius, a * (1.0 - e), rtol=1e-12)

    def test_apoapsis_radius(self):
        """Apoapsis radius r_a = a(1 + e)."""
        a = 8.0e6
        e = 0.1
        elements = KeplerianElements(
            a=a, e=e, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        np.testing.assert_allclose(elements.apoapsis_radius, a * (1.0 + e), rtol=1e-12)

    def test_periapsis_equals_apoapsis_circular(self):
        """For a circular orbit (e=0), periapsis = apoapsis = a."""
        a = 7.0e6
        elements = KeplerianElements(
            a=a, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        np.testing.assert_allclose(elements.periapsis_radius, a, rtol=1e-12)
        np.testing.assert_allclose(elements.apoapsis_radius, a, rtol=1e-12)

    def test_frozen(self):
        """KeplerianElements is frozen (immutable)."""
        elements = KeplerianElements(
            a=7.0e6, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH
        )
        with pytest.raises(AttributeError):
            elements.a = 8.0e6


class TestStateToElements:
    """Tests for state_to_elements conversion (Curtis Algorithm 4.1)."""

    def test_curtis_example_4_3(self):
        """Curtis Example 4.3 (elliptical).

        Input (km -> m): r = [-6045, -3490, 2500] km, v = [-3.457, 6.618, 2.533] km/s
        Expected:
            a ≈ 8788 km (8.788e6 m)
            e ≈ 0.17121
            i ≈ 153.249° (2.6748 rad)
            Ω ≈ 255.279° (4.4552 rad)
            ω ≈ 20.068° (0.3503 rad)
            ν ≈ 28.446° (0.4965 rad)
        Tolerance: 0.5% on a and e; 0.2° on angles.
        """
        r_km = np.array([-6045.0, -3490.0, 2500.0])
        v_kms = np.array([-3.457, 6.618, 2.533])
        r_m = r_km * 1000.0
        v_ms = v_kms * 1000.0

        state = StateVector(r=r_m, v=v_ms, mu=MU_EARTH)
        elem = state_to_elements(state)

        # Expected values
        expected_a = 8.788e6
        expected_e = 0.17121
        expected_i = 2.6748
        expected_raan = 4.4552
        expected_argp = 0.3503
        expected_nu = 0.4965

        # Tolerance
        a_tol = expected_a * 0.005  # 0.5%
        e_tol = expected_e * 0.005  # 0.5%
        angle_tol = np.deg2rad(0.2)  # 0.2°

        assert abs(elem.a - expected_a) < a_tol, (
            f"a {elem.a} not within 0.5% of {expected_a}"
        )
        assert abs(elem.e - expected_e) < e_tol, (
            f"e {elem.e} not within 0.5% of {expected_e}"
        )
        assert abs(elem.i - expected_i) < angle_tol, (
            f"i {elem.i} not within 0.2° of {expected_i}"
        )
        assert abs(elem.raan - expected_raan) < angle_tol, (
            f"Ω {elem.raan} not within 0.2° of {expected_raan}"
        )
        assert abs(elem.argp - expected_argp) < angle_tol, (
            f"ω {elem.argp} not within 0.2° of {expected_argp}"
        )
        assert abs(elem.nu - expected_nu) < angle_tol, (
            f"ν {elem.nu} not within 0.2° of {expected_nu}"
        )


class TestElementsToState:
    """Tests for elements_to_state conversion (Curtis Algorithm 4.2)."""

    def test_curtis_example_4_7_hyperbolic(self):
        """Curtis Example 4.7 (hyperbolic).

        Given h = 80000 km²/s, e = 1.4, i = 30°, Ω = 40°, ω = 60°, ν = 30°
        Expected (km -> m, km/s -> m/s):
            r ≈ [-4040, 4815, 3629] km
            v ≈ [-10.39, -4.772, 1.744] km/s
        Tolerance: 2 km on position, 0.02 km/s on velocity.
        """
        # Compute a from h and e
        h = 80000e6  # convert km²/s to m²/s: 80000 * (1e3)² = 80000e6
        p = h**2 / MU_EARTH
        a = p / (1.0 - 1.4**2)  # negative for hyperbola

        elements = KeplerianElements(
            a=a,
            e=1.4,
            i=np.deg2rad(30.0),
            raan=np.deg2rad(40.0),
            argp=np.deg2rad(60.0),
            nu=np.deg2rad(30.0),
            mu=MU_EARTH,
        )

        state = elements_to_state(elements)

        expected_r_km = np.array([-4040.0, 4815.0, 3629.0])
        expected_v_kms = np.array([-10.39, -4.772, 1.744])
        expected_r_m = expected_r_km * 1000.0
        expected_v_ms = expected_v_kms * 1000.0

        r_tol = 2000.0  # 2 km
        v_tol = 0.02 * 1000.0  # 0.02 km/s

        assert np.allclose(state.r, expected_r_m, atol=r_tol), (
            f"r {state.r / 1000} not within 2 km of {expected_r_km}"
        )
        assert np.allclose(state.v, expected_v_ms, atol=v_tol), (
            f"v {state.v / 1000} not within 0.02 km/s of {expected_v_kms}"
        )


class TestRoundTrip:
    """Round-trip property tests: state ↔ elements ↔ state."""

    @given(
        a=st.floats(min_value=1.0e7, max_value=1.0e8),
        e=st.floats(min_value=0.0, max_value=0.99),
        i=st.floats(min_value=0.0, max_value=np.pi),
        raan=st.floats(min_value=0.0, max_value=2 * np.pi),
        argp=st.floats(min_value=0.0, max_value=2 * np.pi),
        nu=st.floats(min_value=0.0, max_value=2 * np.pi),
    )
    @settings(max_examples=50)
    def test_state_to_elements_to_state(
        self, a, e, i, raan, argp, nu
    ):
        """state → elements → state returns original within 1e-7 relative."""
        elements0 = KeplerianElements(
            a=a, e=e, i=i, raan=raan, argp=argp, nu=nu, mu=MU_EARTH
        )
        state0 = elements_to_state(elements0)
        elements1 = state_to_elements(state0)
        state1 = elements_to_state(elements1)

        # Position and velocity should match to within 1e-7 relative
        r_error = np.linalg.norm(state0.r - state1.r) / np.linalg.norm(state0.r)
        v_error = np.linalg.norm(state0.v - state1.v) / np.linalg.norm(state0.v)

        assert r_error < 1e-7, f"r error {r_error} exceeds 1e-7"
        assert v_error < 1e-7, f"v error {v_error} exceeds 1e-7"

    @given(
        a=st.floats(min_value=1.0e7, max_value=1.0e8),
        e=st.floats(min_value=0.01, max_value=0.99),
        i=st.floats(min_value=0.05, max_value=np.pi - 0.05),
        raan=st.floats(min_value=0.0, max_value=2 * np.pi),
        argp=st.floats(min_value=0.0, max_value=2 * np.pi),
        nu=st.floats(min_value=0.0, max_value=2 * np.pi),
    )
    @settings(max_examples=50)
    def test_elements_to_state_to_elements(
        self, a, e, i, raan, argp, nu
    ):
        """elements -> state -> elements returns original within 1e-7 relative.

        Excludes near-circular (e<0.01) and near-equatorial (i<0.05 rad) orbits
        where angles are degenerate. Those cases are covered by the state round-trip
        test which checks position/velocity directly.
        """
        elements0 = KeplerianElements(
            a=a, e=e, i=i, raan=raan, argp=argp, nu=nu, mu=MU_EARTH
        )
        state = elements_to_state(elements0)
        elements1 = state_to_elements(state)

        a_error = abs(elements0.a - elements1.a) / elements0.a
        e_error = abs(elements0.e - elements1.e) / max(elements0.e, 1e-10)

        assert a_error < 1e-7, f"a error {a_error} exceeds 1e-7"
        assert e_error < 1e-7, f"e error {e_error} exceeds 1e-7"

        def angle_error(a0, a1):
            diff = abs(a0 - a1)
            return min(diff, 2 * np.pi - diff)

        i_error = angle_error(elements0.i, elements1.i)
        raan_error = angle_error(elements0.raan, elements1.raan)
        argp_error = angle_error(elements0.argp, elements1.argp)
        nu_error = angle_error(elements0.nu, elements1.nu)

        assert i_error < 1e-7, f"i error {i_error} exceeds 1e-7"
        assert raan_error < 1e-7, f"raan error {raan_error} exceeds 1e-7"
        assert argp_error < 1e-7, f"argp error {argp_error} exceeds 1e-7"
        assert nu_error < 1e-7, f"nu error {nu_error} exceeds 1e-7"
