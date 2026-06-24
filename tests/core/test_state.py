"""Tests for StateVector (position & velocity in inertial frame)."""
import numpy as np
import pytest
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH


def test_circular_leo_energy():
    """For circular LEO, specific energy ε = v²/2 − μ/r ≈ −μ/(2a)."""
    r_mag = 7.0e6  # meters
    v_circ = np.sqrt(MU_EARTH / r_mag)

    state = StateVector(
        r=np.array([r_mag, 0.0, 0.0]),
        v=np.array([0.0, v_circ, 0.0]),
        mu=MU_EARTH,
    )

    # For circular orbit: ε = −μ/(2r) = −μ/(2a)
    expected_energy = -MU_EARTH / (2.0 * r_mag)
    abs_error = abs(state.specific_energy - expected_energy) / abs(expected_energy)
    assert abs_error < 1e-6, (
        f"specific_energy {state.specific_energy} not within 1e-6 relative "
        f"of {expected_energy}"
    )


def test_circular_leo_angular_momentum():
    """For circular LEO, |h| = r × v = r_mag * v_mag."""
    r_mag = 7.0e6  # meters
    v_circ = np.sqrt(MU_EARTH / r_mag)

    state = StateVector(
        r=np.array([r_mag, 0.0, 0.0]),
        v=np.array([0.0, v_circ, 0.0]),
        mu=MU_EARTH,
    )

    h_mag = np.linalg.norm(state.angular_momentum)
    expected = r_mag * v_circ
    abs_error = abs(h_mag - expected) / expected
    assert abs_error < 1e-6, (
        f"|h| {h_mag} not within 1e-6 relative of {expected}"
    )


def test_state_immutable():
    """StateVector arrays should be immutable."""
    state = StateVector(
        r=np.array([7.0e6, 0.0, 0.0]),
        v=np.array([0.0, 7545.0, 0.0]),
        mu=MU_EARTH,
    )
    with pytest.raises(ValueError):
        state.r[0] = 1.0


def test_state_validation_shape():
    """StateVector should validate that r and v are shape (3,)."""
    with pytest.raises(ValueError):
        StateVector(
            r=np.array([7.0e6, 0.0]),  # wrong shape
            v=np.array([0.0, 7545.0, 0.0]),
            mu=MU_EARTH,
        )


def test_state_validation_dtype():
    """StateVector should convert arrays to float64."""
    state = StateVector(
        r=np.array([7.0e6, 0.0, 0.0], dtype=np.float32),
        v=np.array([0.0, 7545.0, 0.0], dtype=np.float32),
        mu=MU_EARTH,
    )
    assert state.r.dtype == np.float64
    assert state.v.dtype == np.float64
