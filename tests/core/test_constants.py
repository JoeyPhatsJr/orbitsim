"""Tests for physical constants from astropy.constants."""
import pytest
from orbitsim.core import constants


def test_mu_earth():
    """MU_EARTH should be ≈ 3.986004418e14 m³/s² within 0.1% (relative)."""
    expected = 3.986004418e14
    relative_tol = 0.001  # 0.1%
    abs_error = abs(constants.MU_EARTH - expected) / expected
    assert abs_error < relative_tol, (
        f"MU_EARTH {constants.MU_EARTH} not within {relative_tol*100}% of {expected} "
        f"(error: {abs_error*100:.4f}%)"
    )


def test_r_earth():
    """R_EARTH should be ≈ 6.378137e6 m within 0.01% (relative)."""
    expected = 6.378137e6
    relative_tol = 0.0001  # 0.01%
    abs_error = abs(constants.R_EARTH - expected) / expected
    assert abs_error < relative_tol, (
        f"R_EARTH {constants.R_EARTH} not within {relative_tol*100}% of {expected} "
        f"(error: {abs_error*100:.4f}%)"
    )
