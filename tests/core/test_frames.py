"""Tests for rotation matrix helpers in frames module."""
import numpy as np
import pytest
from orbitsim.core.frames import rotation_matrix_3, rotation_matrix_1


def test_r3_identity_zero():
    """R3(0) = identity."""
    R = rotation_matrix_3(0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)


def test_r3_inverse():
    """R3(θ) @ R3(−θ) = I."""
    theta = np.pi / 4
    R = rotation_matrix_3(theta)
    R_inv = rotation_matrix_3(-theta)
    product = R @ R_inv
    np.testing.assert_allclose(product, np.eye(3), atol=1e-15)


def test_r1_identity_zero():
    """R1(0) = identity."""
    R = rotation_matrix_1(0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)


def test_r1_inverse():
    """R1(θ) @ R1(−θ) = I."""
    theta = np.pi / 4
    R = rotation_matrix_1(theta)
    R_inv = rotation_matrix_1(-theta)
    product = R @ R_inv
    np.testing.assert_allclose(product, np.eye(3), atol=1e-15)
