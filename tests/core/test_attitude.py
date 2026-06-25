"""Tests for quaternion attitude helpers."""
import numpy as np
from orbitsim.core.attitude import (
    quat_identity, quat_normalize, quat_from_axis_angle, quat_multiply,
    quat_rotate_vector, angle_between, nose_direction,
)


def test_identity_rotates_nothing():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(quat_rotate_vector(quat_identity(), v), v)


def test_nose_of_identity_is_plus_z():
    assert np.allclose(nose_direction(quat_identity()), [0.0, 0.0, 1.0])


def test_90deg_about_x_maps_z_to_minus_y():
    q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    out = quat_rotate_vector(q, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(out, [0.0, -1.0, 0.0], atol=1e-9)


def test_90deg_about_z_maps_x_to_y():
    q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    out = quat_rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_multiply_composes_rotations():
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    # Applying qx twice == 180 deg about x: z -> -z.
    q2 = quat_multiply(qx, qx)
    assert np.allclose(quat_rotate_vector(q2, np.array([0.0, 0.0, 1.0])),
                       [0.0, 0.0, -1.0], atol=1e-9)


def test_rotation_preserves_length():
    q = quat_from_axis_angle(np.array([1.0, 1.0, 1.0]), 1.234)
    v = np.array([3.0, -2.0, 0.5])
    assert abs(np.linalg.norm(quat_rotate_vector(q, v)) - np.linalg.norm(v)
               ) < 1e-12


def test_angle_between_orthogonal_and_clamped():
    assert (abs(angle_between(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
            - np.pi / 2) < 1e-12)
    # Identical directions -> 0 even with float error (arccos clamp).
    u = np.array([1.0, 1.0, 1.0])
    assert angle_between(u, u) < 1e-7
