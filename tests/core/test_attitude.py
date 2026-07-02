"""Tests for quaternion attitude helpers."""
import math

import numpy as np
import pytest
from orbitsim.core.attitude import (
    quat_identity, quat_normalize, quat_from_axis_angle, quat_multiply,
    quat_rotate_vector, angle_between, nose_direction, slew_toward,
    sas_target_dir, heading_pitch,
)
from orbitsim.core.state import StateVector
from orbitsim.core.constants import MU_EARTH


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


def test_slew_reaches_target_in_expected_time():
    q = quat_identity()                      # nose +Z
    target = np.array([1.0, 0.0, 0.0])       # +X, 90 deg away
    rate = 0.5                               # rad/s
    dt = 0.1
    # pi/2 / 0.5 = ~3.14 s -> ~32 steps; run 40 to be safe.
    for _ in range(40):
        q = slew_toward(q, target, rate, dt)
    assert angle_between(nose_direction(q), target) < 1e-6


def test_slew_single_step_never_overshoots():
    q = quat_identity()
    target = np.array([1.0, 0.0, 0.0])
    before = angle_between(nose_direction(q), target)
    q2 = slew_toward(q, target, max_rate_radps=100.0, dt_s=1.0)  # huge step
    after = angle_between(nose_direction(q2), target)
    assert after <= before + 1e-9            # clamped to the target, no overshoot
    assert after < 1e-6                       # lands exactly on target


def test_slew_handles_antiparallel_target():
    q = quat_identity()                      # nose +Z
    target = np.array([0.0, 0.0, -1.0])      # exactly opposite
    for _ in range(400):
        q = slew_toward(q, target, max_rate_radps=0.5, dt_s=0.1)
    assert angle_between(nose_direction(q), target) < 1e-3


def test_slew_rate_limited_per_step():
    q = quat_identity()
    target = np.array([1.0, 0.0, 0.0])       # 90 deg away
    q2 = slew_toward(q, target, max_rate_radps=0.1, dt_s=1.0)  # only 0.1 rad
    moved = angle_between(nose_direction(q), nose_direction(q2))
    assert abs(moved - 0.1) < 1e-6


def _leo_state() -> StateVector:
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(MU_EARTH / 7.0e6), 0.0])  # +y prograde, h along +z
    return StateVector(r=r, v=v, mu=MU_EARTH)


def test_prograde_is_velocity_direction():
    s = _leo_state()
    assert np.allclose(sas_target_dir("PROGRADE", s), [0.0, 1.0, 0.0],
                       atol=1e-12)
    assert np.allclose(sas_target_dir("RETROGRADE", s), [0.0, -1.0, 0.0],
                       atol=1e-12)


def test_normal_is_angular_momentum_direction():
    s = _leo_state()
    assert np.allclose(sas_target_dir("NORMAL", s), [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(sas_target_dir("ANTINORMAL", s), [0.0, 0.0, -1.0],
                       atol=1e-12)


def test_radial_out_points_away_from_central_body():
    s = _leo_state()
    # RTN radial-out = h_hat x v_hat = +x here.
    assert np.allclose(sas_target_dir("RADIAL_OUT", s), [1.0, 0.0, 0.0],
                       atol=1e-12)
    assert np.allclose(sas_target_dir("RADIAL_IN", s), [-1.0, 0.0, 0.0],
                       atol=1e-12)


def test_target_points_at_target():
    s = _leo_state()
    tgt = np.array([7.0e6, 1.0e6, 0.0])
    d = sas_target_dir("TARGET", s, target_pos=tgt)
    expected = (tgt - s.r) / np.linalg.norm(tgt - s.r)
    assert np.allclose(d, expected, atol=1e-12)


def test_target_without_position_raises():
    with pytest.raises(ValueError):
        sas_target_dir("TARGET", _leo_state())


def test_identity_nose_has_east_heading():
    heading, pitch = heading_pitch(quat_identity(), _leo_state())
    assert abs(pitch) < 1e-9
    assert abs(heading - math.pi / 2) < 1e-9


def test_prograde_nose_has_zero_heading_and_pitch():
    q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -math.pi / 2)
    heading, pitch = heading_pitch(q, _leo_state())
    assert abs(pitch) < 1e-9
    assert abs(heading) < 1e-9


def test_radial_out_nose_has_ninety_degree_pitch():
    q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi / 2)
    _heading, pitch = heading_pitch(q, _leo_state())
    assert abs(pitch - math.pi / 2) < 1e-9


def test_heading_pitch_ranges():
    for angle in np.linspace(-math.pi, math.pi, 7):
        q = quat_from_axis_angle(np.array([0.3, 0.5, 0.8]), float(angle))
        heading, pitch = heading_pitch(q, _leo_state())
        assert 0.0 <= heading < 2.0 * math.pi
        assert -math.pi / 2 <= pitch <= math.pi / 2


# ---------------------------------------------------------------------------
# Degenerate states (landed: v = 0) must never produce NaN attitude data.
# ---------------------------------------------------------------------------
from orbitsim.core.attitude import heading_pitch, local_horizon_basis, sas_target_dir
from orbitsim.core.state import StateVector as _SV
from orbitsim.core.constants import MU_EARTH as _MU_E, R_EARTH as _R_E


def _landed_state():
    return _SV(r=np.array([_R_E, 0.0, 0.0]), v=np.zeros(3), mu=_MU_E, epoch_s=0.0)


def test_sas_orbital_modes_raise_on_zero_velocity():
    for mode in ("PROGRADE", "RETROGRADE", "NORMAL", "ANTINORMAL",
                 "RADIAL_IN", "RADIAL_OUT"):
        with pytest.raises(ValueError):
            sas_target_dir(mode, _landed_state())


def test_sas_target_mode_works_with_zero_velocity():
    d = sas_target_dir("TARGET", _landed_state(), target_pos=np.array([2.0 * _R_E, 0.0, 0.0]))
    assert np.allclose(d, [1.0, 0.0, 0.0])


def test_local_horizon_basis_is_finite_and_orthonormal_when_landed():
    prograde, east, up = local_horizon_basis(_landed_state())
    for vec in (prograde, east, up):
        assert np.all(np.isfinite(vec))
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-12
    assert np.allclose(up, [1.0, 0.0, 0.0])          # radial-out from +x surface point
    assert abs(np.dot(prograde, east)) < 1e-12
    assert abs(np.dot(east, up)) < 1e-12


def test_heading_pitch_finite_when_landed():
    # Nose pointing radially out (+x) from the landing site: pitch +90 deg.
    q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    heading, pitch = heading_pitch(q, _landed_state())
    assert np.isfinite(heading) and np.isfinite(pitch)
    assert abs(pitch - np.pi / 2) < 1e-9
