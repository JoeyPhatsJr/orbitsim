"""Offline tests for the navball's pure pieces: the attitude projection (nose ->
screen center, registration on the unit sphere) and the procedural sky/ground map.
The rendered ball itself is verified at the visual checkpoint, per project convention."""
import numpy as np

from orbitsim.core.attitude import quat_identity, quat_from_axis_angle, nose_direction
from orbitsim.core.state import StateVector
from orbitsim.render.navball import (
    ship_axes, project_direction, horizon_frame, horizon_ball_matrix, _build_navball_texture,
)


def _mat_to_quat(R):
    """Quaternion [w,x,y,z] from a rotation matrix whose columns are the body
    (starboard, dorsal, nose) axes in inertial coords."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


# A circular LEO state with simple axis-aligned vectors: prograde=+Y, radial-out=+X,
# east=+Z. r=7000 km, v ~ circular speed.
_STATE = StateVector(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7546.0, 0.0]), 3.986e14)


def test_nose_projects_to_screen_center():
    # For any orientation, the nose direction must land at (0, -1, 0) under the reticle.
    for q in (quat_identity(),
              quat_from_axis_angle([0, 1, 0], 0.7),
              quat_from_axis_angle([1, 2, 3], 2.1)):
        right, nose, up = ship_axes(q)
        p = project_direction(nose, right, nose, up)
        assert np.allclose(p, [0.0, -1.0, 0.0], atol=1e-9)


def test_identity_axes_map_to_screen():
    right, nose, up = ship_axes(quat_identity())
    # starboard -> screen right (+x), dorsal -> screen up (+z), nose -> into screen (-y).
    assert np.allclose(project_direction(right, right, nose, up), [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(project_direction(up, right, nose, up), [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(project_direction(nose, right, nose, up), [0.0, -1.0, 0.0], atol=1e-9)


def test_projection_preserves_unit_length():
    # The projection is a rotation, so it maps unit vectors to unit vectors (markers
    # stay on the ball surface).
    right, nose, up = ship_axes(quat_from_axis_angle([0.3, -1.0, 0.5], 1.3))
    d = np.array([0.2, -0.7, 0.68]); d /= np.linalg.norm(d)
    assert abs(np.linalg.norm(project_direction(d, right, nose, up)) - 1.0) < 1e-9


def test_horizon_frame_is_orthonormal_rtn():
    v_hat, east, radial_out = horizon_frame(_STATE)
    assert np.allclose(v_hat, [0, 1, 0], atol=1e-9)        # prograde
    assert np.allclose(radial_out, [1, 0, 0], atol=1e-9)   # radial out
    assert np.allclose(east, [0, 0, 1], atol=1e-9)         # right-handed completion


def test_ball_matrix_is_a_rotation():
    m = horizon_ball_matrix(quat_from_axis_angle([0.3, -1.0, 0.5], 1.3), _STATE)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(m) - 1.0) < 1e-9


def test_wings_level_prograde_puts_horizon_centered():
    # Nose=prograde(+Y), dorsal=radial-out(+X): R columns (starboard, dorsal, nose).
    v_hat, east, radial_out = horizon_frame(_STATE)
    starboard = np.cross(radial_out, v_hat)  # dorsal x nose
    q = _mat_to_quat(np.column_stack([starboard, radial_out, v_hat]))
    m = horizon_ball_matrix(q, _STATE)
    # world = local @ m, so row k = where texture axis e_k lands on screen.
    # Texture +X (heading 0 = prograde) must land at screen center (0,-1,0)...
    assert np.allclose(m[0], [0.0, -1.0, 0.0], atol=1e-6)
    # ...and the sky pole (texture +Z = radial-out) must land at screen up (0,0,1).
    assert np.allclose(m[2], [0.0, 0.0, 1.0], atol=1e-6)


def test_texture_is_sky_over_ground():
    tex = _build_navball_texture(64, 64)
    assert tex.get_x_size() == 64 and tex.get_y_size() == 64
    buf = tex.get_ram_image_as("RGB")
    arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(64, 64, 3)
    # RAM image is bottom-up: row 0 is the south pole (ground), top row is sky.
    bottom, top = arr[2].mean(axis=0), arr[-3].mean(axis=0)
    assert top[2] > top[0]        # sky: blue channel dominates
    assert bottom[0] > bottom[2]  # ground: red/brown channel dominates
