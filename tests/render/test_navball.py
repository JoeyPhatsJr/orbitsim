"""Offline tests for the navball's pure pieces: the attitude projection (nose ->
screen center, registration on the unit sphere) and the procedural sky/ground map.
The rendered ball itself is verified at the visual checkpoint, per project convention."""
import numpy as np

from orbitsim.core.attitude import quat_identity, quat_from_axis_angle, nose_direction
from orbitsim.render.navball import ship_axes, project_direction, _build_navball_texture


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


def test_texture_is_sky_over_ground():
    tex = _build_navball_texture(64, 64)
    assert tex.get_x_size() == 64 and tex.get_y_size() == 64
    buf = tex.get_ram_image_as("RGB")
    arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(64, 64, 3)
    # RAM image is bottom-up: row 0 is the south pole (ground), top row is sky.
    bottom, top = arr[2].mean(axis=0), arr[-3].mean(axis=0)
    assert top[2] > top[0]        # sky: blue channel dominates
    assert bottom[0] > bottom[2]  # ground: red/brown channel dominates
