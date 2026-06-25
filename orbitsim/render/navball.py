"""A 3D navball: a sphere oriented to the ship attitude, with velocity / normal /
radial markers and a fixed nose reticle, rendered in a bottom-center display region."""
import numpy as np
from panda3d.core import NodePath, OrthographicLens, TextNode, Vec3

from orbitsim.render.geometry import make_uv_sphere
from orbitsim.core.attitude import nose_direction, sas_target_dir

# Markers shown on the ball, by SAS mode -> color.
_MARKER_COLORS = {
    "PROGRADE": (0.2, 1.0, 0.3, 1), "RETROGRADE": (0.2, 1.0, 0.3, 1),
    "NORMAL": (0.7, 0.3, 1.0, 1), "ANTINORMAL": (0.7, 0.3, 1.0, 1),
    "RADIAL_OUT": (0.2, 0.8, 1.0, 1), "RADIAL_IN": (0.2, 0.8, 1.0, 1),
}
_FILM = 2.6          # orthographic film size (ball has radius 1)
_CAM_DIST = 10.0     # ortho camera distance along -Y


class Navball:
    """Bottom-center attitude sphere with its own camera + display region so it
    overlays the world without being affected by the world camera."""

    def __init__(self, base) -> None:
        self.base = base
        self.root = NodePath("navball_root")
        frame = (0.40, 0.60, 0.0, 0.26)   # left, right, bottom, top (window fraction)
        self.cam = base.make_camera(base.win, displayRegion=frame)
        self.cam.node().get_display_region(0).set_sort(20)
        lens = OrthographicLens()
        lens.set_film_size(_FILM, _FILM)
        self.cam.node().set_lens(lens)
        self.cam.reparent_to(self.root)
        self.cam.set_pos(0, -_CAM_DIST, 0)
        self.cam.look_at(0, 0, 0)

        # The ball (its own pivot we rotate to the ship attitude).
        self.ball = make_uv_sphere(1.0, 18, 36)
        self.ball.reparent_to(self.root)
        self.ball.set_light_off()
        self.ball.set_color(0.25, 0.45, 0.75, 1.0)   # blue ball
        # A bright pole cap so the rotation is legible.
        pole = make_uv_sphere(0.14, 8, 12)
        pole.reparent_to(self.ball)
        pole.set_pos(0, 0, 1.0)
        pole.set_color(0.95, 0.9, 0.4, 1)
        pole.set_light_off()

        # Orbital-frame markers (parented to root, not the ball: they live in the
        # orbital frame, independent of body roll).
        self._markers = {}
        for mode, col in _MARKER_COLORS.items():
            m = make_uv_sphere(0.1, 6, 10)
            m.reparent_to(self.root)
            m.set_color(*col)
            m.set_light_off()
            self._markers[mode] = m

        # Fixed nose reticle (screen center, in front of the ball).
        tn = TextNode("reticle")
        tn.set_text("[ ]")
        tn.set_text_color(1, 1, 1, 1)
        tn.set_align(TextNode.ACenter)
        self.reticle = self.root.attach_new_node(tn)
        self.reticle.set_scale(0.35)
        self.reticle.set_pos(0.0, -_CAM_DIST + 1.0, -0.12)
        self.reticle.set_billboard_point_eye()

    def update(self, *, orientation_q, state, target_pos=None) -> None:
        """Rotate the ball to the ship attitude and place the orbital-frame markers.

        The ball is oriented so its pole follows the ship nose, giving a quick read
        of where the ship points; markers sit on the unit sphere at their orbital
        directions and hide on the far hemisphere (toward the camera at -Y)."""
        nose = np.asarray(nose_direction(orientation_q), dtype=np.float64)
        self.ball.look_at(Vec3(*nose))
        for mode, marker in self._markers.items():
            try:
                d = np.asarray(sas_target_dir(mode, state, target_pos), dtype=np.float64)
            except ValueError:
                marker.hide()
                continue
            marker.set_pos(Vec3(*d))
            # Hide markers on the far side of the ball (those with +Y, away from camera).
            if d[1] > 0.35:
                marker.hide()
            else:
                marker.show()
