"""A 3D attitude navball, KSP-style: a textured sky/ground sphere oriented by the
full ship quaternion, with the nose fixed at screen center, orbital-frame markers
(prograde/retrograde/normal/radial), a bezel ring, and a center reticle.

The ball lives in its own bottom-center display region with an orthographic camera,
so it overlays the world. The display region is kept square in *pixels* (scaling
with the window) so the ball always renders as a circle, never an ellipse."""
from math import sin, cos, pi

import numpy as np
from panda3d.core import (
    NodePath, OrthographicLens, Vec3, Texture, Mat4, LineSegs,
)

from orbitsim.render.geometry import make_uv_sphere
from orbitsim.core.attitude import nose_direction, sas_target_dir, quat_rotate_vector

# Orbital-frame markers shown on the ball -> distinct colors (color is the only cue,
# so prograde/retrograde etc. get clearly different hues).
_MARKER_COLORS = {
    "PROGRADE": (0.25, 1.0, 0.35, 1),
    "RETROGRADE": (1.0, 0.35, 0.30, 1),
    "NORMAL": (0.72, 0.40, 1.0, 1),
    "ANTINORMAL": (1.0, 0.45, 0.90, 1),
    "RADIAL_OUT": (0.30, 0.85, 1.0, 1),
    "RADIAL_IN": (1.0, 0.72, 0.20, 1),
    "TARGET": (1.0, 0.2, 1.0, 1),
    "ANTITARGET": (0.6, 0.2, 0.6, 1),
}
_FILM = 2.6            # orthographic film size (ball has radius 1)
_CAM_DIST = 10.0       # ortho camera distance along -Y
_BALL_FRAC_H = 0.30    # navball height as a fraction of the window height


def ship_axes(q):
    """Body (starboard, nose, dorsal) axes of orientation q, as inertial unit vectors."""
    right = quat_rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    nose = nose_direction(q)
    up = quat_rotate_vector(q, np.array([0.0, 1.0, 0.0]))
    return right, nose, up


def project_direction(d, right, nose, up):
    """Project an inertial direction d into ball/screen space: (d·right, -(d·nose),
    d·up). The nose maps to screen center (0,-1,0); near-hemisphere points have y<0."""
    return np.array([np.dot(d, right), -np.dot(d, nose), np.dot(d, up)])


def horizon_frame(state):
    """Local-horizon basis as inertial unit vectors: (prograde, east, radial-out).
    These become the navball texture's +X (heading 0), +Y, +Z (sky pole)."""
    v = np.asarray(state.v, dtype=np.float64)
    v_hat = v / np.linalg.norm(v)
    radial_out = np.cross(v, np.cross(np.asarray(state.r, dtype=np.float64), v))
    radial_out = radial_out / np.linalg.norm(radial_out)
    east = np.cross(radial_out, v_hat)
    return v_hat, east, radial_out


def horizon_ball_matrix(orientation_q, state):
    """3x3 transform (row-vector convention: world = local @ M) that orients the
    horizon-referenced ball texture for attitude q. Row k is where local axis e_k
    lands on screen, so local +X→proj(prograde), +Y→proj(east), +Z→proj(radial-out),
    keeping the painted sphere registered with the orbital markers."""
    right, nose, up = ship_axes(orientation_q)
    v_hat, east, radial_out = horizon_frame(state)
    return np.array([
        project_direction(v_hat, right, nose, up),
        project_direction(east, right, nose, up),
        project_direction(radial_out, right, nose, up),
    ])


def _build_navball_texture(w: int = 1024, h: int = 512) -> Texture:
    """Procedural equirectangular navball map: sky (top) / ground (bottom) split by
    a horizon line, with a pitch ladder (±30°, ±60°) and heading ticks at the equator.

    Row 0 of the array is the north pole (lat +90°); it is flipped to Panda's
    bottom-up RAM convention before upload."""
    sky_h = np.array([66, 135, 205], float)     # sky at the horizon
    sky_p = np.array([26, 60, 138], float)       # sky at the pole (darker)
    gnd_h = np.array([150, 116, 74], float)       # ground at the horizon
    gnd_p = np.array([92, 66, 42], float)         # ground at the pole
    horizon = np.array([245, 245, 245], float)
    pitch = np.array([220, 224, 230], float)
    tick = np.array([235, 235, 240], float)

    lat = 90.0 - 180.0 * np.arange(h) / (h - 1)   # row 0 = +90°, last row = -90°
    t = lat / 90.0                                # +1 north pole … -1 south pole
    col = np.empty((h, 3), float)
    north = t >= 0.0
    tn = t[north][:, None]
    col[north] = sky_h * (1.0 - tn) + sky_p * tn
    ts = -t[~north][:, None]
    col[~north] = gnd_h * (1.0 - ts) + gnd_p * ts
    img = np.repeat(col[:, None, :], w, axis=1)   # (h, w, 3)

    def row_for(lat_deg: float) -> int:
        return int(round((90.0 - lat_deg) / 180.0 * (h - 1)))

    def band(lat_deg: float, color, half_px: int) -> None:
        r = row_for(lat_deg)
        img[max(0, r - half_px):r + half_px + 1, :, :] = color

    for lat_deg in (30.0, 60.0, -30.0, -60.0):
        band(lat_deg, pitch, 1)
    band(0.0, horizon, 2)

    hrow = row_for(0.0)
    for k in range(24):                           # heading ticks every 15° of longitude
        c = int(round(k / 24.0 * w)) % w
        ext = 18 if k % 6 == 0 else 9             # taller at the four cardinals
        img[max(0, hrow - ext):hrow + ext + 1, max(0, c - 1):c + 2, :] = tick

    data = np.ascontiguousarray(np.flipud(img).astype(np.uint8))
    tex = Texture("navball")
    tex.setup_2d_texture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)
    tex.set_ram_image_as(data.tobytes(), "RGB")
    tex.set_minfilter(Texture.FT_linear)
    tex.set_magfilter(Texture.FT_linear)
    return tex


class Navball:
    """Bottom-center attitude sphere with its own camera + display region.

    The ball is oriented by the full ship quaternion so pitch, yaw, and roll all
    read; the nose is fixed at screen center (the reticle). Orbital markers float
    on the ball at their inertial directions, projected into the ball frame."""

    def __init__(self, base) -> None:
        self.base = base
        self._last_win_size = (0, 0)
        self.root = NodePath("navball_root")
        self.cam = base.make_camera(base.win)
        self.dr = self.cam.node().get_display_region(0)
        self.dr.set_sort(20)
        self.lens = OrthographicLens()
        self.lens.set_film_size(_FILM, _FILM)
        self.cam.node().set_lens(self.lens)
        self.cam.reparent_to(self.root)
        self.cam.set_pos(0, -_CAM_DIST, 0)
        self.cam.look_at(0, 0, 0)

        # The textured ball (we rotate it to the ship attitude each frame).
        self.ball = make_uv_sphere(1.0, 48, 96, with_uv=True)
        self.ball.reparent_to(self.root)
        self.ball.set_light_off()
        self.ball.set_texture(_build_navball_texture())

        # Orbital-frame markers (parented to root; positioned in the ball frame).
        self._markers = {}
        for mode, col in _MARKER_COLORS.items():
            m = make_uv_sphere(0.11, 8, 12)
            m.reparent_to(self.root)
            m.set_color(*col)
            m.set_light_off()
            self._markers[mode] = m

        self._build_bezel()
        self._build_reticle()
        self._layout()

    def _overlay(self, np_: NodePath) -> NodePath:
        """Draw a fixed 2D-ish overlay (bezel/reticle) in front of the ball."""
        np_.set_light_off()
        np_.set_bin("fixed", 30)
        np_.set_depth_test(False)
        np_.set_depth_write(False)
        return np_

    def _build_bezel(self) -> None:
        ls = LineSegs()
        ls.set_thickness(2.5)
        ls.set_color(0.78, 0.81, 0.88, 1.0)
        n, rad, y = 72, 1.18, -1.5
        ls.move_to(rad, y, 0.0)
        for i in range(1, n + 1):
            a = 2.0 * pi * i / n
            ls.draw_to(rad * cos(a), y, rad * sin(a))
        self.bezel = self._overlay(self.root.attach_new_node(ls.create()))

    def _build_reticle(self) -> None:
        ls = LineSegs()
        ls.set_thickness(2.0)
        ls.set_color(1.0, 0.92, 0.25, 1.0)
        n, rad, y = 24, 0.12, -2.0
        ls.move_to(rad, y, 0.0)
        for i in range(1, n + 1):
            a = 2.0 * pi * i / n
            ls.draw_to(rad * cos(a), y, rad * sin(a))
        for ax, az in ((1, 0), (-1, 0), (0, 1), (0, -1)):     # cross ticks
            ls.move_to(ax * rad, y, az * rad)
            ls.draw_to(ax * rad * 1.8, y, az * rad * 1.8)
        self.reticle = self._overlay(self.root.attach_new_node(ls.create()))

    def _layout(self) -> None:
        """Keep the display region square in pixels (so the ball is a circle) and
        scale it with the window: bottom-center, height = _BALL_FRAC_H of the window.

        Called every frame from update() rather than off a 'window-event' handler:
        ShowBase already owns that event (for the main camera's aspect ratio), and
        re-accepting it on the same object would clobber ShowBase's handler."""
        w, h = self.base.win.get_x_size(), self.base.win.get_y_size()
        if w <= 0 or h <= 0 or (w, h) == self._last_win_size:
            return
        self._last_win_size = (w, h)
        side_px = _BALL_FRAC_H * h
        half_w = (side_px / w) / 2.0
        self.dr.set_dimensions(0.5 - half_w, 0.5 + half_w, 0.015, 0.015 + _BALL_FRAC_H)

    def update(self, *, orientation_q, state, target_pos=None) -> None:
        """Orient the ball to the ship attitude and place the orbital markers.

        Projection: an inertial direction d maps into ball/screen space as
        p(d) = (d·right, -(d·nose), d·up), where right/up/nose are the ship body
        axes in inertial coords. So the nose lands at screen center (0,-1,0) under
        the reticle, screen-up follows the ship's dorsal axis (roll reads), and
        markers on the near hemisphere (p_y < 0) are visible.

        The painted ball is referenced to the *local horizon* (RTN), not the
        celestial pole: sky (texture +Z) = radial-out, heading 0 = prograde. So
        flying prograde-and-level puts the horizon across the middle, like KSP."""
        self._layout()   # keep the ball circular if the window was resized
        q = np.asarray(orientation_q, dtype=np.float64)
        b_right, b_nose, b_up = ship_axes(q)

        def proj(d):
            return project_direction(d, b_right, b_nose, b_up)

        # Orient the horizon-referenced ball texture (rows of M = images of the
        # texture's local axes; world = local @ M, Panda's row-vector convention).
        m = horizon_ball_matrix(q, state)
        self.ball.set_mat(Mat4(
            m[0, 0], m[0, 1], m[0, 2], 0.0,
            m[1, 0], m[1, 1], m[1, 2], 0.0,
            m[2, 0], m[2, 1], m[2, 2], 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

        for mode, marker in self._markers.items():
            try:
                d = np.asarray(sas_target_dir(mode, state, target_pos), dtype=np.float64)
            except ValueError:
                marker.hide()
                continue
            p = proj(d)
            marker.set_pos(p[0], p[1], p[2])
            if p[1] > 0.2:               # far hemisphere (behind the ball): hide
                marker.hide()
            else:
                marker.show()
