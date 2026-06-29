"""2D ecliptic-plane map view: top-down projection of the solar system.

Renders planet orbits, the ship trajectory, maneuver nodes, and SOI
boundaries onto a separate Panda3D display region with an orthographic
camera.  Toggled with Tab from the sandbox.
"""
from __future__ import annotations
import math
import numpy as np


# ── pure helpers (unit-testable, no Panda3D) ────────────────────────

def ecliptic_project(pos_m: np.ndarray, origin_m: np.ndarray) -> tuple[float, float]:
    """Project a 3-D physics position onto the ecliptic (XY) plane, relative to origin."""
    d = np.asarray(pos_m, dtype=np.float64) - np.asarray(origin_m, dtype=np.float64)
    return (float(d[0]), float(d[1]))


def circle_points(radius: float, n: int = 128) -> list[tuple[float, float]]:
    """Return *n* (x, y) points forming a circle of the given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(radius * np.cos(a)), float(radius * np.sin(a))) for a in angles]


def view_extents(center_x: float, center_y: float, half_size: float):
    """Return (left, right, bottom, top) for the orthographic film."""
    return (
        center_x - half_size,
        center_x + half_size,
        center_y - half_size,
        center_y + half_size,
    )


# ── Panda3D map scene ───────────────────────────────────────────────

_ORBIT_COLORS = {
    "Mercury": (0.6, 0.6, 0.6, 0.45),
    "Venus": (0.85, 0.75, 0.45, 0.45),
    "Earth": (0.3, 0.5, 1.0, 0.45),
    "Mars": (0.85, 0.4, 0.2, 0.45),
    "Jupiter": (0.8, 0.7, 0.5, 0.35),
    "Saturn": (0.9, 0.8, 0.6, 0.35),
    "Uranus": (0.6, 0.85, 0.9, 0.35),
    "Neptune": (0.3, 0.4, 0.9, 0.35),
}

_BODY_COLORS = {
    "Sun": (1.0, 0.85, 0.2, 1.0),
    "Earth": (0.3, 0.5, 1.0, 1.0),
    "Moon": (0.7, 0.7, 0.72, 1.0),
    "Mercury": (0.6, 0.6, 0.6, 1.0),
    "Venus": (0.9, 0.8, 0.5, 1.0),
    "Mars": (0.9, 0.4, 0.2, 1.0),
    "Jupiter": (0.8, 0.7, 0.5, 1.0),
    "Saturn": (0.9, 0.8, 0.6, 1.0),
    "Uranus": (0.6, 0.85, 0.9, 1.0),
    "Neptune": (0.3, 0.4, 0.9, 1.0),
}

_SOI_COLORS = {
    "Earth": (0.3, 0.5, 1.0, 0.12),
    "Moon": (0.45, 0.65, 1.0, 0.10),
    "Mars": (0.9, 0.5, 0.3, 0.10),
    "Jupiter": (0.8, 0.7, 0.5, 0.08),
    "Saturn": (0.9, 0.8, 0.6, 0.08),
    "Venus": (0.9, 0.8, 0.5, 0.10),
    "Mercury": (0.6, 0.6, 0.6, 0.10),
    "Uranus": (0.6, 0.85, 0.9, 0.08),
    "Neptune": (0.3, 0.4, 0.9, 0.08),
}


def _make_circle_line(radius_scaled, color, thickness, n=128):
    """Build a LineSegs circle on the XY plane."""
    from panda3d.core import LineSegs
    segs = LineSegs()
    segs.set_thickness(thickness)
    segs.set_color(*color)
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        x = radius_scaled * math.cos(a)
        y = radius_scaled * math.sin(a)
        if i == 0:
            segs.move_to(x, y, 0)
        else:
            segs.draw_to(x, y, 0)
    return segs.create()


def _make_dot(size=4.0):
    """A tiny CardMaker quad used as a map dot."""
    from panda3d.core import CardMaker
    cm = CardMaker("dot")
    hs = size * 0.5
    cm.set_frame(-hs, hs, -hs, hs)
    return cm.generate()


class MapView:
    """2D ecliptic map overlay managed by OrbitApp.

    Parameters
    ----------
    base : ShowBase
        The running Panda3D app (for display regions, aspect2d, etc.).
    """

    # Zoom limits in metres (half-width of the visible area).
    MIN_HALF_M = 5e5          # ~500 km (close-up Earth orbit)
    MAX_HALF_M = 8e12         # ~53 AU (whole solar system)
    ZOOM_FACTOR = 1.18        # per scroll step

    def __init__(self, base) -> None:
        from panda3d.core import (
            NodePath, OrthographicLens, TransparencyAttrib, AntialiasAttrib,
            TextNode,
        )
        from direct.gui.OnscreenText import OnscreenText

        self._base = base
        self.visible = False
        self._half_m = 2e9     # initial half-width: ~2 million km (shows Earth-Moon)
        self._center = np.zeros(2)  # ecliptic XY centre in physics metres

        # Separate scene graph for 2D map content.
        self._root = NodePath("map_root")

        # Display region covering the full window, initially inactive.
        self._dr = base.win.make_display_region()
        self._dr.set_sort(50)
        self._dr.set_active(False)
        self._dr.set_clear_color_active(True)
        self._dr.set_clear_color((0.02, 0.02, 0.06, 1.0))
        self._dr.set_clear_depth_active(True)

        # Orthographic camera.
        self._cam_node = base.make_camera(base.win, displayRegion=self._dr)
        lens = OrthographicLens()
        lens.set_near_far(-1000, 1000)
        self._cam_node.node().set_lens(lens)
        self._cam_node.reparent_to(self._root)
        self._cam_node.set_pos(0, 0, 100)
        self._cam_node.look_at(0, 0, 0)
        self._lens = lens

        # Sub-trees for different content layers.
        self._orbit_layer = self._root.attach_new_node("orbits")
        self._orbit_layer.set_transparency(TransparencyAttrib.M_alpha)
        self._orbit_layer.set_antialias(AntialiasAttrib.M_line)

        self._soi_layer = self._root.attach_new_node("soi")
        self._soi_layer.set_transparency(TransparencyAttrib.M_alpha)

        self._body_layer = self._root.attach_new_node("bodies")
        self._body_layer.set_transparency(TransparencyAttrib.M_alpha)

        self._ship_layer = self._root.attach_new_node("ship")
        self._ship_layer.set_transparency(TransparencyAttrib.M_alpha)
        self._ship_layer.set_antialias(AntialiasAttrib.M_line)

        self._label_layer = self._root.attach_new_node("labels")

        # Cached nodes rebuilt each update.
        self._orbit_nodes: list[NodePath] = []
        self._soi_nodes: list[NodePath] = []
        self._body_dots: dict[str, NodePath] = {}
        self._body_labels: dict[str, NodePath] = {}
        self._ship_dot: NodePath | None = None
        self._traj_node: NodePath | None = None
        self._preview_node: NodePath | None = None
        self._node_dot: NodePath | None = None
        self._built = False

        # 2D HUD elements (aspect2d children, only shown when map is visible).
        self._hud_root = base.aspect2d.attach_new_node("map_hud")
        self._hud_root.hide()
        self._title = OnscreenText(
            text="MAP VIEW", pos=(0, 0.92), scale=0.06,
            fg=(0.5, 0.8, 1.0, 0.8), shadow=(0, 0, 0, 1),
            parent=self._hud_root,
        )
        self._scale_text = OnscreenText(
            text="", pos=(-1.2, -0.92), scale=0.045,
            fg=(0.6, 0.7, 0.9, 0.9), align=TextNode.ALeft,
            mayChange=True, parent=self._hud_root,
        )
        self._hint_text = OnscreenText(
            text="Tab: exit  Scroll: zoom  Right-drag: pan",
            pos=(0, -0.92), scale=0.04,
            fg=(0.5, 0.6, 0.7, 0.7), mayChange=False,
            parent=self._hud_root,
        )

        # Pan state.
        self._pan_active = False
        self._pan_last = None

    # ── visibility toggle ────────────────────────────────────────────

    def toggle(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    def show(self) -> None:
        self.visible = True
        self._dr.set_active(True)
        self._hud_root.show()
        # Centre on the ship when opening.
        base = self._base
        if hasattr(base, "world") and base.world.vessels:
            r = base.world.vessels[0].state.r
            self._center = np.array([float(r[0]), float(r[1])])

    def hide(self) -> None:
        self.visible = False
        self._dr.set_active(False)
        self._hud_root.hide()

    # ── input ────────────────────────────────────────────────────────

    def zoom_in(self) -> None:
        self._half_m = max(self.MIN_HALF_M, self._half_m / self.ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self._half_m = min(self.MAX_HALF_M, self._half_m * self.ZOOM_FACTOR)

    def begin_pan(self) -> None:
        self._pan_active = True
        self._pan_last = None

    def end_pan(self) -> None:
        self._pan_active = False
        self._pan_last = None

    def apply_pan(self, mouse_x: float, mouse_y: float) -> None:
        """Pan the map by mouse delta (in NDC -1..1 space)."""
        if not self._pan_active:
            return
        if self._pan_last is not None:
            dx = mouse_x - self._pan_last[0]
            dy = mouse_y - self._pan_last[1]
            self._center[0] -= dx * self._half_m
            self._center[1] -= dy * self._half_m
        self._pan_last = (mouse_x, mouse_y)

    # ── per-frame update ─────────────────────────────────────────────

    def update(self, world, clock, targets=None, node_epoch_s=None, node=None) -> None:
        """Rebuild the map scene for the current frame."""
        if not self.visible:
            return

        scale = self._half_m  # physics metres per render unit
        inv_s = 1.0 / scale if scale > 0 else 1.0

        # Update the orthographic film size.
        ar = self._base.get_aspect_ratio()
        self._lens.set_film_size(2.0 * ar, 2.0)

        # Camera position (centred on self._center, projected to map space).
        cx = self._center[0] * inv_s
        cy = self._center[1] * inv_s
        self._cam_node.set_pos(cx, cy, 100)
        self._cam_node.look_at(cx, cy, 0)

        # Update scale readout.
        self._update_scale_text()

        # Clear previous dynamic content.
        self._clear_dynamic()

        t_now = clock.sim_time_s

        # ── planet orbits (static circles, heliocentric) ─────────────
        # The Sun's geocentric position is the offset from Earth to Sun.
        from orbitsim.core.nbody import _csun
        sun_geo = _csun(t_now).r
        sun_x, sun_y = float(sun_geo[0]) * inv_s, float(sun_geo[1]) * inv_s

        from orbitsim.core.planets import (
            A_MERCURY, A_VENUS, A_EARTH, A_MARS,
            A_JUPITER, A_SATURN, A_URANUS, A_NEPTUNE,
        )
        _orbit_radii = {
            "Mercury": A_MERCURY, "Venus": A_VENUS, "Earth": A_EARTH,
            "Mars": A_MARS, "Jupiter": A_JUPITER, "Saturn": A_SATURN,
            "Uranus": A_URANUS, "Neptune": A_NEPTUNE,
        }
        for name, radius in _orbit_radii.items():
            color = _ORBIT_COLORS.get(name, (0.5, 0.5, 0.5, 0.3))
            node_geom = _make_circle_line(radius * inv_s, color, 1.2, n=192)
            np_ = self._orbit_layer.attach_new_node(node_geom)
            np_.set_pos(sun_x, sun_y, 0)
            self._orbit_nodes.append(np_)

        # ── SOI circles ──────────────────────────────────────────────
        from orbitsim.core.planets import (
            EARTH_SOI_M, MARS_SOI_M, JUPITER_SOI_M, SATURN_SOI_M,
            MERCURY_SOI_M, VENUS_SOI_M, URANUS_SOI_M, NEPTUNE_SOI_M,
        )
        from orbitsim.core.nbody import (
            _cmercury, _cvenus, _cmars,
            _cjupiter, _csaturn, _curanus, _cneptune,
            MOON_SOI_M,
        )
        from orbitsim.core.moon import moon_state_at

        _soi_data = [
            ("Earth", np.zeros(3), EARTH_SOI_M),
            ("Moon", moon_state_at(t_now).r, MOON_SOI_M),
            ("Mercury", _cmercury(t_now).r, MERCURY_SOI_M),
            ("Venus", _cvenus(t_now).r, VENUS_SOI_M),
            ("Mars", _cmars(t_now).r, MARS_SOI_M),
            ("Jupiter", _cjupiter(t_now).r, JUPITER_SOI_M),
            ("Saturn", _csaturn(t_now).r, SATURN_SOI_M),
            ("Uranus", _curanus(t_now).r, URANUS_SOI_M),
            ("Neptune", _cneptune(t_now).r, NEPTUNE_SOI_M),
        ]
        for name, pos, soi_r in _soi_data:
            if soi_r * inv_s < 0.5:
                continue  # too small to see at this zoom
            color = _SOI_COLORS.get(name, (0.5, 0.5, 0.5, 0.1))
            node_geom = _make_circle_line(soi_r * inv_s, color, 1.0, n=64)
            bx, by = float(pos[0]) * inv_s, float(pos[1]) * inv_s
            np_ = self._soi_layer.attach_new_node(node_geom)
            np_.set_pos(bx, by, 0)
            self._soi_nodes.append(np_)

        # ── body dots + labels ───────────────────────────────────────
        _body_positions = {
            "Sun": sun_geo,
            "Earth": np.zeros(3),
            "Moon": moon_state_at(t_now).r,
            "Mercury": _cmercury(t_now).r,
            "Venus": _cvenus(t_now).r,
            "Mars": _cmars(t_now).r,
            "Jupiter": _cjupiter(t_now).r,
            "Saturn": _csaturn(t_now).r,
            "Uranus": _curanus(t_now).r,
            "Neptune": _cneptune(t_now).r,
        }
        from panda3d.core import TextNode
        for name, pos in _body_positions.items():
            bx = float(pos[0]) * inv_s
            by = float(pos[1]) * inv_s
            color = _BODY_COLORS.get(name, (0.8, 0.8, 0.8, 1.0))
            dot_size = 8.0 if name == "Sun" else 5.0
            dot = self._body_layer.attach_new_node(_make_dot(dot_size))
            dot.set_pos(bx, by, 0.5)
            dot.set_color(*color)
            dot.set_billboard_point_eye()
            self._body_dots[name] = dot

            tn = TextNode(f"lbl_{name}")
            tn.set_text(name)
            tn.set_align(TextNode.ACenter)
            tn.set_text_color(*color)
            tn.set_shadow(0.05, 0.05)
            tn.set_shadow_color(0, 0, 0, 0.7)
            lbl = self._label_layer.attach_new_node(tn)
            lbl.set_pos(bx, by + dot_size * 0.6, 0.6)
            lbl.set_scale(0.04)
            lbl.set_billboard_point_eye()
            self._body_labels[name] = lbl

        # ── ship position + trajectory ───────────────────────────────
        if world.vessels:
            v0 = world.vessels[0]
            sx = float(v0.state.r[0]) * inv_s
            sy = float(v0.state.r[1]) * inv_s
            dot = self._ship_layer.attach_new_node(_make_dot(7.0))
            dot.set_pos(sx, sy, 1.0)
            dot.set_color(1.0, 0.9, 0.2, 1.0)
            dot.set_billboard_point_eye()
            self._ship_dot = dot

            # Trajectory: propagate the current orbit forward.
            self._draw_trajectory(v0, t_now, inv_s, (0.2, 0.78, 1.0, 0.8), 2.0)

            # Maneuver preview: if a node is set, draw the post-burn trajectory.
            if node is not None and node.magnitude_mps > 0.0:
                from orbitsim.core.maneuvers import apply_maneuver
                post = apply_maneuver(v0.state, node)
                # Mark the node position.
                nx_ = float(post.r[0]) * inv_s
                ny_ = float(post.r[1]) * inv_s
                ndot = self._ship_layer.attach_new_node(_make_dot(6.0))
                ndot.set_pos(nx_, ny_, 1.5)
                ndot.set_color(0.3, 1.0, 1.0, 1.0)
                ndot.set_billboard_point_eye()
                self._node_dot = ndot
                # Post-burn trajectory.
                self._draw_preview(post, t_now, inv_s, (1.0, 0.25, 0.90, 0.7), 1.8)

    def _draw_trajectory(self, vessel, t_now, inv_s, color, thickness):
        """Sample the vessel's current orbit and draw it on the map."""
        from panda3d.core import LineSegs, TransparencyAttrib
        from orbitsim.core.elements import state_to_elements
        from orbitsim.core.propagate import propagate_kepler

        try:
            elem = state_to_elements(vessel.state)
            if elem.e < 1.0:
                n_pts = 256
                period = elem.period_s
                dts = np.linspace(0, period, n_pts)
            else:
                n_pts = 256
                dts = np.linspace(0, min(400 * 86400, 1e8), n_pts)
        except ValueError:
            return

        segs = LineSegs()
        segs.set_thickness(thickness)
        segs.set_color(*color)
        for i, dt in enumerate(dts):
            try:
                st = propagate_kepler(vessel.state, dt)
            except (ValueError, RuntimeError):
                break
            px = float(st.r[0]) * inv_s
            py = float(st.r[1]) * inv_s
            if i == 0:
                segs.move_to(px, py, 0.3)
            else:
                segs.draw_to(px, py, 0.3)
        node = segs.create()
        np_ = self._ship_layer.attach_new_node(node)
        np_.set_transparency(TransparencyAttrib.M_alpha)
        self._traj_node = np_

    def _draw_preview(self, state, t_now, inv_s, color, thickness):
        """Draw a post-maneuver trajectory preview."""
        from panda3d.core import LineSegs, TransparencyAttrib
        from orbitsim.core.elements import state_to_elements
        from orbitsim.core.propagate import propagate_kepler

        try:
            elem = state_to_elements(state)
            if elem.e < 1.0:
                n_pts = 256
                dts = np.linspace(0, elem.period_s, n_pts)
            else:
                n_pts = 256
                dts = np.linspace(0, min(400 * 86400, 1e8), n_pts)
        except ValueError:
            return

        segs = LineSegs()
        segs.set_thickness(thickness)
        segs.set_color(*color)
        for i, dt in enumerate(dts):
            try:
                st = propagate_kepler(state, dt)
            except (ValueError, RuntimeError):
                break
            px = float(st.r[0]) * inv_s
            py = float(st.r[1]) * inv_s
            if i == 0:
                segs.move_to(px, py, 0.4)
            else:
                segs.draw_to(px, py, 0.4)
        node = segs.create()
        np_ = self._ship_layer.attach_new_node(node)
        np_.set_transparency(TransparencyAttrib.M_alpha)
        self._preview_node = np_

    def _clear_dynamic(self) -> None:
        """Remove all per-frame nodes so update() can rebuild them."""
        for np_ in self._orbit_nodes:
            np_.remove_node()
        self._orbit_nodes.clear()
        for np_ in self._soi_nodes:
            np_.remove_node()
        self._soi_nodes.clear()
        for np_ in self._body_dots.values():
            np_.remove_node()
        self._body_dots.clear()
        for np_ in self._body_labels.values():
            np_.remove_node()
        self._body_labels.clear()
        if self._ship_dot is not None:
            self._ship_dot.remove_node()
            self._ship_dot = None
        if self._traj_node is not None:
            self._traj_node.remove_node()
            self._traj_node = None
        if self._preview_node is not None:
            self._preview_node.remove_node()
            self._preview_node = None
        if self._node_dot is not None:
            self._node_dot.remove_node()
            self._node_dot = None

    def _update_scale_text(self) -> None:
        AU = 1.496e11
        if self._half_m >= 0.01 * AU:
            self._scale_text.setText(f"Scale: {self._half_m / AU:.2f} AU")
        else:
            km = self._half_m / 1000.0
            if km >= 1e6:
                self._scale_text.setText(f"Scale: {km / 1e6:.1f} M km")
            else:
                self._scale_text.setText(f"Scale: {km:,.0f} km")

    def destroy(self) -> None:
        """Clean up all map resources."""
        self._clear_dynamic()
        self._hud_root.remove_node()
        if self._dr is not None:
            self._base.win.remove_display_region(self._dr)
            self._dr = None
        self._root.remove_node()
