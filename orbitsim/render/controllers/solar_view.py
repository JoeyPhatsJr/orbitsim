"""Solar-system viewer mode: Sun-centered planets from the ephemeris, no vessel."""

import numpy as np
from orbitsim.core.planets import heliocentric_state
from orbitsim.render.geometry import make_uv_sphere

# At 1e8x a 30-day orbit bucket passes every ~1.5 frames, but the osculating orbits
# barely move: rebuild them at most this often in real time [s].
SOLAR_ORBIT_MIN_REBUILD_S = 0.5


def solar_orbits_need_rebuild(bucket, last_bucket, px_scale, last_px_scale,
                              now_real_s, last_rebuild_real_s) -> bool:
    """Whether this frame rebuilds the solar orbit lines.

    Always on the first frame and when the line pixel scale changed (a move to a
    display of another density: widths are baked in); on a new sim-time bucket only
    once SOLAR_ORBIT_MIN_REBUILD_S of real time has passed since the last rebuild.
    """
    if last_bucket is None or last_rebuild_real_s is None:
        return True
    if px_scale != last_px_scale:
        return True
    return bucket != last_bucket and now_real_s - last_rebuild_real_s >= SOLAR_ORBIT_MIN_REBUILD_S


class SolarView:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    # Distinct colours for the Sun + 8 planets (constant on-screen marker size).
    _PLANET_COLORS = {
        "Sun": (1.0, 0.85, 0.2, 1.0),
        "Mercury": (0.6, 0.6, 0.6, 1.0),
        "Venus": (0.9, 0.8, 0.5, 1.0),
        "Earth": (0.35, 0.6, 1.0, 1.0),   # flat: a tiny textured Earth read as black
        "Mars": (0.9, 0.4, 0.2, 1.0),
        "Jupiter": (0.8, 0.7, 0.5, 1.0),
        "Saturn": (0.9, 0.8, 0.6, 1.0),
        "Uranus": (0.6, 0.85, 0.9, 1.0),
        "Neptune": (0.3, 0.4, 0.9, 1.0),
    }

    # Orbit lines are baked in these units under one frame node that is rescaled each
    # frame (zoom never rebuilds them); they are rebuilt once per 30 sim-days.
    SOLAR_ORBIT_UNIT_M = 1.0e9
    SOLAR_ORBIT_REFRESH_S = 30 * 86400.0
    SOLAR_ORBIT_POINTS = 256

    # On-screen sizes in framebuffer pixels (x line_px_scale()), constant at any zoom.
    MARKER_RADIUS_PX = {"Sun": 10.0, "Jupiter": 7.0, "Saturn": 7.0}
    DEFAULT_MARKER_RADIUS_PX = 6.0
    LABEL_FONT_PX = 15.0      # pixels per text unit: a name's card is ~22 px tall
    LABEL_GAP_PX = 3.0        # between a dot's edge and its name's card
    # Who keeps a name when names crowd: home first, then its neighbours, the giants,
    # the small or far planets; the Sun last (the big yellow dot needs no label).
    SOLAR_LABEL_PRIORITY = {"Earth": 100, "Venus": 90, "Mars": 85, "Jupiter": 80,
                            "Saturn": 75, "Mercury": 70, "Uranus": 65, "Neptune": 60,
                            "Sun": 10}
    HOVER_TOL_PX = 14.0       # (x ui) a hovered dot always shows its name

    def __init__(self, app) -> None:
        self.app = app

    def build_planets(self) -> None:
        """Create constant-size markers + labels for the Sun and 8 planets."""
        from orbitsim.core.bodies import PLANETS, SUN

        self.app._planet_bodies = [SUN] + list(PLANETS)
        self.app._solar_orbit_frame = self.app.render.attach_new_node("solar_orbit_frame")
        self.app._solar_orbit_nps = {}
        self.app._solar_orbit_bucket = None
        self.app._solar_orbit_px_scale = None
        self.app._solar_orbit_rebuilt_real_s = None
        self.app._planet_nps = []
        self.app._planet_labels = []
        self.app._solar_label_cards = {}
        self.app._solar_marker_px = {}
        self.app._solar_label_slots = {}
        self.app._solar_label_rects = {}
        self.app._solar_hud_text_rect = None   # (key, rect): cached HUD-text bounds
        # The Sun marker below is the only Sun (bodies_view's central body is hidden).
        self.app.central_np.hide()
        from orbitsim.render.world_labels import build_world_label
        for body in self.app._planet_bodies:
            marker = make_uv_sphere(1.0, 12, 16)
            marker.set_name(f"solar_marker_{body.name}")
            marker.reparent_to(self.app.render)
            marker.set_color(*self._PLANET_COLORS.get(body.name, (0.8, 0.8, 0.8, 1.0)))
            marker.set_light_off()
            self.app._planet_nps.append(marker)
            # Billboard label on render; frame_markers_and_labels sizes and places it.
            label = build_world_label(
                self.app.render, body.name, color=(0.8, 0.85, 1.0, 1.0), scale=15.0
            )
            # Drawn over orbit lines; the layout keeps names off the dots.
            label.set_depth_test(False)
            label.set_bin("fixed", 60)
            self.app._planet_labels.append(label)
            self.app._solar_label_cards[body.name] = tuple(label.node().get_card_actual())

    def update_solar_system(self) -> None:
        """Place the Sun + planets for the current sim time (DE440 or fallback)."""
        from datetime import datetime, timedelta
        from orbitsim.core.ephemeris import available as ephemeris_available

        t = self.app.clock.sim_time_s
        self.app.transform.set_origin(np.zeros(3))  # heliocentric: Sun fixed at origin

        states = {body.name: heliocentric_state(body.name, t) for body in self.app._planet_bodies}
        for body, marker in zip(self.app._planet_bodies, self.app._planet_nps):
            rx, ry, rz = self.app.transform.to_render(states[body.name].r)
            marker.set_pos(rx, ry, rz)

        self.app.flight_input.apply_mouse_orbit()
        self.app.bodies_view.update_starfield()
        self.frame_orbits(states)
        self.app.rig.apply()

        self.app.warp_controls.update_warp_readout()
        source = "JPL DE440" if ephemeris_available() else "circular approx (offline)"
        date = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=t)
        self.app.hud.text.setText(
            f"Solar system ({source})\n"
            f"Date: {date:%Y-%m-%d}\n"
            f"',' / '.' or the buttons change warp"
        )
        self.frame_markers_and_labels(self._mouse_px())

    def _mouse_px(self):
        """Mouse position in framebuffer pixels (y up), or None (no mouse; headless)."""
        p = self.app.flight_input.mouse_px()
        if p is None:
            return None
        win = self.app.win
        return (p[0] * win.get_fb_x_size() / max(win.get_x_size(), 1),
                p[1] * win.get_fb_y_size() / max(win.get_y_size(), 1))

    def frame_markers_and_labels(self, mouse_px) -> None:
        """Size every dot and name in pixels for this frame's camera, and lay out names.

        A body behind the camera (or nearer than the lens's near plane) hides its dot and
        name. Otherwise the dot's scale is its pixel radius x ui x units-per-pixel at its
        depth. Names are placed by ``layout_labels``: by ``SOLAR_LABEL_PRIORITY``, the
        hovered body pinned, other dots and the HUD as obstacles, last frame's slot kept
        when still free, and hidden when nothing fits. Dots are never hidden by it.
        """
        from panda3d.core import Point2
        from orbitsim.render.label_layout import (LabelRequest, label_origin_offset_px,
                                                  layout_labels, units_per_pixel)
        from orbitsim.render.picking import nearest_marker
        app = self.app
        fb_w, fb_h = app.win.get_fb_x_size(), app.win.get_fb_y_size()
        if fb_w <= 0 or fb_h <= 0:
            return      # minimised window: nothing to lay out (units_per_pixel would raise)
        ui = app.line_px_scale()
        lens = app.camLens
        near = lens.get_near()
        vfov = lens.get_fov()[1]
        quat = app.camera.get_quat(app.render)
        cam_right, cam_up = quat.get_right(), quat.get_up()
        font_px = self.LABEL_FONT_PX * ui

        requests, placing = [], {}
        for body, marker, label in zip(app._planet_bodies, app._planet_nps, app._planet_labels):
            name = body.name
            card = app._solar_label_cards[name]
            size = ((card[1] - card[0]) * font_px, (card[3] - card[2]) * font_px)
            radius_px = self.MARKER_RADIUS_PX.get(name, self.DEFAULT_MARKER_RADIUS_PX) * ui
            priority = self.SOLAR_LABEL_PRIORITY.get(name, 0)
            pos = marker.get_pos(app.render)
            p = app.cam.get_relative_point(app.render, pos)
            anchor = None
            if p.y <= near:
                marker.hide()
            else:
                upp = units_per_pixel(p.y, vfov, fb_h)
                marker.set_scale(radius_px * upp)
                marker.show()
                q = Point2()
                if lens.project(p, q):
                    anchor = ((q.x * 0.5 + 0.5) * fb_w, (q.y * 0.5 + 0.5) * fb_h)
                    placing[name] = (label, pos, upp, card)
            app._solar_marker_px[name] = None if anchor is None else (*anchor, radius_px)
            requests.append(LabelRequest(name, anchor, size, radius_px, priority))

        hovered = None
        if mouse_px is not None:
            keys = [r.key for r in requests if r.anchor_px is not None]
            i = nearest_marker(mouse_px, [r.anchor_px for r in requests if r.anchor_px is not None],
                               self.HOVER_TOL_PX * ui)
            hovered = None if i is None else keys[i]
        layout = layout_labels(
            requests, (fb_w, fb_h), gap_px=self.LABEL_GAP_PX * ui,
            obstacles=self._ui_obstacles_px(), pinned={hovered} - {None},
            preferred_slots=app._solar_label_slots,
        )
        app._solar_label_slots = {k: v[0] for k, v in layout.items() if v is not None}
        app._solar_label_rects = {k: (None if v is None else v[1]) for k, v in layout.items()}
        for label, name in zip(app._planet_labels, (b.name for b in app._planet_bodies)):
            placed = layout[name]
            if placed is None or name not in placing:
                label.hide()
                continue
            _label, pos, upp, card = placing[name]
            anchor = app._solar_marker_px[name][:2]
            dx, dy = label_origin_offset_px(placed[1], card, font_px, anchor)
            label.set_scale(font_px * upp)
            label.set_pos(pos + (cam_right * dx + cam_up * dy) * upp)
            label.show()

    def _ui_obstacles_px(self):
        """Framebuffer-pixel rects names must keep off: the HUD panels drawn now (the
        warp panel; grown by the panel pass's 4 px margin, so the pass never hides a name
        the layout placed) and the HUD text. Live bounds, so a moved HUD is followed."""
        from orbitsim.render.label_layout import render2d_rect_px
        app = self.app
        win = app.win
        fb_w, fb_h = win.get_fb_x_size(), win.get_fb_y_size()
        sx = fb_w / max(win.get_x_size(), 1)
        sy = fb_h / max(win.get_y_size(), 1)
        m = 4.0
        rects = [((x0 - m) * sx, (y0 - m) * sy, (x1 + m) * sx, (y1 + m) * sy)
                 for x0, y0, x1, y1 in app.readouts.panel_rects_px()]
        text = app.hud.text
        if not text.is_hidden():
            # Measuring forces the text's geometry (~0.14 ms), and the date changes every
            # few frames at warp, so re-measure only when a line's length (or the window
            # or UI scale) changes; m px of padding covers digit-width differences.
            key = (tuple(len(line) for line in text.getText().split("\n")), fb_w, fb_h,
                   win.get_x_size(), win.get_y_size(), getattr(app, "_ui_scale", 1.0))
            cached = app._solar_hud_text_rect
            if cached is None or cached[0] != key:
                bounds = text.get_tight_bounds(app.render2d)
                rect = None
                if bounds:
                    lo, hi = bounds
                    x0, y0, x1, y1 = render2d_rect_px((lo.x, lo.z), (hi.x, hi.z), (fb_w, fb_h))
                    rect = (x0 - m * sx, y0 - m * sy, x1 + m * sx, y1 + m * sy)
                cached = app._solar_hud_text_rect = (key, rect)
            if cached[1] is not None:
                rects.append(cached[1])
        return rects

    def frame_orbits(self, states) -> None:
        """Place and scale the planets' orbit lines; rebuild them once per 30-day bucket
        (at most every SOLAR_ORBIT_MIN_REBUILD_S real seconds) or on a line-scale change.

        ``states`` are this frame's heliocentric states (the ones placing the markers),
        so a rebuild costs no extra ephemeris calls.
        """
        from orbitsim.core.constants import MU_SUN
        from orbitsim.core.state import StateVector
        from orbitsim.render.line_style import PLANET_ORBIT_TINTS, planet_orbit_style
        from orbitsim.render.orbit_lines import build_orbit_node, sample_relative_orbit_points

        import time as _time

        app = self.app
        t = app.clock.sim_time_s
        bucket = int(np.floor(t / self.SOLAR_ORBIT_REFRESH_S))
        px_scale = app.line_px_scale()
        now_real = _time.monotonic()
        if solar_orbits_need_rebuild(bucket, app._solar_orbit_bucket, px_scale,
                                     app._solar_orbit_px_scale, now_real,
                                     app._solar_orbit_rebuilt_real_s):
            app._solar_orbit_bucket = bucket
            app._solar_orbit_px_scale = px_scale
            app._solar_orbit_rebuilt_real_s = now_real
            sun = StateVector(r=np.zeros(3), v=np.zeros(3), mu=MU_SUN, epoch_s=t)
            for name in PLANET_ORBIT_TINTS:
                old = app._solar_orbit_nps.pop(name, None)
                if old is not None:
                    old.remove_node()
                pts = sample_relative_orbit_points(
                    states[name], sun, MU_SUN, n=self.SOLAR_ORBIT_POINTS
                ) / self.SOLAR_ORBIT_UNIT_M
                orbit_np = build_orbit_node(
                    [tuple(p) for p in pts], planet_orbit_style(name),
                    px_scale=px_scale,
                )
                orbit_np.set_name(f"solar_orbit_{name}")
                orbit_np.reparent_to(app._solar_orbit_frame)
                app._solar_orbit_nps[name] = orbit_np
        app._solar_orbit_frame.set_pos(*app.transform.to_render(np.zeros(3)))
        app._solar_orbit_frame.set_scale(self.SOLAR_ORBIT_UNIT_M / app.transform.scale_m_per_unit)
