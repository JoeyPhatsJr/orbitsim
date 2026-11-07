"""Solar-system viewer mode: Sun-centered planets from the ephemeris, no vessel."""

import numpy as np
from orbitsim.core.planets import heliocentric_state
from orbitsim.render.geometry import make_uv_sphere


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

    def __init__(self, app) -> None:
        self.app = app

    def build_planets(self) -> None:
        """Create constant-size markers + labels for the Sun and 8 planets."""
        from orbitsim.core.bodies import PLANETS, SUN

        self.app._planet_bodies = [SUN] + list(PLANETS)
        self.app._solar_orbit_frame = self.app.render.attach_new_node("solar_orbit_frame")
        self.app._solar_orbit_nps = {}
        self.app._solar_orbit_bucket = None
        self.app._planet_nps = []
        self.app._planet_labels = []
        self.app._solar_label_cards = {}
        self.app._solar_marker_px = {}
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

        self.app.central_np.set_pos(*self.app.transform.to_render(np.zeros(3)))
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
        """Size every dot and name in pixels for this frame's camera, and place names.

        A body behind the camera (or nearer than the lens's near plane) hides its dot and
        name. Otherwise the dot's scale is its pixel radius x ui x units-per-pixel at its
        depth, and the name sits "above" its dot.
        """
        from panda3d.core import Point2
        from orbitsim.render.label_layout import (label_origin_offset_px, slot_rect,
                                                  units_per_pixel)
        app = self.app
        ui = app.line_px_scale()
        lens = app.camLens
        near = lens.get_near()
        vfov = lens.get_fov()[1]
        fb_w, fb_h = app.win.get_fb_x_size(), app.win.get_fb_y_size()
        quat = app.camera.get_quat(app.render)
        cam_right, cam_up = quat.get_right(), quat.get_up()
        font_px = self.LABEL_FONT_PX * ui
        for body, marker, label in zip(app._planet_bodies, app._planet_nps, app._planet_labels):
            name = body.name
            pos = marker.get_pos(app.render)
            p = app.cam.get_relative_point(app.render, pos)
            if p.y <= near:
                marker.hide()
                label.hide()
                app._solar_marker_px[name] = None
                continue
            upp = units_per_pixel(p.y, vfov, fb_h)
            radius_px = self.MARKER_RADIUS_PX.get(name, self.DEFAULT_MARKER_RADIUS_PX) * ui
            marker.set_scale(radius_px * upp)
            marker.show()
            q = Point2()
            if not lens.project(p, q):
                label.hide()
                app._solar_marker_px[name] = None
                continue
            anchor = ((q.x * 0.5 + 0.5) * fb_w, (q.y * 0.5 + 0.5) * fb_h)
            app._solar_marker_px[name] = (anchor[0], anchor[1], radius_px)
            card = app._solar_label_cards[name]
            size = ((card[1] - card[0]) * font_px, (card[3] - card[2]) * font_px)
            rect = slot_rect(anchor, size, radius_px + self.LABEL_GAP_PX * ui, "above")
            dx, dy = label_origin_offset_px(rect, card, font_px, anchor)
            label.set_scale(font_px * upp)
            label.set_pos(pos + (cam_right * dx + cam_up * dy) * upp)
            label.show()

    def frame_orbits(self, states) -> None:
        """Place and scale the planets' orbit lines; rebuild them once per 30-day bucket.

        ``states`` are this frame's heliocentric states (the ones placing the markers),
        so a rebuild costs no extra ephemeris calls.
        """
        from orbitsim.core.constants import MU_SUN
        from orbitsim.core.state import StateVector
        from orbitsim.render.line_style import PLANET_ORBIT_TINTS, planet_orbit_style
        from orbitsim.render.orbit_lines import build_orbit_node, sample_relative_orbit_points

        app = self.app
        t = app.clock.sim_time_s
        bucket = int(np.floor(t / self.SOLAR_ORBIT_REFRESH_S))
        if bucket != app._solar_orbit_bucket:
            app._solar_orbit_bucket = bucket
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
                    px_scale=app.line_px_scale(),
                )
                orbit_np.set_name(f"solar_orbit_{name}")
                orbit_np.reparent_to(app._solar_orbit_frame)
                app._solar_orbit_nps[name] = orbit_np
        app._solar_orbit_frame.set_pos(*app.transform.to_render(np.zeros(3)))
        app._solar_orbit_frame.set_scale(self.SOLAR_ORBIT_UNIT_M / app.transform.scale_m_per_unit)
