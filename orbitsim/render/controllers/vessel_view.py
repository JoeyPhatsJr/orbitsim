"""Vessel markers and the cross-fading true-scale ship model + plume."""



class VesselView:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    def __init__(self, app) -> None:
        self.app = app

    def build_vessel_markers(self):
        """Constant-size vessel markers."""
        world = self.app.world
        # Vessel markers + orbit lines.
        self.app.vessel_nps = []
        self.app.orbit_nps = []
        from orbitsim.render.ship_model import build_ship_icon
        for _ in world.vessels:
            # The real ship model, unlit and outlined. The camera always sits
            # RENDER_UNITS_ACROSS_VIEW (1000) units from the focus, so a fixed
            # render-size icon keeps a constant on-screen size at every zoom level.
            m = build_ship_icon()
            m.reparent_to(self.app.render)
            self.app.vessel_nps.append(m)
            self.app.orbit_nps.append(None)

    def apply_opening_camera(self):
        """Snap the first view to the Sun's side of vessel 0, framing its whole orbit."""
        import numpy as np
        from orbitsim.core.elements import state_to_elements
        from orbitsim.core.nbody import _csun, refresh_ephemeris_cache
        from orbitsim.render.camera_rig import opening_camera
        # This runs before the first frame's cache refresh: without it _csun serves the
        # offline circular Sun (~100 deg off the DE440 Sun that lights Earth).
        refresh_ephemeris_cache(self.app.clock.sim_time_s)
        state = self.app.world.vessels[0].state
        try:
            elem = state_to_elements(state)
            r_ap = elem.a * (1.0 + elem.e) if elem.e < 1.0 else float(np.linalg.norm(state.r))
        except ValueError:
            r_ap = float(np.linalg.norm(state.r))
        normal = np.cross(state.r, state.v)
        if not np.any(normal):            # landed / radial-only state: any view will do
            normal = np.array([0.0, 0.0, 1.0])
        az, el, dist = opening_camera(_csun(self.app.clock.sim_time_s).r, normal, r_ap)
        rig = self.app.rig
        rig.set_distance(dist)
        # Initial framing snaps (current AND target); input handlers only set targets.
        rig.azimuth = rig.target_azimuth = az
        rig.elevation = rig.target_elevation = el

    def build_ship_view(self):
        """True-scale ship model + plume for vessel 0."""
        # Ship view: a true-scale, lit, oriented model of vessel 0 that cross-fades
        # in from the constant-size marker as the camera zooms close. (ship_model.py)
        self.app._ship_model_np = None
        if not self.app.solar_system and self.app.world.vessels:
            from orbitsim.render.ship_model import build_ship_model
            from panda3d.core import TransparencyAttrib
            self.app._ship_model_np = build_ship_model()
            self.app._ship_model_np.reparent_to(self.app.render)
            self.app._ship_model_np.hide()  # shown only when model_alpha > 0
            # Marker fades via alpha; enable transparency on vessel 0's marker.
            self.app.vessel_nps[0].set_transparency(TransparencyAttrib.M_alpha)
            # Exhaust plume (parented to the model: inherits its orient + scale).
            from orbitsim.render.ship_model import build_plume
            self.app._plume_np = build_plume()
            self.app._plume_np.reparent_to(self.app._ship_model_np)
            self.app._plume_np.hide()

    def frame_vessels(self):
        """Place vessel markers, refresh trajectory lines, apsis markers."""
        import time as _time
        now_real = _time.monotonic()
        from panda3d.core import LMatrix3f, LQuaternion
        from orbitsim.core.attitude import nose_direction
        from orbitsim.render.ship_model import icon_axes
        cam_q = self.app.camera.get_quat(self.app.render)
        view_dir, up_hint = tuple(cam_q.get_forward()), tuple(cam_q.get_up())
        for idx, vessel in enumerate(self.app.world.vessels):
            vx, vy, vz = self.app.transform.to_render(vessel.state.r)
            self.app.vessel_nps[idx].set_pos(vx, vy, vz)
            # 2.5D map icon: long axis = the nose as seen on screen, side to the camera.
            x, y, z = icon_axes(tuple(nose_direction(vessel.orientation)), view_dir, up_hint)
            icon_q = LQuaternion()
            icon_q.set_from_matrix(LMatrix3f(*x, *y, *z))
            self.app.vessel_nps[idx].set_quat(self.app.render, icon_q)
            self.app.trajectory_view.rebuild_trajectory(idx, vessel, now_real)
        if self.app._preview_np is not None and self.app._preview_origin is not None:
            self.app._preview_np.set_pos(*self.app.transform.to_render(self.app._preview_origin))
        self.app.readouts.place_apsis_markers()

    def frame_ship_view(self):
        """Cross-fade vessel 0 marker <-> true-scale model + plume."""
        # Ship view: cross-fade marker -> true-scale oriented model for vessel 0.
        if self.app._ship_model_np is not None:
            from orbitsim.render.ship_model import view_blend, model_node_scale
            v0 = self.app.world.vessels[0]
            marker_a, model_a = view_blend(self.app.rig.distance_m)
            if marker_a > 0.0:
                self.app.vessel_nps[0].show()
                self.app.vessel_nps[0].set_alpha_scale(marker_a)
            else:
                self.app.vessel_nps[0].hide()
            if model_a > 0.0:
                from panda3d.core import LQuaternion
                self.app._ship_model_np.show()
                self.app._ship_model_np.set_pos(*self.app.transform.to_render(v0.state.r))
                self.app._ship_model_np.set_scale(model_node_scale(self.app.transform.scale_m_per_unit))
                q = v0.orientation  # [w, x, y, z]
                self.app._ship_model_np.set_quat(LQuaternion(float(q[0]), float(q[1]),
                                                         float(q[2]), float(q[3])))
                self.app._ship_model_np.set_alpha_scale(model_a)
                thr = getattr(v0, "throttle", 0.0)
                if thr > 0.0:
                    self.app._plume_np.show()
                    self.app._plume_np.set_sz(0.5 + thr)
                    self.app._plume_np.set_alpha_scale(model_a * thr)
                else:
                    self.app._plume_np.hide()
            else:
                self.app._ship_model_np.hide()
                self.app._plume_np.hide()
