"""Vessel markers and the cross-fading true-scale ship model + plume."""
from orbitsim.render.geometry import make_uv_sphere


class VesselView:
    """Vessel markers and the cross-fading true-scale ship model + plume. Stateless: all state lives on the OrbitApp it is given."""

    def __init__(self, app) -> None:
        self.app = app

    def _build_vessel_markers(self):
        """Constant-size vessel markers."""
        world = self.app.world
        # Vessel markers + orbit lines.
        self.app.vessel_nps = []
        self.app.orbit_nps = []
        for _ in world.vessels:
            m = make_uv_sphere(1.0, 8, 12)
            m.reparent_to(self.app.render)
            m.set_color(1.0, 0.9, 0.2, 1.0)
            m.set_light_off()  # fullbright marker so it is always visible
            # The camera always sits RENDER_UNITS_ACROSS_VIEW (1000) units from the
            # focus, so a fixed render-size marker keeps a constant on-screen size
            # at every zoom level.
            m.set_scale(8.0)
            self.app.vessel_nps.append(m)
            self.app.orbit_nps.append(None)

    def _build_ship_view(self):
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

    def _frame_vessels(self):
        """Place vessel markers, refresh trajectory lines, apsis markers."""
        import time as _time
        now_real = _time.monotonic()
        for idx, vessel in enumerate(self.app.world.vessels):
            vx, vy, vz = self.app.transform.to_render(vessel.state.r)
            self.app.vessel_nps[idx].set_pos(vx, vy, vz)
            self.app.trajectory_view._rebuild_trajectory(idx, vessel, now_real)
        if self.app._preview_np is not None and self.app._preview_origin is not None:
            self.app._preview_np.set_pos(*self.app.transform.to_render(self.app._preview_origin))
        self.app._place_apsis_markers()

    def _frame_ship_view(self):
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
