"""Panda3D ShowBase bootstrap and per-frame loop."""
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import ClockObject, AmbientLight, DirectionalLight, Vec4

from orbitsim.core.elements import state_to_elements
from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.orbit_lines import sample_orbit_points, build_orbit_node
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.hud import Hud

_global_clock = ClockObject.get_global_clock()


class OrbitApp(ShowBase):
    """Renders one central body + vessels with orbit lines; time-warpable."""

    def __init__(self, world, clock) -> None:
        super().__init__()
        self.world = world
        self.clock = clock

        self.transform = RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=2.0e4)
        self.rig = CameraRig(self, self.transform)
        self.disable_mouse()
        self.hud = Hud(self)

        # Central body sphere, sized in render units via the current scale.
        self.central_np = make_uv_sphere(1.0, 24, 48)
        self.central_np.reparent_to(self.render)
        self.central_np.set_color(0.2, 0.4, 0.9, 1.0)

        # Lighting so the sphere is visible.
        amb = AmbientLight("amb")
        amb.set_color(Vec4(0.3, 0.3, 0.3, 1))
        self.render.set_light(self.render.attach_new_node(amb))
        dirl = DirectionalLight("dir")
        dirl.set_color(Vec4(0.9, 0.9, 0.9, 1))
        dnp = self.render.attach_new_node(dirl)
        dnp.set_hpr(45, -45, 0)
        self.render.set_light(dnp)

        # Vessel markers + orbit lines.
        self.vessel_nps = []
        self.orbit_nps = []
        for _ in world.vessels:
            m = make_uv_sphere(0.03, 8, 12)
            m.reparent_to(self.render)
            m.set_color(1.0, 0.9, 0.2, 1.0)
            self.vessel_nps.append(m)
            self.orbit_nps.append(None)

        self._setup_input()
        self.task_mgr.add(self._update, "update")

    def _setup_input(self) -> None:
        self.accept("wheel_up", lambda: self.rig.zoom(0.8))
        self.accept("wheel_down", lambda: self.rig.zoom(1.25))
        self.accept("arrow_left", lambda: self.rig.orbit(-0.1, 0.0))
        self.accept("arrow_right", lambda: self.rig.orbit(0.1, 0.0))
        self.accept("arrow_up", lambda: self.rig.orbit(0.0, 0.1))
        self.accept("arrow_down", lambda: self.rig.orbit(0.0, -0.1))
        self.accept("period", self.clock.warp_up)  # ">" key
        self.accept("comma", self.clock.warp_down)  # "<" key

    def _rebuild_orbit(self, idx, vessel) -> None:
        elem = state_to_elements(vessel.state)
        pts = sample_orbit_points(elem, n=256)
        pts_render = [self.transform.to_render(p) for p in pts]
        if self.orbit_nps[idx] is not None:
            self.orbit_nps[idx].remove_node()
        node = build_orbit_node(pts_render)
        node.reparent_to(self.render)
        self.orbit_nps[idx] = node

    def _update(self, task):
        real_dt = _global_clock.get_dt()
        sim_dt = self.clock.advance(real_dt)
        self.world.step(sim_dt)

        # Focus origin on the first vessel; central body sits relative to it.
        focus = self.world.vessels[0].state.r if self.world.vessels else np.zeros(3)
        self.transform.set_origin(focus)

        # Re-scale + place the central body (at physics origin).
        cx, cy, cz = self.transform.to_render(np.zeros(3))
        self.central_np.set_pos(cx, cy, cz)
        body_render_radius = self.world.central.radius_m / self.transform.scale_m_per_unit
        self.central_np.set_scale(max(body_render_radius, 1e-3))

        for idx, vessel in enumerate(self.world.vessels):
            vx, vy, vz = self.transform.to_render(vessel.state.r)
            self.vessel_nps[idx].set_pos(vx, vy, vz)
            self._rebuild_orbit(idx, vessel)

        self.rig.apply()

        v0 = self.world.vessels[0]
        elem = state_to_elements(v0.state)
        rp = elem.a * (1 - elem.e)
        ra = elem.a * (1 + elem.e)
        try:
            period = elem.period_s
        except ValueError:
            period = float("nan")
        self.hud.update(
            sim_time_s=self.clock.sim_time_s,
            warp=self.clock.warp,
            altitude_m=v0.state.r_mag - self.world.central.radius_m,
            speed_mps=v0.state.v_mag,
            periapsis_m=rp - self.world.central.radius_m,
            apoapsis_m=ra - self.world.central.radius_m,
            period_s=period,
        )
        return task.cont

    def run_app(self) -> None:
        self.run()
