"""Panda3D ShowBase bootstrap and per-frame loop."""
import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectSlider import DirectSlider
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import ClockObject, AmbientLight, DirectionalLight, Vec4, TextNode

from orbitsim.core.elements import state_to_elements
from orbitsim.core.maneuvers import ManeuverNode, predict_elements_after, apply_maneuver
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
            m = make_uv_sphere(1.0, 8, 12)
            m.reparent_to(self.render)
            m.set_color(1.0, 0.9, 0.2, 1.0)
            m.set_light_off()  # fullbright marker so it is always visible
            # The camera always sits RENDER_UNITS_ACROSS_VIEW (1000) units from the
            # focus, so a fixed render-size marker keeps a constant on-screen size
            # at every zoom level.
            m.set_scale(8.0)
            self.vessel_nps.append(m)
            self.orbit_nps.append(None)

        # Maneuver editor (operates on vessel 0). The burn is applied at vessel 0's
        # *current* state (dt=0), so the executed orbit matches the live preview exactly.
        self._preview_np = None
        self._build_maneuver_ui()

        self._setup_input()
        self.task_mgr.add(self._update, "update")

    # Spring-loaded "jog" sliders: displacement from center sets the *rate* of change.
    JOG_MAX_RATE_MPS = 400.0  # delta-V change per second at full deflection
    JOG_CURVE = 4.0  # exponential steepness; higher = gentler near center, sharper at edges
    JOG_DEADZONE = 0.02  # |value| below this contributes no change

    def _build_maneuver_ui(self) -> None:
        """Per-axis spring-loaded jog sliders for RTN delta-V, an Execute button, a readout.

        Each slider is a rate control: hold it right to increase that delta-V component
        (faster the further you push, exponentially), left to decrease. Releasing the
        mouse springs the thumb back to center and the value stops changing.
        """
        self._dv = {"pro": 0.0, "nrm": 0.0, "rad": 0.0}
        self._dv_value_text = {}
        self._jog = {}
        rows = (("pro", "Prograde", 0.40), ("nrm", "Normal", 0.28), ("rad", "Radial", 0.16))
        for axis, label, z in rows:
            OnscreenText(
                text=label,
                pos=(-1.22, z - 0.015),
                scale=0.045,
                fg=(1, 1, 1, 1),
                align=TextNode.ALeft,
                parent=self.a2dBottomRight,
            )
            self._jog[axis] = DirectSlider(
                pos=(-0.62, 0.0, z),
                scale=0.34,
                range=(-1.0, 1.0),
                value=0.0,
                pageSize=0.25,
                parent=self.a2dBottomRight,
            )
            self._dv_value_text[axis] = OnscreenText(
                text="+0",
                pos=(-0.14, z - 0.015),
                scale=0.045,
                fg=(1.0, 0.9, 0.4, 1),
                align=TextNode.ALeft,
                mayChange=True,
                parent=self.a2dBottomRight,
            )
        self._exec_btn = DirectButton(
            text="Execute Burn",
            scale=0.05,
            pos=(-0.5, 0.0, 0.04),
            command=self._execute_burn,
            parent=self.a2dBottomRight,
        )
        self._dv_readout = OnscreenText(
            text="",
            pos=(0.08, -0.36),
            scale=0.045,
            fg=(1.0, 0.4, 1.0, 1),
            shadow=(0, 0, 0, 1),
            align=TextNode.ALeft,
            mayChange=True,
            parent=self.a2dTopLeft,
        )
        # Releasing the mouse springs every jog slider back to its center (zero rate).
        self.accept("mouse1-up", self._release_jogs)
        self._refresh_readout()

    def _jog_rate_mps_per_s(self, value: float) -> float:
        """Map a jog displacement in [-1, 1] to a signed delta-V rate [m/s per s].

        Zero at center, ``JOG_MAX_RATE_MPS`` at full deflection, exponential in between
        so small displacements give fine control and large ones change quickly.
        """
        mag = abs(value)
        if mag <= self.JOG_DEADZONE:
            return 0.0
        shaped = (np.exp(self.JOG_CURVE * mag) - 1.0) / (np.exp(self.JOG_CURVE) - 1.0)
        return float(np.copysign(self.JOG_MAX_RATE_MPS * shaped, value))

    def _apply_jogs(self, real_dt_s: float) -> None:
        """Integrate each jog slider's rate into its delta-V component for this frame."""
        changed = False
        for axis, slider in self._jog.items():
            rate = self._jog_rate_mps_per_s(slider["value"])
            if rate != 0.0:
                self._dv[axis] += rate * real_dt_s
                self._dv_value_text[axis].setText(f"{self._dv[axis]:+.0f}")
                changed = True
        if changed:
            self._refresh_readout()

    def _release_jogs(self) -> None:
        """Spring all jog thumbs back to center so the value stops changing on release."""
        for slider in self._jog.values():
            slider["value"] = 0.0

    def _current_node(self) -> ManeuverNode:
        """Build the node from the current dV values, burning at vessel 0's current epoch."""
        return ManeuverNode(
            epoch_s=self.world.vessels[0].state.epoch_s,
            dv_prograde_mps=self._dv["pro"],
            dv_normal_mps=self._dv["nrm"],
            dv_radial_mps=self._dv["rad"],
        )

    def _refresh_readout(self) -> None:
        node = self._current_node()
        budget = self.world.vessels[0].delta_v_budget_mps
        self._dv_readout.setText(
            f"Maneuver dV: {node.magnitude_mps:,.1f} m/s   (budget {budget:,.0f} m/s)"
        )

    def _execute_burn(self) -> None:
        v0 = self.world.vessels[0]
        node = self._current_node()
        if 0.0 < node.magnitude_mps <= v0.delta_v_budget_mps:
            v0.state = apply_maneuver(v0.state, node)
            v0.delta_v_budget_mps -= node.magnitude_mps
        for axis in self._dv:
            self._dv[axis] = 0.0
            self._dv_value_text[axis].setText("+0")
        self._release_jogs()
        self._refresh_readout()

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

        # Jog sliders accumulate delta-V at a real-time (warp-independent) rate.
        self._apply_jogs(real_dt)

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

        # Live maneuver preview (magenta) for vessel 0.
        node = self._current_node()
        if node.magnitude_mps > 0.0:
            pred = predict_elements_after(self.world.vessels[0].state, node)
            ppts = [self.transform.to_render(p) for p in sample_orbit_points(pred, n=256)]
            if self._preview_np is not None:
                self._preview_np.remove_node()
            self._preview_np = build_orbit_node(ppts, color=(1.0, 0.2, 1.0, 1.0))
            self._preview_np.reparent_to(self.render)
        elif self._preview_np is not None:
            self._preview_np.remove_node()
            self._preview_np = None

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
