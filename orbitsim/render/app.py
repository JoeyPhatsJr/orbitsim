"""Panda3D ShowBase bootstrap and per-frame loop."""
import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectSlider import DirectSlider
from direct.gui.DirectFrame import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    ClockObject,
    AmbientLight,
    DirectionalLight,
    Vec4,
    TextNode,
    CardMaker,
)

from orbitsim.core.elements import state_to_elements
from orbitsim.core.maneuvers import ManeuverNode, predict_elements_after, apply_maneuver
from orbitsim.core.state import StateVector
from orbitsim.core.optimize import porkchop
from orbitsim.render.porkchop import render_porkchop_png
from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.orbit_lines import sample_orbit_points, build_orbit_node
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.hud import Hud

_global_clock = ClockObject.get_global_clock()


class OrbitApp(ShowBase):
    """Renders one central body + vessels with orbit lines; time-warpable."""

    def __init__(self, world, clock, solar_system: bool = False) -> None:
        super().__init__()
        self.world = world
        self.clock = clock
        self.solar_system = solar_system
        self.disable_mouse()
        self._sim_started = False
        self._title_nodes = []
        self._build_title_screen()

    # ------------------------------------------------------------------ title screen

    def _build_title_screen(self) -> None:
        """Show the start menu: title, a delta-V budget slider, and a Play button.

        The sim scene and update loop are not built until Play is clicked, so the
        chosen budget is applied to every vessel before flight begins.
        """
        backdrop = DirectFrame(
            frameColor=(0.02, 0.03, 0.08, 1.0),
            frameSize=(-2.0, 2.0, -1.2, 1.2),
            parent=self.aspect2d,
        )
        title = OnscreenText(
            text="ORBITAL MECHANICS SIM",
            pos=(0.0, 0.55),
            scale=0.13,
            fg=(0.85, 0.92, 1.0, 1.0),
            parent=self.aspect2d,
        )
        subtitle = OnscreenText(
            text="KSP, but the physics are real",
            pos=(0.0, 0.42),
            scale=0.05,
            fg=(0.6, 0.7, 0.85, 1.0),
            parent=self.aspect2d,
        )
        # delta-V budget control.
        default_budget = (
            self.world.vessels[0].delta_v_budget_mps if self.world.vessels else 2000.0
        )
        self._budget_label = OnscreenText(
            text="",
            pos=(0.0, 0.06),
            scale=0.06,
            fg=(1.0, 0.9, 0.4, 1.0),
            mayChange=True,
            parent=self.aspect2d,
        )
        self._budget_slider = DirectSlider(
            pos=(0.0, 0.0, -0.08),
            scale=0.6,
            range=(0.0, 10000.0),
            value=default_budget,
            pageSize=250.0,
            command=self._refresh_budget_label,
            parent=self.aspect2d,
        )
        hint = OnscreenText(
            text="delta-V budget  (drag to set)",
            pos=(0.0, -0.22),
            scale=0.045,
            fg=(0.6, 0.7, 0.85, 1.0),
            parent=self.aspect2d,
        )
        play = DirectButton(
            text="  PLAY  ",
            scale=0.1,
            pos=(0.0, 0.0, -0.45),
            command=self._on_play,
            parent=self.aspect2d,
        )
        self._title_nodes = [backdrop, title, subtitle, self._budget_label, self._budget_slider, hint, play]
        self._refresh_budget_label()

    def _refresh_budget_label(self) -> None:
        self._budget_label.setText(f"delta-V budget: {self._budget_slider['value']:,.0f} m/s")

    def _on_play(self) -> None:
        """Apply the chosen budget to all vessels, tear down the menu, start the sim."""
        budget = float(self._budget_slider["value"])
        for vessel in self.world.vessels:
            vessel.delta_v_budget_mps = budget
        for node in self._title_nodes:
            node.destroy() if hasattr(node, "destroy") else node.remove_node()
        self._title_nodes = []
        self._start_sim()

    # ------------------------------------------------------------------ sim scene

    def _start_sim(self) -> None:
        """Build the world scene (bodies, vessels, HUD, maneuver UI) and start updating."""
        if self._sim_started:
            return
        self._sim_started = True
        world = self.world

        self.transform = RenderTransform(origin_m=np.zeros(3), scale_m_per_unit=2.0e4)
        self.rig = CameraRig(self, self.transform)
        if self.solar_system:
            # The rig owns scale (= distance / 1000). Pull back to ~13 AU so the Sun,
            # inner planets, and Jupiter/Saturn all fit; the user can zoom from there.
            self.rig.set_distance(2.0e12)
        self.hud = Hud(self)

        # Central body sphere. In solar mode it is the Sun (constant on-screen size,
        # fullbright); in sandbox mode it is the planet, lit and scaled to real radius.
        self.central_np = make_uv_sphere(1.0, 24, 48)
        self.central_np.reparent_to(self.render)
        if self.solar_system:
            self.central_np.set_color(1.0, 0.85, 0.2, 1.0)
            self.central_np.set_light_off()
            self.central_np.set_scale(10.0)
        else:
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

        self._preview_np = None
        self._porkchop_card = None
        if self.solar_system:
            self._build_planets()
        else:
            # Maneuver editor (operates on vessel 0). The burn is applied at vessel 0's
            # *current* state (dt=0), so the executed orbit matches the live preview.
            self._build_maneuver_ui()

        self._setup_input()
        self.task_mgr.add(self._update, "update")

    # Distinct colours for the Sun + 8 planets (constant on-screen marker size).
    _PLANET_COLORS = {
        "Sun": (1.0, 0.85, 0.2, 1.0),
        "Mercury": (0.6, 0.6, 0.6, 1.0),
        "Venus": (0.9, 0.8, 0.5, 1.0),
        "Earth": (0.3, 0.5, 1.0, 1.0),
        "Mars": (0.9, 0.4, 0.2, 1.0),
        "Jupiter": (0.8, 0.7, 0.5, 1.0),
        "Saturn": (0.9, 0.8, 0.6, 1.0),
        "Uranus": (0.6, 0.85, 0.9, 1.0),
        "Neptune": (0.3, 0.4, 0.9, 1.0),
    }

    def _build_planets(self) -> None:
        """Create constant-size markers + labels for the Sun and 8 planets."""
        from orbitsim.core.bodies import PLANETS, SUN

        self._planet_bodies = [SUN] + list(PLANETS)
        self._planet_nps = []
        self._planet_labels = []
        for body in self._planet_bodies:
            marker = make_uv_sphere(1.0, 10, 14)
            marker.reparent_to(self.render)
            marker.set_color(*self._PLANET_COLORS.get(body.name, (0.8, 0.8, 0.8, 1.0)))
            marker.set_light_off()
            marker.set_scale(8.0 if body.name == "Sun" else 4.0)
            self._planet_nps.append(marker)
            # 3D billboard label on render (tracks the marker, faces the camera).
            tn = TextNode(f"label_{body.name}")
            tn.set_text(body.name)
            tn.set_text_color(0.8, 0.85, 1.0, 1.0)
            label = self.render.attach_new_node(tn)
            label.set_scale(15.0)
            label.set_billboard_point_eye()
            label.set_light_off()
            self._planet_labels.append(label)

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
        self.accept("p", self._toggle_porkchop)  # porkchop delta-V plot

    def _toggle_porkchop(self) -> None:
        """Build a porkchop plot (vessel 0 -> a higher circular orbit) and overlay it.

        Pressing 'p' again removes it. The departure axis spans a full synodic period
        so the low-delta-V basin (the "banana") is captured.
        """
        existing = getattr(self, "_porkchop_card", None)
        if existing is not None:
            existing.remove_node()
            self._porkchop_card = None
            return

        mu = self.world.central.mu
        dep = self.world.vessels[0].state
        r1 = dep.r_mag
        r2 = r1 * 2.0
        v2 = np.sqrt(mu / r2)
        arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, v2, 0.0]), mu=mu)

        w1 = np.sqrt(mu / r1**3)
        w2 = np.sqrt(mu / r2**3)
        t_syn = 2.0 * np.pi / abs(w1 - w2)
        t_hohmann = np.pi * np.sqrt(((r1 + r2) / 2.0) ** 3 / mu)

        dep_times = np.linspace(0.0, t_syn, 24)
        tof_grid = np.linspace(0.4 * t_hohmann, 1.6 * t_hohmann, 36)
        dv, _ = porkchop(dep, arr, dep_times, tof_grid, mu)

        png = render_porkchop_png(dv, dep_times, tof_grid, "porkchop.png")
        tex = self.loader.load_texture(png)
        cm = CardMaker("porkchop")
        cm.set_frame(-0.95, -0.15, 0.1, 0.85)  # top-right region of aspect2d
        card = self.aspect2d.attach_new_node(cm.generate())
        card.set_texture(tex)
        self._porkchop_card = card

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

        if self.solar_system:
            self._update_solar_system()
            return task.cont

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

    def _update_solar_system(self) -> None:
        """Place the Sun + planets at their DE440 positions for the current sim time."""
        from datetime import datetime, timedelta
        from orbitsim.core.ephemeris import body_state

        t = self.clock.sim_time_s
        self.transform.set_origin(np.zeros(3))  # heliocentric: Sun fixed at origin

        for body, marker, label in zip(self._planet_bodies, self._planet_nps, self._planet_labels):
            pos_m = np.zeros(3) if body.name == "Sun" else body_state(body.name.upper(), t, center="SUN").r
            rx, ry, rz = self.transform.to_render(pos_m)
            marker.set_pos(rx, ry, rz)
            label.set_pos(rx, ry, rz + 6.0)

        self.central_np.set_pos(*self.transform.to_render(np.zeros(3)))
        self.rig.apply()

        date = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=t)
        self.hud.text.setText(
            f"Solar system (JPL DE440)\n"
            f"Date: {date:%Y-%m-%d}\n"
            f"Warp: x{self.clock.warp:,.0f}\n"
            f"',' / '.' change warp"
        )

    def run_app(self) -> None:
        self.run()
