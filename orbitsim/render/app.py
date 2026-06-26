"""Panda3D ShowBase bootstrap and per-frame loop."""
import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectCheckButton import DirectCheckButton
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
from orbitsim.core.propagate import propagate_kepler
from orbitsim.core.moon import MOON_ORBIT, moon_state_at
from orbitsim.core.rendezvous import closest_approach
from orbitsim.core.attitude import (
    quat_from_axis_angle, quat_multiply, quat_normalize, quat_rotate_vector, nose_direction,
)
from orbitsim.core.state import StateVector
from orbitsim.core.optimize import porkchop
from orbitsim.render.porkchop import render_porkchop_png
from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.orbit_lines import sample_orbit_points, build_orbit_node, orbit_shape_changed
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.hud import Hud
from orbitsim.render.earth import build_earth, set_sun_dir
from orbitsim.render.keybind_overlay import KeybindOverlay, SANDBOX_BINDINGS, SOLAR_BINDINGS
from orbitsim.render.settings_panel import SettingsPanel
from orbitsim.sim.persistence import save_scenario, load_scenario
from orbitsim.render.skybox import build_starfield

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
        # Fuel-load control (delta-V is the derived readout via the rocket equation).
        default_fuel = self.world.vessels[0].fuel_mass_kg if self.world.vessels else 800.0
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
            range=(0.0, 20000.0),
            value=default_fuel,
            pageSize=100.0,
            command=self._refresh_budget_label,
            parent=self.aspect2d,
        )
        hint = OnscreenText(
            text="fuel load  (drag to set)",
            pos=(0.0, -0.22),
            scale=0.045,
            fg=(0.6, 0.7, 0.85, 1.0),
            parent=self.aspect2d,
        )
        self._unlimited_check = DirectCheckButton(
            text="Unlimited dV", scale=0.05, pos=(0.0, 0.0, -0.32),
            text_fg=(1, 1, 1, 1), boxPlacement="left", parent=self.aspect2d,
        )
        play = DirectButton(
            text="  PLAY  ",
            scale=0.1,
            pos=(0.0, 0.0, -0.45),
            command=self._on_play,
            parent=self.aspect2d,
        )
        self._title_nodes = [backdrop, title, subtitle, self._budget_label,
                             self._budget_slider, hint, self._unlimited_check, play]
        self._refresh_budget_label()

    def _refresh_budget_label(self) -> None:
        from orbitsim.core.flight import tsiolkovsky_dv

        fuel = float(self._budget_slider["value"])
        if self.world.vessels:
            v = self.world.vessels[0]
            dry, ve = v.dry_mass_kg, v.exhaust_velocity_mps
        else:
            dry, ve = 1000.0, 3000.0
        dv = tsiolkovsky_dv(ve, dry + fuel, dry) if fuel > 0 else 0.0
        self._budget_label.setText(f"Fuel: {fuel:,.0f} kg   (dV {dv:,.0f} m/s)")

    def _on_play(self) -> None:
        """Apply the chosen fuel load to all vessels, tear down the menu, start the sim."""
        fuel = float(self._budget_slider["value"])
        for vessel in self.world.vessels:
            vessel.fuel_mass_kg = fuel
        self._fuel_capacity = fuel if self.world.vessels else 0.0
        if bool(self._unlimited_check["indicatorValue"]):
            self._apply_unlimited(True)   # UI-free: _start_sim hasn't built the HUD yet
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
        bindings = SOLAR_BINDINGS if self.solar_system else SANDBOX_BINDINGS
        self.keybind_overlay = KeybindOverlay(self.aspect2d, bindings)
        self.settings_panel = SettingsPanel(
            self.aspect2d, self.hud.set_units,
            on_unlimited_toggle=self._set_unlimited_dv)
        self._build_warp_controls()

        # Central body. Solar mode: fullbright Sun marker. Sandbox: the textured,
        # day/night-shaded Earth (with an atmosphere shell), or a flat-blue fallback.
        self._atmo_np = None
        if self.solar_system:
            self.central_np = make_uv_sphere(1.0, 24, 48)
            self.central_np.reparent_to(self.render)
            self.central_np.set_color(1.0, 0.85, 0.2, 1.0)
            self.central_np.set_light_off()
            self.central_np.set_scale(10.0)
        else:
            self.central_np, self._atmo_np = build_earth(self)
            self.central_np.reparent_to(self.render)
            if self._atmo_np is not None:
                self._atmo_np.reparent_to(self.central_np)

        # Lighting: low ambient + a sun directional light (aimed at the real Sun each
        # frame in the sandbox). The shadered Earth lights itself; the light is for the
        # flat fallback and any other lit geometry.
        amb = AmbientLight("amb")
        amb.set_color(Vec4(0.3, 0.3, 0.3, 1) if self.solar_system else Vec4(0.12, 0.12, 0.15, 1))
        self.render.set_light(self.render.attach_new_node(amb))
        sun = DirectionalLight("sun")
        sun.set_color(Vec4(1.0, 1.0, 0.95, 1))
        self._sun_light_np = self.render.attach_new_node(sun)
        self._sun_light_np.set_hpr(45, -45, 0)
        self.render.set_light(self._sun_light_np)

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

        # Orbit frame: holds all Earth-centered orbit lines in world meters; repositioned +
        # rescaled once per frame so they track the floating origin without per-vertex rebuilds.
        self._orbit_frame = self.render.attach_new_node("orbit_frame")
        self._orbit_elem_cache = [None for _ in world.vessels]

        self._preview_np = None
        self._porkchop_card = None
        if self.solar_system:
            self._build_planets()
        else:
            # Maneuver editor (operates on vessel 0). The burn is applied at vessel 0's
            # *current* state (dt=0), so the executed orbit matches the live preview.
            self._build_maneuver_ui()
            self._fuel_capacity = (
                self.world.vessels[0].fuel_mass_kg if self.world.vessels else 0.0
            )
            from orbitsim.render.navball import Navball
            self.navball = Navball(self)

            # Targetable bodies (Moon today; ships later). Click a marker to select.
            from orbitsim.render.targets import MoonTarget
            self._targets = [MoonTarget()]
            self._target = None     # current Target or None
            self._ca_recompute_t = 0.0
            self._ca = None
            self._ca_traj = None
            self._ca_abs_epoch = 0.0
            self._ca_marker_ship = None
            self._ca_marker_moon = None
            self._moon_np = make_uv_sphere(1.0, 12, 16)
            self._moon_np.reparent_to(self.render)
            self._moon_np.set_color(0.7, 0.7, 0.72, 1.0)
            self._moon_np.set_light_off()
            self._moon_np.set_scale(7.0)
            self._moon_orbit_np = None      # built lazily in the update loop (scale-dependent)
            self._moon_orbit_scale = None
            self._target_text = OnscreenText(
                text="", pos=(0.08, -0.48), scale=0.045, fg=(1.0, 0.7, 0.4, 1),
                shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True, parent=self.a2dTopLeft,
            )

        # Star background (both modes): inertial, camera-centered, behind everything.
        self.starfield = build_starfield(self)
        self.starfield.reparent_to(self.render)

        self._setup_input()
        self.task_mgr.add(self._update, "update")

    def _update_starfield(self) -> None:
        """Keep the sky centered on the camera so stars sit at infinity (no parallax)."""
        if getattr(self, "starfield", None) is not None:
            self.starfield.set_pos(self.camera.get_pos(self.render))

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
            is_earth = body.name == "Earth"
            marker = make_uv_sphere(1.0, 12, 16, with_uv=is_earth)
            marker.reparent_to(self.render)
            marker.set_color(*self._PLANET_COLORS.get(body.name, (0.8, 0.8, 0.8, 1.0)))
            marker.set_light_off()
            marker.set_scale(8.0 if body.name == "Sun" else 4.0)
            if is_earth:
                from panda3d.core import Filename
                from orbitsim.render.textures import texture_path

                p = texture_path("earth_day")
                if p is not None:
                    marker.set_texture(self.loader.load_texture(Filename.from_os_specific(p)))
                    marker.set_color(1, 1, 1, 1)
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

    # Scheduled maneuver node.
    NODE_TIME_STEP_S = 30.0       # seconds per "Node -/+" press
    AUTO_WARP_LEAD_S = 5.0        # auto-warp-down when within this many real-seconds of the node
    EXECUTE_TOLERANCE_S = 2.0     # execute allowed only within this of the node epoch

    def _build_maneuver_ui(self) -> None:
        """Per-axis spring-loaded jog sliders for RTN delta-V, an Execute button, a readout.

        Each slider is a rate control: hold it right to increase that delta-V component
        (faster the further you push, exponentially), left to decrease. Releasing the
        mouse springs the thumb back to center and the value stops changing.
        """
        self._dv = {"pro": 0.0, "nrm": 0.0, "rad": 0.0}
        self._node_epoch_s = None      # absolute epoch of the scheduled node (None = none)
        self._node_marker_np = None
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
        # Scheduled-node controls: step time-to-node, jump to next apsis, clear.
        self._node_ttn_text = OnscreenText(
            text="", pos=(0.08, -0.42), scale=0.045, fg=(0.4, 1.0, 1.0, 1),
            shadow=(0, 0, 0, 1), align=TextNode.ALeft, mayChange=True, parent=self.a2dTopLeft,
        )
        node_btns = [
            ("Node -", lambda: self._step_node_time(-self.NODE_TIME_STEP_S)),
            ("Node +", lambda: self._step_node_time(self.NODE_TIME_STEP_S)),
            ("Next Pe", self._node_to_pe),
            ("Next Ap", self._node_to_ap),
            ("Clear", self._clear_node),
            ("Clear Tgt", self._clear_target),
        ]
        for i, (label, cmd) in enumerate(node_btns):
            DirectButton(text=label, scale=0.045, pos=(-0.95 + i * 0.34, 0.0, -0.06),
                         command=cmd, parent=self.a2dBottomRight)
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
        """Spring all jog thumbs back to center so the value stops changing on release.

        Also attempt a target pick: a left-click that didn't drag is a tap on a body."""
        self._try_pick_target()
        for slider in self._jog.values():
            slider["value"] = 0.0

    def _on_mouse1_down(self):
        mw = self.mouseWatcherNode
        self._mouse1_down_px = self._mouse_px() if (mw and mw.has_mouse()) else None

    def _mouse_px(self):
        """Current mouse position in pixels, or None if off-window."""
        mw = self.mouseWatcherNode
        if mw is None or not mw.has_mouse():
            return None
        w, h = self.win.get_x_size(), self.win.get_y_size()
        return ((mw.get_mouse_x() * 0.5 + 0.5) * w, (mw.get_mouse_y() * 0.5 + 0.5) * h)

    def _marker_px(self, world_r):
        """Project an inertial position to pixels, or None if behind the camera/off-lens."""
        from panda3d.core import Point2
        rp = self.transform.to_render(world_r)
        p = self.cam.get_relative_point(self.render, rp)
        proj = Point2()
        if not self.camLens.project(p, proj):
            return None
        w, h = self.win.get_x_size(), self.win.get_y_size()
        return ((proj.x * 0.5 + 0.5) * w, (proj.y * 0.5 + 0.5) * h)

    def _try_pick_target(self):
        """On a left-click tap (not a drag), select the nearest target marker."""
        from orbitsim.render.picking import nearest_marker
        down = getattr(self, "_mouse1_down_px", None)
        self._mouse1_down_px = None
        if down is None:
            return
        click = self._mouse_px()
        if click is None:
            return
        if abs(click[0] - down[0]) > 6.0 or abs(click[1] - down[1]) > 6.0:
            return  # was a drag, not a tap
        now = self.clock.sim_time_s
        proj = [self._marker_px(t.state_at(now).r) for t in self._targets]
        idxs = [i for i, p in enumerate(proj) if p is not None]
        hit = nearest_marker(click, [proj[i] for i in idxs], tol_px=22.0)
        if hit is not None:
            self._target = self._targets[idxs[hit]]

    def _current_node(self) -> ManeuverNode:
        """Build the node from the current dV values at the scheduled epoch (or now if none)."""
        epoch = (self._node_epoch_s if self._node_epoch_s is not None
                 else self.world.vessels[0].state.epoch_s)
        return ManeuverNode(
            epoch_s=epoch,
            dv_prograde_mps=self._dv["pro"],
            dv_normal_mps=self._dv["nrm"],
            dv_radial_mps=self._dv["rad"],
        )

    def _time_to_node(self):
        """Seconds until the scheduled node, or None if no node is scheduled."""
        if self._node_epoch_s is None:
            return None
        return self._node_epoch_s - self.clock.sim_time_s

    def _step_node_time(self, delta_s):
        """Nudge the node epoch by delta_s (creating one at now+delta if none), clamped >= now."""
        now = self.clock.sim_time_s
        base = self._node_epoch_s if self._node_epoch_s is not None else now
        self._node_epoch_s = max(now, base + delta_s)

    def _node_to_pe(self):
        from orbitsim.core.maneuvers import time_to_periapsis
        try:
            self._node_epoch_s = self.clock.sim_time_s + time_to_periapsis(self.world.vessels[0].state)
        except ValueError:
            pass  # unbound orbit: no apsis to target

    def _node_to_ap(self):
        from orbitsim.core.maneuvers import time_to_apoapsis
        try:
            self._node_epoch_s = self.clock.sim_time_s + time_to_apoapsis(self.world.vessels[0].state)
        except ValueError:
            pass

    def _clear_node(self):
        self._node_epoch_s = None
        if self._node_marker_np is not None:
            self._node_marker_np.remove_node()
            self._node_marker_np = None

    def _clear_target(self):
        """Deselect the current target; remove its closest-approach markers + readout."""
        self._target = None
        for attr in ("_ca_marker_ship", "_ca_marker_moon"):
            np_ = getattr(self, attr, None)
            if np_ is not None:
                np_.remove_node()
                setattr(self, attr, None)
        self._ca = None
        self._target_text.setText("Target: none")

    def _ca_marker(self, attr, color):
        """Lazily create/reuse a closest-approach marker NodePath."""
        np_ = getattr(self, attr, None)
        if np_ is None:
            np_ = make_uv_sphere(1.0, 8, 12)
            np_.reparent_to(self.render)
            np_.set_color(*color)
            np_.set_light_off()
            np_.set_scale(5.0)
            setattr(self, attr, np_)
        return np_

    def _refresh_readout(self) -> None:
        import math
        node = self._current_node()
        # One budget: the fuel-derived delta-V (rocket equation), shared with flight.
        budget = self.world.vessels[0].delta_v_remaining
        left = "∞" if not math.isfinite(budget) else f"{budget:,.0f} m/s"
        self._dv_readout.setText(
            f"Maneuver dV: {node.magnitude_mps:,.1f} m/s   (dV left {left})"
        )

    UNLIMITED_RESERVE_KG = 1000.0   # min propellant kept while unlimited (so thrust works at empty)

    def _apply_unlimited(self, on: bool) -> None:
        """Set the unlimited-dV flag on all vessels (no UI). When enabling with a
        ~empty tank, top fuel up to a reserve so continuous thrust still produces
        acceleration (integrate_powered needs propellant for its substep impulse;
        fuel never depletes under unlimited, so this is set once)."""
        for vessel in self.world.vessels:
            vessel.unlimited_dv = on
            if on and vessel.fuel_mass_kg < self.UNLIMITED_RESERVE_KG:
                vessel.fuel_mass_kg = self.UNLIMITED_RESERVE_KG

    def _set_unlimited_dv(self, on: bool) -> None:
        if self.solar_system or not self.world.vessels:
            return  # no flyable vessel (solar viewer) — nothing to toggle
        self._apply_unlimited(on)
        self._flash_message(f"Unlimited dV {'ON' if on else 'OFF'}")
        self._refresh_readout()

    def _toggle_unlimited_dv(self) -> None:
        cur = bool(self.world.vessels and self.world.vessels[0].unlimited_dv)
        self._set_unlimited_dv(not cur)

    def _execute_burn(self) -> None:
        from orbitsim.core.flight import fuel_burned_for_dv

        ttn = self._time_to_node()
        if ttn is not None and ttn > self.EXECUTE_TOLERANCE_S:
            return  # scheduled node not due yet
        if ttn is not None and ttn < -self.EXECUTE_TOLERANCE_S:
            self._clear_node()  # node already passed; discard rather than burn backward in time
            return
        v0 = self.world.vessels[0]
        node = self._current_node()
        dv = node.magnitude_mps
        if 0.0 < dv <= v0.delta_v_remaining:
            v0.state = apply_maneuver(v0.state, node)
            # Spend fuel for this impulse, so maneuver nodes and live thrust draw
            # from the same tank (no separate budget pool) — unless unlimited.
            if not v0.unlimited_dv:
                burned = fuel_burned_for_dv(v0.exhaust_velocity_mps, v0.mass_kg, dv)
                v0.fuel_mass_kg = max(0.0, v0.fuel_mass_kg - burned)
        self._clear_node()
        for axis in self._dv:
            self._dv[axis] = 0.0
            self._dv_value_text[axis].setText("+0")
        self._release_jogs()
        self._refresh_readout()

    MOUSE_ORBIT_SENS = 3.0     # radians per unit of normalized mouse travel

    def _setup_input(self) -> None:
        self.accept("wheel_up", lambda: self.rig.zoom(0.8))
        self.accept("wheel_down", lambda: self.rig.zoom(1.25))
        # Right-click + drag orbits the camera (both sandbox and solar modes).
        self._rmb_down = False
        self._last_mouse = None
        self.accept("mouse3", self._rmb, [True])
        self.accept("mouse3-up", self._rmb, [False])
        self.accept("arrow_left", lambda: self.rig.orbit(-0.1, 0.0))
        self.accept("arrow_right", lambda: self.rig.orbit(0.1, 0.0))
        self.accept("arrow_up", lambda: self.rig.orbit(0.0, 0.1))
        self.accept("arrow_down", lambda: self.rig.orbit(0.0, -0.1))
        self.accept("period", self._warp_up_guarded)  # ">" key (blocked while thrusting)
        self.accept("comma", self.clock.warp_down)  # "<" key
        self.accept("p", self._toggle_porkchop)  # porkchop delta-V plot
        self.accept("f1", self.keybind_overlay.toggle)  # keybind help overlay
        self.accept("escape", self.settings_panel.toggle)  # settings panel

        if not self.solar_system and self.world.vessels:
            self._keys = {k: False for k in ("w", "s", "a", "d", "q", "e", "shift", "control")}
            for k in list(self._keys):
                self.accept(k, self._set_key, [k, True])
                self.accept(f"{k}-up", self._set_key, [k, False])
            self._mouse1_down_px = None
            self.accept("mouse1", self._on_mouse1_down)   # tap (no drag) picks a target
            self.accept("z", self._throttle_full)
            self.accept("x", self._throttle_cut)
            self.accept("u", self._toggle_unlimited_dv)
            self.accept("t", self._toggle_sas)
            sas_keys = ["PROGRADE", "RETROGRADE", "NORMAL", "ANTINORMAL",
                        "RADIAL_IN", "RADIAL_OUT", "TARGET"]
            for i, mode in enumerate(sas_keys, start=1):
                self.accept(str(i), self._set_sas, [mode])
            self.accept("f5", self._quicksave)
            self.accept("f9", self._quickload)

    ROTATE_RATE_RADPS = 0.8       # manual pitch/yaw/roll rate
    THROTTLE_STEP = 0.5           # throttle change per second for shift/ctrl

    def _build_warp_controls(self) -> None:
        """Top-center on-screen warp control: slower / faster buttons + readout.
        (The ',' and '.' keys do the same.) Works in both sandbox and solar modes."""
        self._warp_readout = OnscreenText(
            text="", pos=(0.0, -0.09), scale=0.055, fg=(1.0, 1.0, 1.0, 1.0),
            shadow=(0, 0, 0, 1), mayChange=True, parent=self.a2dTopCenter,
        )
        self._warp_btns = [
            DirectButton(text="<<", scale=0.05, pos=(-0.28, 0.0, -0.085),
                         command=self.clock.warp_down, parent=self.a2dTopCenter),
            DirectButton(text=">>", scale=0.05, pos=(0.28, 0.0, -0.085),
                         command=self._warp_up_guarded, parent=self.a2dTopCenter),
        ]
        self._update_warp_readout()

    def _update_warp_readout(self) -> None:
        if getattr(self, "_warp_readout", None) is not None:
            locked = not self.solar_system and self.world.any_thrusting()
            suffix = "  (LOCKED)" if locked else ""
            self._warp_readout.setText(f"Warp  x{self.clock.warp:,.0f}{suffix}")

    def _warp_up_guarded(self):
        if not self.world.any_thrusting():
            self.clock.warp_up()

    def _rmb(self, down):
        self._rmb_down = down
        self._last_mouse = None

    def _apply_mouse_orbit(self):
        """Orbit the camera while the right mouse button is held and dragged."""
        mw = self.mouseWatcherNode
        if mw is None or not (self._rmb_down and mw.has_mouse()):
            self._last_mouse = None
            return
        x, y = mw.get_mouse_x(), mw.get_mouse_y()
        if self._last_mouse is not None:
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self.rig.orbit(dx * self.MOUSE_ORBIT_SENS, dy * self.MOUSE_ORBIT_SENS)
        self._last_mouse = (x, y)

    def _set_key(self, key, down):
        self._keys[key] = down

    def _throttle_full(self):
        self.world.vessels[0].throttle = 1.0

    def _throttle_cut(self):
        self.world.vessels[0].throttle = 0.0

    def _toggle_sas(self):
        v = self.world.vessels[0]
        v.sas_mode = "STABILITY" if v.sas_mode == "OFF" else "OFF"

    def _set_sas(self, mode):
        self.world.vessels[0].sas_mode = mode

    QUICKSAVE_PATH = "saves/quicksave.json"

    def _flash_message(self, text: str) -> None:
        """Transient on-screen user feedback (toast)."""
        self.hud.flash(text)

    def _quicksave(self) -> None:
        """F5: write the current sandbox world + clock to the quicksave slot."""
        try:
            save_scenario(self.world, self.clock, self.QUICKSAVE_PATH)
            self._flash_message("Quicksaved")
        except (OSError, ValueError) as exc:
            self._flash_message(f"Quicksave failed: {exc}")

    def _quickload(self) -> None:
        """F9: restore the quicksave in place onto the live vessel + clock.

        Mutating the existing vessel (rather than swapping self.world) keeps the
        Panda3D scene graph intact. Sandbox-only: one vessel, fixed central body.
        """
        try:
            world, clock = load_scenario(self.QUICKSAVE_PATH)
        except (OSError, ValueError) as exc:
            self._flash_message(f"Quickload failed: {exc}")
            return
        src = world.vessels[0]
        dst = self.world.vessels[0]
        dst.state = src.state
        dst.dry_mass_kg = src.dry_mass_kg
        dst.fuel_mass_kg = src.fuel_mass_kg
        dst.max_thrust_n = src.max_thrust_n
        dst.exhaust_velocity_mps = src.exhaust_velocity_mps
        dst.max_turn_rate_radps = src.max_turn_rate_radps
        dst.throttle = src.throttle
        dst.sas_mode = src.sas_mode
        dst.orientation = src.orientation
        dst.nodes[:] = src.nodes
        self.clock.sim_time_s = clock.sim_time_s
        self.clock.warp = clock.warp
        self._flash_message("Quickloaded")

    def _apply_flight_input(self, dt):
        """Manual throttle trim + rotation from held keys (sandbox flight)."""
        if self.solar_system or not self.world.vessels:
            return
        v = self.world.vessels[0]
        k = self._keys
        if k["shift"]:
            v.throttle = min(1.0, v.throttle + self.THROTTLE_STEP * dt)
        if k["control"]:
            v.throttle = max(0.0, v.throttle - self.THROTTLE_STEP * dt)
        ax = np.zeros(3)
        if k["w"]:
            ax = ax + np.array([1.0, 0.0, 0.0])    # pitch
        if k["s"]:
            ax = ax + np.array([-1.0, 0.0, 0.0])
        if k["a"]:
            ax = ax + np.array([0.0, -1.0, 0.0])   # yaw left
        if k["d"]:
            ax = ax + np.array([0.0, 1.0, 0.0])
        if k["q"]:
            ax = ax + np.array([0.0, 0.0, 1.0])    # roll
        if k["e"]:
            ax = ax + np.array([0.0, 0.0, -1.0])
        if np.linalg.norm(ax) > 0.0:
            v.sas_mode = "OFF"                      # taking manual control
            world_axis = quat_rotate_vector(v.orientation, ax)
            dq = quat_from_axis_angle(world_axis, self.ROTATE_RATE_RADPS * dt)
            v.orientation = quat_normalize(quat_multiply(dq, v.orientation))

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
        scale = self.transform.scale_m_per_unit
        cached = self._orbit_elem_cache[idx]
        if cached is not None and cached[1] == scale and not orbit_shape_changed(cached[0], elem):
            return  # coasting at the same zoom: keep the cached geometry under the orbit frame
        self._orbit_elem_cache[idx] = (elem, scale)
        pts = [tuple(p / scale) for p in sample_orbit_points(elem, n=256)]  # render units
        if self.orbit_nps[idx] is not None:
            self.orbit_nps[idx].remove_node()
        node = build_orbit_node(pts)
        node.reparent_to(self._orbit_frame)
        self.orbit_nps[idx] = node

    def _update(self, task):
        real_dt = _global_clock.get_dt()

        if self.solar_system:
            self.clock.advance(real_dt)
            self._update_solar_system()
            return task.cont

        # Flight input, then lock warp to 1x while thrusting (no RK4 through warp).
        self._apply_flight_input(real_dt)
        if self.world.any_thrusting() and self.clock.warp != 1.0:
            self.clock.warp = 1.0
        # Feed the current target's position to the TARGET/ANTITARGET SAS hold.
        target_pos = (self._target.state_at(self.clock.sim_time_s).r
                      if self._target is not None else None)
        for v in self.world.vessels:
            v.sas_target_pos = target_pos
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

        # Orbit frame tracks the floating origin by translation only (geometry is baked in
        # render units = meters / scale, so a vertex p/scale renders at to_render(0) + p/scale =
        # to_render(p)). Translate-only avoids a tiny node scale (which Panda flags as singular).
        self._orbit_frame.set_pos(cx, cy, cz)  # = to_render(0)
        scale = self.transform.scale_m_per_unit
        if self._moon_orbit_scale != scale:  # rebuild the fixed Moon ring on zoom
            self._moon_orbit_scale = scale
            if self._moon_orbit_np is not None:
                self._moon_orbit_np.remove_node()
            moon_pts = [tuple(p / scale) for p in sample_orbit_points(MOON_ORBIT, n=256)]
            self._moon_orbit_np = build_orbit_node(moon_pts, color=(0.5, 0.5, 0.55, 1.0))
            self._moon_orbit_np.reparent_to(self._orbit_frame)

        # Real Sun direction (Earth->Sun) drives the day/night terminator + sun light.
        try:
            from orbitsim.core.ephemeris import body_state

            sun_r = body_state("SUN", self.clock.sim_time_s, center="EARTH").r
            sun_dir = np.asarray(sun_r, dtype=float)
            n = np.linalg.norm(sun_dir)
            if n > 0:
                sun_dir = sun_dir / n
                set_sun_dir(self.central_np, tuple(sun_dir))
                self._sun_light_np.set_pos(float(sun_dir[0]) * 1000.0,
                                           float(sun_dir[1]) * 1000.0,
                                           float(sun_dir[2]) * 1000.0)
                self._sun_light_np.look_at(0, 0, 0)
        except Exception:
            pass

        for idx, vessel in enumerate(self.world.vessels):
            vx, vy, vz = self.transform.to_render(vessel.state.r)
            self.vessel_nps[idx].set_pos(vx, vy, vz)
            self._rebuild_orbit(idx, vessel)

        # Scheduled maneuver node: preview (magenta), node marker (cyan), auto-warp-down,
        # readout, and a vessel.nodes mirror so quicksave persists the plan.
        node = self._current_node()
        v0 = self.world.vessels[0]
        ttn = self._time_to_node()
        # Discard a node that slipped well past its epoch unexecuted, so it can't linger
        # invisibly or later execute backward in time.
        if ttn is not None and ttn < -self.EXECUTE_TOLERANCE_S:
            self._clear_node()
            ttn = None
            node = self._current_node()
        v0.nodes = [node] if (self._node_epoch_s is not None or node.magnitude_mps > 0.0) else []
        # Post-burn orbit preview.
        if node.magnitude_mps > 0.0:
            pred = predict_elements_after(v0.state, node)
            ppts = [tuple(p / self.transform.scale_m_per_unit)
                    for p in sample_orbit_points(pred, n=256)]
            if self._preview_np is not None:
                self._preview_np.remove_node()
            self._preview_np = build_orbit_node(ppts, color=(1.0, 0.2, 1.0, 1.0))
            self._preview_np.reparent_to(self._orbit_frame)
        elif self._preview_np is not None:
            self._preview_np.remove_node()
            self._preview_np = None
        # Node marker at the node's predicted position on the orbit (held at the vessel
        # through the brief execute window once the node is due).
        if self._node_epoch_s is not None and ttn is not None and ttn >= -self.EXECUTE_TOLERANCE_S:
            npos = propagate_kepler(v0.state, max(0.0, ttn)).r
            mx, my, mz = self.transform.to_render(npos)
            if self._node_marker_np is None:
                self._node_marker_np = make_uv_sphere(1.0, 8, 12)
                self._node_marker_np.reparent_to(self.render)
                self._node_marker_np.set_color(0.3, 1.0, 1.0, 1.0)
                self._node_marker_np.set_light_off()
                self._node_marker_np.set_scale(6.0)
            self._node_marker_np.set_pos(mx, my, mz)
        elif self._node_marker_np is not None:
            self._node_marker_np.remove_node()
            self._node_marker_np = None
        # Auto-warp-down as the node nears (never warps up).
        if ttn is not None and 0.0 < ttn <= self.AUTO_WARP_LEAD_S * self.clock.warp and self.clock.warp > 1.0:
            self.clock.warp_down()
        # Pending-node readout (single node); shows "DUE" once within the execute window.
        if ttn is not None and ttn >= -self.EXECUTE_TOLERANCE_S:
            if ttn <= self.EXECUTE_TOLERANCE_S:
                label = "DUE — press Execute"
            else:
                mm, ss = divmod(int(ttn), 60)
                label = f"in T-{mm:02d}:{ss:02d}"
            self._node_ttn_text.setText(f"Node {label}   dV {node.magnitude_mps:,.1f} m/s")
        else:
            self._node_ttn_text.setText("")

        # Moon position this frame.
        moon_now = moon_state_at(self.clock.sim_time_s)
        self._moon_np.set_pos(*self.transform.to_render(moon_now.r))
        # Closest approach to the current target (throttled recompute). Both the ship
        # trajectory and the target are referenced to the same base epoch (the node epoch
        # when a burn is planned, else now) so they are compared at matching absolute times;
        # the absolute CA epoch is cached so the markers hold steady between recomputes (warp).
        if self._target is not None:
            import time as _time
            now_real = _time.monotonic()
            if self._ca is None or now_real - self._ca_recompute_t > 0.5:
                self._ca_recompute_t = now_real
                if self._node_epoch_s is not None and node.magnitude_mps > 0.0:
                    base_epoch = self._node_epoch_s
                    traj = apply_maneuver(v0.state, node)
                else:
                    base_epoch = self.clock.sim_time_s
                    traj = v0.state
                try:
                    period = state_to_elements(traj).period_s
                except ValueError:
                    period = 14.0 * 86400.0
                window = min(period, 14.0 * 86400.0)
                self._ca = closest_approach(
                    traj, self._target.state_at(base_epoch), window_s=window, coarse_samples=720)
                self._ca_traj = traj
                self._ca_abs_epoch = base_epoch + self._ca.t_ca_s
            ca = self._ca
            ship_at = propagate_kepler(self._ca_traj, ca.t_ca_s).r
            tgt_at = self._target.state_at(self._ca_abs_epoch).r
            self._ca_marker("_ca_marker_ship", (1.0, 0.5, 0.2, 1.0)).set_pos(
                *self.transform.to_render(ship_at))
            self._ca_marker("_ca_marker_moon", (1.0, 0.8, 0.3, 1.0)).set_pos(
                *self.transform.to_render(tgt_at))
            countdown = max(0.0, self._ca_abs_epoch - self.clock.sim_time_s)
            mm, ss = divmod(int(countdown), 60)
            self._target_text.setText(
                f"Target: {self._target.name}   CA T-{mm:02d}:{ss:02d}"
                f"   sep {ca.separation_m / 1000:,.0f} km   rel {ca.rel_speed_mps:,.0f} m/s")

        self._apply_mouse_orbit()
        self._update_starfield()
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
            inclination_rad=elem.i,
        )
        g_local = self.world.central.mu / max(v0.state.r_mag, 1.0) ** 2
        twr = (v0.max_thrust_n / (v0.mass_kg * g_local)) if v0.mass_kg > 0 else 0.0
        cap = getattr(self, "_fuel_capacity", 0.0)
        fuel_frac = v0.fuel_mass_kg / cap if cap > 0 else 0.0
        self.hud.update_flight(
            throttle=v0.throttle,
            fuel_kg=v0.fuel_mass_kg,
            fuel_frac=fuel_frac,
            mass_kg=v0.mass_kg,
            thrust_n=v0.max_thrust_n,
            twr=twr,
            dv_remaining=v0.delta_v_remaining,
            warp_locked=self.world.any_thrusting(),
        )
        self.navball.update(orientation_q=v0.orientation, state=v0.state, target_pos=target_pos)
        self._update_warp_readout()
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
        self._apply_mouse_orbit()
        self._update_starfield()
        self.rig.apply()

        self._update_warp_readout()
        date = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=t)
        self.hud.text.setText(
            f"Solar system (JPL DE440)\n"
            f"Date: {date:%Y-%m-%d}\n"
            f"',' / '.' or the buttons change warp"
        )

    def run_app(self) -> None:
        self.run()
