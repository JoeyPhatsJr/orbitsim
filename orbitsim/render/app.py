"""Panda3D ShowBase bootstrap and per-frame loop."""
from contextlib import nullcontext

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
    Filename,
    Vec4,
    TextNode,
    CardMaker,
)

from orbitsim.core.elements import state_to_elements
from orbitsim.core.maneuvers import ManeuverNode, apply_maneuver
from orbitsim.core.moon import MOON_ORBIT, moon_state_at
from orbitsim.core.rendezvous import closest_approach
from orbitsim.core.attitude import (
    quat_from_axis_angle, quat_multiply, quat_normalize, quat_rotate_vector, nose_direction,
)
from orbitsim.core.state import StateVector
from orbitsim.core.optimize import porkchop
from orbitsim.render.porkchop import render_porkchop_png
from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.geometry import make_uv_sphere, make_ring
from orbitsim.core.nbody import MOON_SOI_M
from orbitsim.core.constants import MU_SUN
from orbitsim.core.planets import (
    EARTH_SOI_M, MERCURY_SOI_M, VENUS_SOI_M, MARS_SOI_M,
    JUPITER_SOI_M, SATURN_SOI_M, URANUS_SOI_M, NEPTUNE_SOI_M,
    A_MERCURY, A_VENUS, A_EARTH, A_MARS,
    A_JUPITER, A_SATURN, A_URANUS, A_NEPTUNE,
)
from orbitsim.render.world_markers import distance_fade
from orbitsim.render.orbit_lines import (
    MANEUVER_COLOR,
    REFERENCE_ORBIT_COLOR,
    build_orbit_node,
    sample_orbit_points,
    sample_relative_orbit_points,
)
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.hud import Hud
from orbitsim.render.earth import build_earth, set_sun_dir
from orbitsim.render.keybind_overlay import KeybindOverlay, SANDBOX_BINDINGS, SOLAR_BINDINGS
from orbitsim.render.settings_panel import SettingsPanel
from orbitsim.sim.persistence import save_scenario, load_scenario
from orbitsim.render.skybox import build_starfield

_global_clock = ClockObject.get_global_clock()


def _fmt_dist(meters: float) -> str:
    AU = 1.496e11
    if meters >= 0.01 * AU:
        return f"{meters / AU:.2f} AU"
    km = meters / 1000.0
    if km >= 1e6:
        return f"{km / 1e6:.2f} M km"
    return f"{km:,.0f} km"


def _fmt_countdown(seconds: float) -> str:
    t = int(seconds)
    if t >= 86400:
        d, rem = divmod(t, 86400)
        h = rem // 3600
        return f"{d}d {h:02d}h"
    if t >= 3600:
        h, rem = divmod(t, 3600)
        m = rem // 60
        return f"{h}h {m:02d}m"
    m, s = divmod(t, 60)
    return f"{m:02d}:{s:02d}"


def _maneuver_preview_key(node, scheduled_epoch_s):
    """Stable preview identity; burn-now epochs move with the vessel and are ignored."""
    return (
        node.dv_prograde_mps,
        node.dv_normal_mps,
        node.dv_radial_mps,
        scheduled_epoch_s,
    )


def _trajectory_horizon_s(state, solar_system: bool) -> float:
    """Prediction horizon that keeps escape trajectories continuous across Earth SOI."""
    if not solar_system:
        return 7.0 * 86400.0
    from orbitsim.core.constants import MU_EARTH
    from orbitsim.core.planets import EARTH_SOI_M

    radius = float(np.linalg.norm(state.r))
    energy = 0.5 * float(np.dot(state.v, state.v)) - MU_EARTH / radius
    if energy >= 0.0:
        return 400.0 * 86400.0
    semi_major = -MU_EARTH / (2.0 * energy)
    eccentricity_vector = (
        ((float(np.dot(state.v, state.v)) - MU_EARTH / radius) * state.r
         - float(np.dot(state.r, state.v)) * state.v)
        / MU_EARTH
    )
    apoapsis = semi_major * (1.0 + float(np.linalg.norm(eccentricity_vector)))
    if apoapsis >= 0.8 * EARTH_SOI_M:
        return 400.0 * 86400.0
    period = 2.0 * np.pi * np.sqrt(semi_major**3 / MU_EARTH)
    return min(2.0 * period, 30.0 * 86400.0)


def _coast_chunks(duration_s: float, max_chunk_s: float = 86400.0) -> list[float]:
    """Split a teleport coast into bounded positive steps that land exactly on its end."""
    if duration_s <= 0.0:
        return []
    count = max(1, int(np.ceil(duration_s / max_chunk_s)))
    chunk = duration_s / count
    return [chunk] * count


def _localize_polyline(points):
    """Return (local points, origin) with float64 subtraction before rendering."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("polyline points must have shape (n, 3) with n > 0")
    origin = pts[0].copy()
    return pts - origin, origin


class OrbitApp(ShowBase):
    """Renders one central body + vessels with orbit lines; time-warpable."""

    def __init__(self, world, clock, solar_system: bool = False) -> None:
        super().__init__()
        from orbitsim.render.ui.theme import install_default_font
        install_default_font(self)
        self.world = world
        self.clock = clock
        self.solar_system = solar_system
        self.disable_mouse()
        self._sim_started = False
        self._title_nodes = []
        self._ui_viewport = None
        from orbitsim.render.ui import OperationController, PanelManager
        self.panel_manager = PanelManager()
        self.operation = OperationController()
        from orbitsim.render.ui.performance import FrameMeter
        self.frame_meter = FrameMeter()
        self._build_title_screen()

    def destroy(self) -> None:
        for attr in ("_preview_executor", "_trajectory_executor", "_planning_executor"):
            executor = getattr(self, attr, None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                setattr(self, attr, None)
        super().destroy()

    # ------------------------------------------------------------------ title screen

    def _build_title_screen(self) -> None:
        """Show the start menu: title, a delta-V budget slider, and a Play button.

        The sim scene and update loop are not built until Play is clicked, so the
        chosen budget is applied to every vessel before flight begins.
        """
        from orbitsim.render.ui.theme import THEME, button_options
        backdrop = DirectFrame(
            frameColor=(0.012, 0.025, 0.040, 1.0),
            frameSize=(-2.0, 2.0, -1.2, 1.2),
            parent=self.aspect2d,
        )
        title = OnscreenText(
            text="ORBITSIM",
            pos=(0.0, 0.55),
            scale=0.13,
            fg=THEME.text,
            parent=self.aspect2d,
        )
        subtitle = OnscreenText(
            text="REAL GRAVITY  /  REAL ORBITS  /  YOUR MISSION",
            pos=(0.0, 0.42),
            scale=0.05,
            fg=THEME.cyan,
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
            text=("  OPEN SOLAR VIEWER  " if self.solar_system else "  LAUNCH SANDBOX  "),
            scale=0.1,
            pos=(0.0, 0.0, -0.45),
            command=self._on_play,
            parent=self.aspect2d,
            **button_options(),
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
        want_unlimited = bool(self._unlimited_check["indicatorValue"])
        for node in self._title_nodes:
            node.destroy() if hasattr(node, "destroy") else node.remove_node()
        self._title_nodes = []
        self._start_sim()                # builds the HUD + settings panel
        if want_unlimited:
            self._set_unlimited_dv(True)  # now applies flag + syncs panel + readout

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
        from orbitsim.render.ui.widgets import OperationOverlay
        self.operation_overlay = OperationOverlay(self, self._cancel_operation)
        bindings = SOLAR_BINDINGS if self.solar_system else SANDBOX_BINDINGS
        self.keybind_overlay = KeybindOverlay(self.aspect2d, bindings)
        self.settings_panel = SettingsPanel(
            self.aspect2d, self.hud.set_units,
            on_unlimited_toggle=self._set_unlimited_dv,
            enable_unlimited=not self.solar_system)
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

        # Ship view: a true-scale, lit, oriented model of vessel 0 that cross-fades
        # in from the constant-size marker as the camera zooms close. (ship_model.py)
        self._ship_model_np = None
        if not self.solar_system and self.world.vessels:
            from orbitsim.render.ship_model import build_ship_model
            from panda3d.core import TransparencyAttrib
            self._ship_model_np = build_ship_model()
            self._ship_model_np.reparent_to(self.render)
            self._ship_model_np.hide()  # shown only when model_alpha > 0
            # Marker fades via alpha; enable transparency on vessel 0's marker.
            self.vessel_nps[0].set_transparency(TransparencyAttrib.M_alpha)
            # Exhaust plume (parented to the model: inherits its orient + scale).
            from orbitsim.render.ship_model import build_plume
            self._plume_np = build_plume()
            self._plume_np.reparent_to(self._ship_model_np)
            self._plume_np.hide()

        # Orbit frame: holds all Earth-centered orbit lines in world meters; repositioned +
        # rescaled once per frame so they track the floating origin without per-vertex rebuilds.
        self._orbit_frame = self.render.attach_new_node("orbit_frame")
        self._traj_world_cache = [None for _ in world.vessels]
        self._traj_epoch_cache = [None for _ in world.vessels]
        self._traj_future = [None for _ in world.vessels]
        self._traj_last_submit_t = [float("-inf") for _ in world.vessels]
        self._traj_last_complete_t = [float("-inf") for _ in world.vessels]
        self._traj_render_scale = [None for _ in world.vessels]
        self._traj_render_origin = [None for _ in world.vessels]
        self._traj_encounters = [None for _ in world.vessels]

        self._preview_np = None
        self._preview_origin = None
        self._preview_submit_t = 0.0
        self._preview_future = None
        self._preview_future_key = None
        self._preview_encounters = []
        if not self.solar_system:
            from concurrent.futures import ThreadPoolExecutor
            self._preview_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="orbit-preview"
            )
            self._trajectory_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="trajectory"
            )
            self._planning_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="planner"
            )
            self._porkchop_future = None
            self._porkchop_request = 0
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
            from orbitsim.render.sas_panel import SasPanel, VelocityReadout
            self.sas_panel = SasPanel(
                self, on_set_mode=self._set_sas, on_toggle=self._toggle_sas
            )
            self.vel_readout = VelocityReadout(self, lambda: self.hud.units)
            from orbitsim.render.ui.widgets import PanelDock
            self.panel_dock = PanelDock(self, [
                ("ORBIT", self.hud.toggle_orbit),
                ("VESSEL", self.hud.toggle_vessel),
                ("MANEUVER", self._toggle_maneuver_drawer),
            ])
            self._maneuver_frame.hide()  # clean-flight default; MNV dock button opens it

            # Targetable bodies. Click a marker to select.
            from orbitsim.render.targets import MoonTarget, LagrangePointTarget, PlanetTarget
            from orbitsim.core.nbody import (
                _csun, _cmercury, _cvenus, _cmars,
                _cjupiter, _csaturn, _curanus, _cneptune,
            )
            self._targets = [MoonTarget()] + [
                LagrangePointTarget(n, n) for n in ("L1", "L2", "L3", "L4", "L5")
            ] + [
                PlanetTarget("Sun", _csun, "SUN"),
                PlanetTarget("Mercury", _cmercury, "MERCURY"),
                PlanetTarget("Venus", _cvenus, "VENUS"),
                PlanetTarget("Mars", _cmars, "MARS"),
                PlanetTarget("Jupiter", _cjupiter, "JUPITER"),
                PlanetTarget("Saturn", _csaturn, "SATURN"),
                PlanetTarget("Uranus", _curanus, "URANUS"),
                PlanetTarget("Neptune", _cneptune, "NEPTUNE"),
            ]
            self._target = None     # current Target or None
            self._ca_recompute_t = 0.0
            self._ca = None
            self._ca_traj = None
            self._ca_abs_epoch = 0.0
            self._ca_marker_ship = None
            self._ca_marker_moon = None
            self._ca_label_positions = {"ship": None, "target": None}
            from orbitsim.render.world_labels import build_labeled_marker, build_world_label
            self._ca_labels = {
                "ship": build_world_label(
                    self.render, "CA: SHIP", color=(1.0, 0.55, 0.25, 1.0), scale=11.0
                ),
                "target": build_world_label(
                    self.render, "CA: TARGET", color=(1.0, 0.82, 0.30, 1.0), scale=11.0
                ),
            }
            for label in self._ca_labels.values():
                label.hide()
            self._target_label = build_world_label(
                self.render, "TARGET", color=(1.0, 0.35, 0.95, 1.0), scale=12.0
            )
            self._target_label.hide()
            self._target_label_position = None
            self._apsis_positions = {"PE": None, "AP": None}
            self._apsis_nps = {}
            self._apsis_labels = {}
            for name, color in (
                ("PE", (0.30, 1.0, 0.58, 1.0)),
                ("AP", (0.38, 0.68, 1.0, 1.0)),
            ):
                marker, label = build_labeled_marker(
                    self.render, name, color=color, marker_scale=5.5, label_scale=11.0
                )
                marker.hide()
                label.hide()
                self._apsis_nps[name] = marker
                self._apsis_labels[name] = label
            # Predicted-encounter periapsis cues: a small pool of gold markers +
            # labels, placed each frame at each upcoming SOI closest approach.
            self._encounter_marker_nps = []
            self._encounter_labels = []
            for _ in range(self.MAX_ENCOUNTER_MARKERS):
                mk, lbl = build_labeled_marker(
                    self.render, "", color=self.ENCOUNTER_COLOR,
                    marker_scale=5.0, label_scale=10.5,
                )
                mk.hide()
                lbl.hide()
                self._encounter_marker_nps.append(mk)
                self._encounter_labels.append(lbl)
            self._moon_np = make_uv_sphere(1.0, 12, 16)
            self._moon_np.reparent_to(self.render)
            self._moon_np.set_color(0.7, 0.7, 0.72, 1.0)
            self._moon_np.set_light_off()
            self._moon_np.set_scale(7.0)
            self._moon_orbit_np = None      # built lazily in the update loop (scale-dependent)
            self._moon_orbit_scale = None
            # Moon sphere-of-influence: faint true-scale translucent tinted shell.
            from panda3d.core import TransparencyAttrib
            self._soi_np = make_uv_sphere(1.0, 24, 32)
            self._soi_np.reparent_to(self.render)
            self._soi_np.set_light_off()                       # flat, unlit tint
            self._soi_np.set_transparency(TransparencyAttrib.M_alpha)
            self._soi_np.set_depth_write(False)                # don't occlude the scene behind it
            self._soi_np.set_two_sided(True)                   # visible from inside too
            self._soi_np.hide()  # shown + placed each frame in _update
            # Lagrange-point markers (constant on-screen size) + billboard labels.
            self._lagrange_nps = []
            self._lagrange_labels = []
            self._lagrange_positions = {}
            for name in ("L1", "L2", "L3", "L4", "L5"):
                mk = make_uv_sphere(1.0, 8, 12)
                mk.reparent_to(self.render)
                mk.set_color(0.3, 0.9, 0.8, 1.0)   # teal — distinct from Moon/CA/node
                mk.set_light_off()
                mk.set_scale(4.0)
                self._lagrange_nps.append(mk)
                lbl = build_world_label(
                    self.render, name, color=(0.5, 1.0, 0.9, 1.0), scale=12.0
                )
                self._lagrange_labels.append(lbl)
            # Inner planets + Sun: markers, true-scale bodies, SOI spheres, orbit lines.
            from orbitsim.core.constants import (
                R_SUN, R_MERCURY, R_VENUS, R_MARS, R_MOON,
                R_JUPITER, R_SATURN, R_URANUS, R_NEPTUNE,
            )
            from orbitsim.render.textures import texture_path
            self._planet_sandbox_nps = {}    # name -> marker NodePath (constant on-screen size)
            self._planet_sandbox_labels = {}
            self._planet_body_nps = {}       # name -> true-scale textured body NodePath
            self._planet_body_radii = {}     # name -> physical radius [m]
            self._planet_soi_nps = {}        # name -> SOI sphere NodePath
            self._planet_soi_radii = {}      # name -> SOI radius [m]
            _planet_defs = [
                ("Sun", (1.0, 0.85, 0.2, 1.0), 10.0, float("inf"), R_SUN, "sun"),
                ("Mercury", (0.6, 0.6, 0.6, 1.0), 4.0, MERCURY_SOI_M, R_MERCURY, "mercury"),
                ("Venus", (0.9, 0.8, 0.5, 1.0), 5.0, VENUS_SOI_M, R_VENUS, "venus_surface"),
                ("Mars", (0.9, 0.4, 0.2, 1.0), 5.0, MARS_SOI_M, R_MARS, "mars"),
                ("Jupiter", (0.8, 0.7, 0.5, 1.0), 7.0, JUPITER_SOI_M, R_JUPITER, "jupiter"),
                ("Saturn", (0.9, 0.8, 0.6, 1.0), 6.5, SATURN_SOI_M, R_SATURN, "saturn"),
                ("Uranus", (0.6, 0.85, 0.9, 1.0), 5.5, URANUS_SOI_M, R_URANUS, "uranus"),
                ("Neptune", (0.3, 0.4, 0.9, 1.0), 5.5, NEPTUNE_SOI_M, R_NEPTUNE, "neptune"),
            ]
            _soi_colors = {
                "Mercury": (0.6, 0.6, 0.6, 1.0),
                "Venus": (0.9, 0.8, 0.5, 1.0),
                "Mars": (0.9, 0.5, 0.3, 1.0),
                "Jupiter": (0.8, 0.7, 0.5, 1.0),
                "Saturn": (0.9, 0.8, 0.6, 1.0),
                "Uranus": (0.6, 0.85, 0.9, 1.0),
                "Neptune": (0.3, 0.4, 0.9, 1.0),
            }
            for pname, color, sz, soi_r, radius_m, tex_name in _planet_defs:
                mk = make_uv_sphere(1.0, 12, 16)
                mk.reparent_to(self.render)
                mk.set_color(*color)
                mk.set_light_off()
                mk.set_scale(sz)
                self._planet_sandbox_nps[pname] = mk
                lbl = build_world_label(
                    self.render, pname, color=(0.8, 0.85, 1.0, 1.0), scale=12.0
                )
                self._planet_sandbox_labels[pname] = lbl
                self._planet_soi_radii[pname] = soi_r
                self._planet_body_radii[pname] = radius_m
                body = make_uv_sphere(1.0, 32, 64, with_uv=True)
                body.reparent_to(self.render)
                body.set_color(*color)
                tp = texture_path(tex_name)
                if tp is not None:
                    body.set_texture(
                        self.loader.load_texture(Filename.from_os_specific(tp)))
                    body.set_color(1, 1, 1, 1)
                if pname == "Sun":
                    body.set_light_off()
                body.hide()
                self._planet_body_nps[pname] = body
                if pname != "Sun" and np.isfinite(soi_r):
                    soi = make_uv_sphere(1.0, 24, 32)
                    soi.reparent_to(self.render)
                    soi.set_light_off()
                    soi.set_transparency(TransparencyAttrib.M_alpha)
                    soi.set_depth_write(False)
                    soi.set_two_sided(True)
                    sc = _soi_colors.get(pname, (0.5, 0.5, 0.5, 1.0))
                    soi.set_color(sc[0], sc[1], sc[2], self.SOI_BASE_ALPHA)
                    soi.hide()
                    self._planet_soi_nps[pname] = soi
            # Saturn ring (true-scale annular disk, inner ~1.11× R, outer ~2.33× R).
            self._saturn_ring_np = make_ring(inner_radius=1.11, outer_radius=2.33, num_segments=64)
            self._saturn_ring_np.reparent_to(self.render)
            self._saturn_ring_np.set_two_sided(True)
            self._saturn_ring_np.set_transparency(TransparencyAttrib.M_alpha)
            self._saturn_ring_np.set_color(0.85, 0.75, 0.6, 0.8)
            ring_tp = texture_path("saturn_ring")
            if ring_tp is not None:
                self._saturn_ring_np.set_texture(
                    self.loader.load_texture(Filename.from_os_specific(ring_tp)))
                self._saturn_ring_np.set_color(1, 1, 1, 0.8)
            self._saturn_ring_np.hide()
            # True-scale Moon body (textured).
            self._moon_body_np = make_uv_sphere(1.0, 32, 64, with_uv=True)
            self._moon_body_np.reparent_to(self.render)
            moon_tp = texture_path("moon")
            if moon_tp is not None:
                self._moon_body_np.set_texture(
                    self.loader.load_texture(Filename.from_os_specific(moon_tp)))
                self._moon_body_np.set_color(1, 1, 1, 1)
            else:
                self._moon_body_np.set_color(0.7, 0.7, 0.72, 1.0)
            self._moon_body_np.hide()
            # Earth SOI sphere (visible once the vessel gets far from Earth).
            earth_soi = make_uv_sphere(1.0, 24, 32)
            earth_soi.reparent_to(self.render)
            earth_soi.set_light_off()
            earth_soi.set_transparency(TransparencyAttrib.M_alpha)
            earth_soi.set_depth_write(False)
            earth_soi.set_two_sided(True)
            earth_soi.set_color(0.3, 0.5, 1.0, self.SOI_BASE_ALPHA)
            earth_soi.hide()
            self._earth_soi_np = earth_soi
            # Planet orbit reference lines (heliocentric circles, positioned at the Sun's
            # geocentric position each frame). Built lazily on first zoom like the Moon line.
            self._planet_orbit_nps = {}
            self._planet_orbit_scale = None
            self._planet_orbit_epoch_bucket = None
            self._sun_orbit_frame = self.render.attach_new_node("sun_orbit_frame")
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
            from orbitsim.render.world_labels import build_world_label
            label = build_world_label(
                self.render, body.name, color=(0.8, 0.85, 1.0, 1.0), scale=15.0
            )
            self._planet_labels.append(label)

    # Spring-loaded "jog" sliders: displacement from center sets the *rate* of change.
    JOG_MAX_RATE_MPS = 400.0  # delta-V change per second at full deflection
    JOG_CURVE = 4.0  # exponential steepness; higher = gentler near center, sharper at edges
    JOG_DEADZONE = 0.02  # |value| below this contributes no change

    # Scheduled maneuver node.
    NODE_TIME_STEP_S = 30.0       # seconds per "Node -/+" press (base step for buttons)
    NODE_JOG_MAX_RATE = 1000.0    # max time jog rate [s/s at full deflection]
    NODE_JOG_CURVE = 4.0
    AUTO_WARP_LEAD_S = 5.0        # auto-warp-down when within this many real-seconds of the node
    EXECUTE_TOLERANCE_S = 2.0     # execute allowed only within this of the node epoch
    PREVIEW_THROTTLE_S = 0.2      # min real-seconds between post-burn preview rebuilds (5 Hz)
    TRAJECTORY_REFRESH_S = 2.0    # prediction cadence; integration runs off the render thread

    def _build_maneuver_ui(self) -> None:
        """Per-axis spring-loaded jog sliders for RTN delta-V, an Execute button, a readout.

        Each slider is a rate control: hold it right to increase that delta-V component
        (faster the further you push, exponentially), left to decrease. Releasing the
        mouse springs the thumb back to center and the value stops changing.
        """
        self._dv = {"pro": 0.0, "nrm": 0.0, "rad": 0.0}
        self._dv_line = self._node_line = self._target_line = self._encounter_line = ""
        self._node_epoch_s = None      # absolute epoch of the scheduled node (None = none)
        self._auto_execute_node = False
        self._node_marker_np = None
        self._node_dv_label = None
        self._dv_value_text = {}
        self._jog = {}
        from orbitsim.render.ui.theme import THEME, button_options
        self._maneuver_frame = DirectFrame(
            frameColor=THEME.panel,
            frameSize=(-1.34, 0.02, -0.29, 0.60),
            parent=self.a2dBottomRight,
        )
        rows = (("pro", "Prograde", 0.40), ("nrm", "Normal", 0.28), ("rad", "Radial", 0.16))
        for axis, label, z in rows:
            OnscreenText(
                text=label,
                pos=(-1.22, z - 0.015),
                scale=0.045,
                fg=(1, 1, 1, 1),
                align=TextNode.ALeft,
                parent=self._maneuver_frame,
            )
            self._jog[axis] = DirectSlider(
                pos=(-0.62, 0.0, z),
                scale=0.34,
                range=(-1.0, 1.0),
                value=0.0,
                pageSize=0.25,
                parent=self._maneuver_frame,
            )
            self._dv_value_text[axis] = OnscreenText(
                text="+0",
                pos=(-0.14, z - 0.015),
                scale=0.045,
                fg=(1.0, 0.9, 0.4, 1),
                align=TextNode.ALeft,
                mayChange=True,
                parent=self._maneuver_frame,
            )
        # Total dV magnitude display.
        self._dv_total_text = OnscreenText(
            text="", pos=(-0.65, 0.52), scale=0.05,
            fg=(1.0, 0.5, 1.0, 1.0), align=TextNode.ACenter,
            mayChange=True, parent=self._maneuver_frame,
        )
        # Node time jog slider (variable time stepping).
        OnscreenText(
            text="Node T", pos=(-1.22, 0.04 - 0.015), scale=0.045,
            fg=(0.3, 1.0, 1.0, 1), align=TextNode.ALeft,
            parent=self._maneuver_frame,
        )
        self._node_time_jog = DirectSlider(
            pos=(-0.62, 0.0, 0.04), scale=0.34,
            range=(-1.0, 1.0), value=0.0, pageSize=0.25,
            parent=self._maneuver_frame,
        )
        self._node_time_text = OnscreenText(
            text="T+0s", pos=(-0.14, 0.04 - 0.015), scale=0.045,
            fg=(0.3, 1.0, 1.0, 1), align=TextNode.ALeft,
            mayChange=True, parent=self._maneuver_frame,
        )
        self._action_frame = DirectFrame(
            frameColor=THEME.panel,
            frameSize=(0.03, 0.78, 0.03, 0.25),
            parent=self.a2dBottomLeft,
        )
        self._exec_btn = DirectButton(
            text="Execute Burn",
            scale=0.045,
            pos=(0.40, 0.0, 0.10),
            command=self._execute_burn,
            parent=self._action_frame,
            **button_options(),
        )
        self._action_status = OnscreenText(
            text="NO MANEUVER PLANNED", pos=(0.40, 0.18), scale=0.036,
            fg=THEME.text_muted, mayChange=True, parent=self._action_frame,
        )
        # Scheduled-node controls: jump to next apsis, clear.
        node_btns = [
            ("Next Pe", self._node_to_pe),
            ("Next Ap", self._node_to_ap),
            ("Clear", self._clear_node),
            ("Clear Tgt", self._clear_target),
            ("Intercept", self._plan_intercept),
        ]
        for i, (label, cmd) in enumerate(node_btns):
            DirectButton(text=label, scale=0.045, pos=(-0.95 + i * 0.34, 0.0, -0.20),
                         command=cmd, parent=self._maneuver_frame, **button_options())
        # Releasing the mouse springs every jog slider back to its center (zero rate).
        self.accept("mouse1-up", self._release_jogs)
        self._refresh_readout()

    def _toggle_maneuver_drawer(self) -> None:
        if self._maneuver_frame.is_hidden():
            self._maneuver_frame.show()
        else:
            self._maneuver_frame.hide()

    def _cancel_operation(self) -> None:
        self.operation.cancel()
        rollback = getattr(self, "_operation_rollback", None)
        if rollback is not None:
            vessel, state, orientation, clock_time = rollback
            vessel.state = state
            vessel.orientation = orientation
            self.clock.sim_time_s = clock_time
        self._operation_rollback = None
        self.operation_overlay.update(self.operation.status)
        self._flash_message("Operation cancelled")

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
        if hasattr(self, "_node_time_jog"):
            tv = self._node_time_jog["value"]
            mag = abs(tv)
            if mag > self.JOG_DEADZONE:
                shaped = (np.exp(self.NODE_JOG_CURVE * mag) - 1.0) / (np.exp(self.NODE_JOG_CURVE) - 1.0)
                rate = float(np.copysign(self.NODE_JOG_MAX_RATE * shaped, tv))
                now = self.clock.sim_time_s
                base = self._node_epoch_s if self._node_epoch_s is not None else now
                self._node_epoch_s = max(now, base + rate * real_dt_s)
                ttn = self._node_epoch_s - now
                if ttn < 60:
                    self._node_time_text.setText(f"T+{ttn:.0f}s")
                elif ttn < 3600:
                    self._node_time_text.setText(f"T+{ttn/60:.1f}m")
                elif ttn < 86400:
                    self._node_time_text.setText(f"T+{ttn/3600:.1f}h")
                else:
                    self._node_time_text.setText(f"T+{ttn/86400:.1f}d")

    def _release_jogs(self) -> None:
        """Spring all jog thumbs back to center so the value stops changing on release.

        Also attempt a target pick: a left-click that didn't drag is a tap on a body."""
        self._try_pick_target()
        for slider in self._jog.values():
            slider["value"] = 0.0
        if hasattr(self, "_node_time_jog"):
            self._node_time_jog["value"] = 0.0

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
        """On a left-click tap, select a body or coast to a future trajectory point."""
        from orbitsim.render.picking import nearest_future_point, nearest_marker
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
            self._clear_target()   # tear down any prior target's CA markers before switching
            self._target = self._targets[idxs[hit]]
            return

        points = self._traj_world_cache[0] if self._traj_world_cache else None
        epochs = self._traj_epoch_cache[0] if self._traj_epoch_cache else None
        if points is None or epochs is None or self.world.any_thrusting():
            return
        trajectory_px = [self._marker_px(point) for point in points]
        point_idx = nearest_future_point(click, trajectory_px, epochs, now, tol_px=16.0)
        if point_idx is None:
            return
        duration_s = float(epochs[point_idx] - now)
        vessel = self.world.vessels[0]
        self.clock.warp = 1.0
        self._operation_rollback = (
            vessel, vessel.state, vessel.orientation.copy(), self.clock.sim_time_s
        )
        self.operation.start(
            f"Coasting {_fmt_countdown(duration_s)} along trajectory",
            self._trajectory_coast_operation(duration_s),
        )
        self.operation_overlay.update(self.operation.status)

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
        self._auto_execute_node = False
        self._node_epoch_s = None
        if self._node_marker_np is not None:
            self._node_marker_np.remove_node()
            self._node_marker_np = None
        if self._node_dv_label is not None:
            self._node_dv_label.remove_node()
            self._node_dv_label = None
        if hasattr(self, "_node_time_text"):
            self._node_time_text.setText("T+0s")

    def _plan_intercept(self):
        """Auto-plan a flyby of the current target via a departure-dV porkchop."""
        import numpy as np
        from orbitsim.core.elements import state_to_elements
        from orbitsim.render.targets import PlanetTarget
        if self._target is None:
            self._flash_message("No target selected")
            return
        v0 = self.world.vessels[0]
        now = self.clock.sim_time_s
        self._flash_message(f"Planning {self._target.name} intercept...")
        if isinstance(self._target, PlanetTarget):
            from orbitsim.core.ephemeris import body_state
            from orbitsim.core.optimize import interplanetary_departure_node
            from orbitsim.core.planets import A_EARTH, _N_EARTH
            target_a = {
                "Mercury": A_MERCURY, "Venus": A_VENUS, "Mars": A_MARS,
                "Jupiter": A_JUPITER, "Saturn": A_SATURN,
                "Uranus": A_URANUS, "Neptune": A_NEPTUNE,
            }.get(self._target.name)
            if target_a is not None:
                w_ship = _N_EARTH
                w_target = np.sqrt(self.world.central.mu / 1.0) if target_a == 0 else 0.0
                from orbitsim.core.constants import MU_SUN
                w_target = np.sqrt(MU_SUN / target_a**3)
                synodic = 2.0 * np.pi / abs(w_ship - w_target)
            else:
                synodic = 365.25 * 86400.0
            dep = np.linspace(0.0, min(synodic, 2 * 365.25 * 86400.0), 24)
            hohmann_tof = np.pi * np.sqrt(((A_EARTH + (target_a or A_EARTH)) / 2.0)**3 / MU_SUN)
            tof = np.linspace(0.3 * hohmann_tof, 2.0 * hohmann_tof, 32)
            try:
                node = interplanetary_departure_node(
                    v0.state,
                    self._target.planning_state_at,
                    lambda epoch: body_state("SUN", epoch, center="EARTH"),
                    dep,
                    tof,
                )
            except ValueError:
                self._flash_message("No intercept found")
                return
        else:
            from orbitsim.core.optimize import intercept_node
            try:
                period = state_to_elements(v0.state).period_s
            except ValueError:
                self._flash_message("Unbound orbit — can't plan intercept")
                return
            dep = np.linspace(0.0, period, 24)
            tof = np.linspace(3.0e3, 14.0 * 86400.0, 48)
            try:
                node = intercept_node(v0.state, self._target.state_at(now),
                                      self.world.central.mu, dep, tof)
            except ValueError:
                self._flash_message("No intercept found")
                return
        self._node_epoch_s = node.epoch_s
        self._auto_execute_node = False
        self._dv["pro"] = node.dv_prograde_mps
        self._dv["nrm"] = node.dv_normal_mps
        self._dv["rad"] = node.dv_radial_mps
        for axis in ("pro", "nrm", "rad"):
            self._dv_value_text[axis].setText(f"{self._dv[axis]:+.0f}")
        self._refresh_readout()
        self._flash_message(f"Intercept planned (dV {node.magnitude_mps:,.0f} m/s)")

    def _clear_target(self):
        """Deselect the current target; remove its closest-approach markers + readout."""
        self._target = None
        for attr in ("_ca_marker_ship", "_ca_marker_moon"):
            np_ = getattr(self, attr, None)
            if np_ is not None:
                np_.remove_node()
                setattr(self, attr, None)
        self._ca = None
        self._target_label_position = None
        self._target_label.hide()
        for key, label in self._ca_labels.items():
            self._ca_label_positions[key] = None
            label.hide()
        self._target_line = ""
        self._sync_maneuver_hud()

    def _ca_marker(self, attr, color):
        """Lazily create/reuse a closest-approach marker NodePath."""
        np_ = getattr(self, attr, None)
        if np_ is None:
            np_ = make_uv_sphere(1.0, 8, 12)
            np_.reparent_to(self.render)
            np_.set_color(*color)
            np_.set_light_off()
            np_.set_scale(5.0)
            from panda3d.core import TransparencyAttrib
            np_.set_transparency(TransparencyAttrib.M_alpha)
            np_.set_depth_write(False)
            setattr(self, attr, np_)
        return np_

    def _place_apsis_markers(self) -> None:
        """Place cached Pe/Ap cues in the current floating-origin frame."""
        for name in ("PE", "AP"):
            world_pos = self._apsis_positions[name]
            marker = self._apsis_nps[name]
            label = self._apsis_labels[name]
            if world_pos is None:
                marker.hide()
                label.hide()
                continue
            rx, ry, rz = self.transform.to_render(world_pos)
            marker.set_pos(rx, ry, rz)
            label.set_pos(rx, ry, rz + 7.0)
            marker.show()

    def _update_marker_readability(self, focus_m) -> None:
        """Fade distant cues and declutter their labels by navigation priority."""
        from panda3d.core import TransparencyAttrib
        from orbitsim.render.world_markers import declutter_indices, distance_fade

        candidates = []

        def add(label, marker, world_pos, priority, minimum=0.22):
            if world_pos is not None:
                candidates.append((label, marker, np.asarray(world_pos), priority, minimum))

        add(self._target_label, None, self._target_label_position, 100, 0.55)
        add(self._ca_labels["ship"], self._ca_marker_ship,
            self._ca_label_positions["ship"], 90, 0.40)
        add(self._ca_labels["target"], self._ca_marker_moon,
            self._ca_label_positions["target"], 85, 0.40)
        add(self._apsis_labels["PE"], self._apsis_nps["PE"],
            self._apsis_positions["PE"], 70)
        add(self._apsis_labels["AP"], self._apsis_nps["AP"],
            self._apsis_positions["AP"], 65)
        for name, marker, label in zip(
            ("L1", "L2", "L3", "L4", "L5"), self._lagrange_nps, self._lagrange_labels
        ):
            add(label, marker, self._lagrange_positions.get(name), 20)

        points_px = [self._marker_px(item[2]) for item in candidates]
        visible = declutter_indices(
            points_px, [item[3] for item in candidates], min_separation_px=52.0
        )
        near = max(self.rig.distance_m * 2.0, 1.0)
        far = max(self.rig.distance_m * 18.0, near + 1.0)
        for index, (label, marker, world_pos, _priority, minimum) in enumerate(candidates):
            alpha = distance_fade(
                float(np.linalg.norm(world_pos - focus_m)), near, far, minimum=minimum
            )
            label.set_alpha_scale(alpha)
            if index in visible:
                label.show()
            else:
                label.hide()
            if marker is not None and not marker.is_empty():
                marker.set_transparency(TransparencyAttrib.M_alpha)
                marker.set_alpha_scale(alpha)

    def _refresh_readout(self) -> None:
        import math
        node = self._current_node()
        budget = self.world.vessels[0].delta_v_remaining
        left = "INF" if not math.isfinite(budget) else f"{budget:,.0f} m/s"
        self._dv_line = (
            f"Maneuver dV: {node.magnitude_mps:,.1f} m/s   (dV left {left})"
            if node.magnitude_mps > 0.0
            else ""
        )
        if hasattr(self, "_dv_total_text"):
            if node.magnitude_mps > 0.0:
                self._dv_total_text.setText(f"Total dV: {node.magnitude_mps:,.1f} m/s")
            else:
                self._dv_total_text.setText("")
        if hasattr(self, "_exec_btn"):
            ttn = self._time_to_node()
            if getattr(self, "_auto_execute_node", False):
                self._exec_btn["text"] = "Cancel Auto Burn"
            elif ttn is not None and ttn > self.EXECUTE_TOLERANCE_S:
                self._exec_btn["text"] = "Jump & Execute"
            else:
                self._exec_btn["text"] = "Execute Burn"
            if hasattr(self, "_action_status"):
                if node.magnitude_mps <= 0.0:
                    self._action_status.setText("NO MANEUVER PLANNED")
                    from direct.gui import DirectGuiGlobals as DGG
                    self._exec_btn["state"] = DGG.DISABLED
                elif ttn is not None and ttn > 0.0:
                    from direct.gui import DirectGuiGlobals as DGG
                    self._exec_btn["state"] = DGG.NORMAL
                    self._action_status.setText(
                        f"NODE T-{_fmt_countdown(ttn)}  /  {node.magnitude_mps:,.0f} m/s"
                    )
                else:
                    from direct.gui import DirectGuiGlobals as DGG
                    self._exec_btn["state"] = DGG.NORMAL
                    self._action_status.setText(f"BURN READY  /  {node.magnitude_mps:,.0f} m/s")
        self._sync_maneuver_hud()

    def _sync_maneuver_hud(self) -> None:
        enc = getattr(self, "_encounter_line", "")
        self.hud.set_maneuver(self._dv_line, self._node_line, self._target_line, enc)

    def _update_encounter_info(self, vessel, dom_body, dom_pos) -> None:
        """Compute and display flyby encounter parameters when approaching a planet."""
        if not self.world.solar_system:
            self._encounter_line = ""
            return
        if dom_body.name in ("Earth", "Moon"):
            self._encounter_line = ""
            return
        if dom_body.name == "Sun":
            self._encounter_line = ""
            return
        from orbitsim.core.flyby import encounter_parameters
        r_rel = vessel.state.r - dom_pos
        v_planet_state = None
        t_now = self.clock.sim_time_s
        from orbitsim.core.nbody import (
            _cmercury, _cvenus, _cmars,
            _cjupiter, _csaturn, _curanus, _cneptune,
        )
        _planet_vel_fns = {
            "Mercury": _cmercury, "Venus": _cvenus,
            "Mars": _cmars, "Jupiter": _cjupiter,
            "Saturn": _csaturn, "Uranus": _curanus,
            "Neptune": _cneptune,
        }
        fn = _planet_vel_fns.get(dom_body.name)
        if fn is None:
            self._encounter_line = ""
            return
        pstate = fn(t_now)
        params = encounter_parameters(
            vessel.state.r, vessel.state.v, dom_pos, pstate.v, dom_body.mu)
        v_inf_kms = params["v_inf_mag"] / 1000.0
        defl_deg = np.degrees(params["deflection_rad"])
        dv_eq = params["dv_equivalent"]
        pe_km = params["periapsis_m"] / 1000.0
        self._encounter_line = (
            f"Flyby {dom_body.name}: v-inf {v_inf_kms:,.1f} km/s  "
            f"δ {defl_deg:,.1f}°  Pe {pe_km:,.0f} km  "
            f"free dV {dv_eq:,.0f} m/s"
        )
        self._sync_maneuver_hud()

    def _apply_unlimited(self, on: bool) -> None:
        """Set the unlimited-dV flag on all vessels (no UI). The sim layer
        floors the working propellant mass for the impulse math while
        unlimited (World.UNLIMITED_FUEL_FLOOR_KG), so thrust works even with
        an empty tank and the real fuel load is never touched."""
        for vessel in self.world.vessels:
            vessel.unlimited_dv = on

    def _set_unlimited_dv(self, on: bool) -> None:
        if self.solar_system or not self.world.vessels:
            return  # no flyable vessel (solar viewer) — nothing to toggle
        self._apply_unlimited(on)
        self.settings_panel.sync(on)   # keep the Esc-panel label in step with key/title
        self._flash_message(f"Unlimited dV {'ON' if on else 'OFF'}")
        self._refresh_readout()

    def _toggle_unlimited_dv(self) -> None:
        cur = bool(self.world.vessels and self.world.vessels[0].unlimited_dv)
        self._set_unlimited_dv(not cur)

    def _execute_burn(self) -> None:
        from orbitsim.core.flight import fuel_burned_for_dv

        ttn = self._time_to_node()
        if ttn is not None and ttn > self.EXECUTE_TOLERANCE_S:
            if self.world.any_thrusting():
                self._flash_message("Cut throttle before jumping to the node")
                return
            self.clock.warp = 1.0
            vessel = self.world.vessels[0]
            self._operation_rollback = (
                vessel, vessel.state, vessel.orientation.copy(), self.clock.sim_time_s
            )
            self.operation.start("Coasting to maneuver", self._jump_operation(ttn))
            self.operation_overlay.update(self.operation.status)
            return
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

    def _jump_operation(self, duration_s):
        """Frame-budgeted coast generator; one ephemeris-aware day per rendered frame."""
        from orbitsim.core.nbody import refresh_ephemeris_cache

        chunks = _coast_chunks(duration_s)
        total = len(chunks)
        for index, dt_s in enumerate(chunks):
            refresh_ephemeris_cache(self.clock.sim_time_s + 0.5 * dt_s)
            self.world.step(dt_s)
            self.clock.sim_time_s += dt_s
            yield (index + 1) / total
        self._operation_rollback = None
        self._execute_burn()

    def _trajectory_coast_operation(self, duration_s):
        """Frame-budgeted coast to a clicked future trajectory sample."""
        from orbitsim.core.nbody import refresh_ephemeris_cache

        chunks = _coast_chunks(duration_s)
        total = len(chunks)
        for index, dt_s in enumerate(chunks):
            refresh_ephemeris_cache(self.clock.sim_time_s + 0.5 * dt_s)
            self.world.step(dt_s)
            self.clock.sim_time_s += dt_s
            yield (index + 1) / total
        self._operation_rollback = None
        self._traj_world_cache[0] = None
        self._traj_epoch_cache[0] = None
        self._flash_message("Arrived at trajectory point")

    MOUSE_ORBIT_SENS = 3.0     # radians per unit of normalized mouse travel

    def _setup_input(self) -> None:
        self.accept("wheel_up", lambda: self.rig.zoom(0.86))
        self.accept("wheel_down", lambda: self.rig.zoom(1.0 / 0.86))
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
        self.accept("0", self._max_warp_guarded)
        self.accept("p", self._toggle_porkchop)  # porkchop delta-V plot
        self.accept("f1", self.keybind_overlay.toggle)  # keybind help overlay
        self.accept("escape", self._handle_escape)

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
            self.accept("m", self._toggle_ship_view)   # snap map <-> ship view
            sas_keys = ["PROGRADE", "RETROGRADE", "NORMAL", "ANTINORMAL",
                        "RADIAL_IN", "RADIAL_OUT", "TARGET", "ANTITARGET", "MANEUVER"]
            for i, mode in enumerate(sas_keys, start=1):
                self.accept(str(i), self._set_sas, [mode])
            self.accept("i", self._plan_intercept)
            self.accept("f5", self._quicksave)
            self.accept("f9", self._quickload)

    ROTATE_RATE_RADPS = 0.8       # manual pitch/yaw/roll rate
    LAGRANGE_REFRESH_S = 10.0     # sim-seconds between L-point recomputes
    THROTTLE_STEP = 0.5           # throttle change per second for shift/ctrl
    SHIP_VIEW_DISTANCE_M = 80.0   # default close framing for ship view
    SOI_COLOR = (0.45, 0.65, 1.0, 1.0)         # cool blue tint (outside)
    SOI_INSIDE_COLOR = (0.45, 1.0, 0.65, 1.0)  # greenish "captured by the Moon"
    SOI_BASE_ALPHA = 0.10                       # faint translucent shell
    SOI_INSIDE_ALPHA = 0.18
    SOI_FADE_NEAR_M = 1.5e9   # camera distance: full alpha when closer than this
    SOI_FADE_FAR_M = 1.5e10   # ... fading to zero past this (tune by screenshot)

    def _toggle_ship_view(self) -> None:
        """Snap the camera between remembered map and ship-view distances ('m')."""
        from orbitsim.render.ship_model import SHIP_VIEW_NEAR_M
        # "In ship view" means already in close framing; anything farther (incl. a
        # mid-fade zoom) counts as the map side and is remembered as such.
        in_ship_view = self.rig.target_distance_m <= SHIP_VIEW_NEAR_M
        if in_ship_view:
            self._ship_distance_m = self.rig.target_distance_m  # remember ship framing
            self.rig.move_to_distance(getattr(self, "_map_distance_m", 2.0e7))
        else:
            self._map_distance_m = self.rig.target_distance_m  # remember map framing
            self.rig.move_to_distance(
                getattr(self, "_ship_distance_m", self.SHIP_VIEW_DISTANCE_M)
            )

    def _build_warp_controls(self) -> None:
        """Top-center on-screen warp control: slower / faster buttons + readout.
        (The ',' and '.' keys do the same.) Works in both sandbox and solar modes."""
        from orbitsim.render.ui.theme import THEME, button_options
        self._warp_frame = DirectFrame(
            frameColor=THEME.panel,
            frameSize=(-0.64, 0.64, -0.16, -0.02),
            parent=self.a2dTopCenter,
        )
        self._warp_readout = OnscreenText(
            text="", pos=(0.0, -0.09), scale=0.045, fg=THEME.text,
            shadow=(0, 0, 0, 1), mayChange=True, parent=self._warp_frame,
        )
        self._warp_btns = [
            DirectButton(text="<<", scale=0.05, pos=(-0.28, 0.0, -0.085),
                         command=self.clock.warp_down, parent=self._warp_frame,
                         **button_options()),
            DirectButton(text=">>", scale=0.05, pos=(0.28, 0.0, -0.085),
                         command=self._warp_up_guarded, parent=self._warp_frame,
                         **button_options()),
            DirectButton(text="MAX", scale=0.042, pos=(0.48, 0.0, -0.085),
                         command=self._max_warp_guarded, parent=self._warp_frame,
                         **button_options()),
        ]
        self._update_warp_readout()

    def _update_warp_readout(self) -> None:
        if getattr(self, "_warp_readout", None) is not None:
            locked = not self.solar_system and self.world.any_thrusting()
            suffix = "  (LOCKED)" if locked else ""
            days = self.clock.sim_time_s / 86400.0
            self._warp_readout.setText(
                f"T+ {days:,.1f} DAYS     WARP  x{self.clock.warp:,.0f}{suffix}"
            )

    def _warp_up_guarded(self):
        if self.world.any_thrusting():
            return
        if self.world.vessels:
            from orbitsim.sim.clock import SimClock
            if self.world.solar_system:
                from orbitsim.core.nbody import max_safe_warp_solar
                cap = max_safe_warp_solar(
                    self.world.vessels[0].state, self.clock.sim_time_s, SimClock.WARP_STEPS)
            else:
                from orbitsim.core.nbody import max_safe_warp
                cap = max_safe_warp(
                    self.world.vessels[0].state, self.clock.sim_time_s, SimClock.WARP_STEPS)
            if self.clock.warp >= cap:
                return
        self.clock.warp_up()

    def _max_warp_guarded(self):
        """Jump directly to the highest safe warp (up to 100,000,000x in deep space)."""
        if self.world.any_thrusting():
            return
        from orbitsim.sim.clock import SimClock

        cap = float(SimClock.WARP_STEPS[-1])
        if self.world.vessels:
            if self.world.solar_system:
                from orbitsim.core.nbody import max_safe_warp_solar

                cap = max_safe_warp_solar(
                    self.world.vessels[0].state,
                    self.clock.sim_time_s,
                    SimClock.WARP_STEPS,
                )
            else:
                from orbitsim.core.nbody import max_safe_warp

                cap = max_safe_warp(
                    self.world.vessels[0].state,
                    self.clock.sim_time_s,
                    SimClock.WARP_STEPS,
                )
        self.clock.warp = float(cap)
        self._update_warp_readout()

    def _rmb(self, down):
        self._rmb_down = down
        self._last_mouse = None

    def _handle_escape(self):
        """Close the topmost transient surface before opening settings."""
        if self.operation.status.running:
            self._cancel_operation()
            return
        if getattr(self.keybind_overlay, "visible", False):
            self.keybind_overlay.hide()
            return
        if getattr(self, "_porkchop_card", None) is not None:
            self._porkchop_card.remove_node()
            self._porkchop_card = None
            if getattr(self, "_porkchop_modal", None) is not None:
                self._porkchop_modal.destroy()
                self._porkchop_modal = None
            return
        self.settings_panel.toggle()

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
        if mode == "MANEUVER" and self._current_node().magnitude_mps <= 0.0:
            self._flash_message("Plan a maneuver first")
            return
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
        self.world.solar_system = world.solar_system
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
        """Build a porkchop plot and overlay it. Uses the current target if one
        is selected (planet or Moon); otherwise falls back to a generic higher-orbit
        demo. Pressing 'p' again removes it.
        """
        existing = getattr(self, "_porkchop_card", None)
        if existing is not None:
            existing.remove_node()
            self._porkchop_card = None
            modal = getattr(self, "_porkchop_modal", None)
            if modal is not None:
                modal.destroy()
                self._porkchop_modal = None
            return

        mu = self.world.central.mu
        ship = self.world.vessels[0].state

        from orbitsim.render.targets import PlanetTarget
        if self._target is not None and isinstance(self._target, PlanetTarget):
            from orbitsim.core.optimize import interplanetary_porkchop
            from orbitsim.core.constants import MU_SUN
            from orbitsim.core.planets import A_EARTH
            target_a = {
                "Mercury": A_MERCURY, "Venus": A_VENUS, "Mars": A_MARS,
                "Jupiter": A_JUPITER, "Saturn": A_SATURN,
                "Uranus": A_URANUS, "Neptune": A_NEPTUNE,
            }.get(self._target.name)
            if target_a is not None:
                from orbitsim.core.planets import _N_EARTH
                w_target = np.sqrt(MU_SUN / target_a**3)
                synodic = 2.0 * np.pi / abs(_N_EARTH - w_target)
                hohmann_tof = np.pi * np.sqrt(((A_EARTH + target_a) / 2.0)**3 / MU_SUN)
            else:
                synodic = 365.25 * 86400.0
                hohmann_tof = 180.0 * 86400.0
            dep_times = np.linspace(0.0, min(synodic, 2 * 365.25 * 86400.0), 36)
            tof_grid = np.linspace(0.3 * hohmann_tof, 2.0 * hohmann_tof, 36)
            self._flash_message(f"Computing {self._target.name} porkchop...")
            dv, _ = interplanetary_porkchop(
                "EARTH",
                self._target.name.upper(),
                ship.epoch_s + dep_times,
                tof_grid,
            )
        elif self._target is not None:
            now = self.clock.sim_time_s
            from orbitsim.core.elements import state_to_elements
            try:
                period = state_to_elements(ship).period_s
            except ValueError:
                self._flash_message("Unbound orbit — can't compute porkchop")
                return
            dep_times = np.linspace(0.0, period, 24)
            tof_grid = np.linspace(3.0e3, 14.0 * 86400.0, 36)
            target_state = self._target.state_at(now)
            dv, _ = porkchop(ship, target_state, dep_times, tof_grid, mu)
        else:
            r1 = ship.r_mag
            r2 = r1 * 2.0
            v2 = np.sqrt(mu / r2)
            arr = StateVector(r=np.array([r2, 0.0, 0.0]), v=np.array([0.0, v2, 0.0]), mu=mu)
            w1 = np.sqrt(mu / r1**3)
            w2 = np.sqrt(mu / r2**3)
            t_syn = 2.0 * np.pi / abs(w1 - w2)
            t_hohmann = np.pi * np.sqrt(((r1 + r2) / 2.0) ** 3 / mu)
            dep_times = np.linspace(0.0, t_syn, 24)
            tof_grid = np.linspace(0.4 * t_hohmann, 1.6 * t_hohmann, 36)
            dv, _ = porkchop(ship, arr, dep_times, tof_grid, mu)

        png = render_porkchop_png(dv, dep_times, tof_grid, "porkchop.png")
        from orbitsim.render.ui.theme import THEME, button_options
        self._porkchop_modal = DirectFrame(
            frameColor=THEME.panel_alt,
            frameSize=(-0.92, 0.92, -0.72, 0.72),
            parent=self.aspect2d,
        )
        OnscreenText(
            text="TRANSFER WINDOW ANALYSIS", pos=(0, 0.64), scale=0.052,
            fg=THEME.cyan, parent=self._porkchop_modal,
        )
        tex = self.loader.load_texture(png)
        cm = CardMaker("porkchop")
        cm.set_frame(-0.78, 0.78, -0.48, 0.50)
        card = self._porkchop_modal.attach_new_node(cm.generate())
        card.set_texture(tex)
        self._porkchop_card = card
        DirectButton(
            text="CREATE MANEUVER", scale=0.048, pos=(0.38, 0, -0.61),
            command=self._plan_intercept, parent=self._porkchop_modal, **button_options(),
        )
        DirectButton(
            text="CLOSE", scale=0.048, pos=(-0.38, 0, -0.61),
            command=self._toggle_porkchop, parent=self._porkchop_modal, **button_options(),
        )

    def _sample_trajectory(
        self, state, n_pts=256, max_horizon_s=7 * 86400, n_orbits=1,
        with_times=False, with_encounters=False,
    ):
        """Forward-integrate state under N-body and return ~n_pts positions [m].

        Horizon is ``n_orbits`` osculating orbital periods capped at max_horizon_s; for an
        Earth-bound orbit this draws that many closed loops (successive loops drift slightly
        under perturbation), for a translunar/hyperbolic arc (no period) it shows the
        next ``max_horizon_s`` of the perturbed path. Returns an (n_pts, 3) float64 array of
        world-meter positions in the Earth-centered inertial frame.
        """
        if self.world.solar_system:
            from orbitsim.core.nbody import (
                osculating_elements_solar,
                propagate_solar_system,
                stable_prediction_ephemeris,
            )
            osc_fn = osculating_elements_solar
            prop_fn = propagate_solar_system
        else:
            from orbitsim.core.nbody import osculating_elements, propagate_earth_moon
            osc_fn = osculating_elements
            prop_fn = propagate_earth_moon
        context = stable_prediction_ephemeris() if self.world.solar_system else nullcontext()
        with context:
            try:
                osc = osc_fn(state, state.epoch_s)
                horizon_s = min(n_orbits * float(osc.period_s), max_horizon_s)
            except (ValueError, AttributeError):
                horizon_s = float(max_horizon_s)
            dt = horizon_s / n_pts
            pts = np.empty((n_pts, 3), dtype=np.float64)
            pts[0] = state.r
            cur = state
            for i in range(1, n_pts):
                if self.world.solar_system:
                    # Visual prediction uses a coarser deep-space ceiling. Adaptive near-body
                    # stepping still tightens this automatically for encounters and flybys.
                    cur = prop_fn(cur, dt, max_step_s=6.0 * 3600.0)
                else:
                    cur = prop_fn(cur, dt)
                pts[i] = cur.r
            epochs = state.epoch_s + np.arange(n_pts, dtype=np.float64) * dt
            encounters = []
            if with_encounters:
                # Classify inside the same ephemeris context that produced the
                # path, so far-future planet positions match what was integrated.
                from orbitsim.core.encounters import (
                    find_encounters, solar_dominant, earth_moon_dominant,
                )
                dominant = solar_dominant if self.world.solar_system else earth_moon_dominant
                encounters = find_encounters(pts, epochs, dominant, primary_name="Earth")
        if with_encounters:
            return pts, epochs, encounters
        if with_times:
            return pts, epochs
        return pts

    def _sample_preview(self, state):
        """Compute preview points + encounters without touching Panda3D scene objects."""
        horizon = _trajectory_horizon_s(state, self.world.solar_system)
        pts, _epochs, encounters = self._sample_trajectory(
            state, n_pts=256, max_horizon_s=horizon, n_orbits=2, with_encounters=True
        )
        return pts, encounters

    def _update_maneuver_preview(self, node, vessel, now_real) -> None:
        """Poll/submit preview work while keeping N-body integration off the render thread."""
        preview_key = _maneuver_preview_key(node, self._node_epoch_s)
        future = self._preview_future
        if future is not None and future.done():
            points, encounters = future.result()
            completed_key = self._preview_future_key
            self._preview_future = None
            self._preview_future_key = None
            if preview_key == completed_key and node.magnitude_mps > 0.0:
                if self._preview_np is not None:
                    self._preview_np.remove_node()
                local_points, self._preview_origin = _localize_polyline(points)
                scale = self.transform.scale_m_per_unit
                render_points = [tuple(point / scale) for point in local_points]
                self._preview_np = build_orbit_node(
                    render_points, color=MANEUVER_COLOR, thickness=2.75
                )
                self._preview_np.reparent_to(self.render)
                self._preview_np.set_pos(*self.transform.to_render(self._preview_origin))
                # Encounter patches ride the preview node (same origin/scale).
                self._preview_encounters = encounters
                self._build_encounter_patches(
                    self._preview_np, points, self._preview_origin, scale, encounters
                )

        if node.magnitude_mps <= 0.0:
            if self._preview_future is not None:
                self._preview_future_key = None
                if self._preview_future.cancel():
                    self._preview_future = None
            if self._preview_np is not None:
                self._preview_np.remove_node()
                self._preview_np = None
                self._preview_origin = None
            self._preview_encounters = []
            return

        if (
            self._preview_future is None
            and (
                self._preview_np is None
                or now_real - self._preview_submit_t > self.PREVIEW_THROTTLE_S
            )
        ):
            post_burn = apply_maneuver(vessel.state, node)
            self._preview_submit_t = now_real
            self._preview_future_key = preview_key
            self._preview_future = self._preview_executor.submit(
                self._sample_preview, post_burn
            )

    # Sphere-of-influence encounter patch: warm gold, distinct from the cyan
    # trajectory and magenta maneuver preview.
    ENCOUNTER_COLOR = (1.0, 0.58, 0.16, 1.0)
    MAX_ENCOUNTER_MARKERS = 4   # rarely more than 1-2 flybys visible on a path

    def _build_encounter_patches(self, parent, world_pts, origin, scale, encounters):
        """Draw each flyby SOI segment as a distinct-colored overlay parented to the
        trajectory node (so it inherits the floating-origin transform for free)."""
        for enc in encounters:
            if enc.body_name in ("Earth", "Sun"):
                continue   # primary / heliocentric-escape: no periapsis patch
            seg = world_pts[enc.start_index:enc.end_index + 1]
            if len(seg) < 2:
                continue
            pts = [tuple((p - origin) / scale) for p in seg]
            node = build_orbit_node(
                pts, color=self.ENCOUNTER_COLOR, thickness=3.0, fade_minimum=0.7
            )
            node.reparent_to(parent)   # local points already relative to origin/scale

    def _install_trajectory(self, idx, world_pts, scale, state) -> None:
        """Swap cached prediction points into Panda3D; called only on the render thread."""
        local_pts, origin = _localize_polyline(world_pts)
        pts = [tuple(p / scale) for p in local_pts]
        if self.orbit_nps[idx] is not None:
            self.orbit_nps[idx].remove_node()   # removes any child encounter patches too
        node = build_orbit_node(pts)
        node.reparent_to(self.render)
        node.set_pos(*self.transform.to_render(origin))
        self.orbit_nps[idx] = node
        self._traj_render_origin[idx] = origin
        encounters = self._traj_encounters[idx] if idx < len(self._traj_encounters) else None
        if encounters:
            self._build_encounter_patches(node, world_pts, origin, scale, encounters)
        if idx == 0:
            try:
                if self.world.solar_system:
                    from orbitsim.core.nbody import osculating_elements_solar
                    osc = osculating_elements_solar(state, state.epoch_s)
                else:
                    osc = state_to_elements(state)
                if osc.e >= 1.0:
                    raise ValueError("unbound trajectory has no apoapsis")
                from orbitsim.render.world_markers import apsis_points_on_path
                pe_point, ap_point = apsis_points_on_path(world_pts)
                self._apsis_positions["PE"] = pe_point
                self._apsis_positions["AP"] = ap_point
            except ValueError:
                self._apsis_positions["PE"] = None
                self._apsis_positions["AP"] = None

    def _rebuild_trajectory(self, idx, vessel, now_real) -> None:
        """Poll/submit N-body prediction work without blocking the render thread."""
        future = self._traj_future[idx]
        if future is not None and future.done():
            try:
                points, epochs, encounters = future.result()
                self._traj_world_cache[idx] = points
                self._traj_epoch_cache[idx] = epochs
                self._traj_encounters[idx] = encounters
            except Exception:
                self._traj_world_cache[idx] = None
                self._traj_epoch_cache[idx] = None
                self._traj_encounters[idx] = []
            self._traj_future[idx] = None
            self._traj_last_complete_t[idx] = now_real
            self._traj_render_scale[idx] = None

        scale = self.transform.scale_m_per_unit
        cached = self._traj_world_cache[idx]
        if cached is not None and self._traj_render_scale[idx] != scale:
            self._install_trajectory(idx, cached, scale, vessel.state)
            self._traj_render_scale[idx] = scale
        origin = self._traj_render_origin[idx]
        if origin is not None and self.orbit_nps[idx] is not None:
            self.orbit_nps[idx].set_pos(*self.transform.to_render(origin))

        if (
            self._traj_future[idx] is None
            and now_real - max(
                self._traj_last_submit_t[idx], self._traj_last_complete_t[idx]
            ) >= self.TRAJECTORY_REFRESH_S
        ):
            horizon = _trajectory_horizon_s(vessel.state, self.world.solar_system)
            n_pts = 256
            self._traj_last_submit_t[idx] = now_real
            self._traj_future[idx] = self._trajectory_executor.submit(
                self._sample_trajectory,
                vessel.state,
                n_pts,
                horizon,
                2,
                True,   # with_times
                True,   # with_encounters
            )

    def _update_responsive_ui(self) -> None:
        """Apply breakpoint changes only when the viewport dimensions change."""
        size = (self.win.get_x_size(), self.win.get_y_size())
        if size == self._ui_viewport or size[0] <= 0 or size[1] <= 0:
            return
        self._ui_viewport = size
        from orbitsim.render.ui.layout import ResponsiveLayout

        layout = ResponsiveLayout.calculate(*size)
        if hasattr(self, "_maneuver_frame"):
            scale = min(layout.ui_scale, 1.08)
            self._maneuver_frame.set_scale(scale)
            self._action_frame.set_scale(scale)

    def _update(self, task):
        real_dt = _global_clock.get_dt()
        self.frame_meter.add(real_dt)
        self.rig.update(real_dt)
        self._update_responsive_ui()

        if self.operation.status.running:
            status = self.operation.tick(steps=1)
            self.operation_overlay.update(status)
            return task.cont
        self.operation_overlay.update(self.operation.status)

        if self.solar_system:
            self.clock.advance(real_dt)
            self._update_solar_system()
            return task.cont

        # Flight input, then bound warp: 1x while thrusting (no integrating through warp),
        # else cap to the largest warp whose per-frame sub-step count stays in budget near
        # bodies (silent — the readout just won't climb past the cap).
        self._apply_flight_input(real_dt)
        if self.world.any_thrusting():
            self.clock.warp = 1.0
        elif self.world.vessels:
            from orbitsim.sim.clock import SimClock
            if self.world.solar_system:
                from orbitsim.core.nbody import max_safe_warp_solar
                cap = max_safe_warp_solar(
                    self.world.vessels[0].state, self.clock.sim_time_s, SimClock.WARP_STEPS)
            else:
                from orbitsim.core.nbody import max_safe_warp
                cap = max_safe_warp(
                    self.world.vessels[0].state, self.clock.sim_time_s, SimClock.WARP_STEPS)
            if self.clock.warp > cap:
                self.clock.warp = cap
        # Feed the current target's position to the TARGET/ANTITARGET SAS hold.
        target_pos = (self._target.state_at(self.clock.sim_time_s).r
                      if self._target is not None else None)
        maneuver_dir = None
        planned_node = self._current_node()
        if planned_node.magnitude_mps > 0.0:
            from orbitsim.core.maneuvers import maneuver_direction

            try:
                maneuver_dir = maneuver_direction(self.world.vessels[0].state, planned_node)
            except ValueError:
                pass
        for v in self.world.vessels:
            v.sas_target_pos = target_pos
            v.sas_maneuver_dir = maneuver_dir
        # An armed automatic burn lands exactly on its epoch instead of stepping past it
        # at high warp. This keeps the impulse and the displayed maneuver direction aligned.
        ttn_before_step = self._time_to_node()
        requested_dt = real_dt * self.clock.warp
        if (
            self._auto_execute_node
            and ttn_before_step is not None
            and 0.0 < ttn_before_step <= requested_dt
        ):
            sim_dt = ttn_before_step
            self.clock.sim_time_s += sim_dt
            self.clock.warp = 1.0
        else:
            sim_dt = self.clock.advance(real_dt)
        if self.world.solar_system:
            from orbitsim.core.nbody import refresh_ephemeris_cache
            refresh_ephemeris_cache(self.clock.sim_time_s)
        self.world.step(sim_dt)
        if self._auto_execute_node and self._time_to_node() is not None:
            if self._time_to_node() <= self.EXECUTE_TOLERANCE_S:
                self._auto_execute_node = False
                self._execute_burn()

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
            self._moon_orbit_np = build_orbit_node(
                moon_pts, color=REFERENCE_ORBIT_COLOR, thickness=1.4, fade_minimum=1.0
            )
            self._moon_orbit_np.reparent_to(self._orbit_frame)

        # Real Sun direction (Earth->Sun) drives the day/night terminator + sun
        # light. The cached wrapper serves DE440 positions when available and
        # falls back to the circular approximation offline — never raises.
        from orbitsim.core.nbody import _csun as _cached_sun
        sun_dir = np.asarray(_cached_sun(self.clock.sim_time_s).r, dtype=float)
        n = np.linalg.norm(sun_dir)
        if n > 0:
            sun_dir = sun_dir / n
            set_sun_dir(self.central_np, tuple(sun_dir))
            self._sun_light_np.set_pos(float(sun_dir[0]) * 1000.0,
                                       float(sun_dir[1]) * 1000.0,
                                       float(sun_dir[2]) * 1000.0)
            self._sun_light_np.look_at(0, 0, 0)

        import time as _time
        now_real = _time.monotonic()
        for idx, vessel in enumerate(self.world.vessels):
            vx, vy, vz = self.transform.to_render(vessel.state.r)
            self.vessel_nps[idx].set_pos(vx, vy, vz)
            self._rebuild_trajectory(idx, vessel, now_real)
        if self._preview_np is not None and self._preview_origin is not None:
            self._preview_np.set_pos(*self.transform.to_render(self._preview_origin))
        self._place_apsis_markers()

        # Ship view: cross-fade marker -> true-scale oriented model for vessel 0.
        if self._ship_model_np is not None:
            from orbitsim.render.ship_model import view_blend, model_node_scale
            v0 = self.world.vessels[0]
            marker_a, model_a = view_blend(self.rig.distance_m)
            if marker_a > 0.0:
                self.vessel_nps[0].show()
                self.vessel_nps[0].set_alpha_scale(marker_a)
            else:
                self.vessel_nps[0].hide()
            if model_a > 0.0:
                from panda3d.core import LQuaternion
                self._ship_model_np.show()
                self._ship_model_np.set_pos(*self.transform.to_render(v0.state.r))
                self._ship_model_np.set_scale(model_node_scale(self.transform.scale_m_per_unit))
                q = v0.orientation  # [w, x, y, z]
                self._ship_model_np.set_quat(LQuaternion(float(q[0]), float(q[1]),
                                                         float(q[2]), float(q[3])))
                self._ship_model_np.set_alpha_scale(model_a)
                thr = getattr(v0, "throttle", 0.0)
                if thr > 0.0:
                    self._plume_np.show()
                    self._plume_np.set_sz(0.5 + thr)
                    self._plume_np.set_alpha_scale(model_a * thr)
                else:
                    self._plume_np.hide()
            else:
                self._ship_model_np.hide()
                self._plume_np.hide()

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
        # N-body preview sampling is CPU-heavy, so it runs on one worker. Only the cheap
        # LineSegs swap happens here on the render thread when a result is ready.
        self._update_maneuver_preview(node, v0, _time.monotonic())
        # Node marker at the node's predicted position on the orbit (held at the vessel
        # through the brief execute window once the node is due).
        if self._node_epoch_s is not None and ttn is not None and ttn >= -self.EXECUTE_TOLERANCE_S:
            npos = apply_maneuver(v0.state, node).r
            mx, my, mz = self.transform.to_render(npos)
            if self._node_marker_np is None:
                self._node_marker_np = make_uv_sphere(1.0, 8, 12)
                self._node_marker_np.reparent_to(self.render)
                self._node_marker_np.set_color(0.3, 1.0, 1.0, 1.0)
                self._node_marker_np.set_light_off()
                self._node_marker_np.set_scale(6.0)
            self._node_marker_np.set_pos(mx, my, mz)
            if self._node_dv_label is None:
                from orbitsim.render.world_labels import build_world_label
                self._node_dv_label = build_world_label(
                    self.render, "", color=(0.3, 1.0, 1.0, 1.0), scale=11.0)
            dv_text = f"dV {node.magnitude_mps:,.0f} m/s"
            if ttn > 0:
                dv_text += f"  T-{_fmt_countdown(ttn)}"
            self._node_dv_label.node().set_text(dv_text)
            self._node_dv_label.set_pos(mx, my, mz + 8.0)
            self._node_dv_label.show()
        elif self._node_marker_np is not None:
            self._node_marker_np.remove_node()
            self._node_marker_np = None
            if self._node_dv_label is not None:
                self._node_dv_label.remove_node()
                self._node_dv_label = None
        # Auto-warp-down as the node nears (never warps up).
        if ttn is not None and 0.0 < ttn <= self.AUTO_WARP_LEAD_S * self.clock.warp and self.clock.warp > 1.0:
            self.clock.warp_down()
        # Pending-node readout (single node); shows "DUE" once within the execute window.
        if ttn is not None and ttn >= -self.EXECUTE_TOLERANCE_S:
            if ttn <= self.EXECUTE_TOLERANCE_S:
                label = "DUE — press Execute"
            else:
                label = f"in T-{_fmt_countdown(ttn)}"
            self._node_line = f"Node {label}   dV {node.magnitude_mps:,.1f} m/s"
        else:
            self._node_line = ""
        self._sync_maneuver_hud()

        # Moon position this frame.
        moon_now = moon_state_at(self.clock.sim_time_s)
        self._moon_np.set_pos(*self.transform.to_render(moon_now.r))
        # True-scale Moon body.
        from orbitsim.core.constants import R_MOON
        moon_render_pos = self.transform.to_render(moon_now.r)
        moon_body_scale = R_MOON / self.transform.scale_m_per_unit
        self._moon_body_np.set_pos(*moon_render_pos)
        self._moon_body_np.set_scale(max(moon_body_scale, 1e-3))
        self._moon_body_np.show()
        # Moon SOI wireframe: true-scale, brighter when the vessel is inside, camera-distance fade.
        soi_scale = MOON_SOI_M / self.transform.scale_m_per_unit
        self._soi_np.set_pos(*self.transform.to_render(moon_now.r))
        self._soi_np.set_scale(soi_scale)
        inside = float(np.linalg.norm(self.world.vessels[0].state.r - moon_now.r)) < MOON_SOI_M
        color = self.SOI_INSIDE_COLOR if inside else self.SOI_COLOR
        base_alpha = self.SOI_INSIDE_ALPHA if inside else self.SOI_BASE_ALPHA
        self._soi_np.set_color(color[0], color[1], color[2], base_alpha)
        fade = distance_fade(self.rig.distance_m, self.SOI_FADE_NEAR_M, self.SOI_FADE_FAR_M, minimum=0.0)
        self._soi_np.set_alpha_scale(fade)
        self._soi_np.show()
        if self._target is not None:
            target_now = self._target.state_at(self.clock.sim_time_s).r
            self._target_label_position = np.asarray(target_now).copy()
            tx, ty, tz = self.transform.to_render(target_now)
            self._target_label.node().set_text(f"TARGET: {self._target.name.upper()}")
            self._target_label.set_pos(tx, ty, tz + 9.0)
        else:
            self._target_label_position = None
            self._target_label.hide()
        # Lagrange points (rotate with the Moon). Each recompute runs three
        # brentq root-solves, so refresh by sim-time bucket instead of every
        # frame: the L-points move ~1 km/s, so a 10 s bucket is at most ~10 km
        # stale (invisible at map zoom) while cutting the per-frame cost to
        # zero at low warp. At high warp the bucket changes every frame and
        # the recompute cadence follows automatically.
        from orbitsim.core.nbody import earth_fixed_lagrange_points
        lag_bucket = int(self.clock.sim_time_s // self.LAGRANGE_REFRESH_S)
        if getattr(self, "_lagrange_bucket", None) != lag_bucket:
            self._lagrange_bucket = lag_bucket
            self._lagrange_cache = earth_fixed_lagrange_points(self.clock.sim_time_s)
        lps = self._lagrange_cache
        for name, mk, lbl in zip(("L1", "L2", "L3", "L4", "L5"),
                                 self._lagrange_nps, self._lagrange_labels):
            rx, ry, rz = self.transform.to_render(lps[name])
            mk.set_pos(rx, ry, rz)
            lbl.set_pos(rx, ry, rz + 6.0)
            self._lagrange_positions[name] = np.asarray(lps[name]).copy()
        # Planets + Sun: position markers, SOI spheres, and orbit reference lines.
        from orbitsim.core.nbody import (
            _csun, _cmercury, _cvenus, _cmars,
            _cjupiter, _csaturn, _curanus, _cneptune,
        )
        _planet_state_fns = {
            "Sun": _csun,
            "Mercury": _cmercury,
            "Venus": _cvenus,
            "Mars": _cmars,
            "Jupiter": _cjupiter,
            "Saturn": _csaturn,
            "Uranus": _curanus,
            "Neptune": _cneptune,
        }
        t_now = self.clock.sim_time_s
        vessel_r = self.world.vessels[0].state.r if self.world.vessels else np.zeros(3)
        for pname, state_fn in _planet_state_fns.items():
            pstate = state_fn(t_now)
            px, py, pz = self.transform.to_render(pstate.r)
            self._planet_sandbox_nps[pname].set_pos(px, py, pz)
            self._planet_sandbox_labels[pname].set_pos(px, py, pz + 8.0)
            if pname in self._planet_body_nps:
                body_np = self._planet_body_nps[pname]
                body_r = self._planet_body_radii[pname]
                body_scale = body_r / self.transform.scale_m_per_unit
                body_np.set_pos(px, py, pz)
                body_np.set_scale(max(body_scale, 1e-3))
                body_np.show()
            if pname in self._planet_soi_nps:
                soi_np = self._planet_soi_nps[pname]
                soi_r = self._planet_soi_radii[pname]
                soi_scale = soi_r / self.transform.scale_m_per_unit
                soi_np.set_pos(px, py, pz)
                soi_np.set_scale(soi_scale)
                dist_to = float(np.linalg.norm(vessel_r - pstate.r))
                inside = dist_to < soi_r
                alpha = self.SOI_INSIDE_ALPHA if inside else self.SOI_BASE_ALPHA
                soi_np.set_alpha_scale(
                    distance_fade(self.rig.distance_m, soi_r * 1.5, soi_r * 15, minimum=0.0)
                )
                soi_np.set_color_scale(1, 1, 1, alpha)
                soi_np.show()
            if pname == "Saturn" and hasattr(self, "_saturn_ring_np"):
                from orbitsim.core.constants import R_SATURN
                ring_scale = R_SATURN / self.transform.scale_m_per_unit
                self._saturn_ring_np.set_pos(px, py, pz)
                self._saturn_ring_np.set_scale(max(ring_scale, 1e-3))
                self._saturn_ring_np.show()
        # Earth SOI sphere (centered at origin, visible when far from Earth).
        earth_soi_scale = EARTH_SOI_M / self.transform.scale_m_per_unit
        ex, ey, ez = self.transform.to_render(np.zeros(3))
        self._earth_soi_np.set_pos(ex, ey, ez)
        self._earth_soi_np.set_scale(earth_soi_scale)
        earth_dist = float(np.linalg.norm(vessel_r))
        earth_inside = earth_dist < EARTH_SOI_M
        earth_alpha = self.SOI_INSIDE_ALPHA if earth_inside else self.SOI_BASE_ALPHA
        self._earth_soi_np.set_alpha_scale(
            distance_fade(self.rig.distance_m, EARTH_SOI_M * 1.5, EARTH_SOI_M * 15, minimum=0.0)
        )
        self._earth_soi_np.set_color_scale(1, 1, 1, earth_alpha)
        self._earth_soi_np.show()
        # Planet orbit reference lines (heliocentric circles, offset by -Earth(t)).
        from orbitsim.core.nbody import _csun
        sun_geo = _csun(t_now).r
        sx, sy, sz = self.transform.to_render(sun_geo)
        self._sun_orbit_frame.set_pos(sx, sy, sz)
        # Refresh monthly in simulation time.  DE440 positions are in tilted ICRF
        # axes and the real orbits are eccentric, so flat mean-radius circles do
        # not describe the paths followed by the rendered planet markers.
        orbit_epoch_bucket = int(np.floor(t_now / (30.0 * 86400.0)))
        if (self._planet_orbit_scale != scale
                or self._planet_orbit_epoch_bucket != orbit_epoch_bucket):
            self._planet_orbit_scale = scale
            self._planet_orbit_epoch_bucket = orbit_epoch_bucket
            _orbit_colors = {
                "Mercury": (0.5, 0.5, 0.5, 0.5),
                "Venus": (0.85, 0.75, 0.45, 0.5),
                "Earth": (0.3, 0.5, 1.0, 0.5),
                "Mars": (0.85, 0.4, 0.2, 0.5),
                "Jupiter": (0.8, 0.7, 0.5, 0.4),
                "Saturn": (0.9, 0.8, 0.6, 0.4),
                "Uranus": (0.6, 0.85, 0.9, 0.4),
                "Neptune": (0.3, 0.4, 0.9, 0.4),
            }
            earth_state = StateVector(
                r=np.zeros(3), v=np.zeros(3), mu=0.0, epoch_s=t_now
            )
            orbit_states = {
                "Mercury": _cmercury(t_now),
                "Venus": _cvenus(t_now),
                "Earth": earth_state,
                "Mars": _cmars(t_now),
                "Jupiter": _cjupiter(t_now),
                "Saturn": _csaturn(t_now),
                "Uranus": _curanus(t_now),
                "Neptune": _cneptune(t_now),
            }
            sun_state = _csun(t_now)
            for oname in ("Mercury", "Venus", "Earth", "Mars",
                          "Jupiter", "Saturn", "Uranus", "Neptune"):
                if oname in self._planet_orbit_nps:
                    self._planet_orbit_nps[oname].remove_node()
                orbit_points = sample_relative_orbit_points(
                    orbit_states[oname], sun_state, MU_SUN, n=256
                )
                pts = [tuple(point / scale) for point in orbit_points]
                color = _orbit_colors[oname]
                node = build_orbit_node(pts, color=color, thickness=1.2, fade_minimum=1.0)
                node.reparent_to(self._sun_orbit_frame)
                self._planet_orbit_nps[oname] = node
        # Closest approach to the current target (throttled recompute). Both the ship
        # trajectory and the target are referenced to the same base epoch (the node epoch
        # when a burn is planned, else now) so they are compared at matching absolute times;
        # the absolute CA epoch is cached so the markers hold steady between recomputes (warp).
        if self._target is not None and not self._target.supports_closest_approach:
            # Lagrange-point target: live distance + relative speed, no closest-approach
            # prediction (an L-point is not Keplerian — you fly to it and null relative velocity).
            self._ca = None
            for key, label in self._ca_labels.items():
                self._ca_label_positions[key] = None
                label.hide()
            L = self._target.state_at(self.clock.sim_time_s)
            dist = float(np.linalg.norm(v0.state.r - L.r))
            relsp = float(np.linalg.norm(v0.state.v - L.v))
            self._target_line = (
                f"Target: {self._target.name}   dist {_fmt_dist(dist)}"
                f"   rel {relsp:,.0f} m/s"
            )
            self._sync_maneuver_hud()
        elif self._target is not None:
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
                # CA stays Keplerian. The target is the on-rails Moon — propagating it under
                # earth_moon_accel is singular (it sits at its own gravity source) — and a
                # per-sample N-body re-integration over a multi-day window is O(samples x window),
                # which would freeze the loop. The live N-body trajectory line already shows the
                # true perturbed path; this readout is an approximate planning aid, consistent
                # with the (also Keplerian) intercept/porkchop seeds.
                self._ca = closest_approach(
                    traj, self._target.state_at(base_epoch), window_s=window, coarse_samples=720)
                self._ca_traj = traj
                self._ca_abs_epoch = base_epoch + self._ca.t_ca_s
            ca = self._ca
            from orbitsim.core.propagate import propagate_kepler
            ship_at = propagate_kepler(self._ca_traj, ca.t_ca_s).r
            tgt_at = self._target.state_at(self._ca_abs_epoch).r
            self._ca_marker("_ca_marker_ship", (1.0, 0.5, 0.2, 1.0)).set_pos(
                *self.transform.to_render(ship_at))
            self._ca_marker("_ca_marker_moon", (1.0, 0.8, 0.3, 1.0)).set_pos(
                *self.transform.to_render(tgt_at))
            self._ca_label_positions["ship"] = np.asarray(ship_at).copy()
            self._ca_label_positions["target"] = np.asarray(tgt_at).copy()
            for key, world_pos in self._ca_label_positions.items():
                rx, ry, rz = self.transform.to_render(world_pos)
                label = self._ca_labels[key]
                label.node().set_text(
                    "CA: SHIP" if key == "ship" else f"CA: {self._target.name.upper()}"
                )
                label.set_pos(rx, ry, rz + 8.0)
            countdown = max(0.0, self._ca_abs_epoch - self.clock.sim_time_s)
            self._target_line = (
                f"Target: {self._target.name}   CA T-{_fmt_countdown(countdown)}"
                f"   sep {ca.separation_m / 1000:,.0f} km   rel {ca.rel_speed_mps:,.0f} m/s"
            )
            self._sync_maneuver_hud()

        self._apply_mouse_orbit()
        self._update_starfield()
        self.rig.apply()
        self._update_marker_readability(focus)

        v0 = self.world.vessels[0]
        # Osculating elements about the dominant body.
        from orbitsim.core.constants import MU_MOON
        from orbitsim.core.bodies import MOON as MOON_BODY
        if self.world.solar_system:
            from orbitsim.core.nbody import osculating_elements_solar, dominant_body_solar
            elem = osculating_elements_solar(v0.state, self.clock.sim_time_s)
            dom_body, dom_pos = dominant_body_solar(v0.state.r, self.clock.sim_time_s)
            ref_radius = dom_body.radius_m
        else:
            from orbitsim.core.nbody import osculating_elements
            elem = osculating_elements(v0.state, self.clock.sim_time_s)
            moon_dominant = elem.mu == MU_MOON
            dom_body = MOON_BODY if moon_dominant else self.world.central
            dom_pos = np.zeros(3) if dom_body == self.world.central else moon_state_at(self.clock.sim_time_s).r
            ref_radius = dom_body.radius_m
        rp = elem.periapsis_radius
        ra = elem.apoapsis_radius
        try:
            period = elem.period_s
        except ValueError:
            period = float("nan")
        dist_from_dom = float(np.linalg.norm(v0.state.r - dom_pos))
        self.hud.update(
            sim_time_s=self.clock.sim_time_s,
            altitude_m=dist_from_dom - ref_radius,
            speed_mps=v0.state.v_mag,
            periapsis_m=rp - ref_radius,
            apoapsis_m=ra - ref_radius,
            period_s=period,
            inclination_rad=elem.i,
            dominant_body=dom_body.name,
        )
        g_local = dom_body.mu / max(dist_from_dom, 1.0) ** 2
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
        self.navball.update(
            orientation_q=v0.orientation,
            state=v0.state,
            target_pos=target_pos,
            maneuver_dir=maneuver_dir,
        )
        from orbitsim.core.attitude import heading_pitch
        heading, pitch = heading_pitch(v0.orientation, v0.state)
        self.sas_panel.update(v0.sas_mode, heading, pitch)
        target_rel_speed = None
        if self._target is not None:
            target_velocity = self._target.state_at(self.clock.sim_time_s).v
            target_rel_speed = float(np.linalg.norm(v0.state.v - target_velocity))
        self.vel_readout.update(v0.state.v_mag, target_rel_speed)
        self._update_encounter_info(v0, dom_body, dom_pos)
        self._update_encounter_cues()
        self._update_warp_readout()
        return task.cont

    def _update_encounter_cues(self) -> None:
        """Place predicted-flyby periapsis markers and set the HUD approach line.

        Shows the post-burn preview's encounters while a maneuver is planned,
        otherwise the live trajectory's. The HUD line is only set when the vessel
        is not already inside an SOI (the live flyby readout takes that slot)."""
        if not hasattr(self, "_encounter_marker_nps"):
            return
        now = self.clock.sim_time_s
        source = (self._preview_encounters
                  if (self._preview_np is not None and self._preview_encounters)
                  else (self._traj_encounters[0] or []))
        flybys = sorted(
            (e for e in source
             if e.body_name not in ("Earth", "Sun")
             and e.periapsis_epoch_s > now - self.EXECUTE_TOLERANCE_S),
            key=lambda e: e.periapsis_epoch_s,
        )
        for i, marker in enumerate(self._encounter_marker_nps):
            label = self._encounter_labels[i]
            if i >= len(flybys):
                marker.hide()
                label.hide()
                continue
            enc = flybys[i]
            rx, ry, rz = self.transform.to_render(enc.periapsis_point_m)
            marker.set_pos(rx, ry, rz)
            marker.show()
            pe_alt_km = (enc.periapsis_radius_m - enc.body_radius_m) / 1000.0
            tag = "IMPACT" if enc.impact else f"Pe {pe_alt_km:,.0f} km"
            countdown = max(0.0, enc.periapsis_epoch_s - now)
            label.node().set_text(f"{enc.body_name} {tag}  T-{_fmt_countdown(countdown)}")
            label.set_pos(rx, ry, rz + 8.0)
            label.show()
        # Predicted approach line — only when not already inside an SOI.
        if flybys and not getattr(self, "_encounter_line", ""):
            enc = flybys[0]
            pe_alt_km = (enc.periapsis_radius_m - enc.body_radius_m) / 1000.0
            tag = "impact" if enc.impact else f"Pe {pe_alt_km:,.0f} km"
            countdown = max(0.0, enc.periapsis_epoch_s - now)
            v_inf = enc.v_inf_mps / 1000.0 if np.isfinite(enc.v_inf_mps) else 0.0
            self._encounter_line = (
                f"Approach {enc.body_name}: {tag}  "
                f"T-{_fmt_countdown(countdown)}  v-inf {v_inf:,.1f} km/s"
            )
            self._sync_maneuver_hud()

    def _helio_body_pos(self, name: str, t_s: float) -> np.ndarray:
        """Heliocentric position [m]: DE440 when available, circular fallback offline."""
        from orbitsim.core.ephemeris import body_state, EphemerisUnavailableError
        try:
            return body_state(name.upper(), t_s, center="SUN").r
        except EphemerisUnavailableError:
            from orbitsim.core import planets as pl
            # planets.py states are geocentric; helio = geocentric - sun_geocentric.
            if name == "Earth":
                return -pl.sun_state_at(t_s).r
            geo_fns = {
                "Mercury": pl.mercury_state_at, "Venus": pl.venus_state_at,
                "Mars": pl.mars_state_at, "Jupiter": pl.jupiter_state_at,
                "Saturn": pl.saturn_state_at, "Uranus": pl.uranus_state_at,
                "Neptune": pl.neptune_state_at,
            }
            return geo_fns[name](t_s).r - pl.sun_state_at(t_s).r

    def _update_solar_system(self) -> None:
        """Place the Sun + planets for the current sim time (DE440 or fallback)."""
        from datetime import datetime, timedelta
        from orbitsim.core.ephemeris import available as ephemeris_available

        t = self.clock.sim_time_s
        self.transform.set_origin(np.zeros(3))  # heliocentric: Sun fixed at origin

        for body, marker, label in zip(self._planet_bodies, self._planet_nps, self._planet_labels):
            pos_m = np.zeros(3) if body.name == "Sun" else self._helio_body_pos(body.name, t)
            rx, ry, rz = self.transform.to_render(pos_m)
            marker.set_pos(rx, ry, rz)
            label.set_pos(rx, ry, rz + 6.0)

        self.central_np.set_pos(*self.transform.to_render(np.zeros(3)))
        self._apply_mouse_orbit()
        self._update_starfield()
        self.rig.apply()

        self._update_warp_readout()
        source = "JPL DE440" if ephemeris_available() else "circular approx (offline)"
        date = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=t)
        self.hud.text.setText(
            f"Solar system ({source})\n"
            f"Date: {date:%Y-%m-%d}\n"
            f"',' / '.' or the buttons change warp"
        )

    def run_app(self) -> None:
        self.run()
