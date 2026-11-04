"""Panda3D ShowBase bootstrap and per-frame loop."""

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject,
)

from orbitsim.core.moon import moon_state_at
from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.world_markers import distance_fade
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.hud import Hud
from orbitsim.render.keybind_overlay import KeybindOverlay, SANDBOX_BINDINGS, SOLAR_BINDINGS
from orbitsim.render.settings_panel import SettingsPanel
from orbitsim.render.process_pool import build_process_pool

from orbitsim.render.controllers.title_screen import TitleScreen

from orbitsim.render.controllers.solar_view import SolarView

from orbitsim.render.controllers.warp_controls import WarpControls

from orbitsim.render.controllers.flight_input import FlightInput

from orbitsim.render.controllers.quicksave import Quicksave

from orbitsim.render.controllers.targeting import Targeting

from orbitsim.render.controllers.porkchop_modal import PorkchopModal

from orbitsim.render.controllers.trajectory_view import TrajectoryView

from orbitsim.render.controllers.maneuver_editor import ManeuverEditor

from orbitsim.render.controllers.bodies_view import BodiesView

from orbitsim.render.controllers.vessel_view import VesselView

_global_clock = ClockObject.get_global_clock()


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
        self.vessel_view = VesselView(self)
        self.bodies_view = BodiesView(self)
        self.maneuver_editor = ManeuverEditor(self)
        self.trajectory_view = TrajectoryView(self)
        self.porkchop_modal = PorkchopModal(self)
        self.targeting = Targeting(self)
        self.quicksave = Quicksave(self)
        self.flight_input = FlightInput(self)
        self.warp_controls = WarpControls(self)
        self.solar_view = SolarView(self)
        self.title_screen = TitleScreen(self)
        self.title_screen._build_title_screen()

    def destroy(self) -> None:
        # De-duplicate: _compute_executor and _grid_executor may both be _pool.
        seen = set()
        for attr in ("_planning_executor", "_compute_executor", "_grid_executor", "_pool"):
            executor = getattr(self, attr, None)
            if executor is not None and id(executor) not in seen:
                seen.add(id(executor))
                executor.shutdown(wait=False, cancel_futures=True)
            setattr(self, attr, None)
        super().destroy()

    # ------------------------------------------------------------------ title screen

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
        self.operation_overlay = OperationOverlay(self, self.trajectory_view._cancel_operation)
        bindings = SOLAR_BINDINGS if self.solar_system else SANDBOX_BINDINGS
        self.keybind_overlay = KeybindOverlay(self.aspect2d, bindings)
        self.settings_panel = SettingsPanel(
            self.aspect2d, self.hud.set_units,
            on_unlimited_toggle=self.maneuver_editor._set_unlimited_dv,
            enable_unlimited=not self.solar_system)
        self.warp_controls._build_warp_controls()

        self.bodies_view._build_central_body()
        self.bodies_view._build_lighting()
        self.vessel_view._build_vessel_markers()
        self.vessel_view._build_ship_view()
        self.bodies_view._build_orbit_frame()
        self._build_background_work()
        if self.solar_system:
            self.solar_view._build_planets()
        else:
            self._build_sandbox_controls()
            self.targeting._build_targeting()
            self._build_apsis_and_encounter_markers()
            self.bodies_view._build_moon_and_lagrange()
            self.bodies_view._build_sandbox_planets()
        self.bodies_view._build_starfield()

        self.flight_input._setup_input()
        self.task_mgr.add(self._update, "update")

    def _build_background_work(self):
        """Preview state, process pool and planning executors."""
        self._preview_np = None
        self._preview_origin = None
        self._preview_submit_t = 0.0
        self._preview_future = None
        self._preview_future_key = None
        self._preview_encounters = []
        if not self.solar_system:
            from concurrent.futures import ThreadPoolExecutor
            # One persistent process pool does all heavy CPU work: the planning grid
            # search (fanned across cores) and the flight-time trajectory/preview
            # sampling (so the live line and the maneuver preview run in separate
            # processes instead of GIL-serialising against each other and the render
            # thread). Falls back to threads where spawning workers isn't possible.
            self._pool = build_process_pool()
            # The planning search always runs on a 1-worker *thread* so it never blocks
            # the render thread; inside that thread it fans grid cells across the pool.
            self._planning_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="planner"
            )
            if self._pool is not None:
                self._compute_executor = self._pool   # preview + live-line sampling
                self._grid_executor = self._pool      # optimize.py cell fan-out
            else:
                # Threaded fallback: two workers keep preview + live-line overlapping
                # as much as the GIL allows; grid search stays serial (executor=None).
                self._compute_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="compute"
                )
                self._grid_executor = None
            self._planning_future = None      # in-flight intercept plan (thread future)
            self._porkchop_future = None      # in-flight porkchop grid (thread future)
            self._porkchop_render = None      # (dep_times, tof_grid, label) for on-poll draw
            self._porkchop_request = 0
        self._porkchop_card = None

    def _build_sandbox_controls(self):
        """Maneuver editor, navball, SAS panel, velocity readout, dock."""
        # Maneuver editor (operates on vessel 0). The burn is applied at vessel 0's
        # *current* state (dt=0), so the executed orbit matches the live preview.
        self.maneuver_editor._build_maneuver_ui()
        self._fuel_capacity = (
            self.world.vessels[0].fuel_mass_kg if self.world.vessels else 0.0
        )
        from orbitsim.render.navball import Navball
        self.navball = Navball(self)
        from orbitsim.render.sas_panel import SasPanel, VelocityReadout
        self.sas_panel = SasPanel(
            self, on_set_mode=self.flight_input._set_sas, on_toggle=self.flight_input._toggle_sas
        )
        self.vel_readout = VelocityReadout(self, lambda: self.hud.units)
        from orbitsim.render.ui.widgets import PanelDock
        self.panel_dock = PanelDock(self, [
            ("ORBIT", self.hud.toggle_orbit),
            ("VESSEL", self.hud.toggle_vessel),
            ("MANEUVER", self.maneuver_editor._toggle_maneuver_drawer),
        ])
        self._maneuver_frame.hide()  # clean-flight default; MNV dock button opens it

    def _build_apsis_and_encounter_markers(self):
        """PE/AP markers + encounter periapsis cues."""
        from orbitsim.render.world_labels import build_labeled_marker, build_world_label
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
    MOON_RING_REFRESH_S = 6 * 3600.0  # sim-seconds between osculating Moon-ring rebuilds
    TRAJECTORY_REFRESH_S = 2.0    # prediction cadence; integration runs off the render thread
    PREDICTION_MIN_SUBSTEP_S = 120.0  # coarse substep floor for the visual prediction only
                                      # (sim is untouched): escape climb-out 4.6s -> ~1.7s

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

        points_px = [self.targeting._marker_px(item[2]) for item in candidates]
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
        self.maneuver_editor._sync_maneuver_hud()

    MOUSE_ORBIT_SENS = 3.0     # radians per unit of normalized mouse travel

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

    QUICKSAVE_PATH = "saves/quicksave.json"

    def _flash_message(self, text: str) -> None:
        """Transient on-screen user feedback (toast)."""
        self.hud.flash(text)

    # Sphere-of-influence encounter patch: warm gold, distinct from the cyan
    # trajectory and magenta maneuver preview.
    ENCOUNTER_COLOR = (1.0, 0.58, 0.16, 1.0)
    MAX_ENCOUNTER_MARKERS = 4   # rarely more than 1-2 flybys visible on a path

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
            self.solar_view._update_solar_system()
            return task.cont

        # Apply any finished background planning work (runs off the render thread).
        self.targeting._poll_planning()
        self.porkchop_modal._poll_porkchop()

        target_pos, maneuver_dir = self._frame_flight(real_dt)

        # Jog sliders accumulate delta-V at a real-time (warp-independent) rate.
        self.maneuver_editor._apply_jogs(real_dt)

        focus, scale = self._frame_origin()
        self.bodies_view._frame_moon_ring(scale)
        self.bodies_view._frame_sun()
        self.vessel_view._frame_vessels()
        self.vessel_view._frame_ship_view()
        node = self.maneuver_editor._frame_maneuver_node()
        self.bodies_view._frame_moon()
        self.targeting._frame_target_label()
        self.bodies_view._frame_lagrange()
        self.bodies_view._frame_planets(scale)
        self.targeting._frame_closest_approach(node)

        self.flight_input._apply_mouse_orbit()
        self.bodies_view._update_starfield()
        self.rig.apply()
        self._update_marker_readability(focus)

        self._frame_readouts(target_pos, maneuver_dir)
        return task.cont

    def _frame_flight(self, real_dt):
        """Flight input, warp bounds, SAS feeds, armed-burn landing, world step."""
        # Flight input, then bound warp: 1x while thrusting (no integrating through warp),
        # else cap to the largest warp whose per-frame sub-step count stays in budget near
        # bodies (silent — the readout just won't climb past the cap).
        self.flight_input._apply_flight_input(real_dt)
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
        planned_node = self.maneuver_editor._current_node()
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
        ttn_before_step = self.maneuver_editor._time_to_node()
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
        if self._auto_execute_node and self.maneuver_editor._time_to_node() is not None:
            if self.maneuver_editor._time_to_node() <= self.EXECUTE_TOLERANCE_S:
                self._auto_execute_node = False
                self.maneuver_editor._execute_burn()
        return target_pos, maneuver_dir

    def _frame_origin(self):
        """Recentre the floating origin on vessel 0; place central body + orbit frame."""
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
        return focus, scale

    def _frame_readouts(self, target_pos, maneuver_dir):
        """HUD, navball, SAS panel, velocity, encounters, warp."""
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
        self.trajectory_view._update_encounter_cues()
        self.warp_controls._update_warp_readout()

    def run_app(self) -> None:
        self.run()
