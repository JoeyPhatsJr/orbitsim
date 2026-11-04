"""Panda3D ShowBase bootstrap and per-frame loop."""

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import ClockObject

from orbitsim.render.floating_origin import RenderTransform
from orbitsim.render.camera_rig import CameraRig
from orbitsim.render.hud import Hud
from orbitsim.render.keybind_overlay import KeybindOverlay, SANDBOX_BINDINGS, SOLAR_BINDINGS
from orbitsim.render.settings_panel import SettingsPanel
from orbitsim.render.process_pool import build_process_pool
from orbitsim.render.line_style import ENCOUNTER_STYLE, px_scale

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
from orbitsim.render.controllers.readouts import Readouts

_global_clock = ClockObject.get_global_clock()


class OrbitApp(ShowBase):
    """Renders one central body + vessels with orbit lines; time-warpable."""

    # Constants shared by several controllers (single-user constants live on their controller).
    EXECUTE_TOLERANCE_S = 2.0     # execute allowed only within this of the node epoch
    PREDICTION_MIN_SUBSTEP_S = 120.0  # coarse substep floor for the visual prediction only
                                      # (sim is untouched): escape climb-out 4.6s -> ~1.7s
    QUICKSAVE_PATH = "saves/quicksave.json"   # the harness redirects this to a temp file
    # Sphere-of-influence encounter patch: warm gold, distinct from the cyan
    # trajectory and magenta maneuver preview.
    ENCOUNTER_COLOR = ENCOUNTER_STYLE.color

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
        self.readouts = Readouts(self)
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
        self.title_screen.build_title_screen()

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
        self.operation_overlay = OperationOverlay(self, self.trajectory_view.cancel_operation)
        bindings = SOLAR_BINDINGS if self.solar_system else SANDBOX_BINDINGS
        self.keybind_overlay = KeybindOverlay(self.aspect2d, bindings)
        self.settings_panel = SettingsPanel(
            self.aspect2d, self.hud.set_units,
            on_unlimited_toggle=self.maneuver_editor.set_unlimited_dv,
            enable_unlimited=not self.solar_system)
        self.warp_controls.build_warp_controls()

        self.bodies_view.build_central_body()
        self.bodies_view.build_lighting()
        self.vessel_view.build_vessel_markers()
        self.vessel_view.build_ship_view()
        self.bodies_view.build_orbit_frame()
        self._build_background_work()
        if self.solar_system:
            self.solar_view.build_planets()
        else:
            self.readouts.build_sandbox_controls()
            self.targeting.build_targeting()
            self.readouts.build_apsis_and_encounter_markers()
            self.bodies_view.build_moon_and_lagrange()
            self.bodies_view.build_sandbox_planets()
            self.vessel_view.apply_opening_camera()
        self.bodies_view.build_starfield()

        self.flight_input.setup_input()
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

    def line_px_scale(self) -> float:
        """Line-width multiplier for the current framebuffer (Retina-aware; 1.0 headless)."""
        if self.win is None:
            return 1.0
        return px_scale(self.win.get_fb_y_size())

    def _flash_message(self, text: str) -> None:
        """Transient on-screen user feedback (toast)."""
        self.hud.flash(text)

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
            self.solar_view.update_solar_system()
            return task.cont

        # Apply any finished background planning work (runs off the render thread).
        self.targeting.poll_planning()
        self.porkchop_modal.poll_porkchop()

        target_pos, maneuver_dir = self._frame_flight(real_dt)

        # Jog sliders accumulate delta-V at a real-time (warp-independent) rate.
        self.maneuver_editor.apply_jogs(real_dt)

        focus, scale = self._frame_origin()
        self.bodies_view.frame_moon_ring(scale)
        self.bodies_view.frame_sun()
        self.vessel_view.frame_vessels()
        self.vessel_view.frame_ship_view()
        node = self.maneuver_editor.frame_maneuver_node()
        self.bodies_view.frame_moon()
        self.targeting.frame_target_label()
        self.bodies_view.frame_lagrange()
        self.bodies_view.frame_planets(scale)
        self.targeting.frame_closest_approach(node)

        self.flight_input.apply_mouse_orbit()
        self.bodies_view.update_starfield()
        self.rig.apply()
        self.readouts.update_marker_readability(focus)

        self.readouts.frame_readouts(target_pos, maneuver_dir)
        return task.cont

    def _frame_flight(self, real_dt):
        """Flight input, warp bounds, SAS feeds, armed-burn landing, world step."""
        # Flight input, then bound warp: 1x while thrusting (no integrating through warp),
        # else cap to the largest warp whose per-frame sub-step count stays in budget near
        # bodies (silent — the readout just won't climb past the cap).
        self.flight_input.apply_flight_input(real_dt)
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
        planned_node = self.maneuver_editor.current_node()
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
        ttn_before_step = self.maneuver_editor.time_to_node()
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
        if self._auto_execute_node and self.maneuver_editor.time_to_node() is not None:
            if self.maneuver_editor.time_to_node() <= self.EXECUTE_TOLERANCE_S:
                self._auto_execute_node = False
                self.maneuver_editor.execute_burn()
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

    def run_app(self) -> None:
        self.run()
