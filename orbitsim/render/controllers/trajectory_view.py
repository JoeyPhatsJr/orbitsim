"""Live trajectory lines, encounter patches/cues, and teleport-coast operations."""

from orbitsim.core.elements import state_to_elements
import numpy as np
from orbitsim.render.hud.format import encounter_label
from orbitsim.render.line_style import ENCOUNTER_STYLE, ORBIT_STYLE
from orbitsim.render.orbit_lines import build_orbit_node
from orbitsim.render.trajectory_helpers import (
    coast_chunks, encounter_source, localize_polyline,
)
from orbitsim.render.trajectory_sampling import (
    sample_trajectory,
    trajectory_horizon_s as _trajectory_horizon_s,
)


class TrajectoryView:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    TRAJECTORY_REFRESH_S = 2.0    # prediction cadence; integration runs off the render thread

    def __init__(self, app) -> None:
        self.app = app

    def invalidate_live_line(self, resubmit: bool = True) -> None:
        """The vessel's path just jumped: the drawn live line (and its target approach)
        shows the old one. ``resubmit`` asks for a fresh line without waiting for the
        refresh cadence (one-off jumps; not every thrusting frame)."""
        app = self.app
        app._orbit_generation += 1
        if resubmit:
            app._traj_last_submit_t = [float("-inf") for _ in app._traj_last_submit_t]
            app._traj_last_complete_t = [float("-inf") for _ in app._traj_last_complete_t]

    def live_line_current(self, idx: int = 0) -> bool:
        """Whether vessel idx's drawn live line was sampled from its current path."""
        return self.app._traj_generation[idx] == self.app._orbit_generation

    def cancel_operation(self) -> None:
        self.app.operation.cancel()
        rollback = getattr(self.app, "_operation_rollback", None)
        if rollback is not None:
            vessel, state, orientation, clock_time = rollback
            vessel.state = state
            vessel.orientation = orientation
            self.app.clock.sim_time_s = clock_time
            self.invalidate_live_line()
        self.app._operation_rollback = None
        self.app.operation_overlay.update(self.app.operation.status)
        self.app._flash_message("Operation cancelled")

    def jump_operation(self, duration_s):
        """Frame-budgeted coast generator; one ephemeris-aware day per rendered frame."""
        from orbitsim.core.nbody import refresh_ephemeris_cache

        chunks = coast_chunks(duration_s)
        total = len(chunks)
        for index, dt_s in enumerate(chunks):
            refresh_ephemeris_cache(self.app.clock.sim_time_s + 0.5 * dt_s)
            self.app.world.step(dt_s)
            self.app.clock.sim_time_s += dt_s
            yield (index + 1) / total
        self.app._operation_rollback = None
        self.app.maneuver_editor.execute_burn()

    def trajectory_coast_operation(self, duration_s):
        """Frame-budgeted coast to a clicked future trajectory sample."""
        from orbitsim.core.nbody import refresh_ephemeris_cache

        chunks = coast_chunks(duration_s)
        total = len(chunks)
        for index, dt_s in enumerate(chunks):
            refresh_ephemeris_cache(self.app.clock.sim_time_s + 0.5 * dt_s)
            self.app.world.step(dt_s)
            self.app.clock.sim_time_s += dt_s
            yield (index + 1) / total
        self.app._operation_rollback = None
        self.app._traj_world_cache[0] = None
        self.app._traj_epoch_cache[0] = None
        self.app._flash_message("Arrived at trajectory point")

    def build_encounter_patches(self, parent, world_pts, origin, scale, encounters,
                                px_scale=1.0):
        """Draw each flyby SOI segment as a distinct-colored overlay parented to the
        trajectory node (so it inherits the floating-origin transform for free)."""
        for enc in encounters:
            if enc.body_name in ("Earth", "Sun"):
                continue   # primary / heliocentric-escape: no periapsis patch
            seg = world_pts[enc.start_index:enc.end_index + 1]
            if len(seg) < 2:
                continue
            pts = [tuple((p - origin) / scale) for p in seg]
            node = build_orbit_node(pts, ENCOUNTER_STYLE, px_scale=px_scale)
            node.reparent_to(parent)   # local points already relative to origin/scale

    def _install_trajectory(self, idx, world_pts, scale, state) -> None:
        """Swap cached prediction points into Panda3D; called only on the render thread."""
        local_pts, origin = localize_polyline(world_pts)
        pts = [tuple(p / scale) for p in local_pts]
        if self.app.orbit_nps[idx] is not None:
            self.app.orbit_nps[idx].remove_node()   # removes any child encounter patches too
        px = self.app.line_px_scale()
        node = build_orbit_node(pts, ORBIT_STYLE, px_scale=px)
        node.reparent_to(self.app.render)
        node.set_pos(*self.app.transform.to_render(origin))
        self.app.orbit_nps[idx] = node
        self.app._traj_render_origin[idx] = origin
        encounters = self.app._traj_encounters[idx] if idx < len(self.app._traj_encounters) else None
        if encounters:
            self.build_encounter_patches(node, world_pts, origin, scale, encounters, px_scale=px)
        if idx == 0:
            try:
                if self.app.world.solar_system:
                    from orbitsim.core.nbody import osculating_elements_solar
                    osc = osculating_elements_solar(state, state.epoch_s)
                else:
                    osc = state_to_elements(state)
                if osc.e >= 1.0:
                    raise ValueError("unbound trajectory has no apoapsis")
                from orbitsim.render.world_markers import apsis_points_on_path
                pe_point, ap_point = apsis_points_on_path(world_pts)
                self.app._apsis_positions["PE"] = pe_point
                self.app._apsis_positions["AP"] = ap_point
                # Same points the ticks sit on; the line is Earth-centred.
                radius = self.app.world.central.radius_m
                self.app._apsis_altitudes["PE"] = float(np.linalg.norm(pe_point)) - radius
                self.app._apsis_altitudes["AP"] = float(np.linalg.norm(ap_point)) - radius
            except ValueError:
                self.app._apsis_positions["PE"] = None
                self.app._apsis_positions["AP"] = None
                self.app._apsis_altitudes["PE"] = None
                self.app._apsis_altitudes["AP"] = None

    def rebuild_trajectory(self, idx, vessel, now_real) -> None:
        """Poll/submit N-body prediction work without blocking the render thread."""
        future = self.app._traj_future[idx]
        if future is not None and future.done():
            try:
                result = future.result()
                points, epochs, encounters = result[:3]
                self.app._traj_world_cache[idx] = points
                self.app._traj_epoch_cache[idx] = epochs
                self.app._traj_encounters[idx] = encounters
                self.app._traj_approach[idx] = result[3] if len(result) > 3 else None
            except Exception:
                self.app._traj_world_cache[idx] = None
                self.app._traj_epoch_cache[idx] = None
                self.app._traj_encounters[idx] = []
                self.app._traj_approach[idx] = None
            self.app._traj_future[idx] = None
            generation = self.app._traj_submit_generation[idx]
            self.app._traj_generation[idx] = generation
            # A line for an outdated path is replaced at once, not after the cadence.
            self.app._traj_last_complete_t[idx] = (
                now_real if generation == self.app._orbit_generation else float("-inf"))
            self.app._traj_render_scale[idx] = None

        scale = self.app.transform.scale_m_per_unit
        cached = self.app._traj_world_cache[idx]
        if cached is not None and self.app._traj_render_scale[idx] != scale:
            self._install_trajectory(idx, cached, scale, vessel.state)
            self.app._traj_render_scale[idx] = scale
        origin = self.app._traj_render_origin[idx]
        if origin is not None and self.app.orbit_nps[idx] is not None:
            self.app.orbit_nps[idx].set_pos(*self.app.transform.to_render(origin))

        if (
            self.app._traj_future[idx] is None
            and now_real - max(
                self.app._traj_last_submit_t[idx], self.app._traj_last_complete_t[idx]
            ) >= self.TRAJECTORY_REFRESH_S
        ):
            horizon = _trajectory_horizon_s(vessel.state, self.app.world.solar_system)
            n_pts = 256
            approach_body = self.app.targeting.aim_target_name() if idx == 0 else None
            self.app._traj_last_submit_t[idx] = now_real
            self.app._traj_submit_generation[idx] = self.app._orbit_generation
            self.app._traj_future[idx] = self.app._compute_executor.submit(
                sample_trajectory,
                vessel.state,
                self.app.world.solar_system,
                self.app.PREDICTION_MIN_SUBSTEP_S,
                n_pts,
                horizon,
                2,
                True,   # with_times
                True,   # with_encounters
                approach_body,
            )

    def update_encounter_cues(self) -> None:
        """Place predicted-flyby periapsis markers.

        Shows the exact post-burn preview's encounters while a maneuver is planned (none
        while it is being edited and the exact result is stale), otherwise the live
        trajectory's. (The target panel's status box says the same in words.)"""
        if not hasattr(self.app, "_encounter_marker_nps"):
            return
        now = self.app.clock.sim_time_s
        node = self.app.maneuver_editor.current_node()
        burn = node.magnitude_mps > 0.0
        source = encounter_source(
            burn, burn and self.app.maneuver_editor.exact_preview_current(node),
            self.app._preview_encounters, self.app._traj_encounters[0])
        flybys = sorted(
            (e for e in source
             if e.body_name not in ("Earth", "Sun")
             and e.periapsis_epoch_s > now - self.app.EXECUTE_TOLERANCE_S),
            key=lambda e: e.periapsis_epoch_s,
        )
        for i, marker in enumerate(self.app._encounter_marker_nps):
            label = self.app._encounter_labels[i]
            if i >= len(flybys):
                marker.hide()
                label.hide()
                continue
            enc = flybys[i]
            rx, ry, rz = self.app.transform.to_render(enc.periapsis_point_m)
            marker.set_pos(rx, ry, rz)
            marker.show()
            text = encounter_label(enc.body_name, enc.periapsis_radius_m - enc.body_radius_m,
                                   enc.impact)
            if label.node().get_text() != text:
                label.node().set_text(text)
            label.set_pos(rx, ry, rz + 8.0)
            label.show()
