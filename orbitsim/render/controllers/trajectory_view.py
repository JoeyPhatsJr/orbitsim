"""Live trajectory lines, encounter patches/cues, and teleport-coast operations."""
from orbitsim.core.elements import state_to_elements
from orbitsim.render.hud.format import fmt_countdown
from orbitsim.render.orbit_lines import build_orbit_node
from orbitsim.render.trajectory_helpers import coast_chunks
from orbitsim.render.trajectory_helpers import localize_polyline
from orbitsim.render.trajectory_sampling import sample_trajectory
from orbitsim.render.trajectory_sampling import trajectory_horizon_s as _trajectory_horizon_s
import numpy as np


class TrajectoryView:
    """Live trajectory lines, encounter patches/cues, and teleport-coast operations. Stateless: all state lives on the OrbitApp it is given."""

    def __init__(self, app) -> None:
        self.app = app

    def _cancel_operation(self) -> None:
        self.app.operation.cancel()
        rollback = getattr(self.app, "_operation_rollback", None)
        if rollback is not None:
            vessel, state, orientation, clock_time = rollback
            vessel.state = state
            vessel.orientation = orientation
            self.app.clock.sim_time_s = clock_time
        self.app._operation_rollback = None
        self.app.operation_overlay.update(self.app.operation.status)
        self.app._flash_message("Operation cancelled")

    def _jump_operation(self, duration_s):
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
        self.app._execute_burn()

    def _trajectory_coast_operation(self, duration_s):
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
                pts, color=self.app.ENCOUNTER_COLOR, thickness=3.0, fade_minimum=0.7
            )
            node.reparent_to(parent)   # local points already relative to origin/scale

    def _install_trajectory(self, idx, world_pts, scale, state) -> None:
        """Swap cached prediction points into Panda3D; called only on the render thread."""
        local_pts, origin = localize_polyline(world_pts)
        pts = [tuple(p / scale) for p in local_pts]
        if self.app.orbit_nps[idx] is not None:
            self.app.orbit_nps[idx].remove_node()   # removes any child encounter patches too
        node = build_orbit_node(pts)
        node.reparent_to(self.app.render)
        node.set_pos(*self.app.transform.to_render(origin))
        self.app.orbit_nps[idx] = node
        self.app._traj_render_origin[idx] = origin
        encounters = self.app._traj_encounters[idx] if idx < len(self.app._traj_encounters) else None
        if encounters:
            self._build_encounter_patches(node, world_pts, origin, scale, encounters)
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
            except ValueError:
                self.app._apsis_positions["PE"] = None
                self.app._apsis_positions["AP"] = None

    def _rebuild_trajectory(self, idx, vessel, now_real) -> None:
        """Poll/submit N-body prediction work without blocking the render thread."""
        future = self.app._traj_future[idx]
        if future is not None and future.done():
            try:
                points, epochs, encounters = future.result()
                self.app._traj_world_cache[idx] = points
                self.app._traj_epoch_cache[idx] = epochs
                self.app._traj_encounters[idx] = encounters
            except Exception:
                self.app._traj_world_cache[idx] = None
                self.app._traj_epoch_cache[idx] = None
                self.app._traj_encounters[idx] = []
            self.app._traj_future[idx] = None
            self.app._traj_last_complete_t[idx] = now_real
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
            ) >= self.app.TRAJECTORY_REFRESH_S
        ):
            horizon = _trajectory_horizon_s(vessel.state, self.app.world.solar_system)
            n_pts = 256
            self.app._traj_last_submit_t[idx] = now_real
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
            )

    def _update_encounter_cues(self) -> None:
        """Place predicted-flyby periapsis markers and set the HUD approach line.

        Shows the post-burn preview's encounters while a maneuver is planned,
        otherwise the live trajectory's. The HUD line is only set when the vessel
        is not already inside an SOI (the live flyby readout takes that slot)."""
        if not hasattr(self.app, "_encounter_marker_nps"):
            return
        now = self.app.clock.sim_time_s
        source = (self.app._preview_encounters
                  if (self.app._preview_np is not None and self.app._preview_encounters)
                  else (self.app._traj_encounters[0] or []))
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
            pe_alt_km = (enc.periapsis_radius_m - enc.body_radius_m) / 1000.0
            tag = "IMPACT" if enc.impact else f"Pe {pe_alt_km:,.0f} km"
            countdown = max(0.0, enc.periapsis_epoch_s - now)
            label.node().set_text(f"{enc.body_name} {tag}  T-{fmt_countdown(countdown)}")
            label.set_pos(rx, ry, rz + 8.0)
            label.show()
        # Predicted approach line — only when not already inside an SOI.
        if flybys and not getattr(self.app, "_encounter_line", ""):
            enc = flybys[0]
            pe_alt_km = (enc.periapsis_radius_m - enc.body_radius_m) / 1000.0
            tag = "impact" if enc.impact else f"Pe {pe_alt_km:,.0f} km"
            countdown = max(0.0, enc.periapsis_epoch_s - now)
            v_inf = enc.v_inf_mps / 1000.0 if np.isfinite(enc.v_inf_mps) else 0.0
            self.app._encounter_line = (
                f"Approach {enc.body_name}: {tag}  "
                f"T-{fmt_countdown(countdown)}  v-inf {v_inf:,.1f} km/s"
            )
            self.app._sync_maneuver_hud()
