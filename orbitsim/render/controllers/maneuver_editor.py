"""Maneuver-node editor: the burn panel, node timing, preview, execute, unlimited dV."""

import numpy as np
from direct.gui.DirectButton import DirectButton
from direct.gui.OnscreenText import OnscreenText
from orbitsim.core.maneuvers import ManeuverNode, apply_maneuver
from orbitsim.render.hud.format import burn_summary
from orbitsim.render.line_style import ESTIMATE_STYLE, PLAN_STYLE
from orbitsim.render.node_edit import (
    NUDGE_DV, NUDGE_KEYS, NUDGE_TIME, nudge_rate, snap_dv, snap_epoch,
)
from orbitsim.render.orbit_lines import build_orbit_node
from orbitsim.render.trajectory_helpers import (
    kepler_estimate_points, localize_polyline, maneuver_preview_key, preview_start_drifted,
    same_burn,
    should_refresh_exact, should_submit_exact,
)
from orbitsim.render.trajectory_sampling import sample_preview
from panda3d.core import TextNode


class ManeuverEditor:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    AUTO_WARP_LEAD_S = 5.0        # auto-warp-down when within this many real-seconds of the node
    PREVIEW_SETTLE_S = 0.25       # burn must be still this long before the exact preview runs
    PREVIEW_REFRESH_S = 0.5       # min real-seconds between drift refreshes of the same burn
    # Burn panel rows inside PANELS["burn"] (a2dBottomRight-relative). The buttons row
    # never moves; the readout above it and Execute below it show only with a burn.
    READOUT_Z = 0.855
    BUTTONS_Z = 0.755
    EXECUTE_Z = 0.645
    IDLE_FRAME_Z = (0.705, 0.815)   # frame bottom/top with only the buttons row

    def __init__(self, app) -> None:
        self.app = app

    def build_maneuver_ui(self) -> None:
        """The burn panel (bottom-right): a one-line readout of the burn, Next Pe /
        Next Ap / Delete burn, and Execute. The burn itself is shaped with the node
        handles and the nudge keys; the panel shows only what exists."""
        self.app._dv = {"pro": 0.0, "nrm": 0.0, "rad": 0.0}
        self.app._nudge_held = {}          # nudge key -> seconds held
        self.app._encounter_line = ""      # in-SOI flyby readout (target panel row)
        self.app._node_epoch_s = None      # absolute epoch of the scheduled node (None = none)
        self.app._auto_execute_node = False
        self.app._node_marker_np = None
        self.app._node_dv_label = None
        self.app._node_label_position = None   # world position of the burn's map label
        from orbitsim.render.ui.hud_layout import PANELS
        from orbitsim.render.ui.theme import THEME, button_options
        from orbitsim.render.ui.widgets import layout_panel_frame
        rect = PANELS["burn"]
        panel = layout_panel_frame(self.app, rect)
        self.app._burn_panel = panel
        self.app._hud_panels["burn"] = panel
        self.app._burn_readout_text = OnscreenText(
            text="", pos=(rect.x0 + 0.04, self.READOUT_Z), scale=0.046,
            fg=(1.0, 0.55, 1.0, 1.0), shadow=(0, 0, 0, 1), align=TextNode.ALeft,
            mayChange=True, parent=panel)
        row = [("Next Pe", -0.92, self._node_to_pe), ("Next Ap", -0.68, self._node_to_ap),
               ("Delete burn", -0.38, self.delete_burn)]
        self.app._burn_buttons = [
            DirectButton(text=label, scale=0.046, pos=(x, 0.0, self.BUTTONS_Z),
                         command=cmd, parent=panel, **button_options())
            for label, x, cmd in row]
        self.app._exec_btn = DirectButton(
            text="Execute Burn", scale=0.052, pos=(-0.57, 0.0, self.EXECUTE_Z),
            command=self.execute_burn, parent=panel, **button_options())
        self.app._burn_buttons.append(self.app._exec_btn)
        self.app._burn_panel_state = None  # last (readout, delete shown, exec label) drawn
        self.refresh_readout()

    # --- Burn editing API shared by the handles and the nudge keys ---

    def set_dv(self, axis: str, value_mps: float) -> None:
        """Set one delta-V component ('pro'/'nrm'/'rad') of the planned burn [m/s]."""
        self.app._dv[axis] = value_mps
        self.refresh_readout()

    def burn_editable(self, flash: bool = True) -> bool:
        """Whether a burn can be planned now: not while landed / at zero velocity (the
        burn frame and the node propagation are undefined there -- they crashed)."""
        v0 = self.app.world.vessels[0]
        ok = v0.landed_on is None and bool(np.any(v0.state.v))
        if not ok and flash:
            self.app._flash_message("Can't plan a burn while landed")
        return ok

    def place_node(self, epoch_s: float) -> None:
        """Schedule the (one) burn at ``epoch_s``; its delta-V is kept."""
        if not self.burn_editable():
            return
        self.app._auto_execute_node = False
        self.set_node_epoch(snap_epoch(epoch_s))

    def set_node_epoch(self, epoch_s: float) -> None:
        """Schedule the burn at ``epoch_s`` (never in the past)."""
        now = self.app.clock.sim_time_s
        self.app._node_epoch_s = max(now, float(epoch_s))

    def _nudge(self, axis: str, delta: float) -> None:
        if axis == "time":
            base = self.app._node_epoch_s
            if base is None:
                base = self.app.clock.sim_time_s
            self.set_node_epoch(base + delta)
        else:
            self.set_dv(axis, self.app._dv[axis] + delta)

    def on_nudge_key(self, key: str, down: bool) -> None:
        """Key down: apply the tap and start the hold ramp. Key up: stop and snap the axis."""
        axis, sign = NUDGE_KEYS[key]
        spec = NUDGE_TIME if axis == "time" else NUDGE_DV
        if down:
            if not self.burn_editable():
                return
            self._nudge(axis, sign * spec.tap)
            self.app._nudge_held[key] = 0.0
            return
        if self.app._nudge_held.pop(key, None) is None:
            return
        self._snap_axis(axis)

    def _snap_axis(self, axis: str) -> None:
        if axis == "time":
            if self.app._node_epoch_s is not None:
                self.set_node_epoch(snap_epoch(self.app._node_epoch_s))
        else:
            self.set_dv(axis, snap_dv(self.app._dv[axis]))

    def apply_nudges(self, real_dt: float) -> None:
        """Ramp every held nudge key; drop keys the input device says aren't held
        (a modifier pressed mid-hold turns the release into e.g. 'shift-j-up')."""
        for key in list(self.app._nudge_held):
            axis, sign = NUDGE_KEYS[key]
            if self.app.flight_input.key_held(key) is False or not self.burn_editable(False):
                del self.app._nudge_held[key]
                self._snap_axis(axis)
                continue
            held = self.app._nudge_held[key] + real_dt
            self.app._nudge_held[key] = held
            spec = NUDGE_TIME if axis == "time" else NUDGE_DV
            rate = nudge_rate(held, spec)
            if rate:
                self._nudge(axis, sign * rate * real_dt)

    def current_node(self) -> ManeuverNode:
        """Build the node from the current dV values at the scheduled epoch (or now if none)."""
        epoch = (self.app._node_epoch_s if self.app._node_epoch_s is not None
                 else self.app.world.vessels[0].state.epoch_s)
        return ManeuverNode(
            epoch_s=epoch,
            dv_prograde_mps=self.app._dv["pro"],
            dv_normal_mps=self.app._dv["nrm"],
            dv_radial_mps=self.app._dv["rad"],
        )

    def exact_preview_current(self, node) -> bool:
        """Whether the drawn exact preview is for this burn and the current target."""
        key = maneuver_preview_key(node, self.app._node_epoch_s,
                                   self.app.targeting.aim_target_name())
        return self.app._preview_exact_key == key

    def time_to_node(self):
        """Seconds until the scheduled node, or None if no node is scheduled."""
        if self.app._node_epoch_s is None:
            return None
        return self.app._node_epoch_s - self.app.clock.sim_time_s

    def _node_to_pe(self):
        from orbitsim.core.maneuvers import time_to_periapsis
        try:
            self.app._node_epoch_s = self.app.clock.sim_time_s + time_to_periapsis(self.app.world.vessels[0].state)
        except ValueError:
            pass  # unbound orbit: no apsis to target

    def _node_to_ap(self):
        from orbitsim.core.maneuvers import time_to_apoapsis
        try:
            self.app._node_epoch_s = self.app.clock.sim_time_s + time_to_apoapsis(self.app.world.vessels[0].state)
        except ValueError:
            pass

    def _clear_node(self):
        self.app._auto_execute_node = False
        self.app._node_epoch_s = None
        if self.app._node_marker_np is not None:
            self.app._node_marker_np.remove_node()
            self.app._node_marker_np = None
        if self.app._node_dv_label is not None:
            self.app._node_dv_label.remove_node()
            self.app._node_dv_label = None

    def delete_burn(self) -> None:
        """Remove the planned burn: no node time and no delta-V."""
        self._clear_node()
        for axis in self.app._dv:
            self.app._dv[axis] = 0.0
        self.refresh_readout()

    def refresh_readout(self) -> None:
        """Redraw the burn panel now (an edit, Execute, unlimited toggle)."""
        self.sync_maneuver_hud()

    def burn_summary_text(self, node=None) -> str:
        """The burn's one-line summary (panel readout and map label): size, time until,
        full-throttle duration."""
        from orbitsim.core.flight import burn_duration_s
        node = self.current_node() if node is None else node
        v0 = self.app.world.vessels[0]
        try:
            duration = burn_duration_s(v0.mass_kg, v0.exhaust_velocity_mps,
                                       v0.max_thrust_n, node.magnitude_mps)
        except ValueError:
            duration = None
        return burn_summary(node.magnitude_mps, self.time_to_node(), duration)

    def sync_maneuver_hud(self) -> None:
        """The burn panel: readout + Delete burn while a burn exists, Execute while it has
        delta-V (its label says what it will do); just Next Pe / Next Ap otherwise."""
        app = self.app
        if not hasattr(app, "_burn_panel"):
            return
        node = self.current_node()
        dv = node.magnitude_mps
        exists = app._node_epoch_s is not None or dv > 0.0
        ttn = self.time_to_node()
        if app._auto_execute_node:
            exec_label = "Cancel Auto Burn"
        elif ttn is not None and ttn > app.EXECUTE_TOLERANCE_S:
            exec_label = "Jump & Execute"
        else:
            exec_label = "Execute Burn"
        readout = self.burn_summary_text(node) if exists else None
        state = (readout, exists, exec_label if dv > 0.0 else None)
        if state == app._burn_panel_state:
            return
        app._burn_panel_state = state
        if readout is None:
            app._burn_readout_text.hide()
        else:
            app._burn_readout_text.setText(readout)
            app._burn_readout_text.show()
        delete_btn = app._burn_buttons[2]
        delete_btn.show() if exists else delete_btn.hide()
        if dv > 0.0:
            if app._exec_btn["text"] != exec_label:
                app._exec_btn["text"] = exec_label
            app._exec_btn.show()
        else:
            app._exec_btn.hide()
        from orbitsim.render.ui.hud_layout import PANELS
        rect = PANELS["burn"]
        bottom, top = (rect.y0, rect.y1) if exists else self.IDLE_FRAME_Z
        if dv <= 0.0 and exists:
            bottom = self.IDLE_FRAME_Z[0]
        app._burn_panel["frameSize"] = (rect.x0, rect.x1, bottom, top)

    def _apply_unlimited(self, on: bool) -> None:
        """Set the unlimited-dV flag on all vessels (no UI). The sim layer
        floors the working propellant mass for the impulse math while
        unlimited (World.UNLIMITED_FUEL_FLOOR_KG), so thrust works even with
        an empty tank and the real fuel load is never touched."""
        for vessel in self.app.world.vessels:
            vessel.unlimited_dv = on

    def set_unlimited_dv(self, on: bool) -> None:
        if self.app.solar_system or not self.app.world.vessels:
            return  # no flyable vessel (solar viewer) — nothing to toggle
        self._apply_unlimited(on)
        self.app.settings_panel.sync(on)   # keep the Esc-panel label in step with key/title
        self.app._flash_message(f"Unlimited dV {'ON' if on else 'OFF'}")
        self.refresh_readout()

    def toggle_unlimited_dv(self) -> None:
        cur = bool(self.app.world.vessels and self.app.world.vessels[0].unlimited_dv)
        self.set_unlimited_dv(not cur)

    def execute_burn(self) -> None:
        from orbitsim.core.flight import fuel_burned_for_dv

        ttn = self.time_to_node()
        if ttn is not None and ttn > self.app.EXECUTE_TOLERANCE_S:
            if self.app.world.any_thrusting():
                self.app._flash_message("Cut throttle before jumping to the node")
                return
            self.app.clock.warp = 1.0
            vessel = self.app.world.vessels[0]
            self.app._operation_rollback = (
                vessel, vessel.state, vessel.orientation.copy(), self.app.clock.sim_time_s
            )
            self.app.operation.start("Coasting to maneuver", self.app.trajectory_view.jump_operation(ttn))
            self.app.operation_overlay.update(self.app.operation.status)
            return
        if ttn is not None and ttn < -self.app.EXECUTE_TOLERANCE_S:
            self._clear_node()  # node already passed; discard rather than burn backward in time
            return
        v0 = self.app.world.vessels[0]
        node = self.current_node()
        dv = node.magnitude_mps
        if 0.0 < dv <= v0.delta_v_remaining:
            v0.state = apply_maneuver(v0.state, node)
            # Spend fuel for this impulse, so maneuver nodes and live thrust draw
            # from the same tank (no separate budget pool) — unless unlimited.
            if not v0.unlimited_dv:
                burned = fuel_burned_for_dv(v0.exhaust_velocity_mps, v0.mass_kg, dv)
                v0.fuel_mass_kg = max(0.0, v0.fuel_mass_kg - burned)
        self._clear_node()
        for axis in self.app._dv:
            self.app._dv[axis] = 0.0
        if self.app.node_handles.dragging() is not None:
            self.app.node_handles.end_drag()     # the burn it was shaping is gone
        self.refresh_readout()

    def _update_maneuver_preview(self, node, vessel, now_real) -> None:
        """Live two-body estimate while the burn changes; exact N-body path once it settles.

        The exact path is sampled off the render thread (pool) and submitted only after
        the burn has been still for PREVIEW_SETTLE_S, once per burn (no resubmits); both
        lines are rebuilt from cached points when the zoom scale changes.
        """
        app = self.app
        target_name = app.targeting.aim_target_name()
        preview_key = maneuver_preview_key(node, app._node_epoch_s, target_name)
        future = app._preview_future
        if future is not None and future.done():
            points, encounters, approach = future.result()
            completed_key = app._preview_future_key
            app._preview_future = None
            app._preview_future_key = None
            if preview_key == completed_key and node.magnitude_mps > 0.0:
                app._preview_points = points
                app._preview_encounters = encounters
                app._preview_approach = approach
                app._preview_exact_key = completed_key
                app._preview_exact_start = app._preview_inflight_start
                self._draw_preview_exact()
                self._remove_preview_estimate()

        if node.magnitude_mps <= 0.0:
            if app._preview_future is not None:
                app._preview_future_key = None
                if app._preview_future.cancel():
                    app._preview_future = None
            self._remove_preview_exact()
            self._remove_preview_estimate()
            app._preview_exact_key = None
            app._preview_exact_start = None
            app._preview_last_key = None
            return

        # Where the burn would start right now (cheap two-body): a burn-now node's start
        # moves with the ship; a manual burn or quickload moves a scheduled node's.
        post_burn = apply_maneuver(vessel.state, node)
        if preview_key != app._preview_last_key:
            app._preview_last_key = preview_key
            app._preview_key_changed_t = now_real
            if same_burn(preview_key, app._preview_exact_key) and app._preview_np is not None:
                # Only the target changed: the drawn path is still this burn's. Keep it
                # (no flicker) but mark it not current, so the approach to the new
                # target is recomputed and the status box says "checking" meanwhile.
                app._preview_exact_key = None
            elif preview_key != app._preview_exact_key:
                self._remove_preview_exact()      # stale: it shows a different burn
                app._preview_exact_key = None
                app._preview_exact_start = None
                self._set_preview_estimate(post_burn)
        elif (app._preview_estimate_np is not None
              and preview_start_drifted(app._preview_estimate_start, post_burn)):
            self._set_preview_estimate(post_burn)

        scale = app.transform.scale_m_per_unit
        if app._preview_np is not None and app._preview_built_scale != scale:
            self._draw_preview_exact()
        if app._preview_estimate_np is not None and app._preview_estimate_scale != scale:
            self._draw_preview_estimate()
        if app._preview_estimate_np is not None:
            app._preview_estimate_np.set_pos(*app.transform.to_render(np.zeros(3)))

        # A job whose burn was cleared mid-flight still occupies the worker: count it.
        inflight_key = (None if app._preview_future is None
                        else (app._preview_future_key or ("cancelled",)))
        settled = should_submit_exact(preview_key, app._preview_exact_key, inflight_key,
                                      app._preview_key_changed_t, now_real,
                                      self.PREVIEW_SETTLE_S)
        # The drawn burn's start moved without an edit: recompute, keeping the line shown.
        refresh = should_refresh_exact(
            preview_key, app._preview_exact_key, inflight_key,
            preview_start_drifted(app._preview_exact_start, post_burn),
            now_real, app._preview_submit_t, self.PREVIEW_REFRESH_S)
        if settled or refresh:
            app._preview_future_key = preview_key
            app._preview_inflight_start = post_burn
            app._preview_submit_t = now_real
            app._preview_future = app._compute_executor.submit(
                sample_preview, post_burn,
                app.world.solar_system, app.PREDICTION_MIN_SUBSTEP_S, target_name,
            )

    def _set_preview_estimate(self, post_burn) -> None:
        """Recompute and redraw the two-body estimate from this post-burn state."""
        self.app._preview_estimate_start = post_burn
        self.app._preview_estimate_points = kepler_estimate_points(post_burn)
        self._draw_preview_estimate()

    def _draw_preview_exact(self) -> None:
        """(Re)build the exact preview line + encounter patches from cached points."""
        app = self.app
        if app._preview_np is not None:
            app._preview_np.remove_node()
        points = app._preview_points
        local_points, app._preview_origin = localize_polyline(points)
        scale = app.transform.scale_m_per_unit
        render_points = [tuple(point / scale) for point in local_points]
        px = app.line_px_scale()
        app._preview_np = build_orbit_node(render_points, PLAN_STYLE, px_scale=px)
        app._preview_np.reparent_to(app.render)
        app._preview_np.set_pos(*app.transform.to_render(app._preview_origin))
        app._preview_built_scale = scale
        # Encounter patches ride the preview node (same origin/scale).
        app.trajectory_view.build_encounter_patches(
            app._preview_np, points, app._preview_origin, scale, app._preview_encounters,
            px_scale=px,
        )

    def _remove_preview_exact(self) -> None:
        app = self.app
        if app._preview_np is not None:
            app._preview_np.remove_node()
        app._preview_np = None
        app._preview_origin = None
        app._preview_built_scale = None
        app._preview_encounters = []
        app._preview_approach = None

    def _draw_preview_estimate(self) -> None:
        """(Re)build the faint two-body estimate (Earth-centred) at the current scale."""
        app = self.app
        self._remove_preview_estimate(keep_points=True)
        points = app._preview_estimate_points
        if points is None:
            return
        scale = app.transform.scale_m_per_unit
        render_points = [tuple(point / scale) for point in points]
        app._preview_estimate_np = build_orbit_node(
            render_points, ESTIMATE_STYLE, px_scale=app.line_px_scale())
        app._preview_estimate_np.reparent_to(app.render)
        app._preview_estimate_np.set_pos(*app.transform.to_render(np.zeros(3)))
        app._preview_estimate_scale = scale

    def _remove_preview_estimate(self, keep_points: bool = False) -> None:
        app = self.app
        if app._preview_estimate_np is not None:
            app._preview_estimate_np.remove_node()
        app._preview_estimate_np = None
        app._preview_estimate_scale = None
        if not keep_points:
            app._preview_estimate_points = None

    def frame_maneuver_node(self):
        """Scheduled node: preview, marker, auto-warp-down, readout."""
        import time as _time
        # Scheduled maneuver node: preview (magenta), node marker (cyan), auto-warp-down,
        # readout, and a vessel.nodes mirror so quicksave persists the plan.
        node = self.current_node()
        v0 = self.app.world.vessels[0]
        ttn = self.time_to_node()
        # Discard a node that slipped well past its epoch unexecuted, so it can't linger
        # invisibly or later execute backward in time.
        if ttn is not None and ttn < -self.app.EXECUTE_TOLERANCE_S:
            self._clear_node()
            ttn = None
            node = self.current_node()
        v0.nodes = [node] if (self.app._node_epoch_s is not None or node.magnitude_mps > 0.0) else []
        # N-body preview sampling is CPU-heavy, so it runs on one worker. Only the cheap
        # LineSegs swap happens here on the render thread when a result is ready.
        self._update_maneuver_preview(node, v0, _time.monotonic())
        # Node marker at the node's predicted position on the orbit (held at the vessel
        # through the brief execute window once the node is due).
        if self.app._node_epoch_s is not None and ttn is not None and ttn >= -self.app.EXECUTE_TOLERANCE_S:
            npos = apply_maneuver(v0.state, node).r
            mx, my, mz = self.app.transform.to_render(npos)
            if self.app._node_marker_np is None:
                # A ring on the orbit (node_handles sizes it to NODE_RADIUS_PX).
                from orbitsim.render.world_labels import build_ring_marker
                self.app._node_marker_np = build_ring_marker(
                    self.app.render, color=(0.3, 1.0, 1.0, 1.0),
                    thickness_px=2.5 * self.app.line_px_scale())
            self.app._node_marker_np.set_pos(mx, my, mz)
            if self.app._node_dv_label is None:
                from orbitsim.render.world_labels import build_world_label
                self.app._node_dv_label = build_world_label(
                    self.app.render, "", color=(0.3, 1.0, 1.0, 1.0), scale=11.0)
            text = self.burn_summary_text(node)   # the burn panel's readout, word for word
            if self.app._node_dv_label.node().get_text() != text:
                self.app._node_dv_label.node().set_text(text)
            self.app._node_dv_label.set_pos(mx, my, mz + 8.0)
            self.app._node_dv_label.show()
            self.app._node_label_position = np.asarray(npos, dtype=np.float64).copy()
        elif self.app._node_marker_np is not None:
            self.app._node_label_position = None
            self.app._node_marker_np.remove_node()
            self.app._node_marker_np = None
            if self.app._node_dv_label is not None:
                self.app._node_dv_label.remove_node()
                self.app._node_dv_label = None
        # Auto-warp-down as the node nears (never warps up).
        if ttn is not None and 0.0 < ttn <= self.AUTO_WARP_LEAD_S * self.app.clock.warp and self.app.clock.warp > 1.0:
            self.app.clock.warp_down()
        self.sync_maneuver_hud()
        return node
