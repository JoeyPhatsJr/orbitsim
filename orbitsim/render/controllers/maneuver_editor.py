"""Maneuver-node editor: jog sliders, node timing, preview, execute, unlimited dV."""
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectSlider import DirectSlider
from direct.gui.OnscreenText import OnscreenText
from orbitsim.core.maneuvers import ManeuverNode
from orbitsim.core.maneuvers import apply_maneuver
from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.hud.format import fmt_countdown
from orbitsim.render.orbit_lines import MANEUVER_COLOR
from orbitsim.render.orbit_lines import build_orbit_node
from orbitsim.render.trajectory_helpers import localize_polyline
from orbitsim.render.trajectory_helpers import maneuver_preview_key
from orbitsim.render.trajectory_sampling import sample_preview
from panda3d.core import TextNode
import numpy as np


class ManeuverEditor:
    """Maneuver-node editor: jog sliders, node timing, preview, execute, unlimited dV. Stateless: all state lives on the OrbitApp it is given."""

    def __init__(self, app) -> None:
        self.app = app

    def _build_maneuver_ui(self) -> None:
        """Per-axis spring-loaded jog sliders for RTN delta-V, an Execute button, a readout.

        Each slider is a rate control: hold it right to increase that delta-V component
        (faster the further you push, exponentially), left to decrease. Releasing the
        mouse springs the thumb back to center and the value stops changing.
        """
        self.app._dv = {"pro": 0.0, "nrm": 0.0, "rad": 0.0}
        self.app._dv_line = self.app._node_line = self.app._target_line = self.app._encounter_line = ""
        self.app._node_epoch_s = None      # absolute epoch of the scheduled node (None = none)
        self.app._auto_execute_node = False
        self.app._node_marker_np = None
        self.app._node_dv_label = None
        self.app._dv_value_text = {}
        self.app._jog = {}
        from orbitsim.render.ui.theme import THEME, button_options
        self.app._maneuver_frame = DirectFrame(
            frameColor=THEME.panel,
            frameSize=(-1.34, 0.02, -0.29, 0.60),
            parent=self.app.a2dBottomRight,
        )
        rows = (("pro", "Prograde", 0.40), ("nrm", "Normal", 0.28), ("rad", "Radial", 0.16))
        for axis, label, z in rows:
            OnscreenText(
                text=label,
                pos=(-1.22, z - 0.015),
                scale=0.045,
                fg=(1, 1, 1, 1),
                align=TextNode.ALeft,
                parent=self.app._maneuver_frame,
            )
            self.app._jog[axis] = DirectSlider(
                pos=(-0.62, 0.0, z),
                scale=0.34,
                range=(-1.0, 1.0),
                value=0.0,
                pageSize=0.25,
                parent=self.app._maneuver_frame,
            )
            self.app._dv_value_text[axis] = OnscreenText(
                text="+0",
                pos=(-0.14, z - 0.015),
                scale=0.045,
                fg=(1.0, 0.9, 0.4, 1),
                align=TextNode.ALeft,
                mayChange=True,
                parent=self.app._maneuver_frame,
            )
        # Total dV magnitude display.
        self.app._dv_total_text = OnscreenText(
            text="", pos=(-0.65, 0.52), scale=0.05,
            fg=(1.0, 0.5, 1.0, 1.0), align=TextNode.ACenter,
            mayChange=True, parent=self.app._maneuver_frame,
        )
        # Node time jog slider (variable time stepping).
        OnscreenText(
            text="Node T", pos=(-1.22, 0.04 - 0.015), scale=0.045,
            fg=(0.3, 1.0, 1.0, 1), align=TextNode.ALeft,
            parent=self.app._maneuver_frame,
        )
        self.app._node_time_jog = DirectSlider(
            pos=(-0.62, 0.0, 0.04), scale=0.34,
            range=(-1.0, 1.0), value=0.0, pageSize=0.25,
            parent=self.app._maneuver_frame,
        )
        self.app._node_time_text = OnscreenText(
            text="T+0s", pos=(-0.14, 0.04 - 0.015), scale=0.045,
            fg=(0.3, 1.0, 1.0, 1), align=TextNode.ALeft,
            mayChange=True, parent=self.app._maneuver_frame,
        )
        self.app._action_frame = DirectFrame(
            frameColor=THEME.panel,
            frameSize=(0.03, 0.78, 0.03, 0.25),
            parent=self.app.a2dBottomLeft,
        )
        self.app._exec_btn = DirectButton(
            text="Execute Burn",
            scale=0.045,
            pos=(0.40, 0.0, 0.10),
            command=self._execute_burn,
            parent=self.app._action_frame,
            **button_options(),
        )
        self.app._action_status = OnscreenText(
            text="NO MANEUVER PLANNED", pos=(0.40, 0.18), scale=0.036,
            fg=THEME.text_muted, mayChange=True, parent=self.app._action_frame,
        )
        # Scheduled-node controls: jump to next apsis, clear.
        node_btns = [
            ("Next Pe", self._node_to_pe),
            ("Next Ap", self._node_to_ap),
            ("Clear", self._clear_node),
            ("Clear Tgt", self.app.targeting._clear_target),
            ("Intercept", self.app.targeting._plan_intercept),
        ]
        for i, (label, cmd) in enumerate(node_btns):
            DirectButton(text=label, scale=0.045, pos=(-0.95 + i * 0.34, 0.0, -0.20),
                         command=cmd, parent=self.app._maneuver_frame, **button_options())
        # Releasing the mouse springs every jog slider back to its center (zero rate).
        self.app.accept("mouse1-up", self._release_jogs)
        self._refresh_readout()

    def _toggle_maneuver_drawer(self) -> None:
        if self.app._maneuver_frame.is_hidden():
            self.app._maneuver_frame.show()
        else:
            self.app._maneuver_frame.hide()

    def _jog_rate_mps_per_s(self, value: float) -> float:
        """Map a jog displacement in [-1, 1] to a signed delta-V rate [m/s per s].

        Zero at center, ``JOG_MAX_RATE_MPS`` at full deflection, exponential in between
        so small displacements give fine control and large ones change quickly.
        """
        mag = abs(value)
        if mag <= self.app.JOG_DEADZONE:
            return 0.0
        shaped = (np.exp(self.app.JOG_CURVE * mag) - 1.0) / (np.exp(self.app.JOG_CURVE) - 1.0)
        return float(np.copysign(self.app.JOG_MAX_RATE_MPS * shaped, value))

    def _apply_jogs(self, real_dt_s: float) -> None:
        """Integrate each jog slider's rate into its delta-V component for this frame."""
        changed = False
        for axis, slider in self.app._jog.items():
            rate = self._jog_rate_mps_per_s(slider["value"])
            if rate != 0.0:
                self.app._dv[axis] += rate * real_dt_s
                self.app._dv_value_text[axis].setText(f"{self.app._dv[axis]:+.0f}")
                changed = True
        if changed:
            self._refresh_readout()
        if hasattr(self.app, "_node_time_jog"):
            tv = self.app._node_time_jog["value"]
            mag = abs(tv)
            if mag > self.app.JOG_DEADZONE:
                shaped = (np.exp(self.app.NODE_JOG_CURVE * mag) - 1.0) / (np.exp(self.app.NODE_JOG_CURVE) - 1.0)
                rate = float(np.copysign(self.app.NODE_JOG_MAX_RATE * shaped, tv))
                now = self.app.clock.sim_time_s
                base = self.app._node_epoch_s if self.app._node_epoch_s is not None else now
                self.app._node_epoch_s = max(now, base + rate * real_dt_s)
                ttn = self.app._node_epoch_s - now
                if ttn < 60:
                    self.app._node_time_text.setText(f"T+{ttn:.0f}s")
                elif ttn < 3600:
                    self.app._node_time_text.setText(f"T+{ttn/60:.1f}m")
                elif ttn < 86400:
                    self.app._node_time_text.setText(f"T+{ttn/3600:.1f}h")
                else:
                    self.app._node_time_text.setText(f"T+{ttn/86400:.1f}d")

    def _release_jogs(self) -> None:
        """Spring all jog thumbs back to center so the value stops changing on release.

        Also attempt a target pick: a left-click that didn't drag is a tap on a body."""
        self.app.targeting._try_pick_target()
        for slider in self.app._jog.values():
            slider["value"] = 0.0
        if hasattr(self.app, "_node_time_jog"):
            self.app._node_time_jog["value"] = 0.0

    def _current_node(self) -> ManeuverNode:
        """Build the node from the current dV values at the scheduled epoch (or now if none)."""
        epoch = (self.app._node_epoch_s if self.app._node_epoch_s is not None
                 else self.app.world.vessels[0].state.epoch_s)
        return ManeuverNode(
            epoch_s=epoch,
            dv_prograde_mps=self.app._dv["pro"],
            dv_normal_mps=self.app._dv["nrm"],
            dv_radial_mps=self.app._dv["rad"],
        )

    def _time_to_node(self):
        """Seconds until the scheduled node, or None if no node is scheduled."""
        if self.app._node_epoch_s is None:
            return None
        return self.app._node_epoch_s - self.app.clock.sim_time_s

    def _step_node_time(self, delta_s):
        """Nudge the node epoch by delta_s (creating one at now+delta if none), clamped >= now."""
        now = self.app.clock.sim_time_s
        base = self.app._node_epoch_s if self.app._node_epoch_s is not None else now
        self.app._node_epoch_s = max(now, base + delta_s)

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
        if hasattr(self.app, "_node_time_text"):
            self.app._node_time_text.setText("T+0s")

    def _refresh_readout(self) -> None:
        import math
        node = self._current_node()
        budget = self.app.world.vessels[0].delta_v_remaining
        left = "INF" if not math.isfinite(budget) else f"{budget:,.0f} m/s"
        self.app._dv_line = (
            f"Maneuver dV: {node.magnitude_mps:,.1f} m/s   (dV left {left})"
            if node.magnitude_mps > 0.0
            else ""
        )
        if hasattr(self.app, "_dv_total_text"):
            if node.magnitude_mps > 0.0:
                self.app._dv_total_text.setText(f"Total dV: {node.magnitude_mps:,.1f} m/s")
            else:
                self.app._dv_total_text.setText("")
        if hasattr(self.app, "_exec_btn"):
            ttn = self._time_to_node()
            if getattr(self.app, "_auto_execute_node", False):
                self.app._exec_btn["text"] = "Cancel Auto Burn"
            elif ttn is not None and ttn > self.app.EXECUTE_TOLERANCE_S:
                self.app._exec_btn["text"] = "Jump & Execute"
            else:
                self.app._exec_btn["text"] = "Execute Burn"
            if hasattr(self.app, "_action_status"):
                if node.magnitude_mps <= 0.0:
                    self.app._action_status.setText("NO MANEUVER PLANNED")
                    from direct.gui import DirectGuiGlobals as DGG
                    self.app._exec_btn["state"] = DGG.DISABLED
                elif ttn is not None and ttn > 0.0:
                    from direct.gui import DirectGuiGlobals as DGG
                    self.app._exec_btn["state"] = DGG.NORMAL
                    self.app._action_status.setText(
                        f"NODE T-{fmt_countdown(ttn)}  /  {node.magnitude_mps:,.0f} m/s"
                    )
                else:
                    from direct.gui import DirectGuiGlobals as DGG
                    self.app._exec_btn["state"] = DGG.NORMAL
                    self.app._action_status.setText(f"BURN READY  /  {node.magnitude_mps:,.0f} m/s")
        self._sync_maneuver_hud()

    def _sync_maneuver_hud(self) -> None:
        enc = getattr(self.app, "_encounter_line", "")
        self.app.hud.set_maneuver(self.app._dv_line, self.app._node_line, self.app._target_line, enc)

    def _apply_unlimited(self, on: bool) -> None:
        """Set the unlimited-dV flag on all vessels (no UI). The sim layer
        floors the working propellant mass for the impulse math while
        unlimited (World.UNLIMITED_FUEL_FLOOR_KG), so thrust works even with
        an empty tank and the real fuel load is never touched."""
        for vessel in self.app.world.vessels:
            vessel.unlimited_dv = on

    def _set_unlimited_dv(self, on: bool) -> None:
        if self.app.solar_system or not self.app.world.vessels:
            return  # no flyable vessel (solar viewer) — nothing to toggle
        self._apply_unlimited(on)
        self.app.settings_panel.sync(on)   # keep the Esc-panel label in step with key/title
        self.app._flash_message(f"Unlimited dV {'ON' if on else 'OFF'}")
        self._refresh_readout()

    def _toggle_unlimited_dv(self) -> None:
        cur = bool(self.app.world.vessels and self.app.world.vessels[0].unlimited_dv)
        self._set_unlimited_dv(not cur)

    def _execute_burn(self) -> None:
        from orbitsim.core.flight import fuel_burned_for_dv

        ttn = self._time_to_node()
        if ttn is not None and ttn > self.app.EXECUTE_TOLERANCE_S:
            if self.app.world.any_thrusting():
                self.app._flash_message("Cut throttle before jumping to the node")
                return
            self.app.clock.warp = 1.0
            vessel = self.app.world.vessels[0]
            self.app._operation_rollback = (
                vessel, vessel.state, vessel.orientation.copy(), self.app.clock.sim_time_s
            )
            self.app.operation.start("Coasting to maneuver", self.app.trajectory_view._jump_operation(ttn))
            self.app.operation_overlay.update(self.app.operation.status)
            return
        if ttn is not None and ttn < -self.app.EXECUTE_TOLERANCE_S:
            self._clear_node()  # node already passed; discard rather than burn backward in time
            return
        v0 = self.app.world.vessels[0]
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
        for axis in self.app._dv:
            self.app._dv[axis] = 0.0
            self.app._dv_value_text[axis].setText("+0")
        self._release_jogs()
        self._refresh_readout()

    def _update_maneuver_preview(self, node, vessel, now_real) -> None:
        """Poll/submit preview work while keeping N-body integration off the render thread."""
        preview_key = maneuver_preview_key(node, self.app._node_epoch_s)
        future = self.app._preview_future
        if future is not None and future.done():
            points, encounters = future.result()
            completed_key = self.app._preview_future_key
            self.app._preview_future = None
            self.app._preview_future_key = None
            if preview_key == completed_key and node.magnitude_mps > 0.0:
                if self.app._preview_np is not None:
                    self.app._preview_np.remove_node()
                local_points, self.app._preview_origin = localize_polyline(points)
                scale = self.app.transform.scale_m_per_unit
                render_points = [tuple(point / scale) for point in local_points]
                self.app._preview_np = build_orbit_node(
                    render_points, color=MANEUVER_COLOR, thickness=2.75
                )
                self.app._preview_np.reparent_to(self.app.render)
                self.app._preview_np.set_pos(*self.app.transform.to_render(self.app._preview_origin))
                # Encounter patches ride the preview node (same origin/scale).
                self.app._preview_encounters = encounters
                self.app.trajectory_view._build_encounter_patches(
                    self.app._preview_np, points, self.app._preview_origin, scale, encounters
                )

        if node.magnitude_mps <= 0.0:
            if self.app._preview_future is not None:
                self.app._preview_future_key = None
                if self.app._preview_future.cancel():
                    self.app._preview_future = None
            if self.app._preview_np is not None:
                self.app._preview_np.remove_node()
                self.app._preview_np = None
                self.app._preview_origin = None
            self.app._preview_encounters = []
            return

        if (
            self.app._preview_future is None
            and (
                self.app._preview_np is None
                or now_real - self.app._preview_submit_t > self.app.PREVIEW_THROTTLE_S
            )
        ):
            post_burn = apply_maneuver(vessel.state, node)
            self.app._preview_submit_t = now_real
            self.app._preview_future_key = preview_key
            self.app._preview_future = self.app._compute_executor.submit(
                sample_preview, post_burn,
                self.app.world.solar_system, self.app.PREDICTION_MIN_SUBSTEP_S,
            )

    def _frame_maneuver_node(self):
        """Scheduled node: preview, marker, auto-warp-down, readout."""
        import time as _time
        # Scheduled maneuver node: preview (magenta), node marker (cyan), auto-warp-down,
        # readout, and a vessel.nodes mirror so quicksave persists the plan.
        node = self._current_node()
        v0 = self.app.world.vessels[0]
        ttn = self._time_to_node()
        # Discard a node that slipped well past its epoch unexecuted, so it can't linger
        # invisibly or later execute backward in time.
        if ttn is not None and ttn < -self.app.EXECUTE_TOLERANCE_S:
            self._clear_node()
            ttn = None
            node = self._current_node()
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
                self.app._node_marker_np = make_uv_sphere(1.0, 8, 12)
                self.app._node_marker_np.reparent_to(self.app.render)
                self.app._node_marker_np.set_color(0.3, 1.0, 1.0, 1.0)
                self.app._node_marker_np.set_light_off()
                self.app._node_marker_np.set_scale(6.0)
            self.app._node_marker_np.set_pos(mx, my, mz)
            if self.app._node_dv_label is None:
                from orbitsim.render.world_labels import build_world_label
                self.app._node_dv_label = build_world_label(
                    self.app.render, "", color=(0.3, 1.0, 1.0, 1.0), scale=11.0)
            dv_text = f"dV {node.magnitude_mps:,.0f} m/s"
            if ttn > 0:
                dv_text += f"  T-{fmt_countdown(ttn)}"
            self.app._node_dv_label.node().set_text(dv_text)
            self.app._node_dv_label.set_pos(mx, my, mz + 8.0)
            self.app._node_dv_label.show()
        elif self.app._node_marker_np is not None:
            self.app._node_marker_np.remove_node()
            self.app._node_marker_np = None
            if self.app._node_dv_label is not None:
                self.app._node_dv_label.remove_node()
                self.app._node_dv_label = None
        # Auto-warp-down as the node nears (never warps up).
        if ttn is not None and 0.0 < ttn <= self.app.AUTO_WARP_LEAD_S * self.app.clock.warp and self.app.clock.warp > 1.0:
            self.app.clock.warp_down()
        # Pending-node readout (single node); shows "DUE" once within the execute window.
        if ttn is not None and ttn >= -self.app.EXECUTE_TOLERANCE_S:
            if ttn <= self.app.EXECUTE_TOLERANCE_S:
                label = "DUE — press Execute"
            else:
                label = f"in T-{fmt_countdown(ttn)}"
            self.app._node_line = f"Node {label}   dV {node.magnitude_mps:,.1f} m/s"
        else:
            self.app._node_line = ""
        self._sync_maneuver_hud()
        return node
