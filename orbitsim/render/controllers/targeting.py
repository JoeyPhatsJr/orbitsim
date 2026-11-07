"""Target selection (click-pick), transfer planning, the target panel (status box) and
the "when you arrive" ghost."""

import numpy as np
from orbitsim.core.constants import MU_SUN, R_MOON
from orbitsim.core.elements import state_to_elements
from orbitsim.core.maneuvers import apply_maneuver
from orbitsim.core.planets import (
    A_EARTH,
    A_JUPITER,
    A_MARS,
    A_MERCURY,
    A_NEPTUNE,
    A_SATURN,
    A_URANUS,
    A_VENUS,
)
from orbitsim.core.rendezvous import closest_approach
from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.hud.format import fmt_countdown, fmt_dist


class Targeting:
    """Stateless controller: holds only ``self.app``; all state lives on OrbitApp."""

    def __init__(self, app) -> None:
        self.app = app

    def build_targeting(self):
        """Targetable bodies, the target panel (status box) and the target label."""
        # Targetable bodies. Click a marker to select.
        from orbitsim.render.targets import MoonTarget, LagrangePointTarget, PlanetTarget
        from orbitsim.core.nbody import (
            _csun, _cmercury, _cvenus, _cmars,
            _cjupiter, _csaturn, _curanus, _cneptune,
        )
        self.app._targets = [MoonTarget()] + [
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
        self.app._target = None     # current Target or None
        # Earth-only Kepler closest approach: the "checking..." estimate (throttled).
        self.app._ca_recompute_t = 0.0
        self.app._ca = None
        self.app._ca_traj = None
        self.app._ca_abs_epoch = 0.0
        self.app._aim_status = None             # AimStatus shown in the target panel
        self.app._aim_ghost_np = None           # "{Target} when you arrive" sphere
        self.app._aim_ghost_position = None     # its world position [m], None = hidden
        from orbitsim.render.world_labels import build_world_label
        self.app._aim_ghost_label = build_world_label(
            self.app.render, "", color=(0.85, 0.90, 0.95, 1.0), scale=11.0)
        self.app._aim_ghost_label.hide()
        self._build_target_panel()
        self.app._target_label = build_world_label(
            self.app.render, "TARGET", color=(1.0, 0.35, 0.95, 1.0), scale=12.0
        )
        self.app._target_label.hide()
        self.app._target_label_position = None

    # Target panel rows, inside PANELS["target"] (a2dTopRight-relative).
    TITLE_Z = -0.095
    STATUS_Z = -0.175
    BUTTONS_Z = -0.255
    FLYBY_Z = -0.335              # in-SOI flyby readout row (only while inside an SOI)
    FLYBY_SCALE = 0.034
    FLYBY_ROW_W = 0.88            # x -0.94 .. -0.06, inside PANELS["target"] (-0.98 .. -0.04)
    BOTTOM_Z = -0.30              # frame bottom without the flyby row
    GHOST_MIN_UNITS = 6.0          # smallest on-screen ghost radius [render units]

    def _build_target_panel(self) -> None:
        """Title row, status row (icon + text) and a buttons row; hidden with no target."""
        from direct.gui.DirectFrame import DirectFrame
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import LineSegs, TextNode
        from orbitsim.render.ui.hud_layout import PANELS
        from orbitsim.render.ui.theme import THEME
        from orbitsim.render.ui.widgets import layout_panel_frame
        panel = layout_panel_frame(self.app, PANELS["target"])
        self.app._target_panel = panel
        self.app._hud_panels["target"] = panel
        self.app._target_title_text = OnscreenText(
            text="", pos=(-0.94, self.TITLE_Z), scale=0.042, fg=THEME.text_muted,
            align=TextNode.ALeft, mayChange=True, parent=panel)
        self.app._aim_status_text = OnscreenText(
            text="", pos=(-0.87, self.STATUS_Z), scale=0.045, fg=THEME.text,
            align=TextNode.ALeft, mayChange=True, parent=panel)
        self.app._flyby_text = OnscreenText(
            text="", pos=(-0.94, self.FLYBY_Z), scale=self.FLYBY_SCALE, fg=(0.4, 1.0, 0.6, 1.0),
            align=TextNode.ALeft, mayChange=True, parent=panel)
        # The bundled font has no tick/cross glyphs: draw them (2 segments each).
        x, z = -0.915, self.STATUS_Z + 0.012
        shapes = {
            "check": [((x - 0.020, z), (x - 0.006, z - 0.016), (x + 0.022, z + 0.020))],
            "cross": [((x - 0.018, z - 0.018), (x + 0.018, z + 0.018)),
                      ((x - 0.018, z + 0.018), (x + 0.018, z - 0.018))],
        }
        self.app._aim_icon_nps = {}
        for kind, strokes in shapes.items():
            segs = LineSegs(f"aim_icon_{kind}")
            segs.set_thickness(2.5)
            for stroke in strokes:
                segs.move_to(stroke[0][0], 0.0, stroke[0][1])
                for px, pz in stroke[1:]:
                    segs.draw_to(px, 0.0, pz)
            icon = panel.attach_new_node(segs.create())
            icon.hide()
            self.app._aim_icon_nps[kind] = icon
        from direct.gui.DirectButton import DirectButton
        from orbitsim.render.ui.theme import button_options
        self.app._plan_transfer_btn = DirectButton(
            text="Plan transfer", scale=0.046, pos=(-0.68, 0.0, self.BUTTONS_Z),
            command=self.plan_intercept, parent=panel, **button_options())
        self.app._clear_target_btn = DirectButton(
            text="Clear target", scale=0.046, pos=(-0.22, 0.0, self.BUTTONS_Z),
            command=self.clear_target, parent=panel, **button_options())
        panel.hide()

    def aim_target_name(self):
        """The target the exact previews measure an approach to: the Moon or a planet.

        None with no target, for Lagrange points (not bodies) and for the Sun."""
        from orbitsim.render.targets import MoonTarget, PlanetTarget
        target = self.app._target
        if isinstance(target, (MoonTarget, PlanetTarget)) and target.name != "Sun":
            return target.name
        return None

    def marker_px(self, world_r):
        """Project an inertial position to pixels, or None if behind the camera/off-lens."""
        from panda3d.core import Point2
        rp = self.app.transform.to_render(world_r)
        p = self.app.cam.get_relative_point(self.app.render, rp)
        proj = Point2()
        if not self.app.camLens.project(p, proj):
            return None
        w, h = self.app.win.get_x_size(), self.app.win.get_y_size()
        return ((proj.x * 0.5 + 0.5) * w, (proj.y * 0.5 + 0.5) * h)

    def project_px(self, points_m):
        """Geocentric positions [m], shape (N, 3), to bottom-up pixels (N, 2); NaN rows
        are behind the camera. One matrix pass (matches marker_px to ~3e-5 px)."""
        from orbitsim.render.picking import project_points_px
        pts = (np.asarray(points_m, dtype=np.float64) - self.app.transform.origin_m) \
            / self.app.transform.scale_m_per_unit

        def m(lm):
            return np.array([[lm.get_cell(r, c) for c in range(4)] for r in range(4)])
        render_to_clip = m(self.app.render.get_mat(self.app.cam)) @ m(
            self.app.camLens.get_projection_mat())
        return project_points_px(pts, render_to_clip,
                                 self.app.win.get_x_size(), self.app.win.get_y_size())

    def pick_body_at(self, click_px) -> bool:
        """Target the body under ``click_px`` (22 px), as a tap on its marker; True on a hit."""
        from orbitsim.render.picking import nearest_marker
        now = self.app.clock.sim_time_s
        proj = [self.marker_px(t.state_at(now).r) for t in self.app._targets]
        idxs = [i for i, p in enumerate(proj) if p is not None]
        hit = nearest_marker(click_px, [proj[i] for i in idxs], tol_px=22.0)
        if hit is None:
            return False
        self.clear_target()   # tear down the prior target's panel/ghost before switching
        self.app._target = self.app._targets[idxs[hit]]
        return True

    def coast_to(self, epoch_s: float) -> bool:
        """Coast (time-skip with rollback) to ``epoch_s`` along the trajectory."""
        if self.app.world.any_thrusting():
            self.app._flash_message("Cut throttle before warping")
            return False
        duration_s = float(epoch_s - self.app.clock.sim_time_s)
        if duration_s <= 0.0:
            self.app._flash_message("That point has passed")
            return False
        vessel = self.app.world.vessels[0]
        self.app.clock.warp = 1.0
        self.app._operation_rollback = (
            vessel, vessel.state, vessel.orientation.copy(), self.app.clock.sim_time_s
        )
        self.app.operation.start(
            f"Coasting {fmt_countdown(duration_s)} along trajectory",
            self.app.trajectory_view.trajectory_coast_operation(duration_s),
        )
        self.app.operation_overlay.update(self.app.operation.status)
        return True

    def plan_intercept(self):
        """Auto-plan an intercept/departure of the current target.

        The Lambert grid search is the expensive part, so it runs on the planning
        thread (fanning cells across the process pool) instead of freezing the
        render thread. This method only builds the grids and submits the job;
        ``_poll_planning`` applies the resulting node when it completes.
        """
        import numpy as np
        from orbitsim.core.elements import state_to_elements
        from orbitsim.render.targets import MoonTarget, PlanetTarget
        if self.app._target is None:
            self.app._flash_message("No target selected")
            return
        if self.app._planning_future is not None and not self.app._planning_future.done():
            self.app._flash_message("Planning already in progress...")
            return
        from orbitsim.render.node_edit import departure_grid_s
        v0 = self.app.world.vessels[0]
        now = self.app.clock.sim_time_s
        grid_executor = self.app._grid_executor
        if isinstance(self.app._target, PlanetTarget):
            from orbitsim.core.optimize import interplanetary_departure_node_by_name
            from orbitsim.core.planets import A_EARTH, _N_EARTH
            from orbitsim.core.constants import MU_SUN, R_MOON
            target_a = {
                "Mercury": A_MERCURY, "Venus": A_VENUS, "Mars": A_MARS,
                "Jupiter": A_JUPITER, "Saturn": A_SATURN,
                "Uranus": A_URANUS, "Neptune": A_NEPTUNE,
            }.get(self.app._target.name)
            if target_a is None:
                self.app._flash_message(f"Can't plan a transfer to {self.app._target.name}")
                return
            sun_target = next((t for t in self.app._targets
                               if isinstance(t, PlanetTarget) and t.name == "Sun"), None)
            if sun_target is None:
                self.app._flash_message("No intercept found")
                return
            # Sweep a full synodic period for departure; time-of-flight brackets the
            # Hohmann transfer time to Earth's neighbour.
            w_target = np.sqrt(MU_SUN / target_a**3)
            synodic = 2.0 * np.pi / abs(_N_EARTH - w_target)
            dep = departure_grid_s(min(synodic, 2 * 365.25 * 86400.0), 24)
            hohmann_tof = np.pi * np.sqrt(((A_EARTH + target_a) / 2.0)**3 / MU_SUN)
            tof = np.linspace(0.3 * hohmann_tof, 2.0 * hohmann_tof, 32)
            ship_state = v0.state
            target_name = self.app._target._ephemeris_name
            sun_name = sun_target._ephemeris_name

            def job():
                return interplanetary_departure_node_by_name(
                    ship_state, target_name, sun_name, dep, tof, executor=grid_executor)
        else:
            from orbitsim.core.optimize import intercept_node
            try:
                period = state_to_elements(v0.state).period_s
            except ValueError:
                self.app._flash_message("Unbound orbit — can't plan intercept")
                return
            dep = departure_grid_s(period, 24)
            tof = np.linspace(3.0e3, 14.0 * 86400.0, 48)
            ship_state = v0.state
            target_now = self.app._target.state_at(now)
            mu = self.app.world.central.mu
            if isinstance(self.app._target, MoonTarget):
                from orbitsim.core.flyby_targeting import moon_flyby_node

                def job():
                    # The Lambert seed aims at the Moon's centre; correct it onto a
                    # ~200 km pass on the same propagator the preview draws.
                    seed = intercept_node(
                        ship_state, target_now, mu, dep, tof, executor=grid_executor)
                    return moon_flyby_node(ship_state, seed, executor=grid_executor)
            else:
                def job():
                    return intercept_node(
                        ship_state, target_now, mu, dep, tof, executor=grid_executor)

        # Planning takes real seconds; at warp they'd be minutes of sim time and the
        # departure would be gone when the plan lands. Plan at 1x (like coasts/burns).
        self.app.clock.warp = 1.0
        self.app._flash_message(f"Planning {self.app._target.name} transfer...")
        self.app._planning_future = self.app._planning_executor.submit(job)

    def _apply_planned_node(self, result) -> None:
        """Install a freshly-planned burn (ManeuverNode, or a FlybyPlan for the Moon)
        into the editor (render thread only)."""
        from orbitsim.core.flyby_targeting import FLYBY_TARGET_ALT_M, FlybyPlan
        from orbitsim.render.node_edit import node_epoch_ok
        node = result.node if isinstance(result, FlybyPlan) else result
        if not node_epoch_ok(node.epoch_s, self.app.clock.sim_time_s):
            # Its departure is (nearly) past: installing it would clamp the burn to now
            # and miss (a Moon flyby 10 s late impacts). Say so instead.
            self.app._flash_message("Plan went stale — press Plan again")
            return
        editor = self.app.maneuver_editor
        self.app._auto_execute_node = False
        editor.set_node_epoch(node.epoch_s)
        editor.set_dv("pro", node.dv_prograde_mps)
        editor.set_dv("nrm", node.dv_normal_mps)
        editor.set_dv("rad", node.dv_radial_mps)
        if not isinstance(result, FlybyPlan):
            self.app._flash_message(f"Intercept planned (dV {node.magnitude_mps:,.0f} m/s)")
            return
        alt_km = (result.moon_pass.periapsis_radius_m - R_MOON) / 1000.0
        if result.converged:
            self.app._flash_message(
                f"Moon transfer planned · Pe {alt_km:,.0f} km · dV {node.magnitude_mps:,.0f} m/s")
        else:
            self.app._flash_message(
                f"Moon transfer planned · closest Pe {alt_km:,.0f} km "
                f"(couldn't reach {FLYBY_TARGET_ALT_M / 1000.0:,.0f} km)")

    def poll_planning(self) -> None:
        """Apply an intercept plan once its background job finishes."""
        future = getattr(self.app, "_planning_future", None)
        if future is None or not future.done():
            return
        self.app._planning_future = None
        try:
            result = future.result()
        except ValueError:
            self.app._flash_message("No intercept found")
            return
        except Exception:
            import traceback
            traceback.print_exc()
            self.app._flash_message("Planning failed")
            return
        self._apply_planned_node(result)

    def clear_target(self):
        """Deselect the current target; hide its panel, ghost and readout."""
        self.app._target = None
        self.app._ca = None
        self.app._aim_status = None
        self.app._target_panel.hide()
        self._hide_ghost()
        self.app._target_label_position = None
        self.app._target_label.hide()

    def frame_target_label(self):
        """Billboard label on the selected target."""
        if self.app._target is not None:
            target_now = self.app._target.state_at(self.app.clock.sim_time_s).r
            self.app._target_label_position = np.asarray(target_now).copy()
            tx, ty, tz = self.app.transform.to_render(target_now)
            self.app._target_label.node().set_text(f"TARGET: {self.app._target.name.upper()}")
            self.app._target_label.set_pos(tx, ty, tz + 9.0)
        else:
            self.app._target_label_position = None
            self.app._target_label.hide()

    def frame_aim(self, node):
        """Target panel: the status box + "when you arrive" ghost for the Moon/planets;
        today's live distance readout for Lagrange points and the Sun."""
        app = self.app
        target = app._target
        flyby = self._sync_flyby_row()
        if target is None:
            app._aim_status = None
            self._hide_ghost()
            if not flyby:
                app._target_panel.hide()
                return
            # Inside a planet's SOI with nothing targeted: the panel carries the flyby only.
            app._target_panel.show()
            app._target_title_text.setText("FLYBY")
            self._show_status(None)
            app._plan_transfer_btn.hide()
            app._clear_target_btn.hide()
            return
        app._target_panel.show()
        app._clear_target_btn.show()
        app._target_title_text.setText(f"TARGET  {target.name.upper()}")
        self._sync_plan_button(target)
        name = self.aim_target_name()
        if name is None:
            self._frame_live_distance(target)
            return
        estimate_m, estimate_pos = self._kepler_estimate(target, node)
        if node.magnitude_mps > 0.0:
            exact = app.maneuver_editor.exact_preview_current(node)
            encounters, approach = app._preview_encounters, app._preview_approach
        else:
            approach = app._traj_approach[0]
            exact = approach is not None and approach.body_name == name
            encounters = app._traj_encounters[0] or []
        from orbitsim.render.aim_status import aim_status
        status = aim_status(name, encounters, approach, exact, estimate_m)
        app._aim_status = status
        self._show_status(status)
        if exact:
            ghost_pos = None if approach is None else approach.body_point_m
        else:
            ghost_pos = estimate_pos
        self._place_ghost(name, ghost_pos, status, exact)

    def _sync_plan_button(self, target) -> None:
        """"Plan {name} transfer" for the current target; none for the Sun."""
        btn = self.app._plan_transfer_btn
        if target.name == "Sun":
            btn.hide()
            return
        label = f"Plan {target.name} transfer"
        if btn["text"] != label:            # re-setting text rebuilds the button geometry
            btn["text"] = label
        btn.show()

    def _sync_flyby_row(self) -> bool:
        """The in-SOI flyby readout as the panel's extra row; True when shown."""
        app = self.app
        flyby = getattr(app, "_encounter_line", "")
        if app._flyby_text.getText() != flyby:
            app._flyby_text.setText(flyby)
        bottom = self.FLYBY_Z - 0.035 if flyby else self.BOTTOM_Z
        size = app._target_panel["frameSize"]
        if size[2] != bottom:
            app._target_panel["frameSize"] = (size[0], size[1], bottom, size[3])
        return bool(flyby)

    def _frame_live_distance(self, target) -> None:
        """Lagrange point / Sun: live distance + relative speed in the status row, no
        status box or ghost."""
        from orbitsim.render.ui.theme import THEME
        app = self.app
        app._aim_status = None
        self._show_status(None)
        self._hide_ghost()
        v0 = app.world.vessels[0]
        here = target.state_at(app.clock.sim_time_s)
        dist = float(np.linalg.norm(v0.state.r - here.r))
        relsp = float(np.linalg.norm(v0.state.v - here.v))
        app._aim_status_text.setText(f"{fmt_dist(dist)} away \u00b7 rel {relsp:,.0f} m/s")
        app._aim_status_text["fg"] = THEME.text_muted

    def _kepler_estimate(self, target, node):
        """(separation [m], target position [m]) of the Earth-only Kepler closest
        approach, recomputed at most every 0.5 s; (None, None) where it isn't defined
        (planets are not Keplerian about Earth)."""
        app = self.app
        if not target.supports_closest_approach:
            return None, None
        import time as _time
        now_real = _time.monotonic()
        if app._ca is None or now_real - app._ca_recompute_t > 0.5:
            app._ca_recompute_t = now_real
            v0 = app.world.vessels[0]
            if app._node_epoch_s is not None and node.magnitude_mps > 0.0:
                base_epoch = app._node_epoch_s
                traj = apply_maneuver(v0.state, node)
            else:
                base_epoch = app.clock.sim_time_s
                traj = v0.state
            try:
                period = state_to_elements(traj).period_s
            except ValueError:
                period = 14.0 * 86400.0
            window = min(period, 14.0 * 86400.0)
            # Keplerian on purpose: a cheap planning estimate shown greyed out until the
            # exact N-body preview for this burn is in.
            app._ca = closest_approach(
                traj, target.state_at(base_epoch), window_s=window, coarse_samples=720)
            app._ca_traj = traj
            app._ca_abs_epoch = base_epoch + app._ca.t_ca_s
        return app._ca.separation_m, target.state_at(app._ca_abs_epoch).r

    def _show_status(self, status) -> None:
        app = self.app
        for kind, icon in app._aim_icon_nps.items():
            if status is not None and status.icon == kind:
                icon.set_color(*status.color)
                icon.show()
            else:
                icon.hide()
        if status is None:
            app._aim_status_text.setText("")
            return
        app._aim_status_text.setText(status.text)
        app._aim_status_text["fg"] = status.color

    def _place_ghost(self, name, world_pos, status, exact) -> None:
        """Translucent target sphere where the target will be at closest approach."""
        app = self.app
        if world_pos is None:
            self._hide_ghost()
            return
        if app._aim_ghost_np is None:
            from panda3d.core import TransparencyAttrib
            ghost = make_uv_sphere(1.0, 16, 24)
            ghost.reparent_to(app.render)
            ghost.set_light_off()
            ghost.set_transparency(TransparencyAttrib.M_alpha)
            ghost.set_depth_write(False)
            app._aim_ghost_np = ghost
        from orbitsim.core.bodies import MOON, PLANETS
        radius_m = next(b.radius_m for b in (MOON, *PLANETS) if b.name == name)
        scale = app.transform.scale_m_per_unit
        r, g, b, _a = status.color
        app._aim_ghost_np.set_color(r, g, b, 0.35 if exact else 0.18)
        app._aim_ghost_np.set_scale(max(radius_m / scale, self.GHOST_MIN_UNITS))
        rx, ry, rz = app.transform.to_render(world_pos)
        app._aim_ghost_np.set_pos(rx, ry, rz)
        app._aim_ghost_np.show()
        app._aim_ghost_position = np.asarray(world_pos, dtype=np.float64).copy()
        label = app._aim_ghost_label
        label.node().set_text(f"{name} when you arrive")
        label.node().set_text_color(r, g, b, 1.0)
        label.set_pos(rx, ry, rz + 9.0)

    def _hide_ghost(self) -> None:
        app = self.app
        if app._aim_ghost_np is not None:
            app._aim_ghost_np.hide()
        app._aim_ghost_position = None
        app._aim_ghost_label.hide()
