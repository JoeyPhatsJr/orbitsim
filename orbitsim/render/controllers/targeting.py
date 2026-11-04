"""Target selection (click-pick), intercept planning, closest-approach markers and readout."""
from orbitsim.core.constants import MU_SUN
from orbitsim.core.elements import state_to_elements
from orbitsim.core.maneuvers import apply_maneuver
from orbitsim.core.planets import A_EARTH
from orbitsim.core.planets import A_JUPITER
from orbitsim.core.planets import A_MARS
from orbitsim.core.planets import A_MERCURY
from orbitsim.core.planets import A_NEPTUNE
from orbitsim.core.planets import A_SATURN
from orbitsim.core.planets import A_URANUS
from orbitsim.core.planets import A_VENUS
from orbitsim.core.rendezvous import closest_approach
from orbitsim.render.geometry import make_uv_sphere
from orbitsim.render.hud.format import fmt_countdown
from orbitsim.render.hud.format import fmt_dist
import numpy as np


class Targeting:
    """Target selection (click-pick), intercept planning, closest-approach markers and readout. Stateless: all state lives on the OrbitApp it is given."""

    def __init__(self, app) -> None:
        self.app = app

    def _build_targeting(self):
        """Targetable bodies, CA labels, target label."""
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
        self.app._ca_recompute_t = 0.0
        self.app._ca = None
        self.app._ca_traj = None
        self.app._ca_abs_epoch = 0.0
        self.app._ca_marker_ship = None
        self.app._ca_marker_moon = None
        self.app._ca_label_positions = {"ship": None, "target": None}
        from orbitsim.render.world_labels import build_labeled_marker, build_world_label
        self.app._ca_labels = {
            "ship": build_world_label(
                self.app.render, "CA: SHIP", color=(1.0, 0.55, 0.25, 1.0), scale=11.0
            ),
            "target": build_world_label(
                self.app.render, "CA: TARGET", color=(1.0, 0.82, 0.30, 1.0), scale=11.0
            ),
        }
        for label in self.app._ca_labels.values():
            label.hide()
        self.app._target_label = build_world_label(
            self.app.render, "TARGET", color=(1.0, 0.35, 0.95, 1.0), scale=12.0
        )
        self.app._target_label.hide()
        self.app._target_label_position = None

    def _on_mouse1_down(self):
        mw = self.app.mouseWatcherNode
        self.app._mouse1_down_px = self._mouse_px() if (mw and mw.has_mouse()) else None

    def _mouse_px(self):
        """Current mouse position in pixels, or None if off-window."""
        mw = self.app.mouseWatcherNode
        if mw is None or not mw.has_mouse():
            return None
        w, h = self.app.win.get_x_size(), self.app.win.get_y_size()
        return ((mw.get_mouse_x() * 0.5 + 0.5) * w, (mw.get_mouse_y() * 0.5 + 0.5) * h)

    def _marker_px(self, world_r):
        """Project an inertial position to pixels, or None if behind the camera/off-lens."""
        from panda3d.core import Point2
        rp = self.app.transform.to_render(world_r)
        p = self.app.cam.get_relative_point(self.app.render, rp)
        proj = Point2()
        if not self.app.camLens.project(p, proj):
            return None
        w, h = self.app.win.get_x_size(), self.app.win.get_y_size()
        return ((proj.x * 0.5 + 0.5) * w, (proj.y * 0.5 + 0.5) * h)

    def _try_pick_target(self):
        """On a left-click tap, select a body or coast to a future trajectory point."""
        from orbitsim.render.picking import nearest_future_point, nearest_marker
        down = getattr(self.app, "_mouse1_down_px", None)
        self.app._mouse1_down_px = None
        if down is None:
            return
        click = self._mouse_px()
        if click is None:
            return
        if abs(click[0] - down[0]) > 6.0 or abs(click[1] - down[1]) > 6.0:
            return  # was a drag, not a tap
        now = self.app.clock.sim_time_s
        proj = [self._marker_px(t.state_at(now).r) for t in self.app._targets]
        idxs = [i for i, p in enumerate(proj) if p is not None]
        hit = nearest_marker(click, [proj[i] for i in idxs], tol_px=22.0)
        if hit is not None:
            self._clear_target()   # tear down any prior target's CA markers before switching
            self.app._target = self.app._targets[idxs[hit]]
            return

        points = self.app._traj_world_cache[0] if self.app._traj_world_cache else None
        epochs = self.app._traj_epoch_cache[0] if self.app._traj_epoch_cache else None
        if points is None or epochs is None or self.app.world.any_thrusting():
            return
        trajectory_px = [self._marker_px(point) for point in points]
        point_idx = nearest_future_point(click, trajectory_px, epochs, now, tol_px=16.0)
        if point_idx is None:
            return
        duration_s = float(epochs[point_idx] - now)
        vessel = self.app.world.vessels[0]
        self.app.clock.warp = 1.0
        self.app._operation_rollback = (
            vessel, vessel.state, vessel.orientation.copy(), self.app.clock.sim_time_s
        )
        self.app.operation.start(
            f"Coasting {fmt_countdown(duration_s)} along trajectory",
            self.app._trajectory_coast_operation(duration_s),
        )
        self.app.operation_overlay.update(self.app.operation.status)

    def _plan_intercept(self):
        """Auto-plan an intercept/departure of the current target.

        The Lambert grid search is the expensive part, so it runs on the planning
        thread (fanning cells across the process pool) instead of freezing the
        render thread. This method only builds the grids and submits the job;
        ``_poll_planning`` applies the resulting node when it completes.
        """
        import numpy as np
        from orbitsim.core.elements import state_to_elements
        from orbitsim.render.targets import PlanetTarget
        if self.app._target is None:
            self.app._flash_message("No target selected")
            return
        if self.app._planning_future is not None and not self.app._planning_future.done():
            self.app._flash_message("Planning already in progress...")
            return
        v0 = self.app.world.vessels[0]
        now = self.app.clock.sim_time_s
        grid_executor = self.app._grid_executor
        if isinstance(self.app._target, PlanetTarget):
            from orbitsim.core.optimize import interplanetary_departure_node_by_name
            from orbitsim.core.planets import A_EARTH, _N_EARTH
            from orbitsim.core.constants import MU_SUN
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
            dep = np.linspace(0.0, min(synodic, 2 * 365.25 * 86400.0), 24)
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
            dep = np.linspace(0.0, period, 24)
            tof = np.linspace(3.0e3, 14.0 * 86400.0, 48)
            ship_state = v0.state
            target_now = self.app._target.state_at(now)
            mu = self.app.world.central.mu

            def job():
                return intercept_node(
                    ship_state, target_now, mu, dep, tof, executor=grid_executor)

        self.app._flash_message(f"Planning {self.app._target.name} intercept...")
        self.app._planning_future = self.app._planning_executor.submit(job)

    def _apply_planned_node(self, node) -> None:
        """Install a freshly-planned maneuver node into the editor (render thread only)."""
        self.app._node_epoch_s = node.epoch_s
        self.app._auto_execute_node = False
        self.app._dv["pro"] = node.dv_prograde_mps
        self.app._dv["nrm"] = node.dv_normal_mps
        self.app._dv["rad"] = node.dv_radial_mps
        for axis in ("pro", "nrm", "rad"):
            self.app._dv_value_text[axis].setText(f"{self.app._dv[axis]:+.0f}")
        self.app._refresh_readout()
        self.app._flash_message(f"Intercept planned (dV {node.magnitude_mps:,.0f} m/s)")

    def _poll_planning(self) -> None:
        """Apply an intercept plan once its background job finishes."""
        future = getattr(self.app, "_planning_future", None)
        if future is None or not future.done():
            return
        self.app._planning_future = None
        try:
            node = future.result()
        except ValueError:
            self.app._flash_message("No intercept found")
            return
        except Exception:
            import traceback
            traceback.print_exc()
            self.app._flash_message("Planning failed")
            return
        self._apply_planned_node(node)

    def _clear_target(self):
        """Deselect the current target; remove its closest-approach markers + readout."""
        self.app._target = None
        for attr in ("_ca_marker_ship", "_ca_marker_moon"):
            np_ = getattr(self.app, attr, None)
            if np_ is not None:
                np_.remove_node()
                setattr(self.app, attr, None)
        self.app._ca = None
        self.app._target_label_position = None
        self.app._target_label.hide()
        for key, label in self.app._ca_labels.items():
            self.app._ca_label_positions[key] = None
            label.hide()
        self.app._target_line = ""
        self.app._sync_maneuver_hud()

    def _ca_marker(self, attr, color):
        """Lazily create/reuse a closest-approach marker NodePath."""
        np_ = getattr(self.app, attr, None)
        if np_ is None:
            np_ = make_uv_sphere(1.0, 8, 12)
            np_.reparent_to(self.app.render)
            np_.set_color(*color)
            np_.set_light_off()
            np_.set_scale(5.0)
            from panda3d.core import TransparencyAttrib
            np_.set_transparency(TransparencyAttrib.M_alpha)
            np_.set_depth_write(False)
            setattr(self.app, attr, np_)
        return np_

    def _frame_target_label(self):
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

    def _frame_closest_approach(self, node):
        """Closest-approach markers + target readout."""
        v0 = self.app.world.vessels[0]
        # Closest approach to the current target (throttled recompute). Both the ship
        # trajectory and the target are referenced to the same base epoch (the node epoch
        # when a burn is planned, else now) so they are compared at matching absolute times;
        # the absolute CA epoch is cached so the markers hold steady between recomputes (warp).
        if self.app._target is not None and not self.app._target.supports_closest_approach:
            # Lagrange-point target: live distance + relative speed, no closest-approach
            # prediction (an L-point is not Keplerian — you fly to it and null relative velocity).
            self.app._ca = None
            for key, label in self.app._ca_labels.items():
                self.app._ca_label_positions[key] = None
                label.hide()
            L = self.app._target.state_at(self.app.clock.sim_time_s)
            dist = float(np.linalg.norm(v0.state.r - L.r))
            relsp = float(np.linalg.norm(v0.state.v - L.v))
            self.app._target_line = (
                f"Target: {self.app._target.name}   dist {fmt_dist(dist)}"
                f"   rel {relsp:,.0f} m/s"
            )
            self.app._sync_maneuver_hud()
        elif self.app._target is not None:
            import time as _time
            now_real = _time.monotonic()
            if self.app._ca is None or now_real - self.app._ca_recompute_t > 0.5:
                self.app._ca_recompute_t = now_real
                if self.app._node_epoch_s is not None and node.magnitude_mps > 0.0:
                    base_epoch = self.app._node_epoch_s
                    traj = apply_maneuver(v0.state, node)
                else:
                    base_epoch = self.app.clock.sim_time_s
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
                self.app._ca = closest_approach(
                    traj, self.app._target.state_at(base_epoch), window_s=window, coarse_samples=720)
                self.app._ca_traj = traj
                self.app._ca_abs_epoch = base_epoch + self.app._ca.t_ca_s
            ca = self.app._ca
            from orbitsim.core.propagate import propagate_kepler
            ship_at = propagate_kepler(self.app._ca_traj, ca.t_ca_s).r
            tgt_at = self.app._target.state_at(self.app._ca_abs_epoch).r
            self._ca_marker("_ca_marker_ship", (1.0, 0.5, 0.2, 1.0)).set_pos(
                *self.app.transform.to_render(ship_at))
            self._ca_marker("_ca_marker_moon", (1.0, 0.8, 0.3, 1.0)).set_pos(
                *self.app.transform.to_render(tgt_at))
            self.app._ca_label_positions["ship"] = np.asarray(ship_at).copy()
            self.app._ca_label_positions["target"] = np.asarray(tgt_at).copy()
            for key, world_pos in self.app._ca_label_positions.items():
                rx, ry, rz = self.app.transform.to_render(world_pos)
                label = self.app._ca_labels[key]
                label.node().set_text(
                    "CA: SHIP" if key == "ship" else f"CA: {self.app._target.name.upper()}"
                )
                label.set_pos(rx, ry, rz + 8.0)
            countdown = max(0.0, self.app._ca_abs_epoch - self.app.clock.sim_time_s)
            self.app._target_line = (
                f"Target: {self.app._target.name}   CA T-{fmt_countdown(countdown)}"
                f"   sep {ca.separation_m / 1000:,.0f} km   rel {ca.rel_speed_mps:,.0f} m/s"
            )
            self.app._sync_maneuver_hud()
